"""The SDK arm's loop agents: diagnose, rank failure classes, rewrite ONE prompt.

The pipeline arm (`teacher.py`) has four prompts and its teacher rewrites one of
them per experiment. The single-session runtime (`backends/agent_sdk.py`) has
**one** prompt, so the loop that improves it reads and writes differently:

- *Diagnose* — the same first-wrong-per-conversation cases, the same gold-derived
  first fault (`stage_scores.first_fault`) computed before any model is asked,
  the same frozen failure taxonomy. What differs is the record the agent reads:
  the qa_agent's own reported trail (turn type, sub-questions, symbolic program,
  retrieved values with their sources) and the calculator tool trajectory, so
  the diagnosis can say *no tool was called* or *the answer matches no tool
  return* — failures the pipeline cannot have.
- *Rank* — failure classes, not agents, are the unit of targeting. Every ledger
  row filed against the exact prompt text is pooled (keyed on the sdk prompt
  hash, so SDK draws never pool with pipeline draws) and classes are ranked on
  the Wilson lower bound of their share of attributed cases — `ledger._score`'s
  formula, not a second one.
- *Rewrite* — the prompt is split by its seven stable headings and the teacher
  agent returns one **edit per failure class** it addresses, each naming the
  section it changes and the diagnoses behind it. Edits replace section bodies;
  the whole-prompt diff and a per-edit hunk are both recorded, one rewrites-ledger
  row per edit sharing a `rewrite_id`. `max_areas=1` is the campaign's fallback
  after two consecutive rejections: attribution is the cost of editing several
  areas at once, and the campaign pays it only while it is winning.
- *Distil* (D8) — the first SDK prompt is drafted by the teacher agent from the
  four pipeline prompts plus the dataset notes and the output contract, with the
  train of thought as its spine and the inter-agent plumbing left behind.

Every model call goes through `evalloop/sdk.py::run_structured` with prompt
references; generated modules (`prompts/sdk_vN.py`) are never hand-edited; runs
log to the optimisation experiment with the `runtime=agent_sdk` tag.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from convfinqa.config import REPO_ROOT
from convfinqa.evalloop import ledgers, prompt_refs, teacher
from convfinqa.evalloop.teacher import (
    AGENTS,
    DIAGNOSTICS_DIR,
    MEMORY_ARTIFACT,
    OPTIMIZATION_EXPERIMENT,
    WRITER_PROMPT_ARTIFACT,
)
from convfinqa.tracking import tracing

RUNTIME = "agent_sdk"

#: Where generated `sdk_vN.py` modules are written. A module attribute so tests
#: can point it at a temp directory and never touch the committed package.
PROMPTS_DIR = REPO_ROOT / "src" / "convfinqa" / "prompts"

#: The seven section headings every SDK prompt carries, in order. They are the
#: `target` areas the teacher edits by, so `propose_version` can split a prompt
#: into sections and rejoin it after replacing some of them.
HEADINGS: tuple[str, ...] = (
    "## 1. Role",
    "## 2. Train of thought",
    "## 3. Triage",
    "## 4. Preprocess",
    "## 5. Retrieve",
    "## 6. Calculate",
    "## 7. Output contract",
)

#: Which pipeline stage a section stands for, for reading gate flips filed by
#: first-fault stage against an edit filed by section.
SECTION_STAGE: dict[str, str] = {
    "## 3. Triage": "triage",
    "## 4. Preprocess": "preprocess",
    "## 5. Retrieve": "retriever",
    "## 6. Calculate": "calculator",
}

#: The tool names and output keys a valid SDK prompt must mention.
TOOL_NAMES: tuple[str, ...] = tuple(
    f"mcp__cfq__{name}"
    for name in ("add", "subtract", "multiply", "divide", "exp", "greater")
)
OUTPUT_KEYS: tuple[str, ...] = (
    "turn_type",
    "conv_type",
    "sub_questions",
    "program",
    "retrieved",
    "answer",
)
PLUMBING_MARKER = "[[ ##"
SDK_MIN_PROMPT_CHARS = 400

DISTIL_PROMPT_ARTIFACT = "distil_prompt.txt"
CHANGES_ARTIFACT = "changes.json"

_HEADING_RE = re.compile(r"^## \d\. .+$", re.MULTILINE)


# ── The frozen taxonomy, reused rather than copied ────────────────────────


def taxonomy_text() -> str:
    """The failure taxonomy block of `teacher.TEACHER_PROMPT`, verbatim.

    Sliced out of the pipeline teacher's prompt rather than copied so the two
    arms cannot drift apart: a label added there is a label here.
    """
    text = teacher.TEACHER_PROMPT
    start = text.index("Use one of these failure modes")
    end = text.index("Then propose ONE targeted rule")
    block = text[start:end].strip()
    if not block:
        raise RuntimeError("the taxonomy block of TEACHER_PROMPT is empty")
    return block


TAXONOMY = taxonomy_text()


# ── Schemas ───────────────────────────────────────────────────────────────


class SdkDiagnosis(BaseModel):
    """One first-wrong question of an SDK run, attributed and explained.

    Same fields as `teacher.Diagnosis` under the ledger's names (`stage`,
    `label`, `fix_hint`), plus nothing: the SDK arm files into the same
    diagnoses ledger and the same taxonomy.
    """

    stage: Literal["triage", "preprocess", "retriever", "calculator"] = Field(
        description="The FIRST stage of the trail that broke"
    )
    label: str = Field(
        description="A failure mode from the frozen taxonomy, or new:<label>"
    )
    what_went_wrong: str = Field(
        description="2-4 sentences: the first mistake and how it produced the wrong answer"
    )
    evidence: str = Field(
        description="Quoted from the document, the history or the trail"
    )
    attribution_reason: str = Field(
        description=(
            "Why this stage and not the one before it. Say so explicitly if "
            "you contradict derived_attribution."
        )
    )
    fix_hint: str = Field(
        description="ONE imperative, general rule for the prompt that would have prevented this"
    )
    gold_suspect: bool = Field(
        description="True if the gold answer itself looks wrong or ambiguous"
    )
    confidence: float = Field(ge=0.0, le=1.0)


class SdkEdit(BaseModel):
    """One section-level change, for one failure class."""

    target: str = Field(description="One of the seven section headings")
    failure_class: str = Field(description="The taxonomy label this edit addresses")
    change_kind: Literal["rule", "example", "criterion", "removal", "reorder"]
    diagnosis_ids: list[str] = Field(default_factory=list)
    edit_text: str = Field(description="One or two sentences naming the change")
    rationale: str = Field(description="Why this change should fix the class")
    new_section_body: str = Field(
        description="The COMPLETE new body of the target section, heading excluded"
    )


class SdkRewrite(BaseModel):
    """The teacher agent's reply: one edit per failure class it addresses."""

    edits: list[SdkEdit]
    summary: str = Field(description="One paragraph on what changed and why")


class PromptDraft(BaseModel):
    """The distillation agent's reply: the first SDK prompt."""

    prompt: str = Field(description="The complete SDK system prompt")
    sections: list[str] = Field(description="The headings, in order")
    dropped: list[str] = Field(
        description="What was in the source prompts and deliberately left out"
    )
    notes: str = Field(description="Judgement calls made while distilling")


# ── Prompts ───────────────────────────────────────────────────────────────

TRAIN_OF_THOUGHT_SPEC = """The SDK prompt has exactly these seven sections, with these
exact markdown headings in this order (the loop edits the prompt by section, so the
headings are load-bearing):

## 1. Role
  The job: answer a sequence of questions about ONE financial report (table + text)
  in one session, where later questions depend on earlier turns. The two turn
  types: a NUMBER turn reads one value straight from the report; a PROGRAM turn
  computes something from values in the report and/or earlier answers.

## 2. Train of thought
  number: triage -> retrieve. program: triage -> preprocess -> retrieve -> calculate.
  Every turn reports the trail it took, even when a stage was trivial.

## 3. Triage
  The turn-type and conversation-type criteria, distilled from the triage prompt.

## 4. Preprocess
  How to turn a program question into sub-questions and a SYMBOLIC program whose
  placeholders (A, B, ...) bind to the sub-questions in order; the rules for
  resolving references to prior turns ("that", "this change", "the difference");
  which values come from history rather than the document.

## 5. Retrieve
  The retrieval conventions: units and scale, signs, table versus text, period
  and column alignment, whole-label matching, totals versus components.

## 6. Calculate
  ALWAYS use the mcp__cfq__add, mcp__cfq__subtract, mcp__cfq__multiply,
  mcp__cfq__divide, mcp__cfq__exp and mcp__cfq__greater tools for EVERY arithmetic
  step; never do arithmetic in the model's head. The final answer must be a value
  a tool returned (or a value read from the report on a number turn). Program
  notation matches the gold DSL: add/subtract/multiply/divide/exp/greater, #0/#1
  references to earlier steps, constants like const_100.

## 7. Output contract
  The structured result: turn_type, conv_type, sub_questions ([] on a number
  turn), program ("" on a number turn), retrieved ([{question, answer, source}]),
  answer. Plain numeric answers with no units or symbols; the percent-formatting
  rules carried over from the source prompts."""


SDK_DISTIL_PROMPT = """You are writing the system prompt for a single Claude session
that answers ConvFinQA conversations: a sequence of questions about one financial
report, where later questions depend on earlier answers. The session sees the
whole report in its first message and every later question as one more message,
so it has the conversation history in context and needs no plumbing to pass it.

You are given the four prompts of the multi-agent pipeline this session replaces
(triage, preprocess, retriever, calculator), the dataset notes, the JSON schema of
the structured reply the session must return, and the required section layout.

Distil, do not copy. The four prompts total tens of thousands of characters and
were written for four STATELESS agents passing `[[ ## name ## ]]` blocks and
"Field guidance" to each other. Keep the knowledge — the turn-type criteria, the
reference-resolution rules for prior turns, the retrieval conventions (units,
signs, table versus text, period alignment, label matching), the calculator
discipline, the answer-formatting rules — and drop the plumbing: input-field
descriptions, inter-agent hand-off formats, anything that only makes sense for an
agent that cannot see the conversation.

Hard constraints:
- Use EXACTLY the seven headings given, in that order, as markdown `## N. Name`
  lines. Nothing else may be a `## ` heading.
- Never write `[[ ##` anywhere.
- Name all six calculator tools in the Calculate section and require them for
  every arithmetic step.
- Name every key of the output schema in the Output contract section.
- Stay general: no company, year or value from any example.
- Aim for a prompt a careful analyst could follow: dense, imperative, ordered.

Return JSON matching the schema: the prompt, the headings you used, what you
dropped, and your notes."""


SDK_DIAGNOSE_PROMPT = f"""You diagnose ONE wrong answer from a single-session financial
Q&A agent. The agent answers a conversation about one financial report in one
session; for every turn it reports a trail: the turn type it chose (number or
program), the sub-questions and symbolic program it planned, the values it
retrieved with their sources, and its answer. Arithmetic must go through six
calculator tools, and every tool call is in the calculator trajectory you are
shown. This is always the FIRST wrong turn of its conversation, so the mistake
originated here, not upstream.

Step 1 has already been done deterministically: `derived_attribution` names the
first stage whose gold-derived check failed (`derived_checks` holds the checks,
`missing_gold_operands` the gold values the trail never retrieved). Treat it as
the default. Set `stage` to something else ONLY when you can point at evidence
that the check misread the case, and say so explicitly in `attribution_reason`.
If the gold answer itself looks wrong, still attribute the divergence, set
gold_suspect=true and lower your confidence.

Step 2 is you. Read the report (table and text), the history, the question, the
trail and the trajectory, and explain the first break, stage by stage:
- triage: what in the wording or the prior turn made it pick the wrong turn type
  or conversation type.
- preprocess: which operand was missing or extra; which reference ("that", "this
  change", "the difference") resolved to the wrong prior answer; whether the
  operator or operand order was wrong. An EMPTY program on a program turn is a
  preprocess failure.
- retriever: the cell or sentence that should have been used and the one that
  was; the unit, scale, sign or period confusion that explains the gap.
- calculator: NO tool call on a program turn is a calculator failure; so is
  inline arithmetic (an answer matching no tool return), a step of the program
  skipped, rounding or percent formatting, or a tool result not carried into the
  answer.

A later stage that faithfully consumed an earlier mistake did not fail. Compare
against the gold program step by step to locate the divergence.

{TAXONOMY}

`evidence` must quote the document, the history or the trail. `fix_hint` is ONE
imperative, general rule for the session's prompt — not about this company or
this year — that would have prevented the failure; if prior diagnoses are
provided, do not repeat a hint already given, sharpen or extend it."""


SDK_WRITER_PROMPT = """You maintain the system prompt of a single-session financial
Q&A agent. The prompt is split into seven sections with fixed headings. You are
given the current text of every section, every failure diagnosed in the run that
just ran grouped by failure class, the pooled ranking of those classes over every
run that used this exact prompt (Wilson lower bound of each class's share of
attributed failures — higher means better evidenced), the evidence and fix hints
behind each, and the history of every previous edit to this prompt lineage with
what the gate said about it.

Return a list of EDITS, one per failure class you decide to address, most
important first. Each edit names ONE section (`target` must be one of the seven
headings exactly), the class it addresses, the diagnosis ids behind it, the kind
of change (rule, example, criterion, removal, reorder), one or two sentences
naming the change, a rationale, and the COMPLETE new body of that section. Two
classes that need the same section must be merged into one edit — a section may
appear once in your list.

Hard constraints, all load-bearing:
- Preserve the output contract: every output key, the six tool names and the
  program notation the current prompt requires must still be required.
- Stay general. Never mention a specific company, year or value.
- Change only the sections you name; return each named section whole.
- Respect `max_areas`: return at most that many edits, taking the classes in
  ranked order. When it is 1, the campaign has had two rejections in a row and
  wants one well-evidenced change it can attribute.
- Read the attempt history. An edit whose class got WORSE after a past attempt
  is flagged "revert or rethink": do not sharpen the same idea, change approach
  or undo it. Do not re-propose a change that was REJECTED unless you can say
  what is different this time.
- Weigh what past edits BROKE as heavily as what they fixed. Prefer a narrow
  rule with a stated precondition over a broad instruction that changes
  behaviour on turns that were never failing.
- A class seen once in one run may be noise; a class recurring in the pooled
  record is the real target.

Return JSON matching the schema: the edits and a one-paragraph summary."""


# ── Section mechanics ─────────────────────────────────────────────────────


def split_sections(prompt: str) -> dict[str, str]:
    """``{heading: body}`` in order of appearance, preamble (if any) under ``""``.

    A body is the raw text between its heading line and the next heading, so
    `join_sections(split_sections(p)) == p` byte for byte. Any `## N. Name`
    line is a section boundary; `validate_sdk_prompt` is what insists on the
    seven canonical ones.
    """
    matches = list(_HEADING_RE.finditer(prompt))
    out: dict[str, str] = {}
    preamble = prompt[: matches[0].start()] if matches else prompt
    if preamble:
        out[""] = preamble
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        heading = m.group(0).strip()
        if heading in out:
            raise ValueError(f"heading {heading!r} appears more than once")
        out[heading] = prompt[m.end() : end]
    return out


def join_sections(sections: dict[str, str]) -> str:
    """The inverse of `split_sections`."""
    parts: list[str] = []
    for heading, body in sections.items():
        parts.append(body if heading == "" else heading + body)
    return "".join(parts)


def normalise_target(target: str) -> str:
    """The canonical heading for a target the writer named, or raise.

    Accepts the exact heading, the heading without the `## ` prefix, the
    number, or the bare name (case-insensitive) — and nothing else.
    """
    wanted = target.strip().lower()
    for heading in HEADINGS:
        number, _, name = heading.removeprefix("## ").partition(". ")
        if wanted in {
            heading.lower(),
            heading.removeprefix("## ").lower(),
            number,
            name.lower(),
        }:
            return heading
    raise ValueError(f"unknown target {target!r}; expected one of {HEADINGS}")


def _section_body(text: str) -> str:
    """A writer-supplied body normalised to the layout the split expects."""
    return "\n" + text.strip("\n") + "\n\n"


def replace_section(prompt: str, heading: str, body: str) -> str:
    """`prompt` with one section's body replaced, everything else untouched."""
    sections = split_sections(prompt)
    if heading not in sections:
        raise ValueError(f"the prompt has no section {heading!r}")
    sections[heading] = _section_body(body)
    return join_sections(sections)


def validate_sdk_prompt(before: str, after: str) -> list[str]:
    """Reasons `after` must not be written to disk. Empty means it may."""
    problems: list[str] = []
    stripped = after.strip()
    if len(stripped) < SDK_MIN_PROMPT_CHARS:
        problems.append(
            f"the prompt is {len(stripped)} characters — under the "
            f"{SDK_MIN_PROMPT_CHARS}-character floor, which is what a collapsed "
            "prompt looks like"
        )
    found = [m.group(0).strip() for m in _HEADING_RE.finditer(after)]
    for heading in HEADINGS:
        n = found.count(heading)
        if n == 0:
            problems.append(f"missing section heading {heading!r}")
        elif n > 1:
            problems.append(f"section heading {heading!r} appears {n} times")
    for extra in sorted(set(found) - set(HEADINGS)):
        problems.append(f"unexpected section heading {extra!r}")
    canonical = [h for h in found if h in HEADINGS]
    if canonical != [h for h in HEADINGS if h in canonical]:
        problems.append("the section headings are out of order")
    for tool in TOOL_NAMES:
        if tool not in after:
            problems.append(f"the calculator tool {tool!r} is not mentioned")
    for key in OUTPUT_KEYS:
        if key not in after:
            problems.append(f"the output key {key!r} is not mentioned")
    if PLUMBING_MARKER in after:
        problems.append(
            f"the prompt contains {PLUMBING_MARKER!r} inter-agent plumbing, "
            "which a single session has no use for"
        )
    if before:
        for token in ("const_", "#0"):
            if token in before and token not in after:
                problems.append(
                    f"the current prompt requires {token!r} program notation and "
                    "the rewrite does not mention it"
                )
    return problems


# ── The generated module ──────────────────────────────────────────────────


def _write_sdk_module(new_version: str, *, prompt: str, header: str) -> Path:
    """Write `prompts/<new_version>.py` exporting `SDK_PROMPT`. Refuses to overwrite."""
    import convfinqa.prompts as prompts_pkg

    if not prompts_pkg.is_sdk_version(new_version):
        raise SystemExit(f"{new_version!r} is not an sdk_vN version name")
    literal = prompt.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if literal.endswith('"'):
        literal += "\\n"
    body = f'''"""Generated by convfinqa.evalloop.sdk_teacher — do not hand-edit.

{header}
"""

__all__ = ["{prompts_pkg.SDK_VAR}"]

{prompts_pkg.SDK_VAR} = """{literal}"""
'''
    path = PROMPTS_DIR / f"{new_version}.py"
    if path.exists():
        raise SystemExit(f"{path} already exists — pick a new version name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    import importlib

    importlib.invalidate_caches()
    return path


def sdk_prompt_hash(version: str) -> str:
    """The content hash of an `sdk_vN` prompt, or "" when it cannot be loaded."""
    try:
        import convfinqa.prompts as prompts_pkg
        from convfinqa.tracking.prompt_ledger import prompt_hash

        return prompt_hash(prompts_pkg.load_sdk(version))
    except Exception:  # noqa: BLE001 — an unloadable version is an empty cell
        return ""


# ── The case payload ──────────────────────────────────────────────────────


def _io(row: Any, stage: str) -> dict[str, Any]:
    raw = ledgers._get(row, f"{stage}_io")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def retrieved_with_sources(row: Any) -> list[dict[str, str]]:
    """The trail's retrieved values with the source each came from."""
    out = _io(row, "retriever").get("output") or {}
    answers = out.get("answers") or []
    sources = out.get("sources") or []
    items: list[dict[str, str]] = []
    for i, a in enumerate(answers):
        if not isinstance(a, dict):
            continue
        items.append(
            {
                "question": str(a.get("question", "")),
                "answer": str(a.get("answer", "")),
                "source": str(sources[i]) if i < len(sources) else "",
            }
        )
    return items


def sdk_flags(row: Any) -> dict[str, Any]:
    """`stage_skips`, `inline_arithmetic` and `tool_calls` for one turn.

    Read from an ``sdk_io`` column when the CSV carries one; otherwise derived
    from the stage captures by the same rules `result_to_capture` applies, so a
    CSV written before the column existed reads the same.
    """
    recorded = _io(row, "sdk")
    if recorded:
        return {
            "stage_skips": list(recorded.get("stage_skips") or []),
            "inline_arithmetic": bool(recorded.get("inline_arithmetic", False)),
            "tool_calls": int(recorded.get("tool_calls") or 0),
        }
    from convfinqa.backends.agent_sdk import _inline_arithmetic

    trajectory = ledgers._calc_trajectory(row)
    tool_calls = sum(1 for e in trajectory if e.get("event") == "tool_call")
    skips: list[str] = []
    inline = False
    if str(ledgers._get(row, "pred_turn_type", "") or "").lower() == "program":
        program = str(ledgers._get(row, "pred_program", "") or "")
        if not program.strip():
            skips.append("preprocess")
        if not retrieved_with_sources(row):
            skips.append("retriever")
        if tool_calls == 0:
            skips.append("calculator")
        inline = _inline_arithmetic(
            str(ledgers._get(row, "pred_answer", "") or ""), trajectory
        )
    return {"stage_skips": skips, "inline_arithmetic": inline, "tool_calls": tool_calls}


def _report_for(report_id: str) -> Any:
    from convfinqa.evalloop import stage_scores

    try:
        raw = stage_scores.report_documents().get(report_id, "")
    except Exception:  # noqa: BLE001 — no dataset on disk is a missing document
        raw = ""
    if not raw:
        return "(document not available)"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def sdk_case_payload(row: Any, doc: Any = None) -> dict[str, Any]:
    """Everything the diagnosis agent needs about one SDK case.

    The deterministic step first (attribution, checks, missing operands), then
    the document, the history, the question, the reported trail and the tool
    trajectory. `doc` is the report (table + text); looked up when omitted.
    """
    from convfinqa.evalloop import stage_scores

    get = ledgers._get
    retrieved = retrieved_with_sources(row)
    return {
        "derived_attribution": stage_scores.attribute(row),
        "derived_checks": {
            "triage_turn_type_ok": get(row, "triage_turn_type_ok"),
            "preprocess_skeleton_ok": get(row, "preprocess_skeleton_ok"),
            "preprocess_plan_ok": get(row, "preprocess_plan_ok"),
            "retriever_operand_recall": get(row, "retriever_operand_recall"),
            "calculator_ok": get(row, "calculator_ok"),
        },
        "missing_gold_operands": stage_scores.missing_operands(
            row, [r["answer"] for r in retrieved]
        ),
        "sdk_flags": sdk_flags(row),
        "report_id": get(row, "report_id", ""),
        "report": doc if doc is not None else _report_for(str(get(row, "report_id"))),
        "conversation_history": get(row, "history_text") or "(no prior turns)",
        "question": get(row, "question", ""),
        "gold_turn_type": get(row, "gold_turn_type", ""),
        "gold_answer": get(row, "gold_answer", ""),
        "gold_program": get(row, "gold_program") or "(number selection — no program)",
        "trail": {
            "turn_type": get(row, "pred_turn_type", ""),
            "conv_type": get(row, "pred_conv_type", ""),
            "sub_questions": stage_scores.planned_sub_questions(row),
            "program": get(row, "pred_program") or "",
            "retrieved": retrieved,
            "answer": get(row, "pred_answer", ""),
        },
        "calculator_trajectory": ledgers._calc_trajectory(row),
    }


def diagnose_prompt_text(payload: dict[str, Any], memory_text: str) -> str:
    """The exact user prompt a diagnosis call sends (same builder as teacher's)."""
    return teacher.diagnose_prompt_text(payload, memory_text)


async def _diagnose_case(
    payload: dict[str, Any],
    memory_text: str,
    refs: dict[str, Any] | None,
) -> tuple[SdkDiagnosis, dict[str, Any]]:
    """One diagnosis on the Agent SDK. `refs` is required — see `run_structured`."""
    from convfinqa.evalloop.sdk import run_structured

    return await run_structured(
        diagnose_prompt_text(payload, memory_text),
        schema=SdkDiagnosis,
        system_prompt=SDK_DIAGNOSE_PROMPT,
        max_turns=4,
        refs=refs,
    )


# ── Memory ────────────────────────────────────────────────────────────────


def sdk_prior_diagnoses(
    experiment: str = OPTIMIZATION_EXPERIMENT, limit: int = 40
) -> list[dict[str, Any]]:
    """Earlier SDK-arm diagnoses, newest first: the ledger, else MLflow.

    Filtered to ``runtime="agent_sdk"`` so the memory a diagnosis reads is the
    memory of the prompt lineage it is diagnosing, not the pipeline's.
    """
    try:
        table = ledgers.load("diagnoses", runtime=RUNTIME)
    except Exception:  # noqa: BLE001 — an unreadable ledger falls back to the store
        table = pd.DataFrame()
    if len(table):
        ordered = table.sort_values("diagnosed_at", ascending=False).head(limit)
        return [
            {
                "version": str(r.version),
                "report_id": str(r.report_id),
                "stage": str(r.stage),
                "label": str(r.label),
                "fix_hint": str(r.fix_hint),
            }
            for r in ordered.itertuples()
        ]
    try:
        from convfinqa.evalloop import ledger as mem

        client = mem._client()
        runs = mem._runs(client, experiment, "sdk_diagnose", limit=5)
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    for run in runs:
        try:
            local = client.download_artifacts(run.info.run_id, "diagnoses.jsonl")
            for line in Path(local).read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                out.append(
                    {
                        "version": str(d.get("version", "")),
                        "report_id": str(d.get("report_id", "")),
                        "stage": str(d.get("stage") or d.get("failed_agent", "")),
                        "label": str(d.get("label") or d.get("failure_mode", "")),
                        "fix_hint": str(
                            d.get("fix_hint") or d.get("proposed_rule", "")
                        ),
                    }
                )
        except Exception:  # noqa: BLE001 — one unreadable run must not block the pass
            continue
    return out[:limit]


def memory_text_from(memory: list[dict[str, Any]]) -> str:
    """The memory block appended to every diagnosis prompt of a pass."""
    if not memory:
        return ""
    lines = [
        f"- [{m['version']}] {m['stage']}/{m['label']}: {m['fix_hint']}" for m in memory
    ]
    return "\n\nPrior diagnoses (do not repeat these hints):\n" + "\n".join(lines)


# ── diagnose_run ──────────────────────────────────────────────────────────


def _kappa(diagnoses: list[dict[str, Any]]) -> float | None:
    """Cohen's κ between the agent's stage and the derived stage, both naming an agent."""
    from convfinqa.evalloop.kappa import cohens_kappa

    pairs = [
        (str(d["derived_agent"]), str(d["stage"]))
        for d in diagnoses
        if d.get("derived_agent") in AGENTS and d.get("stage") in AGENTS
    ]
    if not pairs:
        return None
    return round(cohens_kappa([a for a, _ in pairs], [b for _, b in pairs]), 4)


async def diagnose_run(
    csv_path: Path | str,
    version: str,
    *,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    concurrency: int = 8,
    campaign: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Diagnose every first-wrong case of one SDK eval run; append to the ledger.

    Cases run concurrently under one semaphore and are reassembled in case
    order, exactly as the pipeline teacher does; ambiguous attributions are
    settled by the same binary adjudicator first. Rows go to the diagnoses
    ledger with ``runtime="agent_sdk"`` and the sdk prompt hash, and a per-run
    JSONL is still written for the readers that expect one.
    """
    from convfinqa.evalloop import stage_scores
    from convfinqa.tracking import mlflow_log

    cases = teacher.first_wrong_cases(csv_path)
    memory = sdk_prior_diagnoses(experiment)
    memory_text = memory_text_from(memory)
    prompt_hash = sdk_prompt_hash(version)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sdk-diagnose-{version}-{stamp}"
    tracing.enable()

    diagnoses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
    with mlflow_log.run(
        run_name,
        kind="sdk_diagnose",
        version=version,
        params={
            "runtime": RUNTIME,
            "source_csv": str(csv_path),
            "sdk_prompt_hash": prompt_hash,
            "n_cases": len(cases),
            "n_prior_diagnoses": len(memory),
            "concurrency": concurrency,
            **({"campaign": campaign} if campaign else {}),
            **({"experiment_label": label} if label else {}),
        },
        tags={
            "loop": "evalloop",
            "stage": "diagnose",
            "runtime": RUNTIME,
            **({"campaign": campaign} if campaign else {}),
        },
        experiment=experiment,
        actor_model=teacher.teacher_model(),
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        if memory_text:
            rec.text_artifact(MEMORY_ARTIFACT, memory_text)
        adjudicated = await teacher.resolve_ambiguous(
            cases, csv_path, concurrency=concurrency
        )
        if adjudicated:
            print(  # noqa: T201
                f"  adjudicated {len(adjudicated)} ambiguous case(s): "
                + ", ".join(v["agent"] for v in adjudicated.values())
            )
        sem = asyncio.Semaphore(max(1, concurrency))
        case_rows = list(cases.iterrows())

        async def one(
            order: int, row: pd.Series
        ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
            payload = sdk_case_payload(row)
            derived = str(payload["derived_attribution"])
            settled = adjudicated.get(order)
            if settled is not None:
                derived = str(settled["agent"])
                payload["derived_attribution"] = derived
                payload["adjudication"] = settled
            async with sem:
                with tracing.span(
                    f"sdk-diagnose {row.report_id} q{int(row.turn_index)}",
                    span_type="AGENT",
                    attributes={
                        "runtime": RUNTIME,
                        "report_id": row.report_id,
                        "turn_index": int(row.turn_index),
                        "question": str(row.question),
                        "gold_answer": str(row.gold_answer),
                        "sdk_answer": str(row.pred_answer),
                        "derived_attribution": derived,
                    },
                    trace_tags={
                        "model_version_id": version,
                        "run_name": run_name,
                        "stage": "diagnose",
                        "runtime": RUNTIME,
                    },
                ) as span:
                    try:
                        output, usage = await _diagnose_case(
                            payload,
                            memory_text,
                            {
                                "system_prompt": prompt_refs.teacher_prompt_ref(
                                    "SDK_DIAGNOSE_PROMPT", SDK_DIAGNOSE_PROMPT
                                ),
                                "user_prompt": prompt_refs.diagnose_case_ref(
                                    str(csv_path),
                                    str(row.report_id),
                                    int(row.turn_index),
                                    memory=MEMORY_ARTIFACT if memory_text else "",
                                    text=diagnose_prompt_text(payload, memory_text),
                                    runtime=RUNTIME,
                                ),
                            },
                        )
                    except Exception as exc:  # noqa: BLE001 — one bad case must not sink the pass
                        return (
                            order,
                            None,
                            {"report_id": row.report_id, "error": repr(exc)},
                            {},
                        )
                    span.set(
                        stage=output.stage,
                        label=output.label,
                        reason=output.attribution_reason,
                        what_went_wrong=output.what_went_wrong,
                        evidence=output.evidence,
                        fix_hint=output.fix_hint,
                        confidence=float(output.confidence),
                        gold_suspect=bool(output.gold_suspect),
                        attribution_disputed=output.stage != derived,
                    )
            d = output.model_dump()
            d.update(
                # The pipeline teacher's names too, so `kappa.make_sheet` and
                # the MLflow memory read an SDK pass without a second code path.
                failed_agent=d["stage"],
                failure_mode=d["label"],
                proposed_rule=d["fix_hint"],
                runtime=RUNTIME,
                report_id=row.report_id,
                question_id=row.get("question_id", ""),
                turn_index=int(row.turn_index),
                version=version,
                derived_agent=derived,
                adjudicated=settled is not None,
                adjudication_reason=(settled or {}).get("reason", ""),
                attribution_disputed=d["stage"] != derived,
            )
            return order, d, None, usage

        settled_all = await asyncio.gather(
            *(one(i, row) for i, (_, row) in enumerate(case_rows))
        )
        ledger_inputs: list[tuple[dict[str, Any], pd.Series, dict[str, Any]]] = []
        for order, d, failure, usage in sorted(settled_all, key=lambda r: r[0]):
            if failure is not None:
                failures.append(failure)
                print(f"  [skip] {failure['report_id']}: {failure['error']}")  # noqa: T201
                continue
            assert d is not None
            teacher._accumulate_usage(usage_total, usage)
            d["diagnosis_id"] = ledgers.new_id("d")
            diagnoses.append(d)
            ledger_inputs.append((d, case_rows[order][1], usage))
            mark = " DISPUTED" if d["attribution_disputed"] else ""
            print(  # noqa: T201
                f"  [{d['report_id']} q{d['turn_index']}] gold->{d['derived_agent']}"
                f" agent->{d['stage']}{mark} · {d['label']} (conf {d['confidence']:.2f})"
            )

        counts = {
            a: sum(1 for d in diagnoses if d["derived_agent"] == a) for a in AGENTS
        }
        unattributed = {
            v: sum(1 for d in diagnoses if d["derived_agent"] == v)
            for v in stage_scores.NON_AGENT
        }
        labels: dict[str, int] = {}
        for d in diagnoses:
            labels[d["label"]] = labels.get(d["label"], 0) + 1
        kappa = _kappa(diagnoses)

        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DIAGNOSTICS_DIR / f"sdk_diagnoses_{version}_{stamp}.jsonl"
        out_path.write_text("".join(json.dumps(d) + "\n" for d in diagnoses))
        rec.artifact(out_path)
        rec.dict_artifact(
            "summary.json", {"counts": counts, "labels": labels, "kappa": kappa}
        )
        teacher._log_jsonl_artifact(rec, diagnoses)

        rows = _diagnosis_ledger_rows(
            ledger_inputs,
            version=version,
            prompt_hash=prompt_hash,
            diagnosis_run_id=str(rec.run_id),
            diagnosed_at=datetime.strptime(stamp, "%Y%m%d_%H%M%S").isoformat(
                timespec="seconds"
            ),
        )
        written = ledgers.append("diagnoses", rows)
        ledgers.log_rows_to_run(rec, written, "diagnoses")
        rec.metrics(
            {
                "n_diagnosed": float(len(diagnoses)),
                "n_diagnose_failures": float(len(failures)),
                "n_attribution_disputed": float(
                    sum(1 for d in diagnoses if d["attribution_disputed"])
                ),
                "n_attributed": float(sum(counts.values())),
                "n_adjudicated": float(len(adjudicated)),
                **({"kappa_vs_attribution": kappa} if kappa is not None else {}),
                **{f"faults_{a}": float(counts[a]) for a in AGENTS},
                **{f"unattributed_{v}": float(n) for v, n in unattributed.items()},
                **{f"teacher_{k}": v for k, v in usage_total.items()},
            }
        )
        if failures and len(failures) > len(diagnoses):
            raise SystemExit(
                f"{len(failures)} of {len(failures) + len(diagnoses)} cases failed "
                "to diagnose — that is a broken diagnoser, not a flaky call"
            )
        return {
            "run_id": rec.run_id,
            "run_name": run_name,
            "version": version,
            "diagnoses_path": str(out_path),
            "n_cases": len(cases),
            "n_diagnosed": len(diagnoses),
            "n_failures": len(failures),
            "failures": failures,
            "counts": counts,
            "unattributed": unattributed,
            "labels": labels,
            "ledger_rows": len(written),
            "kappa_vs_attribution": kappa,
            "usage": usage_total,
        }


def _diagnosis_ledger_rows(
    inputs: list[tuple[dict[str, Any], pd.Series, dict[str, Any]]],
    *,
    version: str,
    prompt_hash: str,
    diagnosis_run_id: str,
    diagnosed_at: str,
) -> list[dict[str, Any]]:
    """One diagnoses-ledger row per case, ``runtime="agent_sdk"``, one prompt hash."""
    if not inputs:
        return []
    first_case = inputs[0][1]
    eval_run_id = str(ledgers._get(first_case, "run_id", "") or "")
    seed = ledgers.eval_run_param(eval_run_id, "train_draw_seed")
    try:
        model = teacher.teacher_model()
    except Exception:  # noqa: BLE001
        model = ""
    return [
        ledgers.diagnosis_row(
            d,
            case,
            version=version,
            runtime=RUNTIME,
            prompt_hash=prompt_hash,
            eval_run_id=str(ledgers._get(case, "run_id", "") or eval_run_id),
            diagnosis_run_id=diagnosis_run_id,
            draw_seed=int(seed) if seed and seed.lstrip("-").isdigit() else None,
            diagnoser_model=model,
            usage=usage,
            diagnosed_at=diagnosed_at,
        )
        for d, case, usage in inputs
    ]


# ── rank_classes ──────────────────────────────────────────────────────────


def _wilson_lower(faults: int, n: int) -> float:
    """`ledger._score`'s Wilson lower bound — the one formula targeting uses."""
    from convfinqa.evalloop import ledger

    entry: dict[str, Any] = {"faults": faults, "cases": n}
    ledger._score(entry)
    return round(float(entry["score"]), 6)


def _attributed(table: pd.DataFrame) -> pd.DataFrame:
    """Rows whose verdict names an agent — the targeting population."""
    from convfinqa.evalloop import stage_scores

    def _ok(r: Any) -> bool:
        derived = str(r.derived_agent or "")
        if derived in stage_scores.NON_AGENT:
            return False
        if bool(r.gold_suspect):
            return False
        return (derived in AGENTS) or (str(r.stage) in AGENTS)

    if table.empty:
        return table
    return table[[_ok(r) for r in table.itertuples()]]


def rank_classes(version: str, *, runtime: str = RUNTIME) -> dict[str, dict[str, Any]]:
    """Failure classes ranked on the Wilson lower bound of their pooled share.

    Pools every diagnoses-ledger row whose ``prompt_hash`` is `version`'s sdk
    prompt hash — two versions with the same text pool, a different text does
    not, and SDK draws never pool with pipeline draws because the hash spaces
    differ. The denominator is attributed rows (verdicts naming an agent).
    """
    want = sdk_prompt_hash(version)
    if not want:
        return {}
    table = ledgers.load("diagnoses", runtime=runtime)
    table = table[table["prompt_hash"] == want]
    attributed = _attributed(table)
    n = int(len(attributed))
    if not n:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in attributed.itertuples():
        label = str(r.label or "")
        entry = out.setdefault(
            label, {"faults": 0, "n": n, "stages": {}, "diagnosis_ids": []}
        )
        entry["faults"] += 1
        stage = str(r.stage or "")
        entry["stages"][stage] = entry["stages"].get(stage, 0) + 1
        entry["diagnosis_ids"].append(str(r.diagnosis_id))
    for entry in out.values():
        entry["wilson_lower"] = _wilson_lower(int(entry["faults"]), n)
    ordered = sorted(out, key=lambda k: (-out[k]["wilson_lower"], -out[k]["faults"], k))
    for rank, label in enumerate(ordered, start=1):
        out[label]["rank"] = rank
    return {label: out[label] for label in ordered}


# ── The writer's memory ───────────────────────────────────────────────────


def _class_flips(
    flips_by_class: Any, failure_class: str, target: str
) -> dict[str, int] | None:
    """The gate's fixed/broken counts that bear on one edit, if any were filed.

    Looked up by the edit's class, then by the stage its section stands for,
    then by the class's stage prefix (`retriever/wrong-period` → `retriever`).
    """
    if isinstance(flips_by_class, str):
        try:
            flips_by_class = json.loads(flips_by_class) if flips_by_class else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(flips_by_class, dict):
        return None
    keys = [failure_class, SECTION_STAGE.get(target, ""), failure_class.split("/")[0]]
    for key in keys:
        if key and key in flips_by_class and isinstance(flips_by_class[key], dict):
            got = flips_by_class[key]
            return {
                "fixed": int(got.get("fixed", 0) or 0),
                "broken": int(got.get("broken", 0) or 0),
            }
    return None


def sdk_attempts(limit: int = 40) -> list[dict[str, Any]]:
    """Every previous SDK-arm edit with its gate verdict, newest first.

    Rewrites ⋈ gates on ``rewrite_id`` (falling back to the candidate version).
    An edit whose class got worse — more broken than fixed in that class — is
    flagged ``revert_or_rethink`` so the writer sees it without arithmetic.
    """
    try:
        rewrites = ledgers.load("rewrites", runtime=RUNTIME)
        gates = ledgers.load("gates", runtime=RUNTIME)
    except Exception:  # noqa: BLE001 — an unreadable ledger is no memory
        return []
    if rewrites.empty:
        return []
    by_rewrite: dict[str, Any] = {}
    by_version: dict[str, Any] = {}
    for g in gates.itertuples():
        if g.rewrite_id:
            by_rewrite[str(g.rewrite_id)] = g
        by_version.setdefault(str(g.candidate_version), g)
    out: list[dict[str, Any]] = []
    for r in rewrites.sort_values("proposed_at", ascending=False).itertuples():
        g = by_rewrite.get(str(r.rewrite_id)) or by_version.get(str(r.new_version))
        class_flips = (
            _class_flips(g.flips_by_class, str(r.failure_class), str(r.target))
            if g is not None
            else None
        )
        worse = bool(class_flips and class_flips["broken"] > class_flips["fixed"])
        out.append(
            {
                "version": str(r.new_version),
                "base_version": str(r.base_version),
                "target": str(r.target),
                "failure_class": str(r.failure_class),
                "change_kind": str(r.change_kind),
                "edit_text": str(r.edit_text or "")[:400],
                "rationale": str(r.rationale or "")[:400],
                "at": str(r.proposed_at),
                "outcome": "promoted"
                if g is not None and bool(g.promoted)
                else ("rejected" if g is not None else "not yet gated"),
                "verdict": str(g.reason) if g is not None else "",
                "delta_pp": None
                if g is None or g.delta_pp is None
                else float(g.delta_pp),
                "p_value": None if g is None or g.p_value is None else float(g.p_value),
                "fixed": None if g is None else int(g.fixed),
                "broken": None if g is None else int(g.broken),
                "class_flips": class_flips,
                "revert_or_rethink": worse,
            }
        )
        if len(out) >= limit:
            break
    return out


def sdk_ledger_text(limit: int = 20) -> str:
    """The SDK attempt history as prose for the writer's prompt."""
    rows = sdk_attempts(limit=limit)
    if not rows:
        return (
            "\n\n## Prior edits to this prompt lineage\n"
            "None. This is the first rewrite of the SDK prompt in the recorded history.\n"
        )
    lines = ["\n\n## Prior edits to this prompt lineage (newest first)"]
    for r in rows:
        head = (
            f"- {r['version']} · {r['target']} · {r['failure_class']} "
            f"({r['change_kind']}) — {r['outcome'].upper()}"
        )
        if r.get("delta_pp") is not None:
            head += f" (Δ {float(r['delta_pp']):+.2f}pp"
            if r.get("p_value") is not None:
                head += f", p={float(r['p_value']):.3f}"
            head += ")"
        lines.append(head)
        if r.get("fixed") is not None:
            lines.append(f"  fixed {r['fixed']} questions, broke {r['broken']}")
        if r.get("class_flips"):
            cf = r["class_flips"]
            lines.append(
                f"  in its own class: fixed {cf['fixed']}, broke {cf['broken']}"
                + ("  ← REVERT OR RETHINK" if r["revert_or_rethink"] else "")
            )
        if r.get("edit_text"):
            lines.append(f"  changed: {r['edit_text']}")
    lines.append(
        "\nDo not re-propose a change that was REJECTED unless you can say what is "
        "different this time. An edit marked REVERT OR RETHINK made its own class "
        "worse: change approach or undo it, do not sharpen it."
    )
    return "\n".join(lines)


# ── propose_version ───────────────────────────────────────────────────────


def _group_by_label(diagnoses: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for d in diagnoses:
        label = str(d.get("label") or d.get("failure_mode") or "")
        out.setdefault(label, []).append(
            {
                "diagnosis_id": d.get("diagnosis_id", ""),
                "stage": d.get("stage") or d.get("failed_agent", ""),
                "derived_agent": d.get("derived_agent", ""),
                "question": d.get("question", ""),
                "what_went_wrong": d.get("what_went_wrong", ""),
                "evidence": d.get("evidence", ""),
                "fix_hint": d.get("fix_hint") or d.get("proposed_rule", ""),
                "gold_suspect": d.get("gold_suspect", False),
            }
        )
    return out


def _select_edits(
    edits: list[SdkEdit],
    pooled: dict[str, dict[str, Any]],
    max_areas: int | None,
) -> list[SdkEdit]:
    """Normalise targets, refuse duplicates, order by pooled rank, apply the cap."""
    seen: set[str] = set()
    normalised: list[SdkEdit] = []
    for e in edits:
        heading = normalise_target(e.target)  # raises on an unknown heading
        if heading in seen:
            raise SystemExit(
                f"two edits name the section {heading!r}; merge them into one"
            )
        seen.add(heading)
        normalised.append(e.model_copy(update={"target": heading}))
    ranked = sorted(
        normalised,
        key=lambda e: pooled.get(e.failure_class, {}).get("rank", len(pooled) + 1),
    )
    if max_areas is not None and max_areas >= 1:
        ranked = ranked[:max_areas]
    return ranked


async def propose_version(
    diagnoses_path: Path | str,
    *,
    base_version: str,
    new_version: str,
    experiment: str = OPTIMIZATION_EXPERIMENT,
    campaign: str | None = None,
    label: str | None = None,
    pooled: dict[str, dict[str, Any]] | None = None,
    max_areas: int | None = None,
) -> dict[str, Any]:
    """Edit the SDK prompt by section, one edit per failure class; register it.

    `pooled` is the class ranking (`rank_classes`) when the caller already has
    it; `max_areas=1` is the campaign's one-area mode after two consecutive
    rejections. Every edit becomes a rewrites-ledger row sharing one
    `rewrite_id`; the whole-prompt diff, the change list and the writer prompt
    are logged on the run.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.evalloop.sdk import run_structured
    from convfinqa.tracking import mlflow_log, prompt_ledger

    diagnoses = [
        json.loads(line)
        for line in Path(diagnoses_path).read_text().splitlines()
        if line.strip()
    ]
    if not diagnoses:
        raise SystemExit(f"{diagnoses_path} holds no diagnoses")
    if not prompts_pkg.is_sdk_version(new_version):
        raise SystemExit(f"{new_version!r} is not an sdk_vN version name")
    before = prompts_pkg.load_sdk(base_version)
    sections = split_sections(before)
    pooled = pooled if pooled is not None else rank_classes(base_version)
    by_label = _group_by_label(diagnoses)
    history = sdk_ledger_text()
    prior_attempts = sdk_attempts()
    tracing.enable()

    with mlflow_log.run(
        f"sdk-propose-{new_version}",
        kind="sdk_propose",
        version=base_version,
        params={
            "runtime": RUNTIME,
            "base_version": base_version,
            "new_version": new_version,
            "sdk_prompt_hash_before": sdk_prompt_hash(base_version),
            "n_diagnoses": len(diagnoses),
            "n_classes": len(by_label),
            "n_prior_attempts": len(prior_attempts),
            "max_areas": max_areas if max_areas is not None else "",
            **({"campaign": campaign} if campaign else {}),
            **({"experiment_label": label} if label else {}),
        },
        tags={
            "loop": "evalloop",
            "stage": "propose",
            "runtime": RUNTIME,
            **({"campaign": campaign} if campaign else {}),
        },
        experiment=experiment,
        actor_model=teacher.teacher_model(),
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        with tracing.span(
            f"sdk-propose {new_version}",
            span_type="AGENT",
            attributes={
                "runtime": RUNTIME,
                "base_version": base_version,
                "new_version": new_version,
                "n_diagnoses": len(diagnoses),
                "n_classes": len(by_label),
                "max_areas": max_areas if max_areas is not None else -1,
            },
            trace_tags={"stage": "propose", "runtime": RUNTIME},
        ) as span:
            writer_prompt = (
                json.dumps(
                    {
                        "base_version": base_version,
                        "headings": list(HEADINGS),
                        "current_prompt_sections": {
                            h: b for h, b in sections.items() if h
                        },
                        "max_areas": max_areas,
                        "class_ranking": {
                            k: {
                                "rank": v.get("rank"),
                                "wilson_lower": v.get("wilson_lower"),
                                "faults": v.get("faults"),
                                "n": v.get("n"),
                                "stages": v.get("stages"),
                            }
                            for k, v in pooled.items()
                        },
                        "diagnoses_by_label": by_label,
                    },
                    default=str,
                )
                + history
            )
            rec.text_artifact(WRITER_PROMPT_ARTIFACT, writer_prompt)
            output, usage = await run_structured(
                writer_prompt,
                schema=SdkRewrite,
                system_prompt=SDK_WRITER_PROMPT,
                max_turns=8,
                refs={
                    "system_prompt": prompt_refs.teacher_prompt_ref(
                        "SDK_WRITER_PROMPT", SDK_WRITER_PROMPT
                    ),
                    "user_prompt": prompt_refs.run_artifact_ref(
                        WRITER_PROMPT_ARTIFACT, writer_prompt, run_id=rec.run_id
                    ),
                    "target_prompt": prompt_refs.sdk_prompt_ref(base_version, before),
                },
            )
            try:
                edits = _select_edits(output.edits, pooled, max_areas)
            except ValueError as exc:
                raise SystemExit(f"the rewrite was rejected: {exc}") from exc
            if not edits:
                raise SystemExit("the writer returned no edits")
            after = before
            change_list: list[dict[str, Any]] = []
            for e in edits:
                old_body = sections[e.target]
                new_body = _section_body(e.new_section_body)
                after = replace_section(after, e.target, e.new_section_body)
                change_list.append(
                    {
                        "target": e.target,
                        "failure_class": e.failure_class,
                        "change_kind": e.change_kind,
                        "diagnosis_ids": list(e.diagnosis_ids),
                        "edit_text": e.edit_text,
                        "rationale": e.rationale,
                        "diff": teacher.prompt_diff(
                            old_body, new_body, target=e.target
                        ),
                        "old_body": old_body,
                        "new_body": new_body,
                    }
                )
            span.set(
                summary=output.summary,
                n_edits=len(edits),
                targets=[e.target for e in edits],
                prompt_chars_before=len(before),
                prompt_chars_after=len(after),
            )

        problems = validate_sdk_prompt(before, after)
        diff = teacher.prompt_diff(before, after, target="sdk_prompt")
        rewrite_id = ledgers.new_id("rw")
        rows = [
            ledgers.rewrite_row(
                target=c["target"],
                base_version=base_version,
                new_version=new_version,
                prompt_before=before,
                prompt_after=after,
                diff=c["diff"],
                rationale=c["rationale"],
                edit_text=c["edit_text"],
                failure_class=c["failure_class"],
                diagnosis_ids=c["diagnosis_ids"],
                evidence_summary={
                    "labels_this_run": {k: len(v) for k, v in by_label.items()},
                    "summary": output.summary,
                },
                prior_attempts=[
                    {
                        "version": a["version"],
                        "target": a["target"],
                        "failure_class": a["failure_class"],
                        "outcome": a["outcome"],
                    }
                    for a in prior_attempts
                ],
                wilson_lower=pooled.get(c["failure_class"], {}).get("wilson_lower"),
                rank=pooled.get(c["failure_class"], {}).get("rank"),
                validate_ok=not problems,
                change_kind=c["change_kind"],
                runtime=RUNTIME,
                campaign=campaign,
                label=label,
                teacher_run_id=str(rec.run_id),
                teacher_model=teacher.teacher_model(),
                usage=usage,
                rewrite_id=rewrite_id,
            )
            for c in change_list
        ]
        ledgers.log_rows_to_run(rec, ledgers.append("rewrites", rows), "rewrites")
        for c, row in zip(change_list, rows, strict=True):
            c["edit_id"] = row["edit_id"]
        rec.dict_artifact(
            CHANGES_ARTIFACT,
            {
                "rewrite_id": rewrite_id,
                "base_version": base_version,
                "new_version": new_version,
                "summary": output.summary,
                "edits": [
                    {k: v for k, v in c.items() if k not in {"old_body", "new_body"}}
                    for c in change_list
                ],
            },
        )
        rec.dict_artifact("prompt_diff.json", {"target": "sdk_prompt", "diff": diff})
        if problems:
            raise SystemExit(
                "the rewrite failed its output contract and was not written:\n  - "
                + "\n  - ".join(problems)
            )
        module_path = _write_sdk_module(
            new_version,
            prompt=after,
            header=(
                f"Challenger for {base_version}: {len(edits)} section(s) edited "
                f"({', '.join(e.target for e in edits)}); MLflow run {rec.run_id}. "
                "Regenerate via `convfinqa-evalloop` — never hand-edit."
            ),
        )
        rec.text_artifact("sdk_prompt.txt", after)
        usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
        teacher._accumulate_usage(usage_total, usage)
        rec.metrics(
            {
                "prompt_chars_before": float(len(before)),
                "prompt_chars_after": float(len(after)),
                "n_edits": float(len(edits)),
                "n_prior_attempts": float(len(prior_attempts)),
                **{f"teacher_{k}": v for k, v in usage_total.items()},
            }
        )
        prompt_ledger.ensure_sdk(new_version, source="sdk_teacher", run_id=rec.run_id)
        return {
            "run_id": rec.run_id,
            "module_path": str(module_path),
            "new_version": new_version,
            "base_version": base_version,
            "diff": diff,
            "rewrite_id": rewrite_id,
            "summary": output.summary,
            "edits": [
                {
                    "edit_id": c["edit_id"],
                    "target": c["target"],
                    "failure_class": c["failure_class"],
                    "change_kind": c["change_kind"],
                    "diagnosis_ids": c["diagnosis_ids"],
                    "edit_text": c["edit_text"],
                    "diff": c["diff"],
                    "rationale": c["rationale"],
                }
                for c in change_list
            ],
            "validate_ok": not problems,
            "prompt_chars_before": len(before),
            "prompt_chars_after": len(after),
            "usage": usage_total,
        }


# ── distil_prompt (D8) ────────────────────────────────────────────────────

DATASET_NOTES_HEADING = "## ConvFinQA Dataset Characteristics"


def dataset_notes(path: Path | None = None) -> str:
    """The dataset section of CLAUDE.md, sliced to the next top-level heading."""
    target = path or (REPO_ROOT / "CLAUDE.md")
    try:
        text = target.read_text()
    except OSError:
        return ""
    start = text.find(DATASET_NOTES_HEADING)
    if start == -1:
        return ""
    rest = text[start + len(DATASET_NOTES_HEADING) :]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    end = nxt.start() if nxt else len(rest)
    return (DATASET_NOTES_HEADING + rest[:end]).strip()


def output_contract_schema() -> dict[str, Any]:
    """The JSON schema of the structured reply the session must return."""
    from convfinqa.backends.agent_sdk import SdkTurnResult

    return SdkTurnResult.model_json_schema()


def distil_prompt_text(source_version: str, source_prompts: dict[str, str]) -> str:
    """The exact user prompt a distillation call sends."""
    return json.dumps(
        {
            "source_version": source_version,
            "source_prompts": source_prompts,
            "dataset_notes": dataset_notes(),
            "output_contract_schema": output_contract_schema(),
            "train_of_thought_spec": TRAIN_OF_THOUGHT_SPEC,
            "headings": list(HEADINGS),
        },
        default=str,
    )


async def distil_prompt(
    *,
    source_version: str = "v8",
    new_version: str = "sdk_v1",
    experiment: str = OPTIMIZATION_EXPERIMENT,
) -> dict[str, Any]:
    """Draft the first SDK prompt from the four pipeline prompts of `source_version`.

    Refuses when the module already exists: a distillation is the root of a
    lineage and is never regenerated in place.
    """
    import convfinqa.prompts as prompts_pkg
    from convfinqa.evalloop.sdk import run_structured
    from convfinqa.tracking import mlflow_log, prompt_ledger

    if not prompts_pkg.is_sdk_version(new_version):
        raise SystemExit(f"{new_version!r} is not an sdk_vN version name")
    target_path = PROMPTS_DIR / f"{new_version}.py"
    if target_path.exists():
        raise SystemExit(
            f"{target_path} already exists — a distillation is never redone"
        )
    source_prompts = prompts_pkg.load(source_version)
    user_prompt = distil_prompt_text(source_version, source_prompts)
    tracing.enable()

    with mlflow_log.run(
        f"sdk-distil-{new_version}-from-{source_version}",
        kind="sdk_distil",
        version=source_version,
        params={
            "runtime": RUNTIME,
            "source_version": source_version,
            "new_version": new_version,
        },
        tags={"loop": "evalloop", "stage": "distil", "runtime": RUNTIME},
        experiment=experiment,
        actor_model=teacher.teacher_model(),
        omit_fingerprint=("lm_max", "lm_mini"),
    ) as rec:
        rec.text_artifact(DISTIL_PROMPT_ARTIFACT, user_prompt)
        with tracing.span(
            f"sdk-distil {new_version}",
            span_type="AGENT",
            attributes={"source_version": source_version, "new_version": new_version},
            trace_tags={"stage": "distil", "runtime": RUNTIME},
        ) as span:
            draft, usage = await run_structured(
                user_prompt,
                schema=PromptDraft,
                system_prompt=SDK_DISTIL_PROMPT,
                max_turns=6,
                refs={
                    "system_prompt": prompt_refs.teacher_prompt_ref(
                        "SDK_DISTIL_PROMPT", SDK_DISTIL_PROMPT
                    ),
                    "user_prompt": prompt_refs.run_artifact_ref(
                        DISTIL_PROMPT_ARTIFACT, user_prompt, run_id=rec.run_id
                    ),
                    **{
                        f"source_{agent}": prompt_refs.agent_prompt_ref(
                            agent, source_version, text
                        )
                        for agent, text in source_prompts.items()
                    },
                },
            )
            span.set(
                prompt_chars=len(draft.prompt),
                sections=list(draft.sections),
                dropped=list(draft.dropped),
                notes=draft.notes,
            )
        prompt = draft.prompt.rstrip("\n") + "\n"
        problems = validate_sdk_prompt("", prompt)
        row = ledgers.rewrite_row(
            target="whole",
            base_version=source_version,
            new_version=new_version,
            prompt_before="",
            prompt_after=prompt,
            diff="",
            rationale=draft.notes,
            edit_text=f"distilled from the four {source_version} prompts",
            failure_class="distil",
            evidence_summary={"dropped": list(draft.dropped)},
            validate_ok=not problems,
            change_kind="rewrite",
            runtime=RUNTIME,
            teacher_run_id=str(rec.run_id),
            teacher_model=teacher.teacher_model(),
            usage=usage,
        )
        row["prompt_hash_before"] = ""
        ledgers.log_rows_to_run(rec, ledgers.append("rewrites", [row]), "rewrites")
        rec.dict_artifact(
            "draft.json",
            {
                "sections": draft.sections,
                "dropped": draft.dropped,
                "notes": draft.notes,
                "problems": problems,
            },
        )
        if problems:
            raise SystemExit(
                "the distilled prompt failed its contract and was not written:\n  - "
                + "\n  - ".join(problems)
            )
        module_path = _write_sdk_module(
            new_version,
            prompt=prompt,
            header=(
                f"Distilled from the four {source_version} pipeline prompts by the "
                f"teacher agent; MLflow run {rec.run_id}. The root of the sdk_v* "
                "lineage — later versions are edited by `convfinqa-evalloop`, "
                "never by hand."
            ),
        )
        rec.text_artifact("sdk_prompt.txt", prompt)
        usage_total = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
        teacher._accumulate_usage(usage_total, usage)
        rec.metrics(
            {
                "prompt_chars": float(len(prompt)),
                "source_chars": float(sum(len(t) for t in source_prompts.values())),
                **{f"teacher_{k}": v for k, v in usage_total.items()},
            }
        )
        prompt_ledger.ensure_sdk(new_version, source="distil", run_id=rec.run_id)
        return {
            "run_id": rec.run_id,
            "module_path": str(module_path),
            "new_version": new_version,
            "source_version": source_version,
            "prompt_chars": len(prompt),
            "usage": usage_total,
        }
