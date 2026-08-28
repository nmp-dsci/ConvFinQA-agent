"""DSPy backend: LMs, signatures, sequential agent + multi-conversation runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

# Pin DSPy's LM cache to a repo-local dir before `import dspy` (read at import time).
os.environ.setdefault(
    "DSPY_CACHEDIR",
    str(Path(__file__).resolve().parents[3] / ".dspy_cache"),
)
# DeepSeek reasoning models split structured output across reasoning_content/text;
# merging fixes JSONAdapter parse errors.
os.environ.setdefault("LITELLM_MERGE_REASONING_CONTENT_IN_CHOICES", "true")

import dspy  # noqa: E402

from convfinqa.config import settings  # noqa: E402
from convfinqa.data.schemas import (  # noqa: E402
    AgentResponse,
    ConversationHistory,
    Document,
    QAPair,
)
from convfinqa.pipeline.tools import CALCULATOR_TOOLS  # noqa: E402

# Models are built on first use, not at import. Importing this module must not
# demand an API key: the serving layer reads dataset facts from it, and the
# keyless demo container imports it while being unable to call a model at all.
# `configure_dspy()` is called by the optimizer entry points that actually run.
_lm_cache: dict[str, Any] = {}


def _lm(model: str) -> Any:
    """Build (once) a DSPy LM, routed through the shared demo gate and key."""
    from convfinqa.llm import guard_llm_call

    if model not in _lm_cache:
        guard_llm_call()
        _lm_cache[model] = dspy.LM(
            model=f"deepseek/{model}",
            api_key=settings.require_deepseek_api_key(),
            max_tokens=64000,
            temperature=1,
        )
    return _lm_cache[model]


def lm_mini() -> Any:
    """The fast model the DSPy pipeline runs on."""
    return _lm("deepseek-v4-flash")


def lm_max() -> Any:
    """The flagship model, used where reasoning quality is the product."""
    return _lm("deepseek-v4-pro")


def configure_dspy() -> None:
    """Point DSPy at the mini model. Call before running or optimizing."""
    dspy.configure(lm=lm_mini(), adapter=dspy.ChatAdapter())


class TriageSignature(dspy.Signature):
    """Classify the current turn using the question plus prior conversation history.

    You must predict two labels:

    1. `turn_type`
       - `number`: the answer is a direct value lookup from the document or a
         previously answered value in history. No arithmetic or multi-value
         composition is required.
       - `program`: the answer requires arithmetic, comparison, a rate/change,
         a percentage, a difference between periods, a sum across values, or
         any multi-step reasoning over multiple values.

    2. `conv_type`
       - `Type I`: the question continues the current reasoning thread from the
         same decomposed multi-hop problem.
       - `Type II`: the question switches to a different aspect / sub-problem
         of the same report, even if it still references prior turns.

    Use `history` aggressively. Follow-up questions with references like
    "that", "this", "the change", "the difference", "what about 2010", or
    "what percentage" are often continuations of prior reasoning and are more
    likely to be `program` turns than isolated one-shot lookups. If answering
    the current turn would require combining a value from history with a new
    value, or transforming a prior answer, label it `program`.
    """

    question: str = dspy.InputField()
    history: str = dspy.InputField(
        desc=(
            "Prior Q&A pairs in this session. Use this to resolve follow-up "
            "references and determine whether the current turn is a direct "
            "lookup or a continuation that requires computation."
        )
    )

    turn_type: Literal["number", "program"] = dspy.OutputField(
        desc=(
            "`number` only when the final answer is a single directly retrievable "
            "value. Use `program` when the turn needs arithmetic, comparison, "
            "change-over-time reasoning, percentages, aggregation, or reuse of "
            "a prior answer in a computation."
        ),
    )
    conv_type: Literal["Type I", "Type II"] = dspy.OutputField(
        desc=(
            "`Type I` when the turn continues the current reasoning chain. "
            "Use `Type II` when the turn pivots to a different aspect or a "
            "second decomposed problem about the same report."
        ),
    )


class PreprocessSignature(dspy.Signature):
    """Decompose a program-type question into sub-questions and a calculation program.

    You are given:
      - `question`: the current user question
      - `history`: prior turns with their questions and answers
      - `conv_type`: whether this turn continues the current reasoning chain (Type I)
        or switches to a different aspect of the report (Type II)

    Your job is to produce:
      - `reasoning`: a brief explanation of the decomposition and which cached values
        from `history` can be reused
      - `sub_questions`: value lookups only, not computations
      - `program`: an arithmetic expression over A, B, C, ... using add, subtract,
        multiply, and divide

    Use `conv_type` to guide decomposition:
      - Type I: heavily lean on `history`. Follow-up questions often depend on prior
        answers, so reuse the exact phrasing from relevant earlier turns whenever a
        needed value is already available there.
      - Type II: re-anchor sub-questions on the document because the conversation has
        shifted to a different aspect of the report, but still reuse a cached value
        from `history` if it is clearly the same quantity.

    Reuse `history` whenever possible: if a value needed by the program already appears
    as the answer to a prior turn, restate that earlier question as closely as possible
    so the retriever can return the cached value instead of re-reading the document.
    This reduces drift across turns and is especially important in long conversations.

    Program design rules:
      - If the question asks for a "growth rate" or "percent change", compute the ratio
        and then multiply by 100 so the downstream answer becomes a percentage result.
      - If the question asks what "percentage change this represents", return the raw
        ratio without multiplying by 100.
      - Keep the program numeric only. Do not include units, currency markers, or other
        formatting in the program itself.

    The distinction between `divide(...)` and `multiply(divide(...), 100)` matters and
    should be chosen deliberately based on whether the target answer is a plain ratio or
    a percentage-style result.
    """

    question: str = dspy.InputField()
    history: str = dspy.InputField(
        desc="Prior Q&A pairs in this session — reuse answers when applicable"
    )
    conv_type: Literal["Type I", "Type II"] = dspy.InputField(
        desc="From triage: 'Type I' continues the prior chain; 'Type II' switches aspect",
    )
    sub_questions: list[str] = dspy.OutputField(
        desc=(
            "Self-contained value lookups only, not computations. "
            "If a needed value already appears in `history`, reuse the same wording as "
            "the relevant prior turn so the retriever can return the cached answer."
        ),
    )
    program: str = dspy.OutputField(
        desc=(
            "Arithmetic DSL such as 'subtract(A, B)' or 'divide(subtract(A, B), B)', "
            "where A, B, C... map positionally to `sub_questions`. Use "
            "'multiply(divide(...), 100)' for percentage-style outputs and "
            "'divide(...)' for raw ratios."
        ),
    )


class RetrieverSignature(dspy.Signature):
    """Answer one or more value-lookup questions from the financial document.

    Behavior depends on `turn_type`:
      - `number`: there is exactly one question and it is the user's final question.
        Return the single value that answers it. No downstream calculator stage will
        run, so you may need to do simple arithmetic here when the question asks for
        a change, net increase/decrease, or a percentage.
      - `program`: the questions are sub-questions from Preprocess. Return the raw
        retrieved value for each one, with no arithmetic or aggregation. These values
        are passed to the Calculation stage.

    In both modes, prefer reusing values already present in `history` over re-reading
    the document when the same value has already been answered in a prior turn.

    Retrieval and arithmetic rules:
      - Match both the entity and the date/year carefully. If the document contains
        multiple values from the same year, use the one the question actually refers to.
      - In `number` mode, change questions should use signed arithmetic:
        later minus earlier. Do not take absolute values unless the question asks for
        magnitude explicitly.
      - Percentage change / return-rate questions should use
        `((new - old) / old) * 100` and return a `%` suffixed answer string.
      - Raw factual lookups and all `program` mode outputs should preserve the source
        numeric string as closely as possible, including meaningful trailing zeroes.
      - Computed numeric answers should use sensible precision based on the operands,
        and should never include extraneous units such as `$`, `million`, or `billion`.
    """

    turn_type: Literal["number", "program"] = dspy.InputField(
        desc=(
            "From triage. 'number' = single question, return the final answer. "
            "'program' = sub-questions from preprocess, return raw values for the calculator."
        ),
    )
    questions: list[str] = dspy.InputField(
        desc="One or more self-contained value-lookup questions"
    )
    document: Document = dspy.InputField(
        desc="The financial report: pre_text, post_text, and a structured `table` (column -> row -> value)",
    )
    history: str = dspy.InputField(
        desc="Prior Q&A pairs — reuse cached answers when applicable"
    )
    answers: list[QAPair] = dspy.OutputField(
        desc=(
            "One QAPair per input question, same order as `questions`. "
            "`question` echoes the input question verbatim; `answer` is the retrieved "
            "or computed answer string. In `program` mode, return raw values only. "
            "In `number` mode, return the final answer string, including `%` only when "
            "the question explicitly asks for a percentage-style result."
        ),
    )


class CalculationSignature(dspy.Signature):
    """Execute a DSL program over retrieved values using calculator tools.

    You receive the original `question`, the retrieved sub-question answers, and a
    candidate `program`. The program is a strong hint, but it is not infallible:
    if it conflicts with the question's intent, you should correct the operation,
    argument order, or semantics before finishing.

    Rules for execution:
      - Map placeholders positionally: first retrieved answer = A, second = B, etc.
      - Strip non-numeric decoration from retrieved answers as needed (`%`, `$`,
        commas, units) while preserving the intended numeric value.
      - Treat percentage answers in `retrieved` as whole numbers unless the question
        explicitly asks for a decimal fraction.
      - Sanity-check directionality for changes and differences. If the question asks
        for decline/decrease/change from earlier to later, make sure the subtraction
        order matches that intent.
      - Trust the user's question over the program if the two disagree.
      - The final answer must be a plain numeric string with no units or symbols.
    """

    question: str = dspy.InputField(
        desc="The user's original question (context only — do not re-answer from it)"
    )
    retrieved: list[QAPair] = dspy.InputField(
        desc=(
            "Sub-questions paired with their retrieved values, in placeholder order: "
            "first entry = A, second = B, etc."
        ),
    )
    program: str = dspy.InputField(
        desc=(
            "Candidate DSL to execute, e.g. 'subtract(A, B)' or "
            "'divide(subtract(A, B), B)'. Correct it if it does not match the question."
        ),
    )
    answer: str = dspy.OutputField(
        desc="Final plain numeric result as a string from the calculator workflow, with no units or symbols",
    )


class ConvFinQASequentialAgent(dspy.Module):
    """Sequential pipeline: triage -> preprocess -> retrieve -> calculate."""

    def __init__(self) -> None:
        """Instantiate the four predictors, build the doc lookup, and start a fresh history."""
        super().__init__()
        self.triage = dspy.ChainOfThought(TriageSignature)
        self.preprocess = dspy.ChainOfThought(PreprocessSignature)
        self.retriever = dspy.ChainOfThought(RetrieverSignature)
        self.calculator = dspy.ReAct(
            CalculationSignature,
            tools=CALCULATOR_TOOLS,
            max_iters=8,
        )
        from convfinqa.data.loader import _DOCS  # noqa: PLC0415

        self._docs: dict[str, Document] = _DOCS
        self.conversation: ConversationHistory = ConversationHistory()

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation = ConversationHistory()

    def _retrieve_document(self, report_id: str) -> Document:
        try:
            return self._docs[report_id]
        except KeyError as e:
            msg = f"Unknown report_id: {report_id!r}"
            raise KeyError(msg) from e

    def forward(self, question: str, report_id: str) -> AgentResponse:
        """Run a single turn end-to-end and return an AgentResponse."""
        hist_text = self.conversation.as_text()
        triage = self.triage(question=question, history=hist_text)
        document = self._retrieve_document(report_id)

        if triage.turn_type == "number":
            r = self.retriever(
                turn_type="number",
                questions=[question],
                document=document,
                history=hist_text,
            )
            answer = str(r.answers[0].answer)
            self.conversation.append(
                question=question, answer=answer, report_id=report_id
            )
            return AgentResponse(
                question=question,
                report_id=report_id,
                answer=answer,
                turn_type="number",
                conv_type=triage.conv_type,
                triage_reasoning=getattr(triage, "reasoning", None),
                retriever_reasoning=getattr(r, "reasoning", None),
            )

        pp = self.preprocess(
            question=question, history=hist_text, conv_type=triage.conv_type
        )
        r = self.retriever(
            turn_type="program",
            questions=list(pp.sub_questions),
            document=document,
            history=hist_text,
        )
        calc = self.calculator(
            question=question,
            retrieved=list(r.answers),
            program=pp.program,
        )
        answer = str(calc.answer)
        self.conversation.append(question=question, answer=answer, report_id=report_id)
        return AgentResponse(
            question=question,
            report_id=report_id,
            answer=answer,
            turn_type="program",
            conv_type=triage.conv_type,
            turn_program=str(pp.program),
            triage_reasoning=getattr(triage, "reasoning", None),
            preprocess_reasoning=getattr(pp, "reasoning", None),
            retriever_reasoning=getattr(r, "reasoning", None),
            calc_trajectory=getattr(calc, "trajectory", None),
        )


class ConversationRunner(dspy.Module):
    """Walks all turns of one conversation, with predictors owned directly."""

    def __init__(self) -> None:
        super().__init__()
        self.triage = dspy.ChainOfThought(TriageSignature)
        self.preprocess = dspy.ChainOfThought(PreprocessSignature)
        self.retriever = dspy.ChainOfThought(RetrieverSignature)
        self.calculator = dspy.ReAct(
            CalculationSignature,
            tools=CALCULATOR_TOOLS,
            max_iters=8,
        )

    def _run_turn(
        self,
        question: str,
        report_id: str,
        document: Document,
        conversation: ConversationHistory,
    ) -> AgentResponse:
        hist_text = conversation.as_text()
        triage = self.triage(question=question, history=hist_text)

        if triage.turn_type == "number":
            r = self.retriever(
                turn_type="number",
                questions=[question],
                document=document,
                history=hist_text,
            )
            answer = str(r.answers[0].answer)
            conversation.append(question=question, answer=answer, report_id=report_id)
            return AgentResponse(
                question=question,
                report_id=report_id,
                answer=answer,
                turn_type="number",
                conv_type=triage.conv_type,
                triage_reasoning=getattr(triage, "reasoning", None),
                retriever_reasoning=getattr(r, "reasoning", None),
            )

        pp = self.preprocess(
            question=question, history=hist_text, conv_type=triage.conv_type
        )
        r = self.retriever(
            turn_type="program",
            questions=list(pp.sub_questions),
            document=document,
            history=hist_text,
        )
        calc = self.calculator(
            question=question,
            retrieved=list(r.answers),
            program=pp.program,
        )
        answer = str(calc.answer)
        conversation.append(question=question, answer=answer, report_id=report_id)
        return AgentResponse(
            question=question,
            report_id=report_id,
            answer=answer,
            turn_type="program",
            conv_type=triage.conv_type,
            turn_program=str(pp.program),
            triage_reasoning=getattr(triage, "reasoning", None),
            preprocess_reasoning=getattr(pp, "reasoning", None),
            retriever_reasoning=getattr(r, "reasoning", None),
            calc_trajectory=getattr(calc, "trajectory", None),
        )

    def forward(self, report_id: str, questions: list[str]) -> dspy.Prediction:
        """Walk every turn of one conversation and return per-turn predictions."""
        from convfinqa.data.loader import _DOCS  # noqa: PLC0415

        document = _DOCS[report_id]
        conversation = ConversationHistory()
        responses = [
            self._run_turn(q, report_id, document, conversation) for q in questions
        ]
        return dspy.Prediction(
            predictions=[r.answer for r in responses],
            responses=responses,
            conversation=conversation,
        )


def build_conv_examples(report_ids: list[str]) -> list[dspy.Example]:
    """One dspy.Example per conversation, with all turns in q_order."""
    from convfinqa.data.loader import qa_data  # noqa: PLC0415

    examples: list[dspy.Example] = []
    for rid in report_ids:
        g = qa_data[qa_data["report_id"] == rid].sort_values("q_order")
        examples.append(
            dspy.Example(
                report_id=rid,
                questions=g["conv_questions"].tolist(),
                gold_answers=g["conv_answers"].tolist(),
            ).with_inputs("report_id", "questions")
        )
    return examples


def _build_dspy_data() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Return (train_examples, test_examples) using DSPy's rng-shuffle split.

    The DSPy baseline uses a `random.Random(42)` shuffle of the 200 sampled
    report_ids into a 60/40 train/test split, with no `additional_test_ids`.
    """
    from convfinqa.data.loader import optimizer_split  # noqa: PLC0415

    train_ids, test_ids = optimizer_split()
    return build_conv_examples(train_ids), build_conv_examples(test_ids)


conv_examples_train, conv_examples_test = _build_dspy_data()
