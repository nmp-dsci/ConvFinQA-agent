# S3: V2 Prompt Optimization Plan

## Goal

Use the `v1` evaluation outputs to create a controlled prompt-improvement loop
for a `v2` evaluation run, while avoiding blind overfitting to the existing
evaluation set.

## Lead AI Engineer Recommendations

The core prompt problem is not only "make the model smarter." The system needs
to become a more reliable clause extractor. The prompt strategy should optimize
for exact supporting spans, calibrated no-answer behavior, and consistency
across 41 legal categories.

As lead AI engineer, I would change the v2 prompt work in these ways:

1. Create a reusable base extraction prompt.

   Every category should inherit the same extraction contract:

   - return exact contract span text, not summaries
   - preserve legal wording where possible
   - return multiple spans on separate lines
   - return `NO_ANSWER` only when no supporting clause exists
   - for `Yes/No` categories, return the supporting clause span, not only
     `Yes` or `No`
   - do not infer from general legal knowledge
   - do not answer from the contract title alone unless the category is
     `Document Name`

2. Keep category prompts as small overlays.

   The category-specific prompt should define what evidence counts for that
   category, common synonyms, and what to exclude. It should not repeat the
   entire base prompt. This makes prompt changes easier to review and reduces
   prompt drift across categories.

3. Separate failure modes before rewriting prompts.

   Do not ask an LLM to rewrite a prompt from a pile of failures. First label
   failures into buckets:

   - `classification_instead_of_span`: model answered `Yes` or `No`
   - `false_no_answer`: model returned `NO_ANSWER` but gold has spans
   - `false_positive_span`: model found a plausible but wrong clause
   - `partial_span`: model found nearby text but missed required language
   - `overlong_span`: model returned too much surrounding contract text
   - `format_error`: answer did not follow newline/NO_ANSWER conventions
   - `gold_or_metric_issue`: gold answer or token metric may be questionable

   Prompt changes should target the dominant failure mode per category.

4. Use negative examples.

   For each category, include examples of clauses that look related but should
   not count. This is especially important for categories like:

   - `Anti-Assignment`
   - `Change of Control`
   - `License Grant`
   - `IP Ownership Assignment`
   - `Cap on Liability`
   - `Uncapped Liability`

5. Add category-specific synonym lists.

   Many failures come from vocabulary mismatch. For example:

   - `Anti-Assignment`: assign, transfer, delegate, merger, sale of assets,
     assignment by operation of law, successor
   - `Audit Rights`: inspect, examine books, records, audit, access records
   - `Cap on Liability`: liability cap, aggregate liability, maximum recovery,
     limitation of liability
   - `Most Favored Nation`: no less favorable, better terms, most favored,
     equivalent terms

6. Avoid prompt bloat.

   A longer prompt is not automatically better. Each category overlay should be
   short enough that a reviewer can identify what changed between `v1` and
   `v2`. If a prompt needs many rules, that is a signal to add retrieval,
   examples, or category-specific post-processing rather than continuing to
   expand the prompt.

7. Preserve an unbiased holdout.

   If `outputs/v1/cuad_dspy_eval_results.csv` is used to improve prompts, it is
   no longer an unbiased evaluation set. For v2, create `generator_dev`,
   `evaluator_dev`, and `holdout_eval` splits before declaring improvement.

8. Measure regressions, not only aggregate improvement.

   V2 should be accepted only if it improves the intended failure modes without
   materially damaging strong categories such as `Document Name` and `Parties`.
   Track per-category deltas and review the largest regressions manually.

## Recommended Workflow

1. Add the required harness support code.

   The plan depends on a small amount of implementation work before it can run
   end to end:

   - create `prompt_improve_v2.py` for split creation, generator/evaluator
     loops, prompt candidate writing, and dashboard generation
   - use PydanticAI agents for the generator and evaluator with
     `deepseek/deepseek-v4-pro`
   - extend `dspy_eval_v1.py` to load prompt source from `--prompts-file`
   - extend `dspy_eval_v1.py` to filter evaluation rows from `--eval-split`
   - keep `outputs/<model_id>/system_prompts.py` as a run snapshot only

2. Build a versioned error-analysis dataset from:

   ```text
   outputs/v1/cuad_dspy_eval_results.csv
   ```

   Filter rows where `correct_at_0_5 == False`, grouped by `category`, and
   add a deterministic `row_id` for split tracking.

3. Derive the required answer format from golden answers for each category.

   Do not rely on `data/category_descriptions.csv` answer format as the source
   of truth. Use it only as context. For each question/category, inspect the
   golden answers across the available evaluation rows and derive:

   - whether the gold answers are truly only `Yes` or `No` labels
   - whether the gold answers are verbatim extracts from the contract text
   - whether the expected extract is usually a single sentence, multiple
     sentences, or multiple separate clauses/spans
   - whether `NO_ANSWER` is valid when the gold answer list is empty

   The prompt's required output format should be based only on these golden
   answers. If golden answers are contract text spans, the model must return
   verbatim contract text that matches the clause. If there are multiple gold
   spans, the model should return each matching span on a separate line.

   Save this derived format profile to:

   ```text
   outputs/v2/prompt_harness/answer_format_profiles.json
   ```

4. Label or infer the failure mode for each incorrect row:

   - `classification_instead_of_span`
   - `false_no_answer`
   - `false_positive_span`
   - `partial_span`
   - `overlong_span`
   - `format_error`
   - `gold_or_metric_issue`

5. Create deterministic per-category splits:

   - `generator_dev`: up to 20 failures used by the generator agent
   - `evaluator_dev`: up to 20 different failures used by the evaluator agent
   - `holdout_eval`: remaining rows, never shown to either agent during prompt
     improvement

   Save the row assignments to:

   ```text
   outputs/v2/prompt_harness/splits.json
   ```

6. Run the generator/evaluator harness per category.

   The generator agent receives:

   - current prompt source from `prompts/system_prompts_v1.py`
   - category description from `data/category_descriptions.csv`
   - derived answer format profile from golden answers
   - failure-mode summary
   - only the category's `generator_dev` examples
   - on attempts 2 and 3 only: the previous generated prompt and evaluator
     feedback, not evaluator examples

   The evaluator agent receives:

   - the generator instructions
   - the generator output
   - current prompt source
   - category metadata
   - only the category's separate `evaluator_dev` examples

   Run at most three generator/evaluator loops. On the second and third
   generator attempts, pass the original generator guide, the previous
   generator output prompt, and the evaluator feedback on that prompt. Do not
   reveal `evaluator_dev` examples back to the generator.

7. Write harness artifacts for audit and review:

   ```text
   outputs/v2/prompt_harness/answer_format_profiles.json
   outputs/v2/prompt_harness/category_runs.jsonl
   outputs/v2/prompt_harness/evaluator_reviews.jsonl
   outputs/v2/prompt_harness/accepted_patches.jsonl
   outputs/v2/prompt_harness/rejected_patches.jsonl
   outputs/v2/prompt_harness/prompt_diffs.jsonl
   outputs/v2/prompt_harness/prompts_candidate_v2.py
   outputs/v2/prompt_harness/prompt_review_dashboard.html
   ```

8. Review the dashboard and candidate prompt file manually.

   The dashboard should make the review decision explainable. For each
   category, it should show:

   - v1 prompt overlay
   - proposed v2 prompt overlay
   - human-readable prompt diff
   - question/example detail rows with golden answer, predicted answer,
     failure mode, token F1, and split assignment
   - derived answer format profile based on golden answers
   - generator examples used to propose the change
   - evaluator examples not shown to the generator
   - evaluator feedback for each loop
   - accepted, rejected, or revised decision
   - expected failure modes fixed
   - regression risks
   - links or anchors to the raw harness JSONL records

   Accepted patches should be concise, category-specific, and explainable. A
   rejected category keeps its `v1` prompt. After review, promote the candidate
   prompt file to the editable source:

   ```text
   prompts/system_prompts_v2.py
   ```

   Do not use `outputs/v2/system_prompts.py` as the canonical editable prompt
   source. Files under `outputs/` are run artifacts. The evaluator can still
   copy the active prompt set into `outputs/v2/system_prompts.py` for audit
   traceability, but prompt loading and future edits should happen from
   `prompts/`.

9. Run the real evaluator with the v2 prompt source and `model_id=v2`.

   The evaluator should be extended to accept the holdout row ids from
   `outputs/v2/prompt_harness/splits.json`, or an equivalent filtered dataset,
   so the final score is calculated on `holdout_eval` rather than the examples
   used by the generator/evaluator harness.

   ```bash
   uv run python dspy_eval_v1.py \
     --model-id v2 \
     --prompts-file prompts/system_prompts_v2.py \
     --eval-split outputs/v2/prompt_harness/splits.json:holdout_eval \
     --output-dir outputs
   ```

10. Compare `v1` vs `v2`, using the untouched holdout as the quality signal:

   - overall mean token F1
   - correct at 0.5
   - per-category deltas
   - regressions in categories that were already strong
   - failure-mode deltas, especially `classification_instead_of_span` and
     `false_no_answer`

## Important Evaluation Guidance

Do not optimize only on incorrect answers from the same final evaluation set.
For v2, treat the `v1` incorrect-answer set as prompt-development data and
split it into:

- `generator_dev`: examples used to create prompt patches
- `evaluator_dev`: different examples used to review prompt patches
- `holdout_eval`: examples used only for final generalization scoring

The generator/evaluator loop is still prompt development. It is useful because
it reduces direct overfitting to the generator examples, but it does not replace
the need for an untouched holdout. Final claims about v2 should be based on the
holdout and regression review.

The v1 prompt source should be preserved at:

```text
prompts/system_prompts_v1.py
```

## Answer Format From Golden Answers

The answer format for each prompt should be derived from the dataset's golden
answers, not from the CSV label alone. The CSV `answer_format` field is useful
metadata, but it can make a span-extraction task look like a classification
task. The extraction prompt should follow what the gold answers actually
require.

For each category, compute an answer format profile:

```text
category
csv_answer_format
gold_answer_type
requires_verbatim_contract_span
yes_no_label_only
typical_span_shape
allows_multiple_spans
allows_no_answer
evidence_notes
```

Use these rules:

- If every non-empty golden answer is exactly `Yes` or `No`, then the required
  answer format is a label.
- If non-empty golden answers contain contract language, then the required
  answer format is verbatim extractive span text, even when the CSV says
  `Yes/No`.
- If a golden answer matches one contract sentence, require a single verbatim
  sentence or clause.
- If a golden answer contains multiple sentences, require the full verbatim
  multi-sentence span only when all sentences are needed to answer the
  question.
- If a row has multiple golden answers, require multiple newline-separated
  spans.
- If the golden answer list is empty or the gold row is marked impossible, allow
  `NO_ANSWER`.

The generator and evaluator agents should receive this derived profile instead
of a bare `Answer format: Yes/No` line. For example:

```text
Derived answer format:
- CSV answer format: Yes/No
- Gold answer type: verbatim_contract_span
- Required output: exact clause text, not Yes/No
- Typical span shape: single clause or sentence
- Multiple spans allowed: yes
- NO_ANSWER allowed: yes, only when no supporting span exists
```

The dashboard should show this profile for each category so a human reviewer
can verify that prompt changes are grounded in the actual golden answers.

## Prompt-Improvement Request Shape

The harness should use the stricter generator/evaluator request shape below.
The older one-agent request shape is useful only as a manual fallback, not as
the recommended v2 workflow.

Manual fallback request:

```text
You are improving a CUAD extraction system prompt.

Current prompt:
...

Category:
Anti-Assignment

Category description:
Is consent or notice required of a party if the contract is assigned to a third party?

Derived answer format from golden answers:
- CSV answer format: Yes/No
- Gold answer type: verbatim_contract_span
- Required output: exact clause text, not Yes/No
- Typical span shape: single clause or sentence
- Multiple spans allowed: yes
- NO_ANSWER allowed: yes, only when no supporting span exists

Observed failures:
1.
Contract title: ...
Question: ...
Golden answer: ...
Predicted answer: ...
Gold marked impossible: false
Predicted marked impossible: true
Token F1: 0.0

Task:
Identify recurring failure patterns.
Then produce a revised system prompt that improves extraction for this category.
Keep it concise.
Do not add instructions that conflict with exact-span extraction.
Return only:
- failure_analysis
- revised_system_prompt
- regression_risks
```

Generator request:

```text
You are improving a legal clause extraction prompt.

Current base extraction prompt:
...

Current category overlay:
...

Category:
Anti-Assignment

Category description:
Is consent or notice required of a party if the contract is assigned to a third party?

Derived answer format from golden answers:
- CSV answer format: Yes/No
- Gold answer type: verbatim_contract_span
- Required output: exact clause text, not Yes/No
- Typical span shape: single clause or sentence
- Multiple spans allowed: yes
- NO_ANSWER allowed: yes, only when no supporting span exists

Failure-mode summary:
- classification_instead_of_span: 12
- false_no_answer: 4
- partial_span: 3

Generator examples:
...

Task:
1. Identify the 2-3 prompt issues causing these failures.
2. Revise only the category overlay unless the base extraction prompt is clearly defective.
3. Keep the revised overlay concise.
4. Include positive evidence cues and exclusion cues.
5. Do not optimize for this category in a way that would conflict with exact-span extraction.

Return:
- failure_analysis
- revised_category_overlay
- optional_base_prompt_patch
- regression_risks
```

Evaluator request:

```text
You are reviewing a proposed legal clause extraction prompt patch.

Generator instructions:
...

Current base extraction prompt:
...

Current category overlay:
...

Proposed category overlay:
...

Category:
Anti-Assignment

Category description:
Is consent or notice required of a party if the contract is assigned to a third party?

Derived answer format from golden answers:
- CSV answer format: Yes/No
- Gold answer type: verbatim_contract_span
- Required output: exact clause text, not Yes/No
- Typical span shape: single clause or sentence
- Multiple spans allowed: yes
- NO_ANSWER allowed: yes, only when no supporting span exists

Evaluator examples not shown to generator:
...

Task:
1. Judge whether the proposed patch is likely to generalize to these unseen examples.
2. Identify likely improvements and likely regressions.
3. Reject broad rewrites, prompt bloat, or classification-style answers.
4. Decide: accept, revise, or reject.

Return:
- decision
- generalization_score
- rationale
- likely_fixes
- likely_regressions
- requested_changes
```

## Key Prompt Risk

Many CUAD categories have `Yes/No` in the CSV answer format, but the golden
answers may still be contract text spans. The model should not answer only
`Yes` or `No` unless the golden answers for that category are truly only
`Yes`/`No` labels.

Prompts should explicitly say:

```text
Even when the answer format is Yes/No, return the supporting contract span,
not only "Yes" or "No". Return NO_ANSWER only when no supporting span exists.
```

Use that instruction only when the answer format profile derived from golden
answers says the category requires verbatim contract spans. If the profile says
the gold answers are truly label-only, keep the prompt as a label task.

## Proposed V2 Prompt Architecture

Use a two-layer prompt architecture:

```text
BASE_EXTRACTION_SYSTEM_PROMPT
CATEGORY_SYSTEM_PROMPTS[category]
```

The base prompt should be shared by all agents:

```text
You are a legal contract clause extraction assistant.

Follow the derived answer format profile for this category.
When the profile requires verbatim spans, return exact supporting text span(s)
from the contract.
Do not summarize or answer from legal knowledge outside the contract.
If multiple clauses answer the question, return each span on a separate line.
If no supporting span exists and the profile allows no-answer, return NO_ANSWER.
Return only Yes or No when the gold-answer profile shows that the category is
truly label-only.
Set marked_impossible to true only when returning NO_ANSWER.
```

Each category overlay should contain:

```text
Category: ...
Definition: ...
Derived answer format: ...
Look for: ...
Exclude: ...
Answer span guidance: ...
```

Example category overlay:

```text
Category: Anti-Assignment
Definition: consent, notice, or restriction requirements triggered by assignment
or transfer of the contract or rights.
Look for: assign, assignment, transfer, delegate, merger, sale of assets,
successor, assignment by operation of law.
Exclude: general notices clauses, payment assignment mechanics, or unrelated
IP ownership transfers unless they restrict contract assignment.
Answer span guidance: return the clause that states the consent, notice, or
restriction requirement.
```

## Example Category-Specific Patch

For `Anti-Assignment`:

```text
For this category, look for clauses about assignment, transfer, delegation,
change of control, merger, sale of assets, or assignment by operation of law.
Return the clause text that states whether consent, notice, or restriction is
required. Do not answer only "Yes" or "No"; return the supporting span.
```

## Engineering Recommendation

Create a script such as:

```text
prompt_improve_v2.py
```

The script should:

- load `outputs/v1/cuad_dspy_eval_results.csv`
- group failures by category
- inspect golden answers by category and derive answer format profiles
- label or infer failure modes per row
- create deterministic `generator_dev`, `evaluator_dev`, and `holdout_eval`
  splits
- load source prompts from:

  ```text
  prompts/system_prompts_v1.py
  ```

- run the generator/evaluator loop and write harness artifacts to:

  ```text
  outputs/v2/prompt_harness/
  ```

- use PydanticAI agents with `deepseek/deepseek-v4-pro` when run with
  `--use-llm`; local deterministic mode can remain available for tests and
  offline dry runs

- write derived answer format profiles to:

  ```text
  outputs/v2/prompt_harness/answer_format_profiles.json
  ```

- write the reviewable candidate prompt source to:

  ```text
  outputs/v2/prompt_harness/prompts_candidate_v2.py
  ```

- promote the reviewed candidate to `prompts/system_prompts_v2.py`

The evaluator should then load `prompts/system_prompts_v2.py` and copy the
effective prompts used for the run into:

```text
outputs/v2/system_prompts.py
```

Keep the generated prompt changes reviewable before running `v2`.

## Human Review Dashboard

Add a static HTML dashboard generated by `prompt_improve_v2.py`:

```text
outputs/v2/prompt_harness/prompt_review_dashboard.html
```

This dashboard is the human-facing review surface for the prompt optimization
loop. It should answer one question clearly: "Why did this prompt change, and
did the evaluator evidence support it?"

Recommended structure:

- summary header with model id, source run, split counts, accepted patches,
  rejected patches, and categories requiring human attention
- category table with current score, main failure mode, loop count, evaluator
  decision, derived answer format, generalization score, and regression risk
- category detail view with the v1 prompt, proposed v2 prompt, and a readable
  line-level diff
- answer-format profile panel showing how golden answers determine whether the
  category is label-only or verbatim span extraction
- question/example detail rows showing contract title, question, golden answer,
  v1 predicted answer, failure mode, token F1, and split assignment
- generator panel showing the exact `generator_dev` examples used to propose
  the change
- evaluator panel showing only the separate `evaluator_dev` examples used to
  review generalization
- loop timeline showing generator output, evaluator feedback, and how attempts
  2 and 3 changed the prompt
- final decision panel showing accepted/rejected/revise status and rationale
- holdout results panel after the final evaluator run, showing whether the
  accepted prompt improved unseen examples and which questions still failed

The dashboard should support category filtering, decision filtering, and
failure-mode filtering. It should also support expanding an individual
question/example so the reviewer can see how the golden answer differs from the
model answer and why the prompt was changed. It should make prompt bloat
visible by showing prompt length deltas and changed rules. A reviewer should be
able to inspect one category or one failed question and understand exactly how
evaluator feedback guided the final system prompt.

Dashboard artifacts should be generated from the JSONL files, not from hidden
agent state, so the page can be regenerated and audited.

## Generator/Evaluator Harness

Yes, it is possible and recommended to have a generator agent improve prompts
from one sample and an evaluator agent review those changes against a separate
sample. This is a better design than letting one agent see all failures because
it tests whether the proposed prompt update generalizes beyond the exact
examples used to write it.

Use three data partitions:

- `generator_dev`: 20 failures per category where possible, visible to the
  generator agent
- `evaluator_dev`: 20 different failures per category where possible, visible
  to the evaluator agent
- `holdout_eval`: never shown to either agent, used only for final scoring

If a category has fewer than 40 failures, split what exists deterministically
and record the counts. The evaluator should never see the same examples used
by the generator in that loop.

The harness should run up to three generator/evaluator loops per category:

1. Generator receives the current prompt, category metadata, failure-mode
   summary, and `generator_dev` examples. It proposes a concise category
   overlay patch and optional base prompt patch.
2. Evaluator receives the generator instructions, generator output, current
   prompt, category metadata, and `evaluator_dev` examples. It decides whether
   the patch likely generalizes, identifies regression risks, and either
   accepts, rejects, or requests a revision.
3. If the evaluator requests a revision, the generator receives the original
   generator guide, the previous generated prompt, the evaluator feedback on
   that prompt, and its original `generator_dev` examples. It should not
   receive `evaluator_dev` examples.
4. Stop after acceptance or after three loops. If no patch is accepted, keep
   the previous prompt for that category and write the rejection rationale.

The evaluator can see the generator's prompt-improvement instructions and
proposed output because the purpose is to judge whether the reasoning and
prompt patch would work on unseen examples. It should not reveal the
`evaluator_dev` examples back to the generator, otherwise the loop collapses
into tuning on both sets.

### Harness Artifacts

Write every step to `outputs/v2/prompt_harness/`:

```text
outputs/v2/prompt_harness/
  splits.json
  answer_format_profiles.json
  category_runs.jsonl
  accepted_patches.jsonl
  rejected_patches.jsonl
  evaluator_reviews.jsonl
  prompt_diffs.jsonl
  prompts_candidate_v2.py
  prompt_review_dashboard.html
```

Recommended artifact contents:

- `splits.json`: row ids assigned to `generator_dev`, `evaluator_dev`, and
  `holdout_eval`
- `answer_format_profiles.json`: category-level required output formats derived
  from golden answers, including label-only vs verbatim span requirements
- `category_runs.jsonl`: every generator/evaluator loop with inputs, outputs,
  and decisions
- `accepted_patches.jsonl`: final accepted patches by category
- `rejected_patches.jsonl`: categories where no patch was accepted
- `evaluator_reviews.jsonl`: evaluator decisions and rationale
- `prompt_diffs.jsonl`: category-level prompt diffs and changed-rule summaries
- `prompts_candidate_v2.py`: generated prompt source that a human reviews
  before copying to `prompts/system_prompts_v2.py`
- `prompt_review_dashboard.html`: static review UI explaining how evaluator
  feedback changed each candidate prompt

### PydanticAI Harness Shape

The harness can be implemented as a PydanticAI multi-agent script. The exact
agent model for v2 prompt optimization should be:

```text
deepseek/deepseek-v4-pro
```

Add `pydantic-ai` to the project dependencies before implementing this script
if it is not already present.

```python
from __future__ import annotations

import os

from enum import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider


LLM_MODEL_ID = "deepseek/deepseek-v4-pro"


GENERATOR_SYSTEM_PROMPT = (
    "You improve legal clause extraction prompts. "
    "Patch only what the evidence supports. Keep category overlays concise. "
    "Never convert exact-span extraction into classification."
)

EVALUATOR_SYSTEM_PROMPT = (
    "You are a skeptical prompt evaluator for legal clause extraction. "
    "Review whether a proposed patch generalizes to unseen failures. "
    "Prefer rejecting broad rewrites, prompt bloat, and changes that risk "
    "false positives."
)


class FailureExample(BaseModel):
    row_id: str
    category: str
    contract_title: str
    question: str
    golden_answer: str
    predicted_answer: str
    gold_marked_impossible: bool
    predicted_marked_impossible: bool
    token_f1: float
    failure_mode: str


class AnswerFormatProfile(BaseModel):
    category: str
    csv_answer_format: str
    gold_answer_type: Literal["label_only", "verbatim_contract_span", "mixed", "no_answer_only"]
    requires_verbatim_contract_span: bool
    yes_no_label_only: bool
    typical_span_shape: Literal["none", "single_sentence", "multi_sentence", "multiple_spans", "mixed"]
    allows_multiple_spans: bool
    allows_no_answer: bool
    evidence_notes: list[str]


class PromptPatchRequest(BaseModel):
    model_id: str = "v2"
    category: str
    category_description: str
    answer_format: str
    current_base_prompt: str
    current_category_overlay: str
    answer_format_profile: AnswerFormatProfile
    failure_mode_summary: dict[str, int]
    generator_examples: list[FailureExample] = Field(max_length=20)
    original_generator_guide: str
    previous_generated_prompt: str | None = None
    evaluator_feedback: str | None = None
    loop_index: int


class PromptPatch(BaseModel):
    category: str
    failure_analysis: list[str]
    revised_category_overlay: str
    optional_base_prompt_patch: str | None = None
    expected_improvements: list[str]
    regression_risks: list[str]
    changed_rules: list[str]
    prompt_diff_summary: list[str]


class PromptReviewRequest(BaseModel):
    category: str
    category_description: str
    answer_format: str
    current_base_prompt: str
    current_category_overlay: str
    answer_format_profile: AnswerFormatProfile
    generator_instructions: str
    generator_patch: PromptPatch
    evaluator_examples: list[FailureExample] = Field(max_length=20)
    loop_index: int


class PromptReview(BaseModel):
    decision: Literal["accept", "revise", "reject"]
    generalization_score: float = Field(ge=0.0, le=1.0)
    rationale: list[str]
    likely_fixes: list[str]
    likely_regressions: list[str]
    requested_changes: list[str]


class DashboardExampleRecord(BaseModel):
    row_id: str
    split: Literal["generator_dev", "evaluator_dev", "holdout_eval"]
    contract_title: str
    question: str
    golden_answer: str
    predicted_answer: str
    failure_mode: str
    token_f1: float


class DashboardCategoryRecord(BaseModel):
    category: str
    decision: Literal["accept", "revise", "reject"]
    loop_count: int
    main_failure_mode: str
    answer_format_profile: AnswerFormatProfile
    v1_prompt: str
    candidate_v2_prompt: str
    prompt_diff_summary: list[str]
    generator_example_ids: list[str]
    evaluator_example_ids: list[str]
    evaluator_feedback: list[str]
    regression_risks: list[str]
    generalization_score: float
    examples: list[DashboardExampleRecord]


deepseek_model = OpenAIChatModel(
    LLM_MODEL_ID,
    provider=DeepSeekProvider(api_key=os.environ["DEEPSEEK_API_KEY"]),
)

generator_agent = Agent(
    deepseek_model,
    output_type=PromptPatch,
    system_prompt=GENERATOR_SYSTEM_PROMPT,
)

evaluator_agent = Agent(
    deepseek_model,
    output_type=PromptReview,
    system_prompt=EVALUATOR_SYSTEM_PROMPT,
)


async def improve_category(
    request: PromptPatchRequest,
    evaluator_examples: list[FailureExample],
) -> PromptPatch | None:
    current_request = request
    previous_patch: PromptPatch | None = None

    for loop_index in range(1, 4):
        current_request.loop_index = loop_index
        patch = await generator_agent.run(current_request)

        review_request = PromptReviewRequest(
            category=request.category,
            category_description=request.category_description,
            answer_format=request.answer_format,
            current_base_prompt=request.current_base_prompt,
            current_category_overlay=request.current_category_overlay,
            answer_format_profile=request.answer_format_profile,
            generator_instructions=GENERATOR_SYSTEM_PROMPT,
            generator_patch=patch.output,
            evaluator_examples=evaluator_examples[:20],
            loop_index=loop_index,
        )
        review = await evaluator_agent.run(review_request)

        write_loop_artifacts(
            category=request.category,
            loop_index=loop_index,
            patch=patch.output,
            review=review.output,
        )

        if review.output.decision == "accept":
            return patch.output

        if review.output.decision == "reject":
            return None

        previous_patch = patch.output
        current_request.previous_generated_prompt = (
            previous_patch.revised_category_overlay
        )
        current_request.evaluator_feedback = "\n".join(
            review.output.requested_changes
        )

    return None
```

The script should then assemble accepted category overlays into:

```text
outputs/v2/prompt_harness/prompts_candidate_v2.py
```

A human should review that candidate file before it becomes:

```text
prompts/system_prompts_v2.py
```

After that, run the real evaluator against the untouched holdout:

```bash
uv run python dspy_eval_v1.py \
  --model-id v2 \
  --prompts-file prompts/system_prompts_v2.py \
  --eval-split outputs/v2/prompt_harness/splits.json:holdout_eval \
  --output-dir outputs
```

### Harness Guardrails

- The generator must not see `evaluator_dev` or `holdout_eval` examples.
- The evaluator must judge against unseen examples and regression risk, not
  whether the patch perfectly explains the generator's examples.
- The evaluator can request at most two revisions. The third generator output
  is accept-or-reject.
- Accepted patches should be small enough to review as diffs.
- A category with no accepted patch keeps the v1 prompt.
- Final claims about v2 quality must use the untouched holdout results, not the
  generator/evaluator loop decisions.
- Prompts must follow the answer format profile derived from golden answers,
  not the CSV `answer_format` label alone.

## Acceptance Criteria for V2

Before accepting `v2`, require:

- overall mean token F1 improves over `v1`
- correct at 0.5 improves over `v1`
- no high-confidence category regresses by more than an agreed threshold
- categories with many `Yes`/`No` predictions show fewer
  `classification_instead_of_span` failures
- derived answer format profiles are documented and consistent with golden
  answers
- `NO_ANSWER` behavior improves or remains stable
- reviewed prompt diffs are concise and explainable

Recommended minimum reporting table:

```text
category | v1_f1 | v2_f1 | delta_f1 | v1_correct@0.5 | v2_correct@0.5 | main_failure_mode_fixed | regressions
```

## What Not To Do

- Do not paste all wrong examples into an LLM and accept a full prompt rewrite.
- Do not tune only for aggregate score.
- Do not let `Yes/No` answer format become a classification task.
- Do not use generated `outputs/<model_id>/system_prompts.py` as the editable
  source of truth.
- Do not declare v2 better without a holdout or regression review.

## Prompt Source Layout

Use this structure for editable prompt versions:

```text
prompts/
  system_prompts_v1.py
  system_prompts_v2.py
```

Use this structure for run artifacts:

```text
outputs/
  v1/
    cuad_dspy_eval_results.csv
    cuad_dspy_eval_summary.json
    cuad_dspy_eval.html
    system_prompts.py
  v2/
    prompt_harness/
      splits.json
      answer_format_profiles.json
      category_runs.jsonl
      evaluator_reviews.jsonl
      accepted_patches.jsonl
      rejected_patches.jsonl
      prompt_diffs.jsonl
      prompts_candidate_v2.py
      prompt_review_dashboard.html
    cuad_dspy_eval_results.csv
    cuad_dspy_eval_summary.json
    cuad_dspy_eval.html
    system_prompts.py
```

The `outputs/<model_id>/system_prompts.py` file is a snapshot of the prompts
used for that run. The editable source of truth lives under `prompts/`.
