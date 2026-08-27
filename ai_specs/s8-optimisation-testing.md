name: "ConvFinQA Prompt-Improvement Bench — Comparative Testing of Three Optimisation Techniques Against the s7 Baseline"

## Purpose

The s7 harness (per-case Diagnose → Propose → Verify, with router + 4 specialists) is **one** way to extract rules from `(gold, pred, reasoning trace)` triples and append them to sub-agent system prompts. It's not necessarily the best way. Recent literature (2023–2026) describes several distinct families of teacher-LLM-driven prompt optimisation, each with different granularity, aggregation strategy, and target surface. **This spec defines a benchmark harness that runs the same input data (`pydantic_predictions_v2.csv`) through three alternative techniques and compares them head-to-head against s7 (`v3_1`) and the v2 baseline on the held-out test sample.**

The three techniques are chosen to cover three orthogonal design axes:

| Axis | s7 (baseline) | Tech A: TextGrad | Tech B: ProTeGi | Tech C: PromptWizard |
|---|---|---|---|---|
| **Aggregation** | Per-case | Per-case but with cross-module gradients | Mini-batch of cases | Population (whole prompt set) |
| **Edit surface** | Append a new rule | Edit any module's prompt via textual gradient | Edit current prompt in "opposite direction of error" | Rewrite instruction + synthesize new in-context examples |
| **Scope** | One agent per case | All 4 agents simultaneously (gradient flows through compound pipeline) | One agent at a time | One agent at a time |
| **Promotion criterion** | Verified replay (turns 0..k) | Validation accuracy on held-out batch | Beam-search score on validation batch | Critic loop until score plateaus |

Goal: an empirical answer to "which optimisation technique best improves the ConvFinQA pipeline?" — measured on the same held-out 770-question evaluation that `convfinqa-eval` already reports, so results plug into the existing comparison table alongside `v1`, `v2`, `v3_1`.

## Why these three techniques (and not the others)

Five techniques were considered; two were excluded:

- **OPRO (Optimisation by PROmpting, Yang et al. 2023)** — Uses a "meta-prompt" with the trajectory of `(prompt, score)` pairs and asks an LLM to generate the next candidate. Excluded because GEPA (already in this repo, via `convfinqa-optimize`) is a strict superset: GEPA does the same trajectory-based mutation but adds Pareto-frontier selection and reflective traces. Running OPRO would re-test what GEPA already covers.
- **Reflexion (Shinn et al. 2023)** — Verbal reflections stored in an episodic memory buffer prepended to the prompt. Excluded because it's a *runtime* memory mechanism (each turn sees its own past reflections), not a prompt-optimisation method. It would require redesigning the production pipeline rather than just substituting an optimiser.

The three selected — **TextGrad**, **ProTeGi**, **PromptWizard** — each test a different axis (cross-module gradient flow, mini-batch aggregation, example synthesis) and each have published reference implementations.

---

## Technique A — TextGrad: textual back-propagation through the compound pipeline

### Why it's interesting for ConvFinQA

The ConvFinQA pipeline is a **compound AI system** with four sequential modules: Triage → Preprocess → Retriever → Calculator. Each module has its own `system_prompt`. A wrong final answer can be caused by any one of them, and s7's diagnostic router has to *guess* which one. TextGrad replaces that guess with **explicit textual gradients flowing backwards from the final answer through every intermediate variable, including each module's system prompt**.

Where s7 emits one rule appended to one agent per case, TextGrad emits a coordinated edit across **all four prompts simultaneously**, with each prompt's edit reflecting that module's specific contribution to the error.

### Reference

Yuksekgonul et al., *"TextGrad: Automatic Differentiation via Text"*, Nature 2024. Python framework: `textgrad` (Zou group, Stanford). [Repo](https://github.com/zou-group/textgrad). [Stanford HAI summary](https://hai.stanford.edu/news/textgrad-autograd-text).

### How it works (TL;DR)

1. **Forward pass** — Build a computation graph where each node is one module's call: `triage_io = triage(question, history)`, `preprocess_io = preprocess(question, triage_io, history)`, etc. Each module's `system_prompt` is a *learnable variable* (a "Parameter" in TextGrad terminology). The final node is the executed answer.
2. **Loss computation** — Use a natural-language loss: a teacher LLM compares `pred_answer` to `gold_answer` (and `pred_program` to `gold_program`) and produces a free-text critique. This critique is the "loss" — it lives in language, not numbers.
3. **Backward pass** — A "backward engine" LLM asks, for each parameter (each module's `system_prompt`): *"Given this loss critique and the trace through this module, how should this system prompt change to reduce the loss?"* The answer is a **textual gradient** — a natural-language description of the required edit, attached to that prompt.
4. **Optimiser step** — A TGD ("Textual Gradient Descent") optimiser applies each gradient by asking the optimiser LLM to rewrite that parameter using the gradient as guidance.
5. **Repeat** — Standard training loop: minibatch, forward, loss, backward, step.

### What's different from s7

| s7 | TextGrad |
|---|---|
| Router classifies *which agent* is to blame → one rule for that agent. | Gradients flow to *all four* agents; each one's prompt is edited if its module contributed to the loss. |
| Rule is appended (additive). | Prompt is rewritten (replacement). |
| Per case, sequential. | Per minibatch, with parallel gradients across modules. |
| Verify replay decides promotion. | Validation-set accuracy decides whether the step "sticks". |

### Integration plan

- New `src/convfinqa/diagnosis/techniques/textgrad/` package.
- Wrap the four production sub-agents in TextGrad Parameter blocks (one per `system_prompt`).
- Loss function: teacher LLM (LM_MAX = `deepseek-v4-pro`) compares `(gold_answer, gold_program, pred_answer, pred_program)` and emits a 1–3 sentence critique. Fall back to a simple "matches/doesn't match" string for cases where the critique LLM rate-limits.
- Backward engine + optimiser engine: also LM_MAX.
- Output variant suffix: `--variant v3_1_textgrad` so artifacts (`prompts/v3_1_textgrad.py`, `pydantic_predictions_v3_1_textgrad.csv`) live alongside the s7 outputs.
- Training set: same first-wrong-per-conversation cases s7 uses (so the comparison is apples-to-apples on training data).

### Expected risks

- **Cost** — Forward + loss + backward + step is 4 LLM calls per case minimum; with retries can balloon.
- **Drift** — Replacing prompts wholesale risks regressing previously-correct behaviour. Need a held-out validation set to gate each step.
- **Coordination** — Gradients on all four prompts simultaneously can produce conflicting edits (Triage and Preprocess both "fixing" the same failure). The TextGrad framework includes anti-correlation regularisation; need to verify it's enabled.

---

## Technique B — ProTeGi: textual gradients on mini-batches with beam search

### Why it's interesting for ConvFinQA

s7 reasons about one failing case at a time. ProTeGi reasons about a **batch of failures** at a time, which surfaces *patterns* across cases that any single case wouldn't reveal. A rule extracted from one case ("when the question says 'in percentage', emit `multiply(...,100)`") might be brittle; the same rule extracted from a batch of 20 percentage-related failures is robust by construction.

ProTeGi also adds **beam search over candidate edits**: instead of accepting the first edit a teacher LLM proposes, it generates several candidates, scores them on a validation batch, and keeps the top-`k`.

### Reference

Pryzant et al., *"Automatic Prompt Optimization with 'Gradient Descent' and Beam Search"*, EMNLP 2023. [Paper PDF](https://arxiv.org/pdf/2305.03495). [OpenReview](https://openreview.net/forum?id=WRYhaSrThy).

### How it works (TL;DR)

1. **Sample a minibatch** of failing cases (e.g. 20 first-wrong cases).
2. **Generate textual gradients** — A teacher LLM looks at the batch and emits a 2–4 sentence natural-language critique of the current prompt. The critique points *away* from the error pattern, much like a numerical gradient points away from loss.
3. **Apply the gradient** — A second LLM call edits the current prompt in the *opposite semantic direction* of the gradient. Generate `n` candidate edits per gradient (default n=4).
4. **Beam search** — Of all candidate prompts, evaluate each on a held-out validation batch and keep the top-`k` (default k=2).
5. **Bandit selection** — Use a UCB-style bandit to balance exploring under-evaluated candidates vs. exploiting promising ones.
6. **Iterate** for `T` rounds (default T=6).

### What's different from s7

| s7 | ProTeGi |
|---|---|
| Examines one case at a time. | Examines a batch of N cases. |
| Diagnostic router picks the failing agent. | Beam search over edits; no per-case routing — all edits target the *currently-optimised* agent. |
| Rule is appended to v2 baseline. | Prompt is edited in place; subsequent edits build on the edited prompt. |
| Verify replay (turns 0..k) gates promotion. | Validation-batch score gates promotion via beam-top-`k`. |
| Sequential, deterministic. | Parallel candidate generation + bandit exploration. |

### Integration plan

- New `src/convfinqa/diagnosis/techniques/protegi/` package.
- Run ProTeGi **per agent** (one beam search for Triage's prompt, one for Preprocess's, etc.) since the production pipeline is four separate prompts. This is a notable adaptation — the original ProTeGi optimises one prompt at a time, so the 4-agent loop runs ProTeGi four times.
- Use the same first-wrong-per-conversation training set as s7. Split into 80/20 train/val for the beam search.
- Hyperparameters: batch size 20, beam width 2, candidates per gradient 4, rounds 6. (Match the paper's defaults for direct comparability.)
- Output variant: `--variant v3_1_protegi`.

### Expected risks

- **Adapting ProTeGi to 4-prompt compound systems** — Not in the original paper. Running ProTeGi sequentially per agent (with all other agents fixed) may not converge to the joint optimum.
- **Catastrophic edits** — ProTeGi can rewrite the prompt entirely. Need a regression check against the v2 baseline so an edit that improves on the training batch but tanks unrelated cases is rejected.
- **Validation-set leak** — With only ~95 first-wrong cases total, the 80/20 split leaves ~19 validation cases per agent. Beam search can over-fit to that small validation set. Mitigation: rotate the val split across rounds.

---

## Technique C — PromptWizard: critic + mutator + example synthesis

### Why it's interesting for ConvFinQA

Both s7 and the other two techniques above modify only the **instruction text** of the prompt. PromptWizard adds a third surface: **in-context examples**. Current v2 prompts don't include any worked examples; PromptWizard would synthesize task-specific examples (drawn from failures) and weave them into the prompt alongside any instruction edits.

For a domain-specific task like ConvFinQA (where the DSL `add/subtract/multiply/divide/exp/greater` has tight conventions like "percentage answers need `multiply(...,100)` outermost"), worked examples may carry more information per token than additional instruction rules.

### Reference

Agarwal et al., *"PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework"*, Microsoft Research 2024. [Microsoft blog](https://www.microsoft.com/en-us/research/blog/promptwizard-the-future-of-prompt-optimization-through-feedback-driven-self-evolving-prompts/). [Repo](https://github.com/microsoft/PromptWizard).

### How it works (TL;DR)

PromptWizard has three components running in a loop:

1. **Mutator** — Given the current prompt + task description, generate `n` variations (default n=5). Variations rephrase, restructure, or add emphasis.
2. **Critic** — Score each variation on a small task batch, then have a teacher LLM read the worst-performing variations and write a structured critique: which trigger words underperform, which instructions seem unclear, which omissions surface in the failures.
3. **Synthesis** — Use the critique to generate **new in-context examples** that specifically address the weaknesses identified. The synthesised examples are then integrated into the next round's prompt — the critic guides which examples to add, not just which words to change.

Steps 1–3 repeat for `T` rounds (default T=5) or until score plateaus.

### What's different from s7

| s7 | PromptWizard |
|---|---|
| Only edits instruction text. | Edits instruction text **AND** synthesises in-context examples. |
| One-shot rule proposal per case. | Five candidate mutations per round, scored against a batch. |
| Verify replay of turns 0..k. | Score on a separate batch decides which mutation wins. |
| No example generation. | Critic explicitly drives example synthesis from failure patterns. |

### Integration plan

- New `src/convfinqa/diagnosis/techniques/promptwizard/` package.
- Per-agent loop (same shape as ProTeGi — run PromptWizard four times, once per sub-agent).
- Critic LLM and Mutator LLM both = LM_MAX (`deepseek-v4-pro`); inference still runs on LM_MINI (`deepseek-v4-flash`).
- Example synthesis: synthesised examples are appended under a new `## Worked Examples` section in the generated prompt. This is a structural addition not present in v2.
- Output variant: `--variant v3_1_promptwizard`.

### Expected risks

- **Example hallucination** — Synthesised examples can be subtly wrong (e.g. a "correct" example with a miscomputed answer would actively mislead the production model). Every synthesised example must be verified by executing its program against the example's claimed inputs before inclusion.
- **Token budget** — Adding worked examples to all four prompts inflates the context size for every production call. Need to measure inference cost delta vs s7.
- **Convergence** — PromptWizard has more knobs than the other two; tuning to a sensible operating point will require a few exploratory runs.

---

## Bench design — running all four variants apples-to-apples

The point of s8 is comparability. Every technique must produce a `v3_1_<name>` variant that plugs into the existing `convfinqa-eval` comparison table without special-casing.

### Shared inputs

- **Training cases**: same first-wrong-per-conversation filter used by s7 (`pydantic_predictions_v2.csv` → 95 cases).
- **Validation set** (for techniques that need one): 20% holdout from the training cases, fixed seed (no per-technique leakage).
- **Held-out test set**: the 770-question evaluation sample used by `convfinqa-eval`. This is the **only** scoring surface that counts for the final comparison — no technique gets to see it during optimisation.
- **Base prompts**: v2 (for all techniques).
- **Inference model**: LM_MINI (`deepseek-v4-flash`) for the production sub-agents — same as v2/v3_1, so improvements are attributable to prompt quality alone.
- **Optimiser model**: LM_MAX (`deepseek-v4-pro`) for every technique's teacher / critic / backward engine.

### Outputs (per technique)

Each technique writes its variant artifacts under the existing variant naming convention (§Variants in `s7-prompt-optimisation.md`):

```
src/convfinqa/prompts/v3_1_textgrad.py
src/convfinqa/prompts/v3_1_protegi.py
src/convfinqa/prompts/v3_1_promptwizard.py
evaluation/pydantic_predictions_v3_1_textgrad.csv
evaluation/pydantic_predictions_v3_1_protegi.csv
evaluation/pydantic_predictions_v3_1_promptwizard.csv
```

…and corresponding `_joined` / `.html` reports plus a single comparison summary:

```
evaluation/s8_bench_comparison.{csv,html}
```

The `latest_all()` regex in `prompts/__init__.py` must be widened to include `v3_1_<tag>` patterns (currently `^v\d+(_\d+)?$`). Proposed update: `^v\d+(_\d+)?(_[a-z][a-z0-9]*)?$`.

### Metrics

Beyond raw accuracy on the 770-question test set, compute the following slices for the comparison table:

- **Overall accuracy** (v2, v3_1, v3_1_textgrad, v3_1_protegi, v3_1_promptwizard).
- **Accuracy by turn_type** (Number vs Program) — different techniques may favour different turn shapes.
- **Accuracy by conv_type** (Type I vs Type II).
- **Accuracy by turn_index** (does the technique help deep turns more / less?).
- **Cost** — LLM calls during optimisation (tokens × $/1M). Cheap is good.
- **Prompt size delta** — Output prompt token count vs v2. Cheap inference is good.
- **Regression count** — Cases that were correct in v2 and are wrong in the new variant. Zero is ideal.

### Operator entry point

```bash
# Run all three techniques + score
uv run python scripts/bench_optimisation.py --techniques textgrad,protegi,promptwizard

# Run one technique standalone
uv run python scripts/bench_optimisation.py --techniques textgrad

# Skip optimisation and just re-score existing variants (cached prediction CSVs)
uv run python scripts/bench_optimisation.py --score-only
```

### Comparison report

`evaluation/s8_bench_comparison.{csv,html}` is the human-facing artefact. Table shape:

```
Cut                      Count    v2      v3_1   v3_1_textgrad   v3_1_protegi   v3_1_promptwizard
Overall                    770   77.1%   ?.?%        ?.?%             ?.?%             ?.?%
turn_type=Number           284   87.7%   ?.?%        ?.?%             ?.?%             ?.?%
turn_type=Program          486   71.0%   ?.?%        ?.?%             ?.?%             ?.?%
conv_type=Type I           640   78.8%   ?.?%        ?.?%             ?.?%             ?.?%
conv_type=Type II          130   69.2%   ?.?%        ?.?%             ?.?%             ?.?%
question=0..7              ...    ...    ...         ...              ...              ...
```

…plus a cost table:

```
Technique           Optimiser LLM calls   Total tokens   Wall time   Prompt size delta
v3_1 (s7)                       ~280            ~840k    ~28 min            +X tokens
v3_1_textgrad                   ?               ?        ?                  ?
v3_1_protegi                    ?               ?        ?                  ?
v3_1_promptwizard               ?               ?        ?                  ?
```

…and a head-to-head delta table relative to v2:

```
Technique           Δ Overall   Δ Number   Δ Program   Δ Type I   Δ Type II   Regressions
v3_1 (s7)             +X.X pp     +X.X pp    +X.X pp    +X.X pp     +X.X pp        N
v3_1_textgrad         ?           ?          ?          ?           ?              ?
v3_1_protegi          ?           ?          ?          ?           ?              ?
v3_1_promptwizard     ?           ?          ?          ?           ?              ?
```

Operator decision rule: **a technique only "wins" if Δ Overall is positive AND regressions ≤ 5**. Anything else is recorded as a learning, not a promotion.

---

## File layout

```
src/convfinqa/diagnosis/
  techniques/                          [NEW]
    __init__.py                        [NEW]
    common.py                          [NEW]  shared utilities (training set load,
                                              validation split, LM_MAX teacher,
                                              prompt regression check)
    textgrad/                          [NEW]
      __init__.py
      pipeline_graph.py                [NEW]  wraps 4 production agents as
                                              TextGrad Parameters + computation graph
      loss.py                          [NEW]  gold-vs-pred critique loss fn
      run.py                           [NEW]  training loop
    protegi/                           [NEW]
      __init__.py
      gradient.py                      [NEW]  textual-gradient generation from batch
      beam.py                          [NEW]  beam search + bandit selection
      run.py                           [NEW]
    promptwizard/                      [NEW]
      __init__.py
      mutate.py                        [NEW]  generate prompt variations
      critic.py                        [NEW]  score + critique
      synthesise.py                    [NEW]  generate worked examples; verify each
      run.py                           [NEW]

  bench/                               [NEW]
    __init__.py
    runner.py                          [NEW]  orchestrates all three techniques
    metrics.py                         [NEW]  multi-cut accuracy + cost + regression
    report.py                          [NEW]  s8_bench_comparison.{csv,html} writer

scripts/
  bench_optimisation.py                [NEW]  CLI entry point

src/convfinqa/prompts/__init__.py      [MODIFIED]  widen latest_all() regex to
                                                   accept v3_1_<tag> patterns

src/convfinqa/prompts/v3_1_textgrad.py        [GENERATED]
src/convfinqa/prompts/v3_1_protegi.py         [GENERATED]
src/convfinqa/prompts/v3_1_promptwizard.py    [GENERATED]

evaluation/
  pydantic_predictions_v3_1_textgrad.csv      [GENERATED]
  pydantic_predictions_v3_1_protegi.csv       [GENERATED]
  pydantic_predictions_v3_1_promptwizard.csv  [GENERATED]
  s8_bench_comparison.{csv,html}              [GENERATED]

tests/
  test_bench_optimisation.py           [NEW]  loader + per-technique smoke tests,
                                              all mocked (no API key needed)
```

## Implementation steps

### Step 0 — Shared infrastructure (`techniques/common.py`)

- Load training set (same `pydantic_predictions_v2.csv` first-wrong filter as s7).
- 80/20 train/val split with fixed seed.
- A `teacher_loss(gold, pred)` function that emits a structured natural-language critique using LM_MAX. Cached by `(report_id, turn_index, gold, pred)` hash so the same critique isn't regenerated across techniques.
- A `regression_check(variant_prompts) → list[regressed_cases]` utility that runs the production pipeline with the proposed prompts on a 50-case sanity set drawn from v2-correct cases.

### Step 1 — TextGrad integration

- Vendor or install the `textgrad` PyPI package.
- Build the pipeline as a TextGrad computation graph; each agent's `system_prompt` is a `Parameter`.
- Loss: teacher_loss from Step 0.
- Train 5 epochs over the training set, log per-step prompts to `evaluation/runs/textgrad_<ts>/` (parallel to GEPA's runs dir).
- Final prompts → `prompts/v3_1_textgrad.py`.

### Step 2 — ProTeGi integration

- No external dependency — reimplement the algorithm (it's small). Reference the paper's pseudocode.
- Per-agent loop: 4 separate ProTeGi runs (one for triage, one for preprocess, etc.).
- For each agent, run 6 rounds of (batch sample → gradient → 4 candidate edits → validation score → beam-top-2 → bandit select).
- Final prompts → `prompts/v3_1_protegi.py`.

### Step 3 — PromptWizard integration

- Either install the `promptwizard` package or reimplement the three-component loop.
- Per-agent loop: 4 separate PromptWizard runs.
- Critical: every synthesised in-context example must be verified before inclusion. A "verified example" is one whose program executes to the example's claimed answer when fed into the production calculator stage in isolation.
- Final prompts → `prompts/v3_1_promptwizard.py`.

### Step 4 — Score + compare

- Run `PROMPTS_VERSION=v3_1_<tag> uv run convfinqa-eval-api` for each of the three new variants to produce `pydantic_predictions_v3_1_<tag>.csv`.
- `bench/metrics.py` reads all five CSVs (v1, v2, v3_1, and the three new variants) and computes the cuts.
- `bench/report.py` writes `s8_bench_comparison.{csv,html}`.

### Step 5 — Validation pass

Each technique's variant must pass three gates before being declared a candidate:

1. **No assembly errors** — `prompts.load("v3_1_<tag>")` returns four strings.
2. **No critical regressions** — Fewer than 5 cases where v2 was right and the variant is wrong on the 50-case sanity set from Step 0.
3. **Positive overall delta on the held-out test set** — Otherwise the technique is recorded as a learning, not a promotion.

## Anti-patterns

- **DO NOT** introduce new dependencies in `pyproject.toml` without measuring their footprint. TextGrad pulls a heavy stack; if its install cost is large, vendor only the parts we use.
- **DO NOT** let any technique see the 770-question held-out test set during optimisation. The validation set is the 20% holdout from the **training** cases, not the test set.
- **DO NOT** mix the three techniques in a single run (e.g. "TextGrad followed by PromptWizard"). The bench is an apples-to-apples comparison of each technique starting from the same v2 baseline; pipelining defeats the comparison.
- **DO NOT** auto-promote any of the three variants into `v3_1.py`. The s7 baseline owns that namespace; new variants live as `v3_1_<tag>` until a human reviews the comparison report.
- **DO NOT** hand-edit any of the `v3_1_<tag>.py` files. They're generated by the bench.
- **DO NOT** skip the example-verification step in PromptWizard. A hallucinated worked example is strictly worse than no example — it actively teaches the production model the wrong answer.
- **DO NOT** compare on accuracy alone. Cost, prompt size, and regression count are part of the decision rule.
- **DO NOT** assume one technique will be best for every cut. The output is a per-cut decision matrix, not a single winner.

## References

### Primary papers

- **TextGrad** — Yuksekgonul et al., *"TextGrad: Automatic 'Differentiation' via Text"*, Nature 2024. [PDF](https://arxiv.org/pdf/2406.07496) · [Repo](https://github.com/zou-group/textgrad) · [Stanford HAI](https://hai.stanford.edu/news/textgrad-autograd-text)
- **ProTeGi** — Pryzant et al., *"Automatic Prompt Optimization with 'Gradient Descent' and Beam Search"*, EMNLP 2023. [PDF](https://arxiv.org/pdf/2305.03495) · [ACL Anthology](https://aclanthology.org/2023.emnlp-main.494/) · [OpenReview](https://openreview.net/forum?id=WRYhaSrThy)
- **PromptWizard** — Agarwal et al., *"PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework"*, Microsoft Research 2024. [Blog](https://www.microsoft.com/en-us/research/blog/promptwizard-the-future-of-prompt-optimization-through-feedback-driven-self-evolving-prompts/) · [Repo](https://github.com/microsoft/PromptWizard)

### Related work (covered by GEPA / Reflexion / not in scope)

- **GEPA** — Agrawal et al., *"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"*, ICLR 2026 (Oral). [Paper](https://arxiv.org/abs/2507.19457) · [Repo](https://github.com/gepa-ai/gepa) — already integrated via `convfinqa-optimize`.
- **OPRO** — Yang et al., *"Large Language Models as Optimizers"*. [Repo](https://github.com/google-deepmind/opro) — subsumed by GEPA in scope.
- **Reflexion** — Shinn et al., *"Reflexion: Language Agents with Verbal Reinforcement Learning"*. [Paper](https://arxiv.org/abs/2303.11366) — runtime memory, not optimisation-time; out of scope.
- **EvoPrompt** — Guo et al., *"Connecting LLMs with Evolutionary Algorithms"*. [Repo](https://github.com/beeevita/EvoPrompt) — similar evolutionary scope to GEPA.

## Confidence: 7 / 10

The three techniques are well-supported by published papers and reference implementations, and they exercise three genuinely distinct design axes. Main risks:

1. **TextGrad's compound-pipeline gradient may not converge.** Four simultaneous parameter updates from one loss signal can produce conflicting edits; TextGrad's anti-correlation regularisation helps but isn't a guarantee. If after one full training run the gradients keep "undoing each other", consider one-module-at-a-time TextGrad as a fallback.

2. **ProTeGi's adaptation to multi-prompt systems is not in the paper.** Running it four times sequentially (one per agent) is the obvious adaptation but may miss cross-agent dependencies.

3. **PromptWizard's example synthesis is the highest-risk surface.** A single bad example can derail the production model. The verification gate has to be strict — execute the program, compare to the claimed answer, reject mismatches — and the synthesis budget should be capped (e.g. max 3 verified examples per agent).

4. **All three may underperform s7 on this specific task.** The s7 design (verified per-case rules, no replacement of v2 baseline) is conservative-by-construction. The point of the bench isn't to prove one of the three is better — it's to find out, empirically, where each one's strengths lie. A clean "s7 wins overall but PromptWizard wins on Type II hybrid conversations" result is just as valuable as a single winner.
