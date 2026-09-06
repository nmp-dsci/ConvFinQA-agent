/**
 * Joining the paper's numbers to this deployment's.
 *
 * The rule that shapes this file: the paper's side is a literal (it will never
 * change), our side is `number | null` read from `/eval/runs/<version>/summary`
 * (it changes when a challenger is promoted, and is genuinely absent when the
 * backend is down). `null` must reach the renderer as `null` — a slice that
 * defaulted to 0 would draw a bar claiming the system scored nothing.
 */

import type { ModelAccuracy } from '../../types';
import { SLICE_BASELINES, SLICE_NOT_MEASURED } from './paper';

export interface SliceRow {
  key: string;
  label: string;
  finqanet: number;
  gpt3: number;
  /** Percentage points, or `null` when this deployment does not score it. */
  ours: number | null;
  /** Questions behind `ours`, so the reader can weigh the slice. */
  n: number | null;
  source: string;
  /** Set only on a slice we deliberately do not measure. */
  why?: string;
}

/**
 * Our accuracy for one slice, as a percentage, or `null`.
 *
 * The backend returns fractions in two groupings — `by_turn_type`
 * (`Number` / `Program`) and `by_conv_type` (`Type I` / `Type II`) — with the
 * same label vocabulary as the committed CSVs. Looking in both means the
 * caller does not have to know which grouping a slice lives in, and an unknown
 * label falls through to `null` rather than to the first row that happens to
 * be there.
 */
export function ourSliceAccuracy(
  summary: ModelAccuracy | undefined,
  key: string,
): number | null {
  if (!summary) return null;
  const found = [...summary.by_turn_type, ...summary.by_conv_type].find((s) => s.label === key);
  if (!found || !Number.isFinite(found.accuracy)) return null;
  return found.accuracy * 100;
}

function ourSliceCount(summary: ModelAccuracy | undefined, key: string): number | null {
  if (!summary) return null;
  const found = [...summary.by_turn_type, ...summary.by_conv_type].find((s) => s.label === key);
  return found ? found.n_total : null;
}

/**
 * Every slice the paper reports, including the one we do not measure.
 *
 * Dropping the unmeasured row would turn a comparison into a highlight reel,
 * so it stays in the table with its reason printed where the number would be.
 */
export function sliceRows(summary: ModelAccuracy | undefined): SliceRow[] {
  const rows: SliceRow[] = SLICE_BASELINES.map((slice) => ({
    key: slice.key,
    label: slice.label,
    finqanet: slice.finqanet,
    gpt3: slice.gpt3,
    ours: ourSliceAccuracy(summary, slice.key),
    n: ourSliceCount(summary, slice.key),
    source: slice.source,
  }));

  rows.push({
    key: 'type-ii-second-half',
    label: SLICE_NOT_MEASURED.label,
    finqanet: SLICE_NOT_MEASURED.finqanet,
    gpt3: SLICE_NOT_MEASURED.gpt3,
    ours: null,
    n: null,
    source: SLICE_NOT_MEASURED.source,
    why: SLICE_NOT_MEASURED.why,
  });

  return rows;
}

// ---------------------------------------------------------------------------
// Open work
// ---------------------------------------------------------------------------

export type OpenStatus = 'broken' | 'open' | 'deferred';

export interface OpenItem {
  title: string;
  status: OpenStatus;
  body: string;
  /** The file or command a reader would go to next. */
  where?: string;
}

/**
 * What is wrong or unfinished, in the reader's line of sight.
 *
 * Two of these are things that are broken *right now*. They are first, they say
 * "broken", and they say what does not work as a result. A debrief that lists
 * only future work is a roadmap, and a roadmap is not a debrief.
 */
export const OPEN_WORK: OpenItem[] = [
  {
    title: 'The runtime decision is made; serving has not moved',
    status: 'open',
    body:
      'A single Claude Agent SDK session with the calculator functions as its only tools, running one prompt distilled from the four tuned pipeline prompts, scores 90.5% on the 349-question gate split against the champion pipeline’s 81.7% — paired, one-sided cluster-corrected McNemar p = 0.0003 — and the same prompt on Haiku 4.5 still clears the pipeline at 87.4%. The recommendation on /admin/runtimes is to move the runtime. Nothing about serving has changed: chat, the demo pack and the streamed event contract still run the four-agent champion, and wiring the session runtime into them — including its billing, which is subscription-only by decision — is its own piece of work. The model half of the confound is measured; a four-agent run on a Claude model is not.',
    where: '/admin/runtimes · docs/optimization/agent-sdk.html · backends/agent_sdk.py',
  },
  {
    title: 'DSPy / GEPA runs do not work today',
    status: 'broken',
    body:
      'The provider answers 400 — “thinking mode does not support this tool_choice” — to any request that pins a tool while thinking is on. The fix landed on the pydantic-ai path only: llm.py::model_settings() sends extra_body {"thinking": {"type": "disabled"}} and get_model() applies it. Neither dspy_lm_kwargs() nor backends/dspy.py::_lm() sets extra_body, and neither was testable without a key, so the LiteLLM path DSPy uses still hits the same 400. Every GEPA and DSPy command in the README is broken until that is fixed. The champion’s prompts came from a GEPA run made before the provider changed: they are real, and they are not reproducible from this checkout today.',
    where: 'src/convfinqa/llm.py :: dspy_lm_kwargs() · backends/dspy.py :: _lm()',
  },
  {
    title: 'Latency, token and cost metrics are empty in the demo',
    status: 'broken',
    body:
      'The recorded demo pack carries metrics: {} on every stage event, so every latency, token and cost surface in the demo deployment reads an em dash rather than a number. That is deliberate — a replayed turn timed at replay speed is not a latency measurement, and an unmeasured turn shown as a measured zero is a lie in the flattering direction. Filling them in needs a metered evaluation run followed by re-recording the pack, which costs real API calls and has not been authorised.',
    where: 'REUSE_CACHE=0 uv run convfinqa-eval && uv run convfinqa-demo-pack --n 8',
  },
  {
    title: 'Retrieved values are matched by value, not by cell',
    status: 'open',
    body:
      'The retriever returns { question, answer } pairs and never names the cell it read. The document view therefore highlights every cell whose value matches, which is provenance-shaped but is not provenance: two cells holding 1,240 both light up. The panel says so rather than implying a coordinate it does not have. Returning a row/column reference from the retriever is the fix.',
  },
  {
    title: 'Per-model-call telemetry is constructed and dropped',
    status: 'deferred',
    body:
      'Spans for every request, agent run, model call and tool call are created on every turn through the Logfire SDK, and exported nowhere: no LOGFIRE_TOKEN exists locally or in the demo image, and send_to_logfire is "if-token-present". Scope was decided on 29 August as turns only, so no span capture was built. Setting LOGFIRE_TOKEN in a dev shell turns the whole tree on with no code change — that is the entire recovery path, and it is why this is deferred rather than missing.',
  },
  {
    title: 'The paper’s hardest slice is not scored',
    status: 'open',
    body:
      'Type II second half — the tail of a hybrid conversation, where FinQANet falls to 52.38% — needs the boundary between the two source questions that were concatenated. The sampled prediction CSVs do not carry it, so the slice is shown with the paper’s figures and an empty column rather than an invented one.',
  },
  {
    title: 'The current champion did not come through the gate it is protected by',
    status: 'open',
    body:
      'The registry’s only promotion event is v2, registered by the backfill that rebuilt the history from committed artefacts, with actor "backfill" and comparison null — “the first registered version becomes champion by default”. The comparator and the CI gate are real and can be run against any pair on demand, and v3_1 fails them; but no version has yet been promoted *through* them. The first challenger that is will be the first real exercise of the contract.',
    where: 'evaluation/registry.json · uv run convfinqa-mlflow compare v2 <candidate>',
  },
  {
    title: 'Rate limiting and HTTP route health have no counters',
    status: 'deferred',
    body:
      'The limiter refuses requests but counts nothing, and route latency exists only as a Logfire span. App Runner’s own metrics are the source for both today. Two counters in limits.py is the whole job whenever this serves traffic that matters.',
  },
];
