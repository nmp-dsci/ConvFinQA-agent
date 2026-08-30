/**
 * The ConvFinQA paper, as data.
 *
 * Every figure in this file is read from Chen et al., EMNLP 2022 — the PDF is
 * committed at `data/2210.03849v1.pdf` — and every one of them carries the
 * table or figure it came from. Nothing here is computed, estimated or
 * remembered; if a number has no `source`, it does not belong in this file, and
 * `paper.test.ts` fails the build if one slips in.
 *
 * This is the *static* half of the debrief. The other half — what this system
 * scores — is read live from the backend in `useSystemData.ts` and is never
 * hardcoded here, so the two can never silently disagree.
 */

// ---------------------------------------------------------------------------
// The paper itself
// ---------------------------------------------------------------------------

export const PAPER = {
  title:
    'ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering',
  authors: 'Chen, Li, Smiley, Ma, Shah & Wang',
  venue: 'EMNLP 2022',
  arxivId: '2210.03849',
  arxivUrl: 'https://arxiv.org/abs/2210.03849',
  /** Committed in this repo, so the claim is checkable without the internet. */
  localPath: 'data/2210.03849v1.pdf',
} as const;

// ---------------------------------------------------------------------------
// 1 · The dataset
// ---------------------------------------------------------------------------

export interface DatasetStat {
  label: string;
  value: string;
  note?: string;
  source: string;
}

/** Corpus scale — the panel at the top of the debrief. */
export const DATASET_SCALE: DatasetStat[] = [
  { label: 'conversations', value: '3,892', source: 'Table 2' },
  { label: 'questions', value: '14,115', source: 'Table 2' },
  { label: 'report pages', value: '2,066', source: 'Table 2' },
  {
    label: 'questions / conversation',
    value: '3.67',
    note: 'average question length 10.59 tokens',
    source: 'Table 2',
  },
  {
    label: 'train / dev / test',
    value: '3,037 / 421 / 434',
    note: 'test gold answers were never released',
    source: 'Table 2',
  },
  {
    label: 'simple / hybrid',
    value: '2,715 / 1,177',
    note: 'Type I is one decomposed question; Type II concatenates two',
    source: 'Table 2',
  },
];

export interface Distribution {
  label: string;
  pct: number;
  note?: string;
}

export interface DistributionBlock {
  title: string;
  caption: string;
  source: string;
  bars: Distribution[];
}

export const DISTRIBUTIONS: DistributionBlock[] = [
  {
    title: 'Question type',
    caption:
      'A third of all turns are a look-up. The rest need a program, and most of those are one or two steps.',
    source: 'Table 3',
    bars: [
      { label: 'number selection', pct: 34.73, note: 'retrieve a value, no arithmetic' },
      { label: 'program · 1 step', pct: 35.1 },
      { label: 'program · 2 steps', pct: 25.41 },
      { label: 'program · 3+ steps', pct: 4.75 },
    ],
  },
  {
    title: 'Where the supporting facts live',
    caption:
      'The table carries most of it, but a quarter of questions can only be answered from the prose, and a sixth need both.',
    source: 'Table 3',
    bars: [
      { label: 'table only', pct: 59.18 },
      { label: 'text only', pct: 25.56 },
      { label: 'both', pct: 15.26 },
    ],
  },
  {
    title: 'Operations used',
    caption:
      'Three quarters of all arithmetic is a subtraction or a division — a change and a ratio, which is what an analyst asks a filing.',
    source: 'Table 3',
    bars: [
      { label: 'subtract', pct: 40.49 },
      { label: 'divide', pct: 33.43 },
      { label: 'add', pct: 18.8 },
      { label: 'multiply', pct: 6.92 },
    ],
  },
  {
    title: 'Longest dependency distance',
    caption:
      'Over 60% of questions depend on an earlier one. This is the axis the paper says every model degrades along, and the reason a preprocess stage exists at all.',
    source: 'Figure 4',
    bars: [
      { label: '0 · self-contained', pct: 39.5 },
      { label: '1 turn back', pct: 28.0 },
      { label: '2 turns back', pct: 19.0 },
      { label: '3 turns back', pct: 12.5 },
      { label: '4+ turns back', pct: 1.0 },
    ],
  },
];

/** The six-operation DSL the gold programs are written in. */
export const DSL_OPS: Array<{ op: string; args: string; out: string; meaning: string }> = [
  { op: 'add', args: 'n1, n2', out: 'number', meaning: 'n1 + n2' },
  { op: 'subtract', args: 'n1, n2', out: 'number', meaning: 'n1 − n2' },
  { op: 'multiply', args: 'n1, n2', out: 'number', meaning: 'n1 × n2' },
  { op: 'divide', args: 'n1, n2', out: 'number', meaning: 'n1 ÷ n2' },
  { op: 'exp', args: 'n1, n2', out: 'number', meaning: 'n1 ^ n2' },
  { op: 'greater', args: 'n1, n2', out: 'bool', meaning: 'n1 > n2' },
];

// ---------------------------------------------------------------------------
// 2 · The baselines
// ---------------------------------------------------------------------------

export interface BaselineRow {
  system: string;
  /** What it was trained or tuned on. */
  trained: string;
  /** `null` where the paper does not report the figure. */
  exe: string | null;
  prog: string | null;
  evaluatedOn: string;
  source: string;
  /** True for the human ceiling, which is the line nothing here beats. */
  ceiling?: boolean;
}

/**
 * The paper's leaderboard, in the order the debrief prints it: best first.
 *
 * This system's own row is NOT in this list. It is spliced in at render time
 * from the live backend read, so a promoted challenger moves the row without a
 * code change — and so that no figure describing this system can ever be a
 * literal in a source file.
 */
export const BASELINES: BaselineRow[] = [
  {
    system: 'Human expert',
    trained: '—',
    exe: '89.44',
    prog: '86.34',
    evaluatedOn: '200-question sample',
    source: 'Table 9',
    ceiling: true,
  },
  {
    system: 'FinQANet-Gold (RoBERTa-large, gold retrieval given)',
    trained: '3,037 conversations · full fine-tune',
    exe: '77.32',
    prog: '76.46',
    evaluatedOn: '434-conversation test set',
    source: 'Table 5',
  },
  {
    system: 'FinQANet (RoBERTa-large) — the paper’s best real system',
    trained: '3,037 conversations · full fine-tune',
    exe: '68.90',
    prog: '68.24',
    evaluatedOn: 'test set',
    source: 'Table 5',
  },
  {
    system: 'FinQANet (RoBERTa-base / BERT-large / BERT-base)',
    trained: '3,037 conversations · full fine-tune',
    exe: '64.95 / 61.14 / 55.03',
    prog: '64.16 / 60.55 / 54.57',
    evaluatedOn: 'test set',
    source: 'Table 5',
  },
  {
    system: 'T5-large / GPT-2-medium (generative)',
    trained: '3,037 conversations · full fine-tune',
    exe: '58.66 / 58.19',
    prog: '57.05 / 57.00',
    evaluatedOn: 'test set',
    source: 'Table 5',
  },
  {
    system: 'GPT-3 175B few-shot — best setting (Program-normal, 20 exemplars, gold retrieval)',
    trained: '20 exemplars',
    exe: '50.30',
    prog: '45.10',
    evaluatedOn: 'test set',
    source: 'Table 6',
  },
  {
    system: 'GPT-3 175B few-shot — Program-normal 10 / chain-of-thought / answer-only',
    trained: '10 exemplars',
    exe: '48.85 / 40.63 / 24.09',
    prog: '42.14 / 33.84 / —',
    evaluatedOn: 'test set',
    source: 'Table 6',
  },
  {
    system: 'General crowd (MTurk)',
    trained: '—',
    exe: '46.90',
    prog: '45.52',
    evaluatedOn: '200-question sample',
    source: 'Table 9',
  },
];

/**
 * The four caveats that must sit *with* the table.
 *
 * They are data rather than JSX for one reason: a footnote can be scrolled
 * past, a component that renders a list cannot render the table without them.
 */
export const BENCHMARK_CAVEATS: Array<{ title: string; body: string }> = [
  {
    title: 'Different questions',
    body:
      'The paper scores the 434-conversation test set, whose gold answers were never released. Our 200 conversations are sampled from the public train/dev files. The two sets do not overlap and are not the same difficulty draw.',
  },
  {
    title: 'A kinder tolerance',
    body:
      'The paper’s execution accuracy is an exact match on the executed program. Ours is a numeric match with rounding tolerance, which is slightly more forgiving. Same metric name, not the same test.',
  },
  {
    title: 'A different era of model',
    body:
      'A 2026 hosted LLM in a four-agent pipeline with no fine-tuning, against a 2022 fine-tuned RoBERTa pipeline. The comparison shows what the paradigm shift bought — not that this codebase beats FinQANet on its own terms.',
  },
  {
    title: 'Program accuracy is ~35% against ~77% execution — and that gap is real, not a scoring bug',
    body:
      'The pipeline answers a turn using the conversation’s prior answers, so it writes divide(132, 111), multiply(#0, 100) where the gold program re-derives from raw table values as subtract(243, 111), divide(#0, 111). Same answer, shorter program, scored as a program mismatch. It is a genuine difference in reasoning shape and it is counted honestly here; it is not evidence the system got the turn wrong.',
  },
];

// ---------------------------------------------------------------------------
// The paper's per-slice numbers, for the comparison chart
// ---------------------------------------------------------------------------

export interface SliceBaseline {
  /** Matches the label in our own predictions CSV where one exists. */
  key: string;
  label: string;
  finqanet: number;
  gpt3: number;
  source: string;
}

export const SLICE_BASELINES: SliceBaseline[] = [
  { key: 'Number', label: 'Number-selection questions', finqanet: 82.54, gpt3: 35.32, source: 'Table 4 / Table 6' },
  { key: 'Program', label: 'Program questions', finqanet: 62.14, gpt3: 55.56, source: 'Table 4 / Table 6' },
  { key: 'Type I', label: 'Type I · simple conversations', finqanet: 72.37, gpt3: 52.22, source: 'Table 4 / Table 6' },
  { key: 'Type II', label: 'Type II · hybrid conversations', finqanet: 60.99, gpt3: 41.16, source: 'Table 4 / Table 6' },
];

/**
 * A slice the paper reports and this project does not, kept visible rather
 * than quietly dropped — a comparison that only prints the slices we win is
 * not a comparison.
 */
export const SLICE_NOT_MEASURED = {
  label: 'Type II · second half (the paper’s hardest slice)',
  finqanet: 52.38,
  gpt3: 22.85,
  source: 'Table 4 / Table 6',
  why:
    'Not scored here. It needs the boundary between the two source questions of a hybrid conversation, which the sampled CSVs do not carry.',
};

/** Figure 5 — accuracy against turn position. The curve every system falls down. */
export const PER_TURN_BASELINE: Array<{ turn: number; finqanet: number; gpt3: number }> = [
  { turn: 1, finqanet: 75.6, gpt3: 72.8 },
  { turn: 6, finqanet: 34.4, gpt3: 25.2 },
];

// ---------------------------------------------------------------------------
// The paper's findings, and what this system does about each
// ---------------------------------------------------------------------------

export interface Finding {
  paper: string;
  response: string;
  evidence: string;
  /** Where in this console a reader can go and check it. */
  to?: string;
}

export const FINDINGS: Finding[] = [
  {
    paper: '“The model excels at number selection questions” — 82.54% for FinQANet.',
    response:
      'Triage sends a look-up down a two-stage path — triage → retriever — that skips planning and calculation entirely.',
    evidence: 'The inspector shows the skipped stages on any number turn.',
    to: '/chat',
  },
  {
    paper:
      '“The model suffers from the lack of domain knowledge” — wrong facts retrieved, wrong values chosen, wrong operations applied.',
    response:
      'The s7 harness diagnoses each failure into a bucket, writes a candidate rule for the agent responsible, and keeps it only if the case then passes and the regression set does not move.',
    evidence: 'The per-agent rule stores, with every attempt kept — including the rejected ones.',
    to: '/admin/research',
  },
  {
    paper:
      '“The model struggles with long reasoning chains… if any turn is wrong, there is a very minor chance the subsequent turns are correct.”',
    response:
      'History is threaded as the system’s own prior answers, so a later turn depends on what was actually said rather than on gold it never saw. Per-turn accuracy is a first-class slice, not a summary statistic.',
    evidence: 'The per-turn curve on this page, against the paper’s Figure 5.',
    to: '/admin/evaluations',
  },
  {
    paper:
      'GPT-3 “often fails to make correct references to the context” — asked for “the subsequent year”, it returns the previous year’s value.',
    response:
      'A dedicated preprocess stage rewrites “this / that / the sum of both” into explicit sub-questions before anything is retrieved.',
    evidence: 'The preprocess stage output on any dependent turn, beside the question that produced it.',
    to: '/chat',
  },
  {
    paper: 'GPT-3 “performs better for its familiar program format” — a1 + a2 beats the original DSL.',
    response:
      'The calculator never writes DSL text. It calls add / subtract / multiply / divide / exp / greater as tools, and the program is reconstructed from the tool loop afterwards.',
    evidence: 'The tool loop in every trace.',
    to: '/admin/traces',
  },
  {
    paper:
      'GPT-3 “simply mimics the reasoning steps given in one exemplar but ignores the actual context.”',
    response:
      'No exemplars at all. Instructions are optimised against 120 conversations and measured on the conversations no optimizer ever saw.',
    evidence: 'The never-seen figure reported beside — never inside — the overall figure.',
    to: '/admin/experiments',
  },
];
