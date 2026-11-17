/**
 * Optimise › Fine-tuning, in sentences and columns (s32). Pure functions over the committed record,
 * so the claims the page prints are computed rather than typed, and a test can hold them.
 */
import { NO_VALUE } from '../landing/format';
import type { CaseRow, FineTuningRecord, FtPass, FtRound, Harness, Paired, SplitName, Stage } from './api';

/** The lens: which view, which cases, which turn type. Absent `round` is the overall story. */
export const CASE_FILTERS = ['changed', 'fixed', 'broken', 'wrong', 'all'] as const;
export type CaseFilter = (typeof CASE_FILTERS)[number];
export const TURN_TYPES = ['', 'program', 'number'] as const;
export type TurnType = (typeof TURN_TYPES)[number];

export function isCaseFilter(v: string | null): v is CaseFilter {
  return (CASE_FILTERS as readonly string[]).includes(v ?? '');
}

export function isTurnType(v: string | null): v is TurnType {
  return (TURN_TYPES as readonly string[]).includes(v ?? '');
}

export function passIndex(record: FineTuningRecord | undefined): Map<string, FtPass> {
  return new Map((record?.passes ?? []).map((p) => [p.id, p]));
}

/** `e0p5` → `epoch 0.5`, `e2` → `epoch 2`. */
export function epochLabel(key: string | null | undefined): string {
  if (!key) return NO_VALUE;
  return `epoch ${key.replace(/^e/, '').replace('p', '.')}`;
}

/** A round's name from the record (`sft2` → `Round 2`, `dpo1full` → `DPO v1`), else from its key. */
export function roundName(key: string | null | undefined, record?: FineTuningRecord): string {
  if (!key) return NO_VALUE;
  const r = record?.rounds.find((x) => x.key === key);
  return r ? r.label : `Round ${key.replace(/^sft/, '')}`;
}

export function harnessLabel(record: FineTuningRecord | undefined, harness: Harness | null | undefined): string {
  if (!harness) return NO_VALUE;
  return record?.harnesses[harness]?.label ?? harness;
}

export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return NO_VALUE;
  return `${(x * 100).toFixed(digits)}%`;
}

export function pp(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return NO_VALUE;
  const sign = x > 0 ? '+' : x < 0 ? '−' : '±';
  return `${sign}${Math.abs(x).toFixed(digits)}pp`;
}

/** A one-sided p at the precision it deserves; the record rounds to four places. */
export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return NO_VALUE;
  return p < 0.001 ? '< 0.001' : p.toFixed(3);
}

/** Significant on the gate's rule: improved, and one-sided cluster p under 0.05. */
export function significant(v: Paired | null | undefined): boolean {
  return !!v && v.delta_pp > 0 && v.p_one_sided < 0.05;
}

/** A step in words: `+21.2pp, 91 fixed / 17 broken, p < 0.001`. */
export function stepPhrase(v: Paired | null | undefined): string {
  if (!v) return NO_VALUE;
  return `${pp(v.delta_pp)}, ${v.fixed} fixed / ${v.broken} broken, p ${fmtP(v.p_one_sided)}`;
}

/** The record's link into Campaigns › Runs for one pass: the MLflow run behind it. */
export function campaignsLink(runId: string | null | undefined): string | null {
  return runId ? `/optimise/campaigns?run=${encodeURIComponent(runId)}` : null;
}

// ---------------------------------------------------------------------------
// The overall story
// ---------------------------------------------------------------------------

export interface Headline {
  base: FtPass | null;
  final: FtPass | null;
  finalStage: Stage | null;
  teacher: FtPass | null;
  gapClosed: number | null;
  nRounds: number;
}

export function headline(record: FineTuningRecord | undefined): Headline {
  const passes = passIndex(record);
  const stages = record?.progression.test ?? [];
  const drawn = stages.filter((s) => s.pass && passes.has(s.pass));
  const finalStage = drawn[drawn.length - 1] ?? null;
  return {
    base: drawn[0] ? passes.get(drawn[0].pass!) ?? null : null,
    final: finalStage ? passes.get(finalStage.pass!) ?? null : null,
    finalStage,
    teacher: record?.teacher ? passes.get(record.teacher.pass) ?? null : null,
    gapClosed: record?.teacher?.gap_closed ?? null,
    // the rounds the test number covers: a round whose pick was never opened on test is not in it
    nRounds: drawn.filter((s) => s.kind === 'round').length,
  };
}

/** A stage's name on two lines: what was trained, then how it was scored. */
export function stageLines(stage: Stage, record: FineTuningRecord | undefined): [string, string] {
  const harness = `${harnessLabel(record, stage.harness)} harness`;
  if (stage.kind === 'base') return ['Base', harness];
  if (stage.kind === 'harness') {
    const from = stage.key.split('-')[0];
    return [`${from === 'base' ? 'Base' : roundName(from, record)}, re-scored`, harness];
  }
  const round = record?.rounds.find((r) => r.key === stage.round);
  return [`${roundName(stage.round, record)} · ${epochLabel(round?.best)}`, harness];
}

/** The newest stage on val with a pass, and its paired step from the stage before: where the student is now. */
export function latestVal(record: FineTuningRecord | undefined): { stage: Stage; pass: FtPass } | null {
  const passes = passIndex(record);
  const stages = (record?.progression.val ?? []).filter((s) => s.pass && passes.has(s.pass));
  const stage = stages[stages.length - 1];
  return stage ? { stage, pass: passes.get(stage.pass!)! } : null;
}

/** The overall step that bought the most, by paired delta on test. */
export function biggestStep(record: FineTuningRecord | undefined): Stage | null {
  const steps = (record?.progression.test ?? []).filter((s) => s.vs_prev);
  return steps.reduce<Stage | null>((best, s) => (!best || s.vs_prev!.delta_pp > best.vs_prev!.delta_pp ? s : best), null);
}

// ---------------------------------------------------------------------------
// A round
// ---------------------------------------------------------------------------

export interface CheckpointRow {
  key: string;
  epoch: number;
  step: number;
  val: FtPass | null;
  test: FtPass | null;
  vsStart: Paired | null;
  best: boolean;
}

export function checkpointRows(round: FtRound, passes: Map<string, FtPass>): CheckpointRow[] {
  return round.checkpoints.map((c) => ({
    key: c.key,
    epoch: c.epoch,
    step: c.step,
    val: c.val_pass ? passes.get(c.val_pass) ?? null : null,
    test: c.test_pass ? passes.get(c.test_pass) ?? null : null,
    vsStart: c.vs_start,
    best: c.key === round.best,
  }));
}

/** The round's claim: where it started, what it trained on, and which checkpoint val picked. */
export function roundClaim(round: FtRound, passes: Map<string, FtPass>): string {
  const best = round.checkpoints.find((c) => c.key === round.best);
  const bestVal = best?.val_pass ? passes.get(best.val_pass) : undefined;
  const from = round.start.round ? `from ${round.start.label}` : 'from the untrained base';
  const rows =
    round.method === 'dpo'
      ? `${round.dataset.n_rows.toLocaleString()} preference pairs`
      : `${round.dataset.n_rows} conversations`;
  const epochs = `${round.settings.epochs} epoch${round.settings.epochs === 1 ? '' : 's'}`;
  const pick = bestVal ? `${epochLabel(round.best)} scored highest on val at ${pct(bestVal.accuracy)}` : 'no checkpoint was scored on val';
  return `${round.label} started ${from} and trained ${epochs} on ${rows}; ${pick}`;
}

/** The start's test pass on the round's harness, when there is one: what the best is paired with on test. */
export function startTest(round: FtRound, record: FineTuningRecord | undefined): FtPass | null {
  const hits = (record?.passes ?? []).filter(
    (p) =>
      p.complete &&
      p.split === 'test' &&
      p.model === round.start.model &&
      p.harness === round.sweep_harness &&
      (p.temperature === null || p.temperature === record?.default_temperature),
  );
  return hits.sort((a, b) => (a.id < b.id ? 1 : -1))[0] ?? null;
}

/** Why a re-scored pass is not on the round's line: its harness or its temperature differs. */
export function rescoreReason(p: FtPass, round: FtRound, record: FineTuningRecord | undefined): string {
  const parts: string[] = [];
  if (p.harness !== round.sweep_harness) parts.push(`${harnessLabel(record, p.harness)} harness`);
  if (p.temperature !== null && p.temperature !== record?.default_temperature) parts.push(`temperature ${p.temperature}`);
  if (p.split === 'test' && !parts.length) parts.push('test');
  return parts.join(', ') || 'a repeat';
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

export interface CaseColumn {
  pass: string;
  label: string;
  sub: string;
}

export interface CaseSet {
  split: SplitName;
  columns: CaseColumn[];
  /** The column a case is judged from, and the one it is judged at. */
  from: number;
  to: number;
  fromLabel: string;
  toLabel: string;
  /** How many leading columns "changed" looks across: the teacher is a reference, not a step. */
  span: number;
}

/** Overall: every stage's test pass in order, then the teacher's. */
export function overallCases(record: FineTuningRecord | undefined): CaseSet | null {
  if (!record) return null;
  const stages = record.progression.test.filter((s) => s.pass);
  if (!stages.length) return null;
  const columns: CaseColumn[] = stages.map((s) => {
    const [label, sub] = stageLines(s, record);
    return { pass: s.pass!, label, sub };
  });
  const to = columns.length - 1;
  if (record.teacher) {
    columns.push({ pass: record.teacher.pass, label: 'Teacher', sub: `${record.teacher.name} · plain harness` });
  }
  return { split: 'test', columns, from: 0, to, fromLabel: columns[0].label, toLabel: columns[to].label, span: to + 1 };
}

/** A round on val: the start, then every checkpoint of its sweep. On test: the start and the pick. */
export function roundCases(round: FtRound, record: FineTuningRecord | undefined, split: SplitName): CaseSet | null {
  const harness = `${harnessLabel(record, round.sweep_harness)} harness`;
  if (split === 'val') {
    if (!round.start.val_pass) return null;
    const columns: CaseColumn[] = [{ pass: round.start.val_pass, label: `Start · ${round.start.label}`, sub: harness }];
    let to = 0;
    for (const c of round.checkpoints) {
      if (!c.val_pass) continue;
      if (c.key === round.best) to = columns.length;
      columns.push({ pass: c.val_pass, label: epochLabel(c.key), sub: c.key === round.best ? 'picked' : harness });
    }
    return { split, columns, from: 0, to, fromLabel: columns[0].label, toLabel: columns[to].label, span: columns.length };
  }
  const best = round.checkpoints.find((c) => c.key === round.best);
  const start = startTest(round, record);
  if (!best?.test_pass) return null;
  const columns: CaseColumn[] = [];
  if (start) columns.push({ pass: start.id, label: `Start · ${round.start.label}`, sub: harness });
  columns.push({ pass: best.test_pass, label: epochLabel(best.key), sub: 'picked' });
  const to = columns.length - 1;
  return { split, columns, from: 0, to, fromLabel: columns[0].label, toLabel: columns[to].label, span: columns.length };
}

function rightAt(row: CaseRow, i: number): boolean | null {
  const a = row.answers[i];
  return a ? a.correct : null;
}

export function matches(row: CaseRow, set: CaseSet, filter: CaseFilter, turn: TurnType): boolean {
  if (turn && row.turn_type !== turn) return false;
  const from = rightAt(row, set.from);
  const to = rightAt(row, set.to);
  switch (filter) {
    case 'all':
      return true;
    case 'changed':
      return new Set(row.answers.slice(0, set.span).map((a) => a?.correct)).size > 1;
    case 'fixed':
      return from === false && to === true;
    case 'broken':
      return from === true && to === false;
    case 'wrong':
      return to === false;
  }
}

export function caseCounts(rows: readonly CaseRow[], set: CaseSet, turn: TurnType): Record<CaseFilter, number> {
  const out = Object.fromEntries(CASE_FILTERS.map((f) => [f, 0])) as Record<CaseFilter, number>;
  for (const row of rows) for (const f of CASE_FILTERS) if (matches(row, set, f, turn)) out[f] += 1;
  return out;
}

export function filterLabel(filter: CaseFilter, set: CaseSet): string {
  switch (filter) {
    case 'changed':
      return set.columns.length > 2 ? 'changed along the way' : 'changed';
    case 'fixed':
      return `fixed: wrong at ${set.fromLabel}, right at ${set.toLabel}`;
    case 'broken':
      return `broken: right at ${set.fromLabel}, wrong at ${set.toLabel}`;
    case 'wrong':
      return `still wrong at ${set.toLabel}`;
    case 'all':
      return 'every question';
  }
}
