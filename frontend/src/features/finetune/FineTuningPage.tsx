import { useState } from 'react';
import type { ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Clip, Figure, Rate, Status } from '@/kit/ui';
import { useLens } from '@/kit/url';
import { AdminPage, ErrorNote, LoadingRows, Panel } from '../admin/ui';
import { formatUsd } from '../landing/format';
import { getFineTuning, getFineTuningCases } from './api';
import type { FineTuningRecord, FtPass, FtRound, Paired, SplitName } from './api';
import { CheckpointChart, LossChart, ProgressChart } from './charts';
import {
  CASE_FILTERS,
  TURN_TYPES,
  biggestStep,
  campaignsLink,
  caseCounts,
  checkpointRows,
  epochLabel,
  filterLabel,
  fmtP,
  harnessLabel,
  headline,
  isCaseFilter,
  isTurnType,
  matches,
  overallCases,
  passIndex,
  pct,
  pp,
  rescoreReason,
  roundCases,
  roundClaim,
  significant,
  stageLines,
  stepPhrase,
} from './model';
import type { CaseSet, Headline } from './model';
import './finetune.css';

/**
 * Optimise › Fine-tuning (s32): the SFT student's whole record, from `evaluation/fine_tuning.json`.
 *
 * Overall first — base to each round's pick on test 100, one change per step, the teacher above —
 * then any round on its own: its checkpoints on val and which one val picked, its loss, its data
 * and settings, and the questions behind every number. The lens is the URL: `?round=sft2`,
 * `?cases=fixed`, `?type=program`, `?split=test`. Each pass links to its MLflow run on Campaigns.
 */
export default function FineTuningPage() {
  const query = useQuery({
    queryKey: ['fine-tuning'],
    queryFn: () => getFineTuning(),
    staleTime: Infinity, // a committed record: it does not change under the page
  });
  return <FineTuningView data={query.data} loading={query.isLoading} error={query.error} />;
}

/** The page for a record — split from the route so a test can render it from a fixture. */
export function FineTuningView({
  data,
  loading = false,
  error = null,
}: {
  data: FineTuningRecord | undefined;
  loading?: boolean;
  error?: unknown;
}) {
  const [lens, setLens] = useLens();
  const round = data?.rounds.find((r) => r.key === lens.get('round')) ?? null;
  const h = headline(data);
  return (
    <AdminPage testId="optimise-fine-tuning" eyebrow="optimise · fine-tuning" title={<Claim h={h} />} sub={lead(data, h)}>
      {error ? <ErrorNote error={error} /> : null}
      {loading && !data ? (
        <LoadingRows rows={4} />
      ) : !data ? (
        <p className="empty">— no fine-tuning record is committed at evaluation/fine_tuning.json</p>
      ) : (
        <>
          <div className="chips ft-views" role="group" aria-label="pick a view">
            {[{ key: '', label: 'Overall' }, ...data.rounds.map((r) => ({ key: r.key, label: `${r.label} · ${r.key}` }))].map((v) => (
              <button
                key={v.key || 'overall'}
                type="button"
                className={`chip${(round?.key ?? '') === v.key ? ' on' : ''}`}
                aria-pressed={(round?.key ?? '') === v.key}
                onClick={() => setLens({ round: v.key || null, cases: null, split: null })}
                data-testid={`ft-view-${v.key || 'overall'}`}
              >
                {v.label}
              </button>
            ))}
          </div>
          {round ? <RoundView round={round} record={data} /> : <OverallView record={data} />}
        </>
      )}
    </AdminPage>
  );
}

/** The h1: from base to the last round's pick on test, and how far that is toward the teacher. */
function Claim({ h }: { h: Headline }) {
  if (!h.base || !h.final || h.base === h.final) {
    return (
      <>
        No fine-tuned checkpoint has been <em>scored</em> on test yet
      </>
    );
  }
  const rounds = `${h.nRounds === 1 ? 'One round' : `${h.nRounds === 2 ? 'Two' : h.nRounds} rounds`} of fine-tuning`;
  const toward = h.gapClosed != null ? `, ${Math.round(h.gapClosed * 100)}% of the way to its teacher` : '';
  return (
    <>
      {rounds} took the student from {pct(h.base.accuracy)} to <em>{pct(h.final.accuracy)}</em> on test 100{toward}
    </>
  );
}

function lead(data: FineTuningRecord | undefined, h: Headline): ReactNode {
  if (!data) return 'The student’s rounds, checkpoints and passes, read from the committed record.';
  const val = data.splits.val;
  const test = data.splits.test;
  return (
    <>
      {data.student} with a LoRA adapter, trained on llm_react_v1’s own conversations. Val ({val.n_conversations}{' '}
      conversations, never trained on) picks each round’s checkpoint; test 100 ({test.n_conversations} conversations) is
      the holdout, opened only for a picked checkpoint. The teacher scored {pct(h.teacher?.accuracy)} on the same test.
      Each step is paired with the one before it on the same questions.
    </>
  );
}

// ---------------------------------------------------------------------------
// Overall
// ---------------------------------------------------------------------------

function OverallView({ record }: { record: FineTuningRecord }) {
  const passes = passIndex(record);
  const big = biggestStep(record);
  const rounds = record.progression.test.filter((s) => s.kind === 'round');
  const last = rounds[rounds.length - 1];
  const set = overallCases(record);
  return (
    <>
      <section className="opt-sec">
        <Figure
          testId="ft-progress"
          title="accuracy at each stage, test 100 and val, with the teacher's test score above"
          caption={
            <>
              {big ? `${stageLines(big, record)[0]} bought the most: ${stepPhrase(big.vs_prev)} on test. ` : ''}
              {last?.vs_prev && last !== big
                ? `${stageLines(last, record)[0]} added ${pp(last.vs_prev.delta_pp)} (p ${fmtP(last.vs_prev.p_one_sided)}${significant(last.vs_prev) ? '' : ', not significant'}). `
                : ''}
              The y axis is cropped to the values drawn.
            </>
          }
          source={
            <>
              <code>evaluation/fine_tuning.json</code> · <code>progression</code>
            </>
          }
        >
          <ProgressChart record={record} />
          <ProgressKey record={record} />
        </Figure>
      </section>

      <Panel
        testId="ft-steps"
        title={stepsClaim(record)}
        endpoint="fine_tuning.json · progression.test"
        note="A harness change is a stage of its own: sft1 was swept on the plain harness and sft2 on the checked one, so sft1’s pick re-scored on the checked harness stands between them. One-sided cluster-corrected McNemar and a conversation-bootstrap 95% interval, as the gate computes them."
      >
        <StageTable record={record} />
      </Panel>

      <Panel testId="ft-rounds" title={`${record.rounds.length} training rounds, each picked on val`} endpoint="runs/sft/*/run.json">
        <div className="cards ft-rounds">
          {record.rounds.map((r) => (
            <RoundCard key={r.key} round={r} record={record} passes={passes} />
          ))}
        </div>
      </Panel>

      {set && (
        <CasesPanel
          set={set}
          record={record}
          title={`Every test question, at every stage and the teacher’s`}
          note="Judged from the base to the last round’s pick; the teacher’s column is a reference, not a step."
        />
      )}

      <LeftOut record={record} />
    </>
  );
}

function stepsClaim(record: FineTuningRecord): string {
  const steps = record.progression.test.filter((s) => s.vs_prev);
  const sig = steps.filter((s) => significant(s.vs_prev)).length;
  return `${steps.length} steps on test 100, ${sig} of them significant on the gate’s rule`;
}

function ProgressKey({ record }: { record: FineTuningRecord }) {
  return (
    <ul className="ft-key" aria-label="key">
      <li>
        <svg viewBox="0 0 22 14" className="glyph" aria-hidden>
          <line className="test" x1="0" y1="7" x2="22" y2="7" />
          <circle className="dot-test" cx="11" cy="7" r="4" />
        </svg>
        test 100, {record.splits.test.n_conversations} conversations
      </li>
      <li>
        <svg viewBox="0 0 22 14" className="glyph" aria-hidden>
          <line className="val" x1="0" y1="7" x2="22" y2="7" />
          <circle className="dot-val" cx="11" cy="7" r="3.5" />
        </svg>
        val, {record.splits.val.n_conversations} conversations
      </li>
      <li>
        <svg viewBox="0 0 22 14" className="glyph" aria-hidden>
          <circle className="dot-hi" cx="11" cy="7" r="4" />
        </svg>
        the last round’s pick on test
      </li>
      <li>
        <svg viewBox="0 0 22 14" className="glyph" aria-hidden>
          <line className="ref" x1="0" y1="7" x2="22" y2="7" />
        </svg>
        the teacher, {record.teacher?.name ?? '—'}, on test
      </li>
    </ul>
  );
}

function RunLink({ pass }: { pass: FtPass | null | undefined }) {
  const to = campaignsLink(pass?.run_id);
  if (!to) return <span className="muted">—</span>;
  return (
    <Link to={to} title={pass?.id}>
      run record →
    </Link>
  );
}

function Step({ v }: { v: Paired | null }) {
  if (!v) return <span className="muted">— the first stage</span>;
  return (
    <>
      <span className={significant(v) ? 'v-ok' : ''}>{pp(v.delta_pp)}</span> · {v.fixed} fixed / {v.broken} broken · p{' '}
      {fmtP(v.p_one_sided)}
      <span className="ci">
        95% CI {pp(v.ci_pp[0])} to {pp(v.ci_pp[1])} · {v.n} questions{v.same_harness ? '' : ' · harness changed'}
      </span>
    </>
  );
}

function StageTable({ record }: { record: FineTuningRecord }) {
  const passes = passIndex(record);
  const [, setLens] = useLens();
  const val = new Map(record.progression.val.map((s) => [s.key, s]));
  return (
    <div className="tw">
      <table className="ft-table" data-testid="ft-stage-table">
        <thead>
          <tr>
            <th>stage</th>
            <th>test 100</th>
            <th className="num">program</th>
            <th className="num">number</th>
            <th>step from the stage before</th>
            <th>val</th>
            <th>MLflow</th>
          </tr>
        </thead>
        <tbody>
          {record.progression.test.map((s) => {
            const p = s.pass ? passes.get(s.pass) : undefined;
            const v = val.get(s.key)?.pass ? passes.get(val.get(s.key)!.pass!) : undefined;
            const [a, b] = stageLines(s, record);
            return (
              <tr key={s.key} data-stage={s.key}>
                <td className="sub stagecell">
                  {s.round ? (
                    <button type="button" className="linkish" onClick={() => setLens({ round: s.round, cases: null, split: null })}>
                      {a}
                    </button>
                  ) : (
                    a
                  )}
                  <span className="path">{b}</span>
                </td>
                <td>{p ? <Rate passed={p.correct} n={p.n} digits={1} /> : <span className="muted">— not opened on test</span>}</td>
                <td className="num">{pct(p?.program.accuracy)}</td>
                <td className="num">{pct(p?.number.accuracy)}</td>
                <td className="step">
                  <Step v={s.vs_prev} />
                </td>
                <td>{v ? <Rate passed={v.correct} n={v.n} digits={1} /> : <span className="muted">— not scored on val</span>}</td>
                <td>
                  <RunLink pass={p} />
                </td>
              </tr>
            );
          })}
          {record.teacher && <TeacherRow record={record} />}
        </tbody>
      </table>
    </div>
  );
}

function TeacherRow({ record }: { record: FineTuningRecord }) {
  const p = passIndex(record).get(record.teacher!.pass);
  const v = record.teacher!.vs_final;
  return (
    <tr className="dim" data-stage="teacher">
      <td className="sub stagecell">
        Teacher · {record.teacher!.name}
        <span className="path">
          {record.teacher!.model} · {record.teacher!.effort} · plain harness
        </span>
      </td>
      <td>{p ? <Rate passed={p.correct} n={p.n} digits={1} /> : '—'}</td>
      <td className="num">{pct(p?.program.accuracy)}</td>
      <td className="num">{pct(p?.number.accuracy)}</td>
      <td className="step">
        {v ? (
          <>
            {pp(v.delta_pp)} over the last pick · p {fmtP(v.p_one_sided)}
            <span className="ci">
              a reference, not a step · 95% CI {pp(v.ci_pp[0])} to {pp(v.ci_pp[1])}
            </span>
          </>
        ) : (
          '—'
        )}
      </td>
      <td>
        <span className="muted">— not scored on val</span>
      </td>
      <td>
        <RunLink pass={p} />
      </td>
    </tr>
  );
}

function RoundCard({ round, record, passes }: { round: FtRound; record: FineTuningRecord; passes: Map<string, FtPass> }) {
  const [, setLens] = useLens();
  const best = round.checkpoints.find((c) => c.key === round.best);
  const test = best?.test_pass ? passes.get(best.test_pass) : undefined;
  return (
    <article className="card" data-round={round.key}>
      <h3>
        {round.label} <span className="tag">{round.key}</span>
      </h3>
      <p className="claim">{roundClaim(round, passes)}.</p>
      <dl className="kv">
        <dt>data</dt>
        <dd>
          {round.dataset.n_rows} rows · {round.dataset.n_turns.toLocaleString()} questions · dataset {round.dataset.version}
        </dd>
        <dt>training</dt>
        <dd>
          {round.settings.epochs} epochs · lr {round.settings.learning_rate} {round.settings.schedule} · LoRA r
          {round.settings.lora_r}
        </dd>
        <dt>pick on test</dt>
        <dd>{test ? <Rate passed={test.correct} n={test.n} digits={1} /> : '— not opened on test'}</dd>
        <dt>harness</dt>
        <dd>{harnessLabel(record, round.sweep_harness)}</dd>
        <dt>cost</dt>
        <dd>
          {formatUsd(round.cost_usd)} · {hours(round.duration_s)} on {round.instance}
        </dd>
      </dl>
      <p className="open">
        <button type="button" className="linkish" onClick={() => setLens({ round: round.key, cases: null, split: null })}>
          Open {round.label}: its checkpoints and cases →
        </button>
      </p>
    </article>
  );
}

function hours(seconds: number): string {
  return `${(seconds / 3600).toFixed(1)} h`;
}

function LeftOut({ record }: { record: FineTuningRecord }) {
  if (!record.excluded.length && !record.skipped_runs.length) return null;
  return (
    <Panel
      testId="ft-left-out"
      title={`${record.excluded.length + record.skipped_runs.length} records left off every chart, and why`}
      endpoint="fine_tuning.json · excluded"
    >
      <ul className="small">
        {record.excluded.map((e) => (
          <li key={e.id}>
            <code className="wrap-any">{e.id}</code>: {e.reason}
          </li>
        ))}
        {record.skipped_runs.map((e) => (
          <li key={e.run_id}>
            <code className="wrap-any">{e.run_id}</code>: {e.reason}
          </li>
        ))}
      </ul>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// A round
// ---------------------------------------------------------------------------

function RoundView({ round, record }: { round: FtRound; record: FineTuningRecord }) {
  const passes = passIndex(record);
  const [lens, setLens] = useLens();
  const split: SplitName = lens.get('split') === 'test' ? 'test' : 'val';
  const set = roundCases(round, record, split);
  const testSet = roundCases(round, record, 'test');
  const best = round.checkpoints.find((c) => c.key === round.best);
  const start = round.start.val_pass ? passes.get(round.start.val_pass) : undefined;
  return (
    <>
      <section className="opt-sec">
        <h2 data-testid="ft-round-claim">{roundClaim(round, passes)}</h2>
        <Figure
          testId="ft-checkpoints"
          title={`${round.label}: val accuracy of its start and each checkpoint, ${harnessLabel(record, round.sweep_harness)} harness`}
          caption={
            <>
              Each checkpoint against the start ({round.start.label}, {pct(start?.accuracy)}), paired on the same{' '}
              {start?.n ?? '—'} questions.{' '}
              {best?.vs_start ? `The pick, ${epochLabel(best.key)}: ${stepPhrase(best.vs_start)}. ` : ''}
              The y axis is cropped to the values drawn.
            </>
          }
          source={
            <>
              <code>fine_tuning.json</code> · <code>rounds[{round.key}].checkpoints</code>
            </>
          }
        >
          <CheckpointChart round={round} record={record} />
        </Figure>
      </section>

      <Panel
        testId="ft-checkpoint-table"
        title={`${round.checkpoints.length} checkpoints saved; val picked ${epochLabel(round.best)}`}
        endpoint={round.record}
        note="Val picks; test is opened only for the pick. A pass on another harness or temperature is listed under the table, not on the line."
      >
        <CheckpointTable round={round} record={record} />
        <Rescores round={round} record={record} />
      </Panel>

      <section className="opt-sec">
        <Figure
          testId="ft-loss"
          title={`${round.label}: training loss at each of ${round.steps} steps, with the saved checkpoints`}
          caption={
            <>
              Mean training loss {round.train_loss.toFixed(3)}, last step {round.final_loss.toFixed(3)}. Loss is on the
              teacher’s tokens, so it says how closely the student copies them, not how often it is right; val decides.
            </>
          }
          source={
            <>
              <code>{round.record.replace('run.json', 'metrics.jsonl')}</code>
            </>
          }
        >
          <LossChart round={round} />
        </Figure>
      </section>

      <Panel testId="ft-setup" title={setupClaim(round)} endpoint={round.dataset.card}>
        <Setup round={round} />
      </Panel>

      <div className="chips ft-split" role="group" aria-label="cases on which split">
        {(['val', 'test'] as const).map((s) => (
          <button
            key={s}
            type="button"
            className={`chip${split === s ? ' on' : ''}`}
            aria-pressed={split === s}
            disabled={s === 'test' && !testSet}
            onClick={() => setLens({ split: s === 'val' ? null : s })}
            data-testid={`ft-cases-split-${s}`}
          >
            cases on {s}
            {s === 'test' && !testSet ? ' — the pick was not opened on test' : ''}
          </button>
        ))}
      </div>
      {set ? (
        <CasesPanel
          key={`${round.key}-${split}`}
          set={set}
          record={record}
          title={
            split === 'val'
              ? `Every val question, at the start and at each checkpoint`
              : `Every test question, at the start and at the pick`
          }
          note={`Judged from the start to the pick (${epochLabel(round.best)}).`}
        />
      ) : (
        <p className="empty">— this round has no {split} passes to line up</p>
      )}
    </>
  );
}

function setupClaim(round: FtRound): string {
  return `${round.dataset.n_rows} rows, ${(round.dataset.tokens_trained / 1e6).toFixed(1)}M tokens trained, ${formatUsd(round.cost_usd)} on one GPU`;
}

function CheckpointTable({ round, record }: { round: FtRound; record: FineTuningRecord }) {
  const passes = passIndex(record);
  const rows = checkpointRows(round, passes);
  const start = round.start.val_pass ? passes.get(round.start.val_pass) : undefined;
  return (
    <div className="tw">
      <table className="ft-table" data-testid="ft-checkpoints-table">
        <thead>
          <tr>
            <th>checkpoint</th>
            <th>val</th>
            <th className="num">program</th>
            <th className="num">number</th>
            <th>step from the start</th>
            <th>test 100</th>
            <th>MLflow</th>
          </tr>
        </thead>
        <tbody>
          <tr className="dim">
            <td className="sub stagecell">
              Start · {round.start.label}
              <span className="path">{start?.model ?? '—'}</span>
            </td>
            <td>{start ? <Rate passed={start.correct} n={start.n} digits={1} /> : '—'}</td>
            <td className="num">{pct(start?.program.accuracy)}</td>
            <td className="num">{pct(start?.number.accuracy)}</td>
            <td className="step">
              <span className="muted">— where the round began</span>
            </td>
            <td>
              <span className="muted">—</span>
            </td>
            <td>
              <RunLink pass={start} />
            </td>
          </tr>
          {rows.map((r) => (
            <tr key={r.key} className={r.best ? 'pro' : undefined} data-checkpoint={r.key}>
              <td className="sub stagecell">
                {epochLabel(r.key)} {r.best && <Status tone="acc">picked</Status>}
                <span className="path">
                  step {r.step} · {r.val?.model ?? '—'}
                </span>
              </td>
              <td>{r.val ? <Rate passed={r.val.correct} n={r.val.n} digits={1} /> : <span className="muted">— not scored</span>}</td>
              <td className="num">{pct(r.val?.program.accuracy)}</td>
              <td className="num">{pct(r.val?.number.accuracy)}</td>
              <td className="step">
                <Step v={r.vsStart} />
              </td>
              <td>
                {r.test ? (
                  <Rate passed={r.test.correct} n={r.test.n} digits={1} />
                ) : (
                  <span className="muted">— {r.best ? 'not opened' : 'only the pick is opened'}</span>
                )}
              </td>
              <td>
                <RunLink pass={r.val} />
                {r.test && (
                  <>
                    {' '}
                    · <RunLink pass={r.test} />
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Rescores({ round, record }: { round: FtRound; record: FineTuningRecord }) {
  const passes = passIndex(record);
  const extra = round.rescores.map((id) => passes.get(id)).filter((p): p is FtPass => !!p);
  if (!extra.length) return null;
  return (
    <ul className="small ft-more" data-testid="ft-rescores">
      {extra.map((p) => (
        <li key={p.id}>
          {epochLabel(p.model.split('-').pop())} on {p.split}, {rescoreReason(p, round, record)}:{' '}
          <Rate passed={p.correct} n={p.n} digits={1} /> · <RunLink pass={p} />
        </li>
      ))}
    </ul>
  );
}

function Setup({ round }: { round: FtRound }) {
  const s = round.settings;
  const d = round.dataset;
  return (
    <div className="grid2">
      <dl className="kv">
        <dt>dataset</dt>
        <dd>
          {d.version} · manifest {d.manifest}
        </dd>
        <dt>rows</dt>
        <dd>
          {d.n_rows} conversations{d.n_prefix_rows ? `, ${d.n_prefix_rows} of them prefixes cut before a first wrong turn` : ''}
        </dd>
        <dt>questions</dt>
        <dd>
          {d.n_turns.toLocaleString()} · {d.n_type2} Type II conversations
        </dd>
        <dt>dropped</dt>
        <dd>{d.n_dropped} conversations the gates refused</dd>
        <dt>tokens</dt>
        <dd>{d.tokens_trained.toLocaleString()} trained</dd>
        <dt>started from</dt>
        <dd>{round.start.round ? `${round.start.label} (${round.start.model})` : 'the untrained base'}</dd>
      </dl>
      <dl className="kv">
        <dt>learning rate</dt>
        <dd>
          {s.learning_rate} · {s.schedule}
        </dd>
        <dt>epochs</dt>
        <dd>
          {s.epochs} · {round.steps} steps · batch {s.effective_batch}
        </dd>
        <dt>LoRA</dt>
        <dd>
          r {s.lora_r} · α {s.lora_alpha} · dropout {s.lora_dropout} · {s.target_modules}
        </dd>
        <dt>max length</dt>
        <dd>{s.max_length.toLocaleString()} tokens</dd>
        <dt>saved</dt>
        <dd>{s.save}</dd>
        <dt>run</dt>
        <dd>
          <code className="wrap-any">{round.run_id}</code> · code {round.code_sha} · {hours(round.duration_s)} on {round.instance}
        </dd>
        <dt>cost</dt>
        <dd>
          {formatUsd(round.cost_usd)} <span className="muted">({round.cost_basis})</span>
        </dd>
      </dl>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

const PAGE = 60;

function CasesPanel({ set, record, title, note }: { set: CaseSet; record: FineTuningRecord; title: string; note: string }) {
  const passes = passIndex(record);
  const [lens, setLens] = useLens();
  const rawFilter = lens.get('cases');
  const filter = isCaseFilter(rawFilter) ? rawFilter : 'changed';
  const rawType = lens.get('type');
  const turn = isTurnType(rawType) ? rawType : '';
  const [limit, setLimit] = useState(PAGE);
  const ids = set.columns.map((c) => c.pass);
  const query = useQuery({
    queryKey: ['fine-tuning-cases', ...ids],
    queryFn: () => getFineTuningCases(ids),
    staleTime: Infinity,
  });
  const rows = query.data?.rows ?? [];
  const counts = caseCounts(rows, set, turn);
  const shown = rows.filter((r) => matches(r, set, filter, turn));

  return (
    <Panel
      testId="ft-cases"
      title={`${title}: ${shown.length} ${filterLabel(filter, set)}`}
      endpoint={`/eval/fine-tuning/questions · ${set.split}`}
      note={note}
    >
      <div className="ft-filters">
        <div className="chips" role="group" aria-label="which cases">
          <span className="label">cases</span>
          {CASE_FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              className={`chip${filter === f ? ' on' : ''}`}
              aria-pressed={filter === f}
              onClick={() => {
                setLimit(PAGE);
                setLens({ cases: f === 'changed' ? null : f });
              }}
              data-testid={`ft-cases-${f}`}
            >
              {filterLabel(f, set)} · {query.data ? counts[f] : '…'}
            </button>
          ))}
        </div>
        <div className="chips" role="group" aria-label="which turn type">
          <span className="label">turn type</span>
          {TURN_TYPES.map((t) => (
            <button
              key={t || 'all'}
              type="button"
              className={`chip${turn === t ? ' on' : ''}`}
              aria-pressed={turn === t}
              onClick={() => {
                setLimit(PAGE);
                setLens({ type: t || null });
              }}
            >
              {t || 'both'}
            </button>
          ))}
        </div>
      </div>
      {query.error ? <ErrorNote error={query.error} /> : null}
      {query.isLoading ? (
        <LoadingRows rows={5} />
      ) : !shown.length ? (
        <p className="empty">— no question is {filterLabel(filter, set)}{turn ? ` among ${turn} turns` : ''}</p>
      ) : (
        <>
          <div className="tw">
            <table className="ft-cases" data-testid="ft-cases-table">
              <thead>
                <tr>
                  <th>question</th>
                  <th>gold</th>
                  <th>type</th>
                  {set.columns.map((c, i) => {
                    const to = campaignsLink(passes.get(c.pass)?.run_id);
                    return (
                      <th key={c.pass} className={`pass${i === set.to ? ' to' : ''}`} title={c.pass}>
                        {c.label}
                        <span className="path">
                          {c.sub}
                          {to && (
                            <>
                              {' · '}
                              <Link to={to}>run</Link>
                            </>
                          )}
                        </span>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {shown.slice(0, limit).map((r) => (
                  <tr key={`${r.report_id}-${r.turn_index}`}>
                    <td className="q">
                      <Clip text={r.question} at={140} />
                      <span className="rid">
                        {r.report_id} · q{r.turn_index + 1}
                      </span>
                    </td>
                    <td className="gold">
                      {r.gold_answer}
                      {r.gold_program && <span className="prog">{r.gold_program}</span>}
                    </td>
                    <td>{r.turn_type}</td>
                    {r.answers.map((a, i) => (
                      <td key={set.columns[i].pass} className={`ans${i === set.to ? ' to' : ''}`}>
                        {a ? (
                          <>
                            <Status tone={a.correct ? 'ok' : 'err'}>{a.correct ? 'right' : 'wrong'}</Status>
                            {a.answer || <span className="muted">— no answer</span>}
                            {a.error && (
                              <span className="err">
                                <Clip text={a.error} at={60} />
                              </span>
                            )}
                          </>
                        ) : (
                          <span className="muted">— not in this pass</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {shown.length > limit && (
            <p className="ft-more">
              <button type="button" className="linkish" onClick={() => setLimit(shown.length)}>
                Show all {shown.length} ({shown.length - limit} more)
              </button>
            </p>
          )}
        </>
      )}
    </Panel>
  );
}

