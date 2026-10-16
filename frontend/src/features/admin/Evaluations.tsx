import { useMemo, useState } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { Link, useSearchParams } from 'react-router-dom';
import { Figure } from '@/kit/ui';
import { NO_VALUE, formatFilingId, formatPercent } from '../landing/format';
import { getAnswers } from './api';
import { FlipsDrawer, FlipsSummary } from './FlipsDrawer';
import { CHAMPION_ROW, InstrumentTable } from './InstrumentTable';
import { PROG_ACC_CAVEAT, clip, formatCount, fraction, versionLabel } from './lib';
import { SliceChart } from './SliceChart';
import type { SliceSeries } from './SliceChart';
import {
  AdminPage,
  Caveat,
  EmptyState,
  ErrorNote,
  LoadingRows,
  Panel,
  TwoUp,
} from './ui';
import { ak, useComparison, useSplits, useVersionRows } from './useAdminData';
import './runs.css';
import { getEvalSummary } from '../../lib/api';
import { qk } from '../../lib/queryClient';
import type { AnswerRow, SplitSummary } from '../../types';

/**
 * Evaluations: the splits, the slices, every answer beside its gold, and the
 * flips between two versions.
 *
 * The order is an argument. Splits come first because no accuracy figure below
 * means anything until you know which population it was measured on; the flips
 * drawer sits above the answers table because a promotion decision is made on
 * the flips, not on the mean.
 */

/** What each split can support, as a status word — never colour alone. */
const SPLIT_STATUS: Record<string, { tone: 'ok' | 'warn' | 'no'; word: string }> = {
  optimizer_train: { tone: 'warn', word: 'seen by the optimizer' },
  never_seen: { tone: 'ok', word: 'never seen by any optimizer' },
  sampled: { tone: 'no', word: 'seen and unseen mixed' },
};

function SplitCard({
  split,
  active,
  onSelect,
}: {
  split: SplitSummary;
  active: boolean;
  onSelect: () => void;
}) {
  const status = SPLIT_STATUS[split.name];
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      data-testid={`split-${split.name}`}
      className={`card pick link${active ? ' hi' : ''}`}
    >
      <span className="head">
        <span className="label">{split.name.replace(/_/g, ' ')}</span>
        {status && <span className={`status ${status.tone} small`}>{status.word}</span>}
      </span>
      <span className="n">{formatCount(split.n_questions)}</span>
      <span className="small muted">
        questions · {formatCount(split.n_conversations)} conversations
      </span>
      <span className="small">{split.description}</span>
      <span className="foot">
        <span className="act">{active ? 'filtering the answers table ✕' : 'filter the answers table →'}</span>
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------

export default function Evaluations() {
  const [params, setParams] = useSearchParams();
  const { rows, champion, isLoading, error } = useVersionRows();
  const splits = useSplits();

  const versionNames = rows.map((r) => r.version);
  const selectedVersion =
    params.get('version') ??
    (champion && versionNames.includes(champion) ? champion : versionNames.slice(-1)[0]) ??
    '';
  // The champion may have no legacy predictions CSV (every version from v4 on
  // was promoted on the gate split, not here), so it is the default only when
  // this page can actually read it; otherwise the latest legacy version is.
  const baseline =
    params.get('baseline') ??
    (champion && versionNames.includes(champion) ? champion : versionNames.slice(-1)[0]) ??
    '';
  const candidate =
    params.get('candidate') ?? versionNames.filter((v) => v !== baseline).slice(-1)[0] ?? '';
  const flipsOpen = params.get('flips') === 'open';
  const activeSplit = params.get('split') ?? '';
  const reportFilter = params.get('report') ?? '';
  const onlyDisagreements = params.get('disagree') === '1';

  function setParam(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  }

  const comparison = useComparison(baseline, candidate);

  // One summary per version, for the per-question-position slice. The slices
  // in `/admin/experiments` carry accuracy but no denominator; the summary
  // carries both, and a bar without its denominator is not a measurement.
  const summaries = useQueries({
    queries: versionNames.map((version) => ({
      queryKey: qk.evalSummary(version),
      queryFn: () => getEvalSummary(version),
    })),
  });

  const [sliceKind, setSliceKind] = useState<'by_q_order' | 'by_turn_type' | 'by_conv_type'>(
    'by_q_order',
  );
  const [model, setModel] = useState('pydantic');

  const availableModels = useMemo(() => {
    const seen = new Set<string>();
    for (const s of summaries) for (const m of s.data?.available_models ?? []) seen.add(m);
    return [...seen].sort();
  }, [summaries]);

  const sliceSeries: SliceSeries[] = useMemo(
    () =>
      versionNames
        .map((version, i) => {
          const accuracy = summaries[i]?.data?.models?.[model];
          if (!accuracy) return null;
          return { version, points: accuracy[sliceKind] };
        })
        .filter((s): s is SliceSeries => s !== null),
    [versionNames, summaries, model, sliceKind],
  );

  // The n the slice chart was measured on — different backends scored
  // different subsets, so it is printed rather than assumed.
  const sliceDenominator = useMemo(() => {
    const idx = versionNames.indexOf(selectedVersion);
    return summaries[idx]?.data?.models?.[model]?.overall.n_total ?? null;
  }, [versionNames, selectedVersion, summaries, model]);

  const answers = useQuery({
    queryKey: ak.answers(reportFilter, onlyDisagreements, 2000),
    queryFn: () => getAnswers({ reportId: reportFilter, onlyDisagreements, limit: 2000 }),
  });

  const splitMembers = useMemo(() => {
    const split = splits.data?.find((s) => s.name === activeSplit);
    return split ? new Set(split.report_ids) : null;
  }, [splits.data, activeSplit]);

  const filteredAnswers = useMemo(() => {
    const all = answers.data ?? [];
    return splitMembers ? all.filter((r) => splitMembers.has(r.report_id)) : all;
  }, [answers.data, splitMembers]);

  const [visible, setVisible] = useState(150);
  const shown = filteredAnswers.slice(0, visible);

  const answerColumns = useMemo<Array<ColumnDef<AnswerRow, unknown>>>(() => {
    const base: Array<ColumnDef<AnswerRow, unknown>> = [
      {
        id: 'filing',
        header: 'filing · turn',
        accessorFn: (r) => r.report_id,
        meta: { align: 'left', mono: false, width: '148px' },
        cell: ({ row }) => (
          <Link
            to={`/runs/traces/eval?version=${selectedVersion}&report_id=${encodeURIComponent(
              row.original.report_id,
            )}&turn_index=${row.original.turn_index}`}
            title={row.original.report_id}
          >
            {formatFilingId(row.original.report_id)}
            <span className="muted"> · {row.original.turn_index}</span>
          </Link>
        ),
      },
      {
        id: 'question',
        header: 'question',
        accessorFn: (r) => r.question,
        meta: { align: 'left', mono: false, wrap: true, width: '230px' },
        cell: ({ row }) => clip(row.original.question, 130),
      },
      {
        id: 'type',
        header: 'type',
        accessorFn: (r) => `${r.gold_turn_type} ${r.gold_conv_type}`,
        meta: { align: 'left', width: '92px' },
        cell: ({ row }) => (
          <span className="muted">
            {row.original.gold_turn_type || '—'} · {row.original.gold_conv_type || '—'}
          </span>
        ),
      },
      {
        id: 'gold',
        header: 'gold',
        accessorFn: (r) => r.gold_answer,
        meta: { width: '86px' },
        cell: ({ row }) => (
          <span title={row.original.gold_program}>
            {row.original.gold_answer || NO_VALUE}
          </span>
        ),
      },
    ];

    for (const version of versionNames) {
      base.push({
        id: `v-${version}`,
        header: versionLabel(version),
        accessorFn: (r) => r.versions.find((v) => v.version === version)?.pred_answer ?? '',
        meta: { width: '92px' },
        cell: ({ row }) => {
          const answer = row.original.versions.find((v) => v.version === version);
          if (!answer) return <span className="muted" title="this version did not score this turn">{NO_VALUE}</span>;
          return (
            <span
              className={answer.correct ? 'v-ok' : 'v-fail'}
              title={
                answer.pred_program && answer.pred_program !== 'nan'
                  ? answer.pred_program
                  : 'no program recorded for this turn'
              }
            >
              {answer.pred_answer || NO_VALUE} {answer.correct ? '✓' : '✗'}
            </span>
          );
        },
      });
    }
    return base;
  }, [versionNames, selectedVersion]);

  /**
   * The denominators the two `n` columns used to hold.
   *
   * They were the same two numbers on every row — 770 sampled, 309 never-seen —
   * repeated down the table and already stated in the Splits panel above, so
   * they cost two columns to say nothing that varied. Said once here instead,
   * and computed from the rows rather than written down, so a version scored on
   * a different set makes the sentence change rather than makes it wrong.
   */
  const denominators = useMemo(() => {
    const overall = new Set(
      rows.map((r) => r.nQuestions).filter((n): n is number => typeof n === 'number'),
    );
    const neverSeen = new Set(
      rows.map((r) => r.holdoutN).filter((n): n is number => typeof n === 'number'),
    );
    if (overall.size !== 1 || neverSeen.size !== 1) {
      return 'denominators differ between versions — the per-version counts are on /admin/versions';
    }
    return `every version scored on the same ${formatCount([...overall][0])} sampled questions, ${formatCount([...neverSeen][0])} of them never seen by any optimizer`;
  }, [rows]);

  const versionColumns = useMemo<Array<ColumnDef<(typeof rows)[number], unknown>>>(
    () => [
      {
        id: 'version',
        header: 'version',
        accessorFn: (r) => r.version,
        meta: { align: 'left', mono: false, width: '104px' },
        cell: ({ row }) => (
          <button
            type="button"
            onClick={() => setParam('version', row.original.version)}
            className="linkish"
            aria-current={row.original.version === selectedVersion ? 'true' : undefined}
          >
            {row.original.version === selectedVersion ? <b>{versionLabel(row.original.version)}</b> : versionLabel(row.original.version)}
            {row.original.isChampion && <span className="muted"> · champion</span>}
          </button>
        ),
      },
      {
        id: 'overall',
        header: 'overall exe',
        accessorFn: (r) => r.overall ?? -1,
        cell: ({ row }) => formatPercent(row.original.overall),
      },
      {
        id: 'holdout',
        header: 'never-seen exe',
        accessorFn: (r) => r.holdout ?? -1,
        cell: ({ row }) => formatPercent(row.original.holdout),
      },
      {
        id: 'prog',
        header: 'program',
        accessorFn: (r) => r.progAcc ?? -1,
        cell: ({ row }) => formatPercent(row.original.progAcc),
      },
      {
        id: 'progN',
        header: 'program turns',
        accessorFn: (r) => r.nProgramTurns ?? -1,
        cell: ({ row }) =>
          fraction(row.original.nProgramCorrect, row.original.nProgramTurns),
      },
    ],
    // `setParam` closes over `params`; the columns only need to re-render when
    // the selection actually changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selectedVersion],
  );

  // The claim the page opens on: the selected version's score, with its denominator.
  const selectedRow = rows.find((r) => r.version === selectedVersion);
  const title =
    selectedRow && selectedRow.overall !== null && selectedRow.nQuestions ? (
      <>
        <em>{versionLabel(selectedRow.version)}</em> answered{' '}
        {formatCount(Math.round(selectedRow.overall * selectedRow.nQuestions))} of{' '}
        {formatCount(selectedRow.nQuestions)} sampled questions ({formatPercent(selectedRow.overall)})
        {selectedRow.holdout !== null && selectedRow.holdoutN
          ? ` — ${formatPercent(selectedRow.holdout)} of the ${formatCount(selectedRow.holdoutN)} never seen`
          : ''}
      </>
    ) : (
      <>
        Every early version, <em>scored</em> on the same questions
      </>
    );

  const SLICE_KINDS = [
    ['by_q_order', 'turn position'],
    ['by_turn_type', 'turn type'],
    ['by_conv_type', 'conversation'],
  ] as const;
  const sliceKindLabel = SLICE_KINDS.find(([key]) => key === sliceKind)?.[1] ?? sliceKind;

  return (
    <AdminPage
      testId="admin-evaluations"
      eyebrow="runs · legacy v1–v3_1"
      title={title}
      sub="The legacy 770-question scoring of the early versions (v1–v3_1): which questions each was measured on, how it scored by slice, what it answered beside gold, and every result that changed between two. Versions from v4 on are gated on the 349-question split — see Experiments for the pipeline track and Runtimes for the single-session challenger."
    >
      {error ? <ErrorNote error={error} /> : null}

      <Panel
        title="01 · Splits — only never_seen supports a generalisation claim"
        endpoint="/eval/splits"
        note="the 60/40 split is seeded 42; click a split to filter the answers table"
      >
        {splits.isLoading ? (
          <LoadingRows rows={3} />
        ) : splits.error ? (
          <ErrorNote error={splits.error} />
        ) : (
          <>
            <div className="cards pick-cards">
              {(splits.data ?? []).map((split) => (
                <SplitCard
                  key={split.name}
                  split={split}
                  active={activeSplit === split.name}
                  onSelect={() => setParam('split', activeSplit === split.name ? '' : split.name)}
                />
              ))}
            </div>
            <Caveat>
              Overall accuracy is measured on <code>sampled</code>, which contains every
              conversation in <code>optimizer_train</code>. It is never reported as held out, and
              the two figures are never averaged into one.
            </Caveat>
          </>
        )}
      </Panel>

      <TwoUp>
        <Panel
          title="02 · Accuracy per version — execution and program are different questions"
          endpoint="/admin/versions"
          note={`Click a version to open its turns from the answers table — ${denominators}.`}
        >
          {isLoading ? (
            <LoadingRows />
          ) : (
            <>
              <InstrumentTable
                data={rows}
                columns={versionColumns}
                rowKey={(r) => r.version}
                rowClass={(r) => (r.isChampion ? CHAMPION_ROW : undefined)}
                minWidth={480}
                testId="evaluations-versions"
              />
              <Caveat>
                {PROG_ACC_CAVEAT} The full explanation lives on the <Link to="/agent">Agent page</Link>.
              </Caveat>
            </>
          )}
        </Panel>

        <Panel
          title={`03 · Accuracy by ${sliceKindLabel} — where each version wins and loses`}
          endpoint={`/eval/runs/${selectedVersion}/summary`}
        >
          <div className="filters">
            <div className="seg" role="group" aria-label="Slice by">
              {SLICE_KINDS.map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  aria-pressed={sliceKind === key}
                  onClick={() => setSliceKind(key)}
                  className={sliceKind === key ? 'on' : undefined}
                >
                  {label}
                </button>
              ))}
            </div>
            {availableModels.length > 1 &&
              availableModels.map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={model === m}
                  onClick={() => setModel(m)}
                  className={`tog${model === m ? ' on' : ''}`}
                >
                  {m} backend
                </button>
              ))}
          </div>
          {summaries.some((s) => s.isLoading) ? (
            <LoadingRows rows={5} />
          ) : sliceSeries.length === 0 ? (
            <EmptyState>
              No {model} predictions for these versions. Pick another backend above.
            </EmptyState>
          ) : (
            <Figure
              title={`Fig · accuracy by ${sliceKindLabel}, one bar per version`}
              caption={
                <>
                  {sliceKind === 'by_q_order'
                    ? 'Accuracy falls with turn position because errors compound: a wrong intermediate answer is carried into every question that depends on it.'
                    : 'An aggregate hides which turns a version actually fixed; the same questions cut by slice show it.'}{' '}
                  {sliceDenominator
                    ? `${model} backend · ${formatCount(sliceDenominator)} scored questions for ${versionLabel(selectedVersion)}; hover a bar for its n.`
                    : 'Hover a bar for its n.'}
                </>
              }
              source={<code>/eval/runs/&lt;version&gt;/summary</code>}
            >
              <SliceChart series={sliceSeries} testId="evaluations-slices" />
            </Figure>
          )}
        </Panel>
      </TwoUp>

      <Panel
        testId="evaluations-flips"
        title="04 · Two versions compared — promotion is decided on the flips, not the mean"
        endpoint="/admin/compare"
      >
        <div className="filters">
          <label className="pick">
            <span className="label">baseline</span>
            <select value={baseline} onChange={(e) => setParam('baseline', e.target.value)}>
              {versionNames.map((v) => (
                <option key={v} value={v}>
                  {versionLabel(v)}
                </option>
              ))}
            </select>
          </label>
          <span aria-hidden className="muted">→</span>
          <label className="pick">
            <span className="label">candidate</span>
            <select value={candidate} onChange={(e) => setParam('candidate', e.target.value)}>
              {versionNames.map((v) => (
                <option key={v} value={v}>
                  {versionLabel(v)}
                </option>
              ))}
            </select>
          </label>
        </div>
        {baseline === candidate ? (
          <EmptyState>Pick two different versions to compare.</EmptyState>
        ) : comparison.isLoading ? (
          <LoadingRows rows={3} />
        ) : comparison.error ? (
          <ErrorNote error={comparison.error} />
        ) : comparison.data ? (
          <>
            <FlipsSummary
              comparison={comparison.data}
              onOpen={() => setParam('flips', 'open')}
            />
            <FlipsDrawer
              comparison={comparison.data}
              open={flipsOpen}
              onOpenChange={(open) => setParam('flips', open ? 'open' : '')}
            />
          </>
        ) : null}
      </Panel>

      <Panel
        testId="evaluations-answers"
        title="05 · Every answer, beside its gold"
        endpoint="/eval/answers"
      >
        <div className="filters">
          <input
            type="search"
            value={reportFilter}
            onChange={(e) => setParam('report', e.target.value)}
            placeholder="filter by report id"
            aria-label="Filter by report id"
          />
          <label className="pick small">
            <input
              type="checkbox"
              checked={onlyDisagreements}
              onChange={(e) => setParam('disagree', e.target.checked ? '1' : '')}
            />
            only where versions disagree
          </label>
          {activeSplit && (
            <button type="button" onClick={() => setParam('split', '')} className="chip on">
              {activeSplit} ✕ clear split
            </button>
          )}
          {answers.data && (
            <span className="count">
              {formatCount(filteredAnswers.length)} of {formatCount(answers.data.length)} scored
              questions{activeSplit ? ` in ${activeSplit}` : ''}
            </span>
          )}
        </div>
        {answers.isLoading ? (
          <LoadingRows rows={6} />
        ) : answers.error ? (
          <ErrorNote error={answers.error} />
        ) : filteredAnswers.length === 0 ? (
          <EmptyState>No scored question matches these filters.</EmptyState>
        ) : (
          <>
            <InstrumentTable
              data={shown}
              columns={answerColumns}
              rowKey={(r) => `${r.report_id}:${r.turn_index}`}
              minWidth={760}
              maxHeight={520}
            />
            {shown.length < filteredAnswers.length && (
              <p className="small">
                <button type="button" className="linkbtn" onClick={() => setVisible((n) => n + 250)}>
                  show 250 more
                </button>{' '}
                <span className="muted">
                  · {formatCount(filteredAnswers.length - shown.length)} remaining
                </span>
              </p>
            )}
            <p className="small muted">
              A cell shows what that version answered, ✓ or ✗ against gold; hover it for the program
              it produced. Click a filing to open the reconstructed stage trace for that scored turn
              in {versionLabel(selectedVersion) || 'the selected version'}.
            </p>
          </>
        )}
      </Panel>
    </AdminPage>
  );
}
