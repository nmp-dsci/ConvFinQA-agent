import { useMemo, useState } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import { Link, useSearchParams } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { NO_VALUE, formatFilingId, formatPercent } from '../landing/format';
import { getAnswers } from './api';
import { FlipsDrawer, FlipsSummary } from './FlipsDrawer';
import { CHAMPION_ROW, InstrumentTable } from './InstrumentTable';
import { PROG_ACC_CAVEAT, clip, formatCount, fraction } from './lib';
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

const SPLIT_TONE: Record<string, string> = {
  optimizer_train: 'border-amber-line',
  never_seen: 'border-good-line',
  sampled: 'border-line-2',
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
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      data-testid={`split-${split.name}`}
      className={cn(
        // `flex flex-col`: a stretched <button> would otherwise centre its
        // content and leave the three split cards misaligned.
        'flex min-w-0 flex-col rounded-md border bg-panel p-3 text-left transition-colors',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber',
        SPLIT_TONE[split.name] ?? 'border-line',
        active ? 'bg-panel-2 ring-1 ring-amber-line' : 'hover:bg-panel-2',
      )}
    >
      <div className="mono-caps">{split.name.replace(/_/g, ' ')}</div>
      <div className="type-hud mt-1.5 text-text">{formatCount(split.n_questions)}</div>
      <div className="type-meta mt-0.5 text-muted">
        questions · {formatCount(split.n_conversations)} conversations
      </div>
      <p className="type-meta mt-2 text-faint">{split.description}</p>
      <div className="type-meta mt-auto pt-2 text-faint underline underline-offset-2">
        {active ? 'filtering the answers table' : 'filter the answers table'}
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------

export default function Evaluations() {
  const [params, setParams] = useSearchParams();
  const { rows, champion, isLoading, error } = useVersionRows();
  const splits = useSplits();

  const versionNames = rows.map((r) => r.version);
  const selectedVersion = params.get('version') ?? champion ?? versionNames[0] ?? '';
  const baseline = params.get('baseline') ?? champion ?? versionNames[0] ?? '';
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
            to={`/admin/traces/eval?version=${selectedVersion}&report_id=${encodeURIComponent(
              row.original.report_id,
            )}&turn_index=${row.original.turn_index}`}
            className="hover:text-amber"
            title={row.original.report_id}
          >
            {formatFilingId(row.original.report_id)}
            <span className="text-faint"> · {row.original.turn_index}</span>
          </Link>
        ),
      },
      {
        id: 'question',
        header: 'question',
        accessorFn: (r) => r.question,
        meta: { align: 'left', mono: false, wrap: true, width: '230px' },
        cell: ({ row }) => <span className="text-muted">{clip(row.original.question, 130)}</span>,
      },
      {
        id: 'type',
        header: 'type',
        accessorFn: (r) => `${r.gold_turn_type} ${r.gold_conv_type}`,
        meta: { align: 'left', width: '92px' },
        cell: ({ row }) => (
          <span className="text-faint">
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
          <span className="text-text" title={row.original.gold_program}>
            {row.original.gold_answer || NO_VALUE}
          </span>
        ),
      },
    ];

    for (const version of versionNames) {
      base.push({
        id: `v-${version}`,
        header: version,
        accessorFn: (r) => r.versions.find((v) => v.version === version)?.pred_answer ?? '',
        meta: { width: '92px' },
        cell: ({ row }) => {
          const answer = row.original.versions.find((v) => v.version === version);
          if (!answer) return <span className="text-faint">{NO_VALUE}</span>;
          return (
            <span
              className={answer.correct ? 'text-good' : 'text-bad'}
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
            className={cn('hover:text-amber', row.original.version === selectedVersion && 'text-amber')}
          >
            {row.original.version}
            {row.original.isChampion && <span className="text-faint"> · champion</span>}
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
        id: 'n',
        header: 'n',
        accessorFn: (r) => r.nQuestions ?? -1,
        cell: ({ row }) => formatCount(row.original.nQuestions),
      },
      {
        id: 'holdout',
        header: 'never-seen exe',
        accessorFn: (r) => r.holdout ?? -1,
        cell: ({ row }) => (
          <span className="text-good">{formatPercent(row.original.holdout)}</span>
        ),
      },
      {
        id: 'holdoutN',
        header: 'n',
        accessorFn: (r) => r.holdoutN ?? -1,
        cell: ({ row }) => formatCount(row.original.holdoutN),
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

  return (
    <AdminPage
      testId="admin-evaluations"
      eyebrow="admin · evaluations"
      title="Evaluations"
      sub="Which questions each version was measured on, how it scored by slice, what it answered beside gold, and every result that changed between two versions."
    >
      {error ? <ErrorNote error={error} /> : null}

      <Panel
        title="Splits"
        endpoint="/eval/splits"
        note="the 60/40 split is seeded 42; only never_seen supports a generalisation claim"
      >
        {splits.isLoading ? (
          <LoadingRows rows={3} />
        ) : splits.error ? (
          <ErrorNote error={splits.error} />
        ) : (
          <>
            <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
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
          title="Accuracy per version"
          endpoint="/admin/versions"
          note="execution accuracy and program accuracy are different questions about the same run"
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
                minWidth={560}
                testId="evaluations-versions"
              />
              <Caveat>{PROG_ACC_CAVEAT} The full explanation lives on the System page.</Caveat>
            </>
          )}
        </Panel>

        <Panel
          title="Slices"
          endpoint={`/eval/runs/${selectedVersion}/summary`}
          note={
            sliceDenominator
              ? `${model} backend · ${formatCount(sliceDenominator)} scored questions for ${selectedVersion}`
              : 'accuracy per slice, per version'
          }
          right={
            <div className="flex flex-wrap gap-1">
              {(
                [
                  ['by_q_order', 'turn position'],
                  ['by_turn_type', 'turn type'],
                  ['by_conv_type', 'conversation'],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setSliceKind(key)}
                  className={cn(
                    'rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px] tracking-[0.06em] uppercase',
                    sliceKind === key
                      ? 'border-amber-line bg-amber-soft text-amber'
                      : 'border-line text-faint hover:text-text',
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          }
        >
          {summaries.some((s) => s.isLoading) ? (
            <LoadingRows rows={5} />
          ) : sliceSeries.length === 0 ? (
            <EmptyState>
              No {model} predictions for these versions. Pick another backend below.
            </EmptyState>
          ) : (
            <SliceChart series={sliceSeries} testId="evaluations-slices" />
          )}
          {availableModels.length > 1 && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <span className="mono-caps">backend</span>
              {availableModels.map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setModel(m)}
                  className={cn(
                    'rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px]',
                    model === m
                      ? 'border-amber-line bg-amber-soft text-amber'
                      : 'border-line text-faint hover:text-text',
                  )}
                >
                  {m}
                </button>
              ))}
            </div>
          )}
          <p className="type-meta mt-2 text-faint">
            Accuracy falls with turn position because errors compound: a wrong intermediate answer
            is carried into every question that depends on it.
          </p>
        </Panel>
      </TwoUp>

      <Panel
        testId="evaluations-flips"
        title="Version comparison — the promotion contract"
        endpoint="/admin/compare"
        right={
          <div className="flex flex-wrap items-center gap-1.5">
            <label className="mono-caps flex items-center gap-1">
              baseline
              <select
                value={baseline}
                onChange={(e) => setParam('baseline', e.target.value)}
                className="rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text"
              >
                {versionNames.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label className="mono-caps flex items-center gap-1">
              candidate
              <select
                value={candidate}
                onChange={(e) => setParam('candidate', e.target.value)}
                className="rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text"
              >
                {versionNames.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          </div>
        }
      >
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
        title="Every answer, beside its gold"
        endpoint="/eval/answers"
        note={
          answers.data
            ? `${formatCount(filteredAnswers.length)} of ${formatCount(answers.data.length)} scored questions${
                activeSplit ? ` in ${activeSplit}` : ''
              }`
            : undefined
        }
        right={
          <div className="flex flex-wrap items-center gap-1.5">
            <input
              value={reportFilter}
              onChange={(e) => setParam('report', e.target.value)}
              placeholder="filter by report id"
              aria-label="Filter by report id"
              className="w-40 rounded-[4px] border border-line-2 bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-text placeholder:text-faint"
            />
            <label className="mono-caps flex items-center gap-1">
              <input
                type="checkbox"
                checked={onlyDisagreements}
                onChange={(e) => setParam('disagree', e.target.checked ? '1' : '')}
                className="accent-[var(--amber)]"
              />
              versions disagree
            </label>
            {activeSplit && (
              <button
                type="button"
                onClick={() => setParam('split', '')}
                className="rounded-[4px] border border-amber-line bg-amber-soft px-1.5 py-0.5 font-mono text-[10px] text-amber"
              >
                {activeSplit} ✕
              </button>
            )}
          </div>
        }
      >
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
              <button
                type="button"
                onClick={() => setVisible((n) => n + 250)}
                className="mono-caps mt-2 w-full rounded-[4px] border border-line py-1.5 hover:border-amber-line hover:text-amber"
              >
                show 250 more · {formatCount(filteredAnswers.length - shown.length)} remaining
              </button>
            )}
            <p className="type-meta mt-2 text-faint">
              A cell shows what that version answered; hover it for the program it produced. Click a
              filing to open the reconstructed stage trace for that scored turn in{' '}
              {selectedVersion || 'the selected version'}.
            </p>
          </>
        )}
      </Panel>
    </AdminPage>
  );
}
