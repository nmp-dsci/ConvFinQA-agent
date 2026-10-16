import type { ReactNode } from 'react';
import { versionLabel } from '../admin/lib';
import { NO_VALUE, formatCount, formatPercent, formatUsd } from '../landing/format';
import { NO_FILTERS, OUTCOMES, filterRows, formatDay, formatP, formatPp, shortModel } from './timeline';
import type { LedgerFilters, LedgerRow } from './timeline';
import './story.css';

/**
 * Every pass as one row: the ledger the timeline is drawn from (s21).
 *
 * The rows are the story's `test_timeline`, oldest first, and — behind the Runs page's toggle —
 * its `train_passes`, marked as train in their own column. Filter chips narrow by runtime, model
 * and outcome; each group lists only what the rows hold, so no chip ever filters to nothing.
 *
 * An absent value prints `—` with its reason: in the cell when it is particular to the row (the
 * champion's own pass is not compared with itself; an incomplete pass is never compared), once in
 * the note under the table when it is systematic (pipeline passes recorded no cost and have no
 * effort setting).
 */
export function PassLedger({
  rows,
  filters = NO_FILTERS,
  onFilters,
  showSplit = false,
  caption,
  testId = 'pass-ledger',
}: {
  rows: readonly LedgerRow[];
  filters?: LedgerFilters;
  onFilters?: (next: LedgerFilters) => void;
  showSplit?: boolean;
  caption?: ReactNode;
  testId?: string;
}) {
  const shown = filterRows(rows, filters);
  const runtimes = unique(rows.map((r) => r.runtime));
  const models = unique(rows.map((r) => shortModel(r.model)).filter(Boolean));
  const outcomes = unique(rows.map((r) => r.outcome));
  const set = (key: keyof LedgerFilters, value: string) =>
    onFilters?.({ ...filters, [key]: filters[key] === value ? '' : value });

  return (
    <div className="ledger" data-testid={testId}>
      {onFilters && (
        <div className="ledger-filters" role="group" aria-label="filter the passes">
          <ChipGroup
            name="runtime"
            options={runtimes.map((r) => ({ value: r, label: r === 'agent_sdk' ? 'single session' : 'pipeline' }))}
            current={filters.runtime}
            onPick={(v) => set('runtime', v)}
          />
          <ChipGroup
            name="model"
            options={models.map((m) => ({ value: m, label: m }))}
            current={filters.model}
            onPick={(v) => set('model', v)}
          />
          <ChipGroup
            name="outcome"
            options={outcomes.map((o) => ({ value: o, label: o === 'train' ? 'train' : OUTCOMES[o].word }))}
            current={filters.outcome}
            onPick={(v) => set('outcome', v)}
          />
        </div>
      )}
      <div className="tw">
        <table>
          {caption && <caption>{caption}</caption>}
          <thead>
            <tr>
              <th>day</th>
              <th>pass</th>
              {showSplit && <th>split</th>}
              <th>runtime</th>
              <th>model</th>
              <th>effort</th>
              <th className="num">accuracy</th>
              <th className="num">n</th>
              <th>vs the champion of its day</th>
              <th>outcome</th>
              <th>experiment</th>
              <th className="num">cost</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <Row key={r.key} row={r} showSplit={showSplit} />
            ))}
          </tbody>
        </table>
      </div>
      {shown.length === 0 && (
        <p className="empty" data-testid="ledger-empty">
          — no pass matches these filters; clear one to see more
        </p>
      )}
      <p className="ledger-note">
        {formatCount(shown.length)} of {formatCount(rows.length)} passes shown. Δ is the gate’s
        question-by-question figure where a gate or a challenger verdict paired the two passes
        (“paired”, with its one-sided p), and a plain difference of accuracies otherwise
        (“unpaired”). Pipeline passes recorded no cost and have no effort setting, so those cells
        read {NO_VALUE}.
      </p>
    </div>
  );
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function ChipGroup({
  name,
  options,
  current,
  onPick,
}: {
  name: string;
  options: { value: string; label: string }[];
  current: string;
  onPick: (value: string) => void;
}) {
  if (options.length < 2) return null;
  return (
    <div className="chips" data-filter={name}>
      <span className="label">{name}</span>
      <button type="button" className={`chip${current ? '' : ' on'}`} aria-pressed={!current} onClick={() => current && onPick(current)}>
        all
      </button>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={`chip${current === o.value ? ' on' : ''}`}
          aria-pressed={current === o.value}
          onClick={() => onPick(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Row({ row: r, showSplit }: { row: LedgerRow; showSplit: boolean }) {
  const isTrain = r.split === 'train';
  const outcome = r.outcome === 'train' ? null : OUTCOMES[r.outcome];
  const cls = [r.championRun ? 'pro' : '', r.outcome === 'incomplete' ? 'dim' : '', isTrain ? 'train' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <tr
      className={cls || undefined}
      data-testid={isTrain ? 'train-row' : 'pass-row'}
      data-outcome={r.outcome}
      data-run={r.run_name}
    >
      <td className="mono day">{formatDay(r.date)}</td>
      <td className="sub">
        {r.label}
        <span className="path">{r.run_name}</span>
      </td>
      {showSplit && (
        <td className="splitcell">
          <span className={`tag split${isTrain ? ' warn' : ''}`}>{r.split}</span>
          {isTrain && (
            <span className="path">
              {r.seed ? `seed ${r.seed}` : 'manifest train list'}
              {r.stopAtFirstWrong ? ' · stops at first wrong' : ''}
            </span>
          )}
        </td>
      )}
      <td>{r.runtime === 'agent_sdk' ? 'single session' : 'pipeline'}</td>
      <td className="mono nowrap">{shortModel(r.model) || NO_VALUE}</td>
      <td className="mono">{r.effort ?? NO_VALUE}</td>
      <td className="num">{formatPercent(r.accuracy)}</td>
      <td className="num">{formatCount(r.n)}</td>
      <td className="wrap">
        <Versus row={r} />
      </td>
      <td className="wrap">
        {outcome ? (
          <span className={`status ${outcome.tone}`} title={r.reason || outcome.meaning}>
            {outcome.word}
          </span>
        ) : (
          <span className="status no" title="train passes write versions and never promote them">
            train
          </span>
        )}
      </td>
      <td className="mono">{r.experiment ?? NO_VALUE}</td>
      <td className="num">{formatUsd(r.cost)}</td>
    </tr>
  );
}

function Versus({ row: r }: { row: LedgerRow }) {
  if (r.split === 'train') {
    return (
      <>
        {NO_VALUE}
        <span className="path">train: never set against the champion</span>
      </>
    );
  }
  if (r.outcome === 'incomplete') {
    return (
      <>
        {NO_VALUE}
        <span className="path">never compared: {r.reason || 'turns unscored'}</span>
      </>
    );
  }
  const vs = r.vs;
  if (!vs) {
    return (
      <>
        {NO_VALUE}
        <span className="path">{r.championRun ? 'the champion’s own pass' : 'no champion was in force'}</span>
      </>
    );
  }
  return (
    <>
      <span className="num">{formatPp(vs.delta_pp)}</span>
      <span className="path">
        vs {versionLabel(vs.version)}
        {vs.model && vs.version?.startsWith('sdk_') ? ` · ${shortModel(vs.model)}` : ''} ·{' '}
        {vs.paired
          ? `paired · p ${formatP(vs.p_value)}${vs.p_kind ? ` ${vs.p_kind.replace(/^one-sided /, '')}` : ''}`
          : 'unpaired difference'}
      </span>
    </>
  );
}
