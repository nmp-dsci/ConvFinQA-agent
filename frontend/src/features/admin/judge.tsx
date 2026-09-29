import { Status } from '@/kit/ui';
import type { JudgeSummary, JudgeVerdict as JudgeVerdictSummary } from './api';
import { EmptyState } from './ui';
import { formatPp } from './runtimeStory';
import { judgeRows } from './runtimeStory';
import type { JudgeRow } from './runtimeStory';
import { NO_VALUE, formatCount, formatPercent } from '../landing/format';
import { versionLabel } from './lib';
import './runtimes.css';

/**
 * s12/s13: the confidence judge — tried on the Agent SDK runtime, measured on the sealed split,
 * not adopted. It lives on Scoring (slot 3, "how a turn is judged"), read from the same
 * `/eval/campaigns` payload the Runtimes page reads; Runtimes re-exports both components so
 * the story tests keep importing them from there.
 */

/**
 * What the judge experiment settled, in the terms the decision needs.
 *
 * The band is only worth acting on if it beats the policy it replaces —
 * releasing every answer — so this leads with that comparison and with
 * whether the interval separates them, not with the high-band accuracy on
 * its own, which reads as a pass mark when it is not one.
 */
export function JudgeVerdictBanner({ verdict }: { verdict: JudgeVerdictSummary }) {
  const adopt = verdict.significant && verdict.meets_target;
  const [lo, hi] = verdict.high_band_accuracy_ci ?? [null, null];
  return (
    <section
      data-testid="judge-verdict"
      data-significant={verdict.significant ? 'true' : 'false'}
      data-tone={adopt ? 'ok' : 'warn'}
      className="card rt-verdict"
    >
      <div className="rt-verdict-head">
        <span className="label">recommendation</span>
        <Status tone={adopt ? 'ok' : 'warn'}>{adopt ? 'adopt as a gate' : 'advisory only'}</Status>
        <span className="mono muted">
          {versionLabel(verdict.version)} on {verdict.split}
        </span>
      </div>
      <p className="lead">
        Tried at the end of the SDK work, and{' '}
        {verdict.significant ? 'it beats' : <strong>not statistically significant</strong>}
        {verdict.significant ? '' : ' against the policy it replaces'}: releasing everything
        scores <span className="num">{formatPercent(verdict.baseline_accuracy, 2)}</span> on
        the same turns, the high band{' '}
        <span className="num">{formatPercent(verdict.high_band_accuracy, 2)}</span> —{' '}
        <span className="num">{formatPp(verdict.delta_pp)}</span>, and the 95% interval{' '}
        {lo === null || hi === null
          ? 'is unavailable'
          : `${formatPercent(lo, 1)}–${formatPercent(hi, 1)}`}{' '}
        {verdict.significant ? 'excludes' : 'contains'} that baseline.
      </p>
      <p className="small muted">
        It withholds {formatCount(verdict.n_withheld)} answers to remove{' '}
        {formatCount(verdict.n_failures_caught)} wrong ones, and{' '}
        {formatCount(verdict.n_false_alarms)} of those withheld were correct. The band is
        recorded and shown; it is not evidence to gate on.
      </p>
    </section>
  );
}

const JUDGE_HEADS: Array<[string, boolean]> = [
  ['judge', false],
  ['split', false],
  ['turns', true],
  ['runtime acc.', true],
  ['coverage', true],
  ['high band correct', true],
  ['wrong / released', true],
  ['unseen error (95%)', true],
  ['failures caught', true],
  ['AUROC', true],
  ['target', false],
];

/**
 * The judge, one row per (version, split). The champion's test row is the headline: the gate
 * split, scored once by the frozen judge. "Unseen error" is the one-sided 95% bound on the high
 * band's true error rate — zero misses among n released proves ≤ ~3/n, never 0%.
 */
export function JudgeTable({ rows, target }: { rows: JudgeRow[]; target: number }) {
  if (!rows.length) {
    return (
      <EmptyState>
        not yet run — no judge has been scored on the calibrate or test split
      </EmptyState>
    );
  }
  return (
    <div className="tw">
      <table data-testid="judge-table" className="rt-table">
        <thead>
          <tr>
            {JUDGE_HEADS.map(([h, num]) => (
              <th key={h} scope="col" className={num ? 'num' : undefined}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={`${row.version}-${row.split}`}
              data-judge={row.version}
              data-split={row.split}
              data-champion={row.isChampion ? 'true' : 'false'}
              className={row.isChampion && row.split === 'test' ? 'pro' : undefined}
            >
              <td className="sub">
                <span className="mono">{versionLabel(row.version)}</span>
                {row.isChampion && <span className="path">champion</span>}
              </td>
              <td className="mono">{row.split}</td>
              <td className="num">
                {row.n} · {row.nWrong} wrong
              </td>
              <td className="num">{formatPercent(row.runtimeAccuracy)}</td>
              <td className="num rt-strong">{formatPercent(row.coverage)}</td>
              <td className={`num rt-strong ${row.meetsTarget ? 'v-ok' : 'v-warn'}`}>
                {row.highBandAccuracy == null ? NO_VALUE : `${(row.highBandAccuracy * 100).toFixed(2)}%`}
              </td>
              <td className="num">
                {row.nHighWrong} / {row.nHigh}
              </td>
              <td className="num">
                {row.unseenErrorUpper == null ? NO_VALUE : `≤ ${(row.unseenErrorUpper * 100).toFixed(2)}%`}
              </td>
              <td className="num">
                {formatPercent(row.failureCapture)} ({row.nCaught}/{row.nWrong})
              </td>
              <td className="num">{row.auroc == null ? NO_VALUE : row.auroc.toFixed(3)}</td>
              <td className="mono">
                <Status tone={row.meetsTarget ? 'ok' : 'warn'}>
                  {row.meetsTarget ? `≤ ${(target * 100).toFixed(0)}% met` : 'missed'}
                </Status>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The judge's whole record, as Scoring shows it: the verdict, then every scored split. */
export function JudgeEvidence({ judge }: { judge: JudgeSummary | null | undefined }) {
  const rows = judgeRows(judge);
  const target = judge?.error_target ?? 0.01;
  return (
    <>
      <p className="small muted">
        judge_champion{' '}
        <span className="mono">{judge?.champion ? versionLabel(judge.champion) : NO_VALUE}</span> ·
        calibrated to{' '}
        <span className="mono">
          {judge?.runtime_version ? versionLabel(judge.runtime_version) : NO_VALUE}
        </span>
      </p>
      {judge?.verdict && <JudgeVerdictBanner verdict={judge.verdict} />}
      <JudgeTable rows={rows} target={target} />
      {rows.length > 0 && (
        <p className="callout">
          Zero wrong answers among <em>n</em> released proves the true error rate is below about
          3/<em>n</em> at 95% confidence, not that it is zero — that is the &ldquo;unseen
          error&rdquo; column. Some misses are not the runtime&rsquo;s: a gold answer rounded
          differently from the calculator&rsquo;s exact result leaves a trace that passes every
          check, and no judge without gold can withhold it.
        </p>
      )}
    </>
  );
}
