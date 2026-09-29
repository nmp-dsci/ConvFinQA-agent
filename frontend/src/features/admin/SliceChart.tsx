import { formatPercent } from '../landing/format';
import { versionLabel } from './lib';
import './runs.css';

/**
 * Accuracy by slice, one group per label, one bar per version — hand-drawn SVG on the house
 * `.dia` vocabulary (ui/DESIGN.md §3 Figures).
 *
 * Two things this chart refuses to do. It does not draw a bar for a slice with no questions in
 * it — an empty bucket is not zero accuracy — and every bar carries its denominator in its
 * `<title>`, because "33% on q_order 6" is six questions and reads very differently once you
 * know that.
 */

export interface SlicePoint {
  label: string;
  accuracy: number;
  n_correct: number;
  n_total: number;
}

export interface SliceSeries {
  version: string;
  points: SlicePoint[];
}

/** The first series is the accent, the rest step down the ink ladder. */
// Not the kit's `.bar`: its HTML rule sets `height: .6em`, which overrides an SVG rect's height.
const SERIES_BAR = ['sbar s1', 'sbar s2', 'sbar s3'];
const SERIES_SWATCH = ['sw a', 'sw b', 'sw c'];

function labelsOf(series: SliceSeries[]): string[] {
  const labels: string[] = [];
  for (const s of series) {
    for (const p of s.points) if (!labels.includes(p.label)) labels.push(p.label);
  }
  return labels;
}

export function SliceChart({
  series,
  height = 220,
  testId,
}: {
  series: SliceSeries[];
  height?: number;
  testId?: string;
}) {
  const labels = labelsOf(series);

  if (labels.length === 0) {
    return (
      <p className="empty" data-testid={testId}>
        no slice in this payload
      </p>
    );
  }

  // Sized for a half-width column (~520px), so 13px labels render at ~13px on a desktop.
  const W = 520;
  const H = Math.max(160, height);
  const left = 44;
  const right = 8;
  const top = 14;
  const bottom = 30;
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const groupW = plotW / labels.length;
  const gap = 2;
  const barW = Math.max(3, Math.min(22, (groupW * 0.78 - gap * (series.length - 1)) / series.length));
  const clusterW = barW * series.length + gap * (series.length - 1);
  const y = (fraction: number) => top + plotH * (1 - fraction);

  return (
    <div data-testid={testId} className="slice-fig">
      <div className="figscroll">
        <svg
          className="dia"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Accuracy by slice for ${series.map((s) => versionLabel(s.version)).join(', ')}`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line className={f === 0 ? 'ax' : 'gl'} x1={left} x2={W - right} y1={y(f)} y2={y(f)} />
              <text className="tx k" x={left - 6} y={y(f) + 4} textAnchor="end">
                {f * 100}%
              </text>
            </g>
          ))}
          {labels.map((label, gi) => {
            const x0 = left + gi * groupW + (groupW - clusterW) / 2;
            return (
              <g key={label}>
                {series.map((s, si) => {
                  const point = s.points.find((p) => p.label === label);
                  // No questions in this bucket means no bar, not a bar at zero.
                  if (!point || point.n_total <= 0) return null;
                  const h = Math.max(1, plotH * point.accuracy);
                  return (
                    <rect
                      key={s.version}
                      className={SERIES_BAR[si % SERIES_BAR.length]}
                      x={x0 + si * (barW + gap)}
                      y={top + plotH - h}
                      width={barW}
                      height={h}
                      rx={2}
                    >
                      <title>
                        {versionLabel(s.version)} · {label}: {formatPercent(point.accuracy)} ·{' '}
                        {point.n_correct}/{point.n_total} questions
                      </title>
                    </rect>
                  );
                })}
                <text className="tx k" x={left + gi * groupW + groupW / 2} y={H - 10} textAnchor="middle">
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="legend">
        {series.map((s, i) => (
          <span key={s.version}>
            <span aria-hidden className={SERIES_SWATCH[i % SERIES_SWATCH.length]} />
            {versionLabel(s.version)}
          </span>
        ))}
      </p>
    </div>
  );
}
