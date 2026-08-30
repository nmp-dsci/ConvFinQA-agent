import { cn } from '@/lib/utils';
import { NO_VALUE } from '../landing/format';
import type { Distribution } from './paper';
import type { SliceRow } from './benchmark';

/**
 * Hand-authored inline SVG charts.
 *
 * No charting library: this page needs four shapes, all of them simple, and a
 * dependency would cost the lazy chunk more than the shapes are worth. Bars are
 * SVG rects sized in percentages so they reflow with the column instead of
 * scaling their own text down at 420px; only the per-turn curve uses a viewBox,
 * because a line chart genuinely needs a coordinate space.
 *
 * The rule the charts share with the rest of the page: `null` is not zero. A
 * series with no measurement draws no bar and prints its reason. A zero-length
 * bar claiming a system scored nothing would be the worst possible failure mode
 * for a page whose subject is honest measurement.
 */

// ---------------------------------------------------------------------------
// Distribution — one dataset property, as proportions of a whole
// ---------------------------------------------------------------------------

export function DistributionBars({ bars }: { bars: Distribution[] }) {
  return (
    <ul className="space-y-1.5">
      {bars.map((bar) => (
        <li key={bar.label} className="min-w-0">
          <div className="flex items-baseline justify-between gap-3">
            <span className="type-small truncate text-text">{bar.label}</span>
            <span className="type-num shrink-0 text-[11px] text-muted">{bar.pct}%</span>
          </div>
          <svg
            width="100%"
            height="5"
            role="img"
            aria-label={`${bar.label}: ${bar.pct} percent`}
            className="mt-1 block overflow-hidden rounded-[2px]"
          >
            <rect width="100%" height="5" fill="var(--panel-2)" />
            <rect width={`${bar.pct}%`} height="5" fill="var(--amber)" opacity="0.75" />
          </svg>
          {bar.note && <p className="type-meta mt-0.5 text-faint">{bar.note}</p>}
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// The slice comparison — the paper's systems beside ours
// ---------------------------------------------------------------------------

const SERIES = [
  { key: 'finqanet', label: 'FinQANet RoBERTa-large', fill: 'var(--info)', opacity: 0.55 },
  { key: 'gpt3', label: 'GPT-3 Program-normal', fill: 'var(--muted)', opacity: 0.45 },
  { key: 'ours', label: 'This deployment', fill: 'var(--amber)', opacity: 1 },
] as const;

export function SliceLegend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {SERIES.map((s) => (
        <li key={s.key} className="flex items-center gap-1.5">
          <svg width="10" height="6" aria-hidden className="shrink-0">
            <rect width="10" height="6" rx="1" fill={s.fill} opacity={s.opacity} />
          </svg>
          <span className="type-meta">{s.label}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * One slice: three bars on a 0–100 point scale, sharing one axis.
 *
 * Our bar is amber and last so the eye reads the paper's baselines first —
 * this is a comparison against prior work, not a scoreboard for us.
 */
export function SliceChart({ rows }: { rows: SliceRow[] }) {
  return (
    <ul className="space-y-3.5">
      {rows.map((row) => (
        <li key={row.key} className="min-w-0">
          <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
            <span className="type-small text-text">{row.label}</span>
            <span className="type-meta shrink-0">
              {row.ours === null ? (
                <span className="text-faint">not scored here</span>
              ) : (
                <>
                  <span className="type-num text-amber">{row.ours.toFixed(1)}</span>
                  <span className="text-faint">
                    {' '}
                    vs <span className="type-num">{row.finqanet.toFixed(2)}</span> FinQANet
                    {row.n !== null && (
                      <>
                        {' · '}
                        <span className="type-num">{row.n}</span> q
                      </>
                    )}
                  </span>
                </>
              )}
            </span>
          </div>

          <svg
            width="100%"
            height="20"
            role="img"
            aria-label={`${row.label}: FinQANet ${row.finqanet}, GPT-3 ${row.gpt3}, this deployment ${
              row.ours === null ? 'not scored' : row.ours.toFixed(1)
            }`}
            className="mt-1 block"
          >
            <rect y="0" width="100%" height="20" fill="var(--panel-2)" opacity="0.5" rx="2" />
            <rect y="1" width={`${row.finqanet}%`} height="5" fill="var(--info)" opacity="0.55" />
            <rect y="7.5" width={`${row.gpt3}%`} height="5" fill="var(--muted)" opacity="0.45" />
            {row.ours !== null && (
              <rect y="14" width={`${row.ours}%`} height="5" fill="var(--amber)" />
            )}
            {/* The human ceiling, so no bar is read as "good" in isolation. */}
            <line
              x1="89.44%"
              x2="89.44%"
              y1="0"
              y2="20"
              stroke="var(--good)"
              strokeWidth="1"
              strokeDasharray="2 2"
              opacity="0.8"
            />
          </svg>

          {row.why && <p className="type-meta mt-1 text-faint">{row.why}</p>}
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Accuracy against turn position — the paper's Figure 5, and ours
// ---------------------------------------------------------------------------

export interface TurnPoint {
  /** 0-based question order, as the predictions CSV stores it. */
  order: number;
  accuracy: number;
  n: number;
}

/**
 * The curve the paper says every system falls down, drawn against ours.
 *
 * Two honesty constraints shape this chart. Only the endpoints of Figure 5 are
 * transcribed in this repo — turn 1 and turn 6 — so the paper's line is drawn
 * as a dashed segment between the two known points and labelled as such rather
 * than interpolated into a curve it never had. And our points carry their
 * sample size, because turn 7 is one question and a 0% there is noise, not a
 * cliff.
 */
export function PerTurnChart({
  ours,
  paper,
}: {
  ours: TurnPoint[];
  paper: Array<{ turn: number; finqanet: number; gpt3: number }>;
}) {
  const W = 320;
  const H = 150;
  const padL = 26;
  const padR = 8;
  const padT = 8;
  const padB = 22;

  // A shared axis with the paper's, which is 1-based over six turns.
  const maxTurn = Math.max(6, ...ours.map((p) => p.order + 1));
  const x = (turn: number) => padL + ((turn - 1) / (maxTurn - 1)) * (W - padL - padR);
  const y = (acc: number) => padT + (1 - acc / 100) * (H - padT - padB);

  // The line is drawn only through turns with a usable sample. The deepest
  // turn in the 200-conversation sample holds a single question, and letting
  // the polyline dive to its 0% would draw a cliff that is one wrong answer
  // wide. Those points are still plotted, faded and unconnected, because
  // dropping them would be its own kind of edit.
  const SAMPLE_FLOOR = 5;
  const oursPoints = ours
    .filter((p) => p.n >= SAMPLE_FLOOR)
    .map((p) => `${x(p.order + 1).toFixed(1)},${y(p.accuracy).toFixed(1)}`);
  const paperPoints = paper.map((p) => `${x(p.turn).toFixed(1)},${y(p.finqanet).toFixed(1)}`);

  return (
    <div className="min-w-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Accuracy by turn position: this deployment against the paper's Figure 5 endpoints"
        className="block h-auto w-full"
      >
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line
              x1={padL}
              x2={W - padR}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--line)"
              strokeWidth="0.5"
            />
            <text
              x={padL - 4}
              y={y(tick) + 3}
              textAnchor="end"
              fontSize="7"
              fill="var(--faint)"
              fontFamily="var(--font-mono)"
            >
              {tick}
            </text>
          </g>
        ))}

        {Array.from({ length: maxTurn }, (_, i) => i + 1).map((turn) => (
          <text
            key={turn}
            x={x(turn)}
            y={H - 8}
            textAnchor="middle"
            fontSize="7"
            fill="var(--faint)"
            fontFamily="var(--font-mono)"
          >
            {turn}
          </text>
        ))}

        {paperPoints.length > 1 && (
          <polyline
            points={paperPoints.join(' ')}
            fill="none"
            stroke="var(--info)"
            strokeWidth="1.2"
            strokeDasharray="4 3"
            opacity="0.8"
          />
        )}
        {paper.map((p) => (
          <circle key={p.turn} cx={x(p.turn)} cy={y(p.finqanet)} r="2" fill="var(--info)" />
        ))}

        {oursPoints.length > 1 && (
          <polyline
            points={oursPoints.join(' ')}
            fill="none"
            stroke="var(--amber)"
            strokeWidth="1.6"
          />
        )}
        {ours.map((p) => (
          <circle
            key={p.order}
            cx={x(p.order + 1)}
            cy={y(p.accuracy)}
            r={p.n < SAMPLE_FLOOR ? 1.5 : 2.4}
            fill="var(--amber)"
            opacity={p.n < SAMPLE_FLOOR ? 0.4 : 1}
          />
        ))}
      </svg>

      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="type-meta flex items-center gap-1.5">
          <svg width="12" height="4" aria-hidden>
            <line x1="0" y1="2" x2="12" y2="2" stroke="var(--amber)" strokeWidth="1.6" />
          </svg>
          this deployment, per turn
        </span>
        <span className="type-meta flex items-center gap-1.5">
          <svg width="12" height="4" aria-hidden>
            <line
              x1="0"
              y1="2"
              x2="12"
              y2="2"
              stroke="var(--info)"
              strokeWidth="1.2"
              strokeDasharray="3 2"
            />
          </svg>
          FinQANet, Figure 5 endpoints only
        </span>
      </div>

      {ours.length === 0 && (
        <p className="type-meta mt-1 text-faint">
          {NO_VALUE} per-turn slices unavailable — <code>/eval/runs/…/summary</code> did not answer.
        </p>
      )}
      {ours.some((p) => p.n < SAMPLE_FLOOR) && (
        <p className="type-meta mt-1 text-faint">
          Faded points have fewer than {SAMPLE_FLOOR} questions behind them — the deepest turn in
          the sample holds one — so the line stops rather than diving through them. They are still
          plotted, because dropping them would be its own kind of edit.
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// A single proportion, for a version row
// ---------------------------------------------------------------------------

export function MiniBar({ pct, tone = 'amber' }: { pct: number | null; tone?: 'amber' | 'info' }) {
  if (pct === null) {
    return <span className="type-num text-faint">{NO_VALUE}</span>;
  }
  return (
    <svg width="100%" height="4" aria-hidden className={cn('block overflow-hidden rounded-[2px]')}>
      <rect width="100%" height="4" fill="var(--panel-2)" />
      <rect
        width={`${Math.max(0, Math.min(100, pct))}%`}
        height="4"
        fill={tone === 'amber' ? 'var(--amber)' : 'var(--info)'}
      />
    </svg>
  );
}
