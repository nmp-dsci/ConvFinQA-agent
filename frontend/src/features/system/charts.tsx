import { NO_VALUE } from '../landing/format';
import { Dia } from './diagrams';
import type { Distribution } from './paper';
import type { SliceRow } from './benchmark';

/**
 * The debrief's charts, on house tokens.
 *
 * Proportion bars are HTML, not SVG: their labels are page text at reading size and the bars reflow
 * with the column. Only the per-turn curve is an SVG, because a line chart genuinely needs a
 * coordinate space; it uses the kit's `.dia` vocabulary and never draws a label under 12.8px.
 *
 * The rule the charts share with the rest of the page: `null` is not zero. A series with no
 * measurement draws no bar and prints its reason.
 */

// ---------------------------------------------------------------------------
// Distribution — one dataset property, as proportions of a whole
// ---------------------------------------------------------------------------

export function DistributionBars({ bars }: { bars: Distribution[] }) {
  return (
    <ul className="sys-bars">
      {bars.map((bar) => (
        <li key={bar.label}>
          <div className="hd">
            <span>{bar.label}</span>
            <span className="num">{bar.pct}%</span>
          </div>
          <span className="sys-track" role="img" aria-label={`${bar.label}: ${bar.pct} percent`}>
            <i style={{ width: `${bar.pct}%` }} />
          </span>
          {bar.note && <p className="why">{bar.note}</p>}
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// The slice comparison — the paper's systems beside ours
// ---------------------------------------------------------------------------

export function SliceLegend() {
  return (
    <ul className="sys-legend">
      <li>
        <span className="sw b" aria-hidden /> FinQANet RoBERTa-large
      </li>
      <li>
        <span className="sw g" aria-hidden /> GPT-3 Program-normal
      </li>
      <li>
        <span className="sw a" aria-hidden /> this deployment
      </li>
      <li>
        <span className="ln ref" aria-hidden /> human expert, 89.44
      </li>
    </ul>
  );
}

/**
 * One slice: three bars on a 0–100 point scale, sharing one axis, with the human reference drawn
 * through all three so no bar is read as "good" in isolation. Ours is last, so the eye reads the
 * paper's baselines first — this is a comparison against prior work, not a scoreboard for us.
 */
export function SliceChart({ rows }: { rows: SliceRow[] }) {
  return (
    <ul className="sys-bars">
      {rows.map((row) => (
        <li key={row.key} className="sys-slice">
          <div className="hd">
            <span>{row.label}</span>
            <span className="small">
              {row.ours === null ? (
                <span className="muted">not scored here</span>
              ) : (
                <>
                  <span className="num">{row.ours.toFixed(1)}</span>
                  <span className="muted">
                    {' '}
                    vs <span className="num">{row.finqanet.toFixed(2)}</span> FinQANet
                    {row.n !== null && (
                      <>
                        {' · '}
                        <span className="num">{row.n}</span> questions
                      </>
                    )}
                  </span>
                </>
              )}
            </span>
          </div>
          <div
            className="tracks"
            role="img"
            aria-label={`${row.label}: FinQANet ${row.finqanet}, GPT-3 ${row.gpt3}, this deployment ${
              row.ours === null ? 'not scored' : row.ours.toFixed(1)
            }`}
          >
            <i className="f" style={{ width: `${row.finqanet}%` }} />
            <i className="g" style={{ width: `${row.gpt3}%` }} />
            {row.ours !== null ? <i className="o" style={{ width: `${row.ours}%` }} /> : <i />}
            <span className="ref" style={{ left: '89.44%' }} />
          </div>
          {row.why && <p className="why">{row.why}</p>}
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
 * Only the endpoints of Figure 5 are transcribed in this repo — turn 1 and turn 6 — so the paper's
 * line is a dashed segment between the two known points, labelled as such rather than interpolated.
 * Our points carry their sample size, because the deepest turn is one question and a 0% there is
 * noise, not a cliff.
 */
export function PerTurnChart({
  ours,
  paper,
}: {
  ours: TurnPoint[];
  paper: Array<{ turn: number; finqanet: number; gpt3: number }>;
}) {
  const W = 540;
  const H = 280;
  const padL = 52;
  const padR = 20;
  const padT = 20;
  const padB = 44;

  // A shared axis with the paper's, which is 1-based over six turns.
  const maxTurn = Math.max(6, ...ours.map((p) => p.order + 1));
  const x = (turn: number) => padL + ((turn - 1) / (maxTurn - 1)) * (W - padL - padR);
  const y = (acc: number) => padT + (1 - acc / 100) * (H - padT - padB);

  // The line runs only through turns with a usable sample; thinner turns are still plotted, faded
  // and unconnected, because dropping them would be its own kind of edit.
  const SAMPLE_FLOOR = 5;
  const oursPoints = ours
    .filter((p) => p.n >= SAMPLE_FLOOR)
    .map((p) => `${x(p.order + 1).toFixed(1)},${y(p.accuracy).toFixed(1)}`);
  const paperPoints = paper.map((p) => `${x(p.turn).toFixed(1)},${y(p.finqanet).toFixed(1)}`);

  return (
    <div>
      <Dia
        id="perturn"
        w={W}
        h={H}
        label="Accuracy by turn position: this deployment against the paper's Figure 5 endpoints"
      >
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line className="gl" x1={padL} x2={W - padR} y1={y(tick)} y2={y(tick)} />
            <text className="tx k" x={padL - 8} y={y(tick) + 4} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}
        {Array.from({ length: maxTurn }, (_, i) => i + 1).map((turn) => (
          <text key={turn} className="tx k" x={x(turn)} y={H - 20} textAnchor="middle">
            {turn}
          </text>
        ))}
        <text className="tx s" x={W - padR} y={H - 2} textAnchor="end">
          turn in the conversation
        </text>

        {paperPoints.length > 1 && (
          <polyline className="ed dash" points={paperPoints.join(' ')} />
        )}
        {paper.map((p) => (
          <circle key={p.turn} className="bar b" cx={x(p.turn)} cy={y(p.finqanet)} r="3.5" />
        ))}

        {oursPoints.length > 1 && <polyline className="spark" points={oursPoints.join(' ')} />}
        {ours.map((p) => (
          <circle
            key={p.order}
            className="bar hi"
            cx={x(p.order + 1)}
            cy={y(p.accuracy)}
            r={p.n < SAMPLE_FLOOR ? 2.5 : 4}
            opacity={p.n < SAMPLE_FLOOR ? 0.4 : 1}
          />
        ))}
      </Dia>

      <ul className="sys-legend">
        <li>
          <span className="ln" aria-hidden /> this deployment, per turn
        </li>
        <li>
          <span className="ln b" aria-hidden /> FinQANet, Figure 5 endpoints only
        </li>
      </ul>

      {ours.length === 0 && (
        <p className="small muted">
          {NO_VALUE} per-turn slices unavailable: <code className="lit">/eval/runs/…/summary</code>{' '}
          did not answer.
        </p>
      )}
      {ours.some((p) => p.n < SAMPLE_FLOOR) && (
        <p className="small muted">
          Faded points have fewer than {SAMPLE_FLOOR} questions behind them, so the line stops
          rather than diving through them.
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
    return <span className="num muted">{NO_VALUE}</span>;
  }
  return (
    <span className="sys-track mini" aria-hidden>
      <i
        className={tone === 'amber' ? 'acc' : undefined}
        style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
      />
    </span>
  );
}
