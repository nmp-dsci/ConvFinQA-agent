import { useId } from 'react';
import type { ChampionStep, TestPass } from '../admin/api';
import { versionLabel } from '../admin/lib';
import { formatPercent } from '../landing/format';
import {
  OUTCOMES,
  agentLabel,
  formatDay,
  formatP,
  formatPp,
  markOf,
  shortModel,
  timelineModel,
} from './timeline';
import type { MarkShape, TimelineModel } from './timeline';
import './story.css';

/**
 * Every complete test pass on the gate split as one point, in scoring order, with the champion
 * of the day as a step line under them (s21).
 *
 * Hand-drawn SVG on the kit's `.dia` vocabulary; every colour is a token (`story.css`). What it
 * holds on purpose:
 *
 *  - **x is scoring order, banded by day** (owner's D5): shaded bands group the passes of one day,
 *    and a break mark with the gap in days sits between two days that are not consecutive.
 *  - **The y axis is cropped and says so.** The passes span 77–91%; a dot's position carries the
 *    value, not a bar's length, so the axis starts near the lowest pass and the caption names where.
 *  - **The shape carries the outcome as well as the colour**: a filled dot is a champion's pass,
 *    a ring was judged and refused or never judged, a square passed its rule, a hatched diamond is
 *    a challenger agent that fell short. The key under the figure says it in words.
 *  - **Labels never collide**: each label takes the first of up to four rows where it clears its
 *    neighbour, with a leader line to its point; the compact form labels only the passes that moved
 *    or challenged the champion.
 *  - **An incomplete pass is not drawn**; the key says how many were left off and why.
 */
export function TimelineChart({
  passes,
  steps,
  compact = false,
  gateQuestions,
}: {
  passes: readonly TestPass[] | undefined;
  steps: readonly ChampionStep[] | undefined;
  compact?: boolean;
  gateQuestions?: number | null;
}) {
  const model = timelineModel(passes, steps);
  const hatchId = `tl-hatch-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  if (!model.points.length) {
    return (
      <p className="empty" data-testid="timeline-empty">
        — no complete test pass is on record yet: the story has nothing to draw until a pass scores
        every question of the gate split
      </p>
    );
  }
  return (
    <div className="tl" data-testid="timeline-chart" data-compact={compact ? 'true' : 'false'}>
      <div className="figscroll">
        <TimelineSvg model={model} compact={compact} hatchId={hatchId} />
      </div>
      <TimelineKey model={model} hatchId={`${hatchId}-key`} gateQuestions={gateQuestions ?? null} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

const CHAR_W = 7.9; // a 13px IBM Plex Mono glyph, in viewBox units
const ROW_H = 34;
const MAX_ROWS = 4;

interface Label {
  index: number;
  lines: string[];
  /** The point it names, and where the label itself sits (clamped inside the frame). */
  x: number;
  at: number;
  row: number;
}

/** A label centre kept inside the frame, so an edge label is moved in rather than cut off. */
export function clampCentre(x: number, width: number, min: number, max: number): number {
  return Math.min(Math.max(x, min + width / 2), max - width / 2);
}

/** The two lines under a point: its version, then its model (single session) and accuracy. */
export function labelLines(p: TestPass): string[] {
  const head = p.agent_name || versionLabel(p.version) || p.version;
  const pct = formatPercent(p.accuracy);
  return [head, p.runtime !== 'pipeline' && p.model ? `${shortModel(p.model)} · ${pct}` : pct];
}

/** Which points the compact chart labels: the ones that moved or challenged the champion, and the last. */
function labelled(p: TestPass, i: number, n: number, compact: boolean): boolean {
  if (!compact) return true;
  return (
    i === n - 1 ||
    p.champion_run ||
    ['promoted', 'beats', 'matches', 'passed', 'falls_short'].includes(p.outcome)
  );
}

/**
 * Greedy row assignment: each label takes the first row where it clears the previous label in that
 * row by `gap`; past `MAX_ROWS` it takes the row with the most room. Exported for the test.
 */
export function assignRows(items: { x: number; width: number }[], gap = 8): number[] {
  const rightEdge: number[] = [];
  return items.map(({ x, width }) => {
    const left = x - width / 2;
    let row = rightEdge.findIndex((edge) => edge + gap <= left);
    if (row === -1) {
      if (rightEdge.length < MAX_ROWS) row = rightEdge.length;
      else row = rightEdge.indexOf(Math.min(...rightEdge));
    }
    rightEdge[row] = x + width / 2;
    return row;
  });
}

function TimelineSvg({ model, compact, hatchId }: { model: TimelineModel; compact: boolean; hatchId: string }) {
  const { points, bands, line } = model;
  const n = points.length;
  const w = compact ? 680 : 980;
  const left = 54;
  const right = compact ? 18 : 184;
  const top = 44;
  const plotH = compact ? 190 : 250;
  // Wide enough that the gap's "39d" clears the day labels either side of it.
  const breakW = compact ? 44 : 36;
  const nBreaks = bands.filter((b) => b.breakBefore).length;
  const plotW = w - left - right;
  const slot = (plotW - nBreaks * breakW) / n;
  const breaksBefore = (i: number) => bands.filter((b) => b.breakBefore && b.from <= i).length;
  const xOf = (i: number) => left + breaksBefore(i) * breakW + slot * (i + 0.5);

  const values = [...points.map((p) => p.accuracy as number), ...line.map((s) => s.accuracy)];
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.02) * 20) / 20);
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.015) * 20) / 20);
  const yOf = (v: number) => top + ((hi - v) / (hi - lo || 1)) * plotH;
  const plotBottom = top + plotH;
  const ticks: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += 0.05) ticks.push(Math.round(v * 100) / 100);

  const labels: Label[] = [];
  const toLabel = points
    .map((p, i) => ({ p, i }))
    .filter(({ p, i }) => labelled(p, i, n, compact))
    .map(({ p, i }) => {
      const width = Math.max(...labelLines(p).map((l) => l.length)) * CHAR_W + 4;
      return { p, i, width, at: clampCentre(xOf(i), width, 4, w - 4) };
    });
  const rows = assignRows(toLabel.map(({ at, width }) => ({ x: at, width })));
  toLabel.forEach(({ p, i, at }, k) => labels.push({ index: i, lines: labelLines(p), x: xOf(i), at, row: rows[k] }));
  const nRows = Math.max(1, ...rows.map((r) => r + 1));
  const labelTop = plotBottom + 28;
  const h = labelTop + nRows * ROW_H + 6;

  // The champion line as one polyline: level across each reign, rising at the pass that won.
  const linePts: string[] = [];
  line.forEach((s, k) => {
    const xs = k === 0 ? xOf(s.from) - slot / 2 : xOf(s.from);
    const xe = k === line.length - 1 ? xOf(n - 1) + slot / 2 : xOf(line[k + 1].from);
    linePts.push(`${xs},${yOf(s.accuracy)}`, `${xe},${yOf(s.accuracy)}`);
  });
  const current = line[line.length - 1];

  const description = points
    .map(
      (p) =>
        `${agentLabel(p)} ${formatPercent(p.accuracy)} — ${OUTCOMES[p.outcome].word}${
          p.vs?.delta_pp != null ? `, ${formatPp(p.vs.delta_pp)} vs ${versionLabel(p.vs.version)}` : ''
        }`,
    )
    .join('; ');

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="dia tl-svg"
      // Never drawn narrower than its viewBox: on a phone the figure scrolls in its own wrapper
      // rather than shrinking 13px labels under the house's 12.8px floor.
      style={{ minWidth: w }}
      role="img"
      aria-label={`Test accuracy of ${n} complete passes in scoring order, with the champion line`}
    >
      <title>Every complete test pass on the gate split, in scoring order</title>
      <desc>{description}</desc>
      <defs>
        <pattern id={hatchId} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line className="hatch" x1="0" y1="0" x2="0" y2="5" />
        </pattern>
      </defs>

      {/* day bands, alternating, with the day over each and a break mark between distant days */}
      {bands.map((b, k) => {
        const x1 = xOf(b.from) - slot / 2;
        const x2 = xOf(b.to) + slot / 2;
        return (
          <g key={`${b.date}-${b.from}`} data-band={b.date ?? ''}>
            <rect className={`band${k % 2 ? ' alt' : ''}`} x={x1} y={top - 6} width={x2 - x1} height={plotH + 6} />
            <text className="tx k" x={(x1 + x2) / 2} y={top - 14} textAnchor="middle">
              {formatDay(b.date, false)}
            </text>
            {b.breakBefore && (
              <g data-break={b.gapDays ?? ''}>
                <line className="brk" x1={x1 - breakW / 2} y1={top - 6} x2={x1 - breakW / 2} y2={plotBottom} />
                <path
                  className="brk-z"
                  d={`M${x1 - breakW / 2 - 6},${plotBottom + 6} l6,-8 l0,8 l6,-8`}
                />
                <text className="tx s" x={x1 - breakW / 2} y={top - 14} textAnchor="middle">
                  {b.gapDays != null ? `${b.gapDays}d` : '…'}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {ticks.map((v) => (
        <g key={v}>
          <line className="gl" x1={left} y1={yOf(v)} x2={xOf(n - 1) + slot / 2} y2={yOf(v)} />
          <text className="tx k" x={left - 8} y={yOf(v) + 4} textAnchor="end">
            {Math.round(v * 100)}%
          </text>
        </g>
      ))}
      <line className="ax" x1={left} y1={plotBottom} x2={xOf(n - 1) + slot / 2} y2={plotBottom} />

      <polyline className="champ" data-testid="champion-line" points={linePts.join(' ')} />
      {!compact && current && (
        <g className="champ-label">
          <text className="tx hi" x={xOf(n - 1) + slot / 2 + 10} y={yOf(current.accuracy) - 18}>
            champion {formatPercent(current.accuracy)}
          </text>
          <text className="tx" x={xOf(n - 1) + slot / 2 + 10} y={yOf(current.accuracy) - 2}>
            {versionLabel(current.step.version)}
            {current.step.model ? ` · ${shortModel(current.step.model)}` : ''}
          </text>
          {current.step.effort && (
            <text className="tx s" x={xOf(n - 1) + slot / 2 + 10} y={yOf(current.accuracy) + 14}>
              effort {current.step.effort}
            </text>
          )}
        </g>
      )}

      {labels.map((l) => {
        const y = labelTop + l.row * ROW_H;
        return (
          <g key={`label-${l.index}`} className="tl-label">
            <line className="lead" x1={l.x} y1={plotBottom + 3} x2={l.at} y2={y - 12} />
            <text className="tx k" x={l.at} y={y} textAnchor="middle">
              {l.lines[0]}
            </text>
            <text className="tx s" x={l.at} y={y + 15} textAnchor="middle">
              {l.lines[1]}
            </text>
          </g>
        );
      })}

      {points.map((p, i) => {
        const mark = markOf(p.outcome);
        const x = xOf(i);
        const y = yOf(p.accuracy as number);
        const vs = p.vs;
        const tip = `${agentLabel(p)} — ${formatPercent(p.accuracy)} on ${p.n_scored ?? p.n_questions ?? 'the'} questions · ${
          OUTCOMES[p.outcome].word
        }${
          vs?.delta_pp != null
            ? ` · ${formatPp(vs.delta_pp)} vs ${versionLabel(vs.version)} (${vs.paired ? `paired, p ${formatP(vs.p_value)}` : 'unpaired'})`
            : ''
        }${p.label ? ` · ${p.label}` : ''} · ${p.date ?? ''}`;
        return (
          <g
            key={p.run_name}
            className="pt"
            data-testid="timeline-point"
            data-outcome={p.outcome}
            data-run={p.run_name}
          >
            <title>{tip}</title>
            <Mark shape={mark.shape} cls={mark.cls} x={x} y={y} hatchId={hatchId} />
          </g>
        );
      })}
    </svg>
  );
}

function Mark({ shape, cls, x, y, hatchId }: { shape: MarkShape; cls: string; x: number; y: number; hatchId: string }) {
  switch (shape) {
    case 'dot':
      return <circle className={`mark ${cls}`} cx={x} cy={y} r={6.5} />;
    case 'ring':
      return <circle className={`mark ${cls}`} cx={x} cy={y} r={5.5} />;
    case 'square':
      return <rect className={`mark ${cls}`} x={x - 5.5} y={y - 5.5} width={11} height={11} />;
    case 'diamond':
      return (
        <polygon
          className={`mark ${cls}`}
          points={`${x},${y - 8} ${x + 8},${y} ${x},${y + 8} ${x - 8},${y}`}
          // CSS outranks a presentation attribute, so the per-render pattern id goes inline.
          style={{ fill: `url(#${hatchId})` }}
        />
      );
  }
}

// ---------------------------------------------------------------------------
// The key: every mark in words, and what was left off
// ---------------------------------------------------------------------------

const KEY_ORDER = ['champion', 'promoted', 'beats', 'matches', 'passed', 'rejected', 'falls_short', 'baseline', 'scored'] as const;

function TimelineKey({ model, hatchId, gateQuestions }: { model: TimelineModel; hatchId: string; gateQuestions: number | null }) {
  const present = new Set(model.points.map((p) => p.outcome));
  const lo = model.points.reduce((m, p) => Math.min(m, p.accuracy as number), 1);
  return (
    <div className="tl-key">
      <ul>
        <li>
          <svg viewBox="0 0 24 14" className="glyph" aria-hidden>
            <polyline className="champ" points="1,11 11,11 11,4 23,4" />
          </svg>
          <span>champion of the day — moves only on a promotion</span>
        </li>
        {KEY_ORDER.filter((o) => present.has(o)).map((o) => {
          const mark = markOf(o);
          return (
            <li key={o} data-key={o}>
              <svg viewBox="0 0 20 20" className="glyph" aria-hidden>
                <defs>
                  <pattern id={`${hatchId}-${o}`} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                    <line className="hatch" x1="0" y1="0" x2="0" y2="5" />
                  </pattern>
                </defs>
                <Mark shape={mark.shape} cls={mark.cls} x={10} y={10} hatchId={`${hatchId}-${o}`} />
              </svg>
              <span>
                {OUTCOMES[o].word} <span className="sub">· {OUTCOMES[o].meaning}</span>
              </span>
            </li>
          );
        })}
      </ul>
      <p className="small muted">
        Test accuracy on the {gateQuestions ?? 'fixed'}-question gate split, one point per complete
        pass in the order it was scored; shaded bands are the day each ran, and a break mark with its
        gap in days sits between days that are not consecutive. The axis starts near the lowest pass (
        {formatPercent(lo)}), not at zero: a point’s position carries its value.
        {model.excluded.length > 0 && (
          <>
            {' '}
            <span data-testid="timeline-excluded">
              Left off the axis: {model.excluded.length} incomplete pass
              {model.excluded.length === 1 ? '' : 'es'} (
              {model.excluded.map((p) => `${agentLabel(p)}, ${formatPercent(p.accuracy)}`).join('; ')}) — turns
              unscored, so it measured a subset of the split. It stays in the ledger with its reason.
            </span>
          </>
        )}
      </p>
    </div>
  );
}
