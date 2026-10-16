import { useId } from 'react';
import './optimise.css';
import { EmptyState } from './ui';
import { versionLabel } from './lib';
import { ARMS, PAPER_HUMAN, agentShortName, formatDay, outcomeWord } from './runtimeStory';
import type { ProgressionPoint } from './runtimeStory';
import { NO_VALUE, formatPercent } from '../landing/format';

/**
 * One or two words per stage, on both widths.
 *
 * The full chart used to print the whole stage name — `multi-agent, optimised`
 * — and at five columns those labels touched their neighbours. They also
 * repeated what the bracket underneath already says: which runtime each pair
 * belongs to. The runtime is said once, under the bracket; the column says only
 * what happened at that stage.
 */
const STAGE_LABEL: Record<ProgressionPoint['key'], string> = {
  pipeline_raw: 'raw',
  pipeline_optimised: 'after the loop',
  sdk_distilled: 'distilled',
  sdk_optimised: 'loop attempt',
  sdk_model_swap: 'model swap',
  challenger: 'challenger',
};

/**
 * The full chart's lines under one column, top to bottom: what happened, the
 * version, the agent, its effort, the day and the verdict. Each is short by
 * construction — a version head, a model slug, one effort word, a date, a
 * status word — so no line is wider than `SLOT_CHARS`, which is what keeps
 * neighbouring columns from touching however many challengers the record
 * holds: the chart widens, the labels never do.
 */
export function columnLines(p: ProgressionPoint): string[] {
  const absent = !p.present || p.accuracy === null;
  return [
    p.key === 'challenger' ? p.stage : STAGE_LABEL[p.key],
    p.version ? versionLabel(p.version).split(' · ')[0] : 'no version',
    absent && !p.model ? NO_VALUE : agentShortName(p),
    p.runtime === 'pipeline' ? NO_VALUE : (p.effort ?? NO_VALUE),
    formatDay(p.date),
    absent ? '○ not yet run' : outcomeWord(p),
  ];
}

/** The widest a column label may be, in characters of 13px type, at `SLOT` units a column. */
export const SLOT_CHARS = 15;
const SLOT = 112;

/**
 * Accuracy across the stages of the story, as bars on a zero baseline.
 *
 * Hand-drawn SVG on the kit's `.dia` vocabulary (bar colours per stage live in
 * `optimise.css`: raw `--line-2`, pipeline after the loop `--ink-2`, distilled
 * `--accent`, the loop attempt `--line-3`, the model swap `--ok`; every column is
 * named in words under it). Three properties it holds on purpose:
 *
 *  - Bars start at zero. The interesting range is 77–91%, and a chart cropped to
 *    it would make a nine-point difference look like a fourfold one.
 *  - An absent stage is a dashed empty slot labelled "not yet run", never a bar
 *    of no height, which would read as a measured zero.
 *  - Two reference lines: the paper's human-expert *execution* accuracy (amber),
 *    marked as a published figure, and the incumbent pipeline champion, marked as
 *    measured here. The paper's program-accuracy figure is deliberately not
 *    drawn — it is a different quantity from the one on this axis, and a second
 *    horizontal line in the same units would invite exactly the comparison it
 *    does not support. It is quoted in the caveats instead.
 */
interface RefLine {
  key: 'human' | 'incumbent';
  /** A ratio in 0..1, never null — an absent reference is simply not pushed. */
  value: number;
  label: string;
  sub: string;
}

export function ProgressionChart({
  points,
  pipelineBaseline,
  compact = false,
  className,
}: {
  points: ProgressionPoint[];
  pipelineBaseline: number | null;
  /**
   * The landing's version: a narrower viewBox with the reference lines
   * labelled in the caption instead of beside the plot, so the type stays
   * legible when the figure is drawn at half the width of the Runtimes page.
   */
  compact?: boolean;
  className?: string;
}) {
  const hatchId = `hatch-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const drawable = points.filter((p) => p.present && p.accuracy !== null);
  if (!drawable.length) {
    return <EmptyState>not yet run — no stage of the progression has been scored</EmptyState>;
  }

  const left = 52;
  // Wide enough for the longest reference caption at 13px.
  const right = compact ? 20 : 200;
  // The full chart grows a column per agent rather than squeezing them: its
  // labels are fixed-width by construction (`columnLines`), so a narrower slot
  // is the only way two could touch.
  const w = compact ? 600 : Math.max(940, left + right + SLOT * points.length);
  const top = 26;
  const bottom = compact ? 68 : 150;
  const h = compact ? 304 : 404;
  const plotW = w - left - right;
  const plotH = h - top - bottom;
  const slot = plotW / points.length;
  const barW = Math.min(74, slot * 0.52);
  const yOf = (v: number) => top + (1 - v) * plotH;
  const xOf = (i: number) => left + slot * (i + 0.5);

  // Built by pushing rather than filtering: a reference line with no value is
  // not a line at zero, it is no line, and the two are one `if` apart.
  const references: RefLine[] = [];
  if (typeof PAPER_HUMAN.exe === 'number' && Number.isFinite(PAPER_HUMAN.exe)) {
    references.push({
      key: 'human',
      value: PAPER_HUMAN.exe,
      label: `human expert ${formatPercent(PAPER_HUMAN.exe)}`,
      sub: 'published, not measured here',
    });
  }
  if (typeof pipelineBaseline === 'number' && Number.isFinite(pipelineBaseline)) {
    references.push({
      key: 'incumbent',
      value: pipelineBaseline,
      label: `pipeline champion ${formatPercent(pipelineBaseline)}`,
      sub: 'measured on this gate split',
    });
  }
  const pipelineIdx = points.flatMap((p, i) => (p.runtime === 'pipeline' ? [i] : []));
  const sessionIdx = points.flatMap((p, i) => (p.runtime === 'agent_sdk' ? [i] : []));
  const groups = [
    { label: ARMS.pipeline.title, idx: pipelineIdx },
    { label: ARMS.agent_sdk.title, idx: sessionIdx },
  ].filter((g) => g.idx.length);

  return (
    <figure className="prog">
      <svg
        data-testid="progression-chart"
        data-compact={compact ? 'true' : 'false'}
        viewBox={`0 0 ${w} ${h}`}
        className={`dia${compact ? '' : ' prog-wide'}${className ? ` ${className}` : ''}`}
        role="img"
        aria-labelledby="progression-title progression-desc"
      >
        <title id="progression-title">
          {`Execution accuracy on the gate split across ${points.length} stages, from the raw four-agent pipeline to the single-session agent and every challenger agent scored against it.`}
        </title>
        <desc id="progression-desc">
          {drawable
            .map(
              (p) =>
                `${p.stage}${p.version ? ` (${versionLabel(p.version)}` : ''}${
                  p.model && p.runtime === 'agent_sdk'
                    ? ` on ${p.model}${p.effort ? ` · ${p.effort}` : ''}`
                    : ''
                }${p.version ? ')' : ''} ${formatPercent(p.accuracy)}${
                  p.date ? `, ${formatDay(p.date)}` : ''
                } — ${outcomeWord(p).slice(2)}`,
            )
            .join('; ')}
          . Reference lines: {references.map((r) => `${r.label} — ${r.sub}`).join('; ')}.
          {points.some((p) => !p.present)
            ? ` Not yet run: ${points
                .filter((p) => !p.present)
                .map((p) => p.stage)
                .join(', ')}.`
            : ''}
        </desc>
        <defs>
          <pattern
            id={hatchId}
            width="6"
            height="6"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line className="hatch" x1="0" y1="0" x2="0" y2="6" />
          </pattern>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line className="gl" x1={left} y1={yOf(v)} x2={w - right} y2={yOf(v)} />
            <text className="tx k" x={left - 8} y={yOf(v) + 4} textAnchor="end">
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* Two reference lines can sit within a couple of points of each other
            (human 89% over champion 82%), which is closer than two lines of
            label. The highest line labels itself above the line, every other
            one below, so the captions never cross. */}
        {references.map((r) => {
          const highest = references.every((o) => o.value <= r.value);
          const y = yOf(r.value);
          const labelY = highest ? y - 20 : y + 16;
          const subY = highest ? y - 5 : y + 31;
          return (
            <g key={r.key} data-reference={r.key} className={`s-${r.key}`}>
              <line className="ref" x1={left} y1={y} x2={w - right + 6} y2={y} />
              {!compact && (
                <>
                  <text className="tx ref-t" x={w - right + 12} y={labelY}>
                    {r.label}
                  </text>
                  <text className="tx s" x={w - right + 12} y={subY}>
                    {r.sub}
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* The stepped line over the bar tops: the progression as one movement,
            broken wherever a stage is absent rather than bridged across it. */}
        {(() => {
          const runs: string[][] = [];
          let current: string[] = [];
          points.forEach((p, i) => {
            if (!p.present || p.accuracy === null) {
              if (current.length > 1) runs.push(current);
              current = [];
              return;
            }
            const y = yOf(p.accuracy);
            current.push(`${xOf(i) - barW / 2},${y}`, `${xOf(i) + barW / 2},${y}`);
            if (i < points.length - 1) {
              const next = points[i + 1];
              if (next.present && next.accuracy !== null) {
                current.push(`${xOf(i + 1) - barW / 2},${y}`);
              }
            }
          });
          if (current.length > 1) runs.push(current);
          return runs.map((pts, i) => <polyline key={i} className="step" points={pts.join(' ')} />);
        })()}

        {points.map((p, i) => {
          const x = xOf(i) - barW / 2;
          const absent = !p.present || p.accuracy === null;
          const failed = p.tone === 'fail';
          return (
            <g
              key={p.id}
              data-stage={p.key}
              data-present={absent ? 'false' : 'true'}
              data-tone={absent ? 'absent' : p.tone}
            >
              {absent ? (
                <>
                  <rect className="slot" x={x} y={top} width={barW} height={plotH} rx={2} />
                  <text className="tx s" x={xOf(i)} y={top + plotH / 2} textAnchor="middle">
                    not yet run
                  </text>
                </>
              ) : (
                <>
                  <rect
                    className={`bar ${p.key} t-${p.tone}`}
                    x={x}
                    y={yOf(p.accuracy as number)}
                    width={barW}
                    height={plotH - (yOf(p.accuracy as number) - top)}
                    // CSS outranks a presentation attribute, so the per-render pattern id goes inline.
                    style={failed ? { fill: `url(#${hatchId})` } : undefined}
                    rx={2}
                  />
                  <text
                    className="tx num"
                    x={xOf(i)}
                    y={yOf(p.accuracy as number) - 8}
                    textAnchor="middle"
                  >
                    {formatPercent(p.accuracy)}
                  </text>
                </>
              )}
              {compact ? (
                // Three short lines per column: the runtime, the stage, the
                // version. Five 105px slots cannot hold "multi-agent, optimised".
                <>
                  <text className="tx s" x={xOf(i)} y={h - bottom + 18} textAnchor="middle">
                    {p.runtime === 'pipeline' ? 'pipeline' : 'session'}
                  </text>
                  <text className="tx" x={xOf(i)} y={h - bottom + 34} textAnchor="middle">
                    {STAGE_LABEL[p.key]}
                  </text>
                  {/*
                    Head only — `sdk-v2 · rejected` and `sdk-v1 · haiku-4-5`
                    are wider than a 120-unit slot and ran into each other. The
                    hatched bar and the caption below already say "rejected",
                    and the stage line above already says "model swap"; the two
                    columns both reading `sdk-v1` is the point of that stage.
                  */}
                  <text className="tx k" x={xOf(i)} y={h - bottom + 50} textAnchor="middle">
                    {p.version ? versionLabel(p.version).split(' · ')[0] : 'no version'}
                  </text>
                </>
              ) : (
                columnLines(p).map((line, k) => (
                  <text
                    key={k}
                    className={`tx${k === 0 ? '' : k === 5 ? ` st t-${absent ? 'absent' : p.tone}` : ' k'}`}
                    x={xOf(i)}
                    y={h - bottom + 20 + k * 16}
                    textAnchor="middle"
                  >
                    {line}
                  </text>
                ))
              )}
            </g>
          );
        })}

        {/* Which runtime each run of columns belongs to, bracketed under the labels. */}
        {!compact &&
          groups.map((group) => {
            const x1 = left + slot * Math.min(...group.idx) + 6;
            const x2 = left + slot * (Math.max(...group.idx) + 1) - 6;
            const y = h - bottom + 118;
            return (
              <g key={group.label}>
                <line className="ax" x1={x1} y1={y} x2={x2} y2={y} />
                <text className="tx s" x={(x1 + x2) / 2} y={y + 18} textAnchor="middle">
                  {group.label}
                </text>
              </g>
            );
          })}
      </svg>
      {!compact && (
        // Under 760px the columns would shrink their 13px labels to a few
        // pixels, so the same points are listed as rows instead (CSS swaps the
        // two): one agent per row, every word of its column on one line.
        <>
          {references.length > 0 && (
            // The rows draw no reference lines, so the two figures are said in words.
            <p className="prog-rows-ref">
              {references.map((r) => `${r.label} (${r.sub})`).join(' · ')}
            </p>
          )}
          <ol className="prog-rows" data-testid="progression-rows">
            {points.map((p) => {
              const absent = !p.present || p.accuracy === null;
              const [what, version, agent, effort, day] = columnLines(p);
              return (
                <li key={p.id} data-tone={absent ? 'absent' : p.tone}>
                  <span className="prog-row-head">
                    <span className="mono">{version}</span> · {what}
                  </span>
                  <span className="prog-row-sub mono">
                    {p.runtime === 'pipeline' ? agent : `${agent} · ${effort}`} · {day}
                  </span>
                  <span className="prog-row-bar">
                    <span className="track">
                      {!absent && (
                        <i
                          className={`t-${p.tone}`}
                          style={{ width: `${(p.accuracy as number) * 100}%` }}
                        />
                      )}
                    </span>
                    <span className="mono">
                      {absent ? 'not yet run' : formatPercent(p.accuracy)}
                    </span>
                  </span>
                  <span className={`prog-row-st st t-${absent ? 'absent' : p.tone}`}>
                    {absent ? '○ not yet run' : outcomeWord(p)}
                  </span>
                </li>
              );
            })}
          </ol>
        </>
      )}
      <figcaption>
        {compact && references.length > 0 && (
          <span className="opt-key">
            {references.map((r) => (
              <span key={r.key} className={`s-${r.key} dash`}>
                {r.label} <span className="sub">· {r.sub}</span>
              </span>
            ))}
          </span>
        )}
        Execution accuracy on the fixed gate split, on a zero baseline. The dashed amber line is the
        human-expert execution accuracy reported by {PAPER_HUMAN.citation} on a{' '}
        {PAPER_HUMAN.evaluatedOn || 'sample of the paper’s test set'} — a published figure about a
        different question set, not a measurement of this system. The dashed grey line is the
        incumbent pipeline champion measured here. The accent bar is the agent serving today; a
        green bar passed its gate; a red hatched bar was rejected or fell short of the champion — it
        is what was tried, not what the runtime does.
        {!compact &&
          ' Effort applies to the Claude agents only, so the pipeline’s columns print — for it. Each date is the day the pass’s own run name stamps.'}
      </figcaption>
    </figure>
  );
}
