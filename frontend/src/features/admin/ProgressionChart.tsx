import { EmptyState } from './ui';
import { ARMS, PAPER_HUMAN } from './runtimeStory';
import type { ProgressionPoint } from './runtimeStory';
import { formatPercent } from '../landing/format';
import { cn } from '@/lib/utils';

/** One or two words per stage for the compact chart's five narrow columns. */
const COMPACT_STAGE: Record<ProgressionPoint['key'], string> = {
  pipeline_raw: 'raw',
  pipeline_optimised: 'after the loop',
  sdk_distilled: 'distilled',
  sdk_optimised: 'loop attempt',
  sdk_model_swap: 'model swap',
};

const STAGE_FILL: Record<ProgressionPoint['key'], string> = {
  pipeline_raw: 'var(--muted)',
  pipeline_optimised: 'var(--amber)',
  sdk_distilled: 'var(--info)',
  sdk_optimised: 'var(--violet)',
  sdk_model_swap: 'var(--good)',
};

/**
 * Accuracy across the four stages of the story, as bars on a zero baseline.
 *
 * Hand-drawn SVG, like the champion chart on the Campaigns page and for the same
 * reasons: four columns do not justify a charting dependency, and the CSP on the
 * published surfaces allows only a short list of CDNs. Three properties it holds
 * on purpose:
 *
 *  - Bars start at zero. The interesting range is 77–91%, and a chart cropped to
 *    it would make a nine-point difference look like a fourfold one.
 *  - An absent stage is a dashed empty slot labelled "not yet run", never a bar
 *    of no height, which would read as a measured zero.
 *  - Two reference lines: the paper's human-expert *execution* accuracy, marked
 *    as a published figure, and the incumbent pipeline champion, marked as
 *    measured here. The paper's program-accuracy figure is deliberately not
 *    drawn — it is a different quantity from the one on this axis, and a second
 *    horizontal line in the same units would invite exactly the comparison it
 *    does not support. It is quoted in the caveats instead.
 */
interface RefLine {
  key: string;
  /** A ratio in 0..1, never null — an absent reference is simply not pushed. */
  value: number;
  line: string;
  text: string;
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
  const drawable = points.filter((p) => p.present && p.accuracy !== null);
  if (!drawable.length) {
    return <EmptyState>not yet run — no stage of the progression has been scored</EmptyState>;
  }

  const w = compact ? 600 : 940;
  const h = compact ? 300 : 340;
  const left = 52;
  const right = compact ? 20 : 178;
  const top = 24;
  const bottom = compact ? 64 : 78;
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
      line: 'var(--violet)',
      text: 'var(--violet)',
      label: `human expert ${formatPercent(PAPER_HUMAN.exe)}`,
      sub: 'published, not measured here',
    });
  }
  if (typeof pipelineBaseline === 'number' && Number.isFinite(pipelineBaseline)) {
    references.push({
      key: 'incumbent',
      value: pipelineBaseline,
      line: 'var(--amber-line)',
      text: 'var(--amber)',
      label: `pipeline champion ${formatPercent(pipelineBaseline)}`,
      sub: 'measured on this gate split',
    });
  }

  return (
    <figure className="m-0 min-w-0">
      <svg
        data-testid="progression-chart"
        data-compact={compact ? 'true' : 'false'}
        viewBox={`0 0 ${w} ${h}`}
        className={cn('w-full', className)}
        role="img"
        aria-labelledby="progression-title progression-desc"
      >
        <title id="progression-title">
          Accuracy across five stages: multi-agent raw, multi-agent optimised, single-session SDK
          distilled, the SDK optimisation attempt, and the same SDK prompt on a second model.
        </title>
        <desc id="progression-desc">
          {drawable
            .map((p) => `${p.stage}${p.version ? ` (${p.version})` : ''} ${formatPercent(p.accuracy)}`)
            .join('; ')}
          . Reference lines: {references.map((r) => `${r.label} — ${r.sub}`).join('; ')}.
          {points.some((p) => !p.present)
            ? ` Not yet run: ${points
                .filter((p) => !p.present)
                .map((p) => p.stage)
                .join(', ')}.`
            : ''}
        </desc>

        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={left}
              y1={yOf(v)}
              x2={w - right}
              y2={yOf(v)}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text
              x={left - 8}
              y={yOf(v) + 3}
              textAnchor="end"
              fill="var(--faint)"
              className="type-num type-meta"
            >
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
          const labelY = highest ? y - 14 : y + 12;
          const subY = highest ? y - 4 : y + 23;
          return (
            <g key={r.key} data-reference={r.key}>
              <line
                x1={left}
                y1={y}
                x2={w - right + 6}
                y2={y}
                stroke={r.line}
                strokeWidth={1.4}
                strokeDasharray="6 4"
              />
              {!compact && (
                <>
                  <text x={w - right + 12} y={labelY} fill={r.text} className="type-num type-meta">
                    {r.label}
                  </text>
                  <text x={w - right + 12} y={subY} fill="var(--faint)" className="type-num type-meta">
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
          return runs.map((pts, i) => (
            <polyline
              key={i}
              points={pts.join(' ')}
              fill="none"
              stroke="var(--text)"
              strokeOpacity={0.35}
              strokeWidth={1.2}
            />
          ));
        })()}

        {points.map((p, i) => {
          const x = xOf(i) - barW / 2;
          const absent = !p.present || p.accuracy === null;
          const rejected = p.promoted === false;
          return (
            <g key={p.key} data-stage={p.key} data-present={absent ? 'false' : 'true'}>
              {absent ? (
                <>
                  <rect
                    x={x}
                    y={top}
                    width={barW}
                    height={plotH}
                    fill="none"
                    stroke="var(--line-2)"
                    strokeDasharray="4 4"
                    rx={2}
                  />
                  <text
                    x={xOf(i)}
                    y={top + plotH / 2}
                    textAnchor="middle"
                    fill="var(--faint)"
                    className="type-num type-meta"
                  >
                    not yet run
                  </text>
                </>
              ) : (
                <>
                  <rect
                    x={x}
                    y={yOf(p.accuracy as number)}
                    width={barW}
                    height={plotH - (yOf(p.accuracy as number) - top)}
                    fill={STAGE_FILL[p.key]}
                    fillOpacity={rejected ? 0.3 : 0.85}
                    stroke={STAGE_FILL[p.key]}
                    strokeDasharray={rejected ? '4 3' : undefined}
                    rx={2}
                  />
                  <text
                    x={xOf(i)}
                    y={yOf(p.accuracy as number) - 7}
                    textAnchor="middle"
                    fill="var(--text)"
                    className="type-num type-small"
                  >
                    {formatPercent(p.accuracy)}
                  </text>
                </>
              )}
              {compact ? (
                // Three short lines per column: the runtime, the stage, the
                // version. Five 105px slots cannot hold "multi-agent, optimised".
                <>
                  <text x={xOf(i)} y={h - bottom + 16} textAnchor="middle" fill="var(--faint)" className="type-num type-meta">
                    {p.runtime === 'pipeline' ? 'pipeline' : 'session'}
                  </text>
                  <text x={xOf(i)} y={h - bottom + 30} textAnchor="middle" fill="var(--text)" className="type-num type-meta">
                    {COMPACT_STAGE[p.key]}
                  </text>
                  <text x={xOf(i)} y={h - bottom + 44} textAnchor="middle" fill="var(--faint)" className="type-num type-meta">
                    {p.version ?? 'no version'}
                    {rejected ? ' · rejected' : ''}
                  </text>
                </>
              ) : (
                <>
                  <text x={xOf(i)} y={h - bottom + 18} textAnchor="middle" fill="var(--text)" className="type-num type-meta">
                    {p.stage}
                  </text>
                  <text x={xOf(i)} y={h - bottom + 32} textAnchor="middle" fill="var(--faint)" className="type-num type-meta">
                    {p.version ?? 'no version'}
                    {rejected ? ' · rejected' : ''}
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* Which runtime each pair belongs to, bracketed under the labels. */}
        {!compact && [
          { label: ARMS.pipeline.title, from: 0, to: 1 },
          { label: ARMS.agent_sdk.title, from: 2, to: points.length - 1 },
        ].map((group) => {
          const x1 = left + slot * group.from + 6;
          const x2 = left + slot * (group.to + 1) - 6;
          const y = h - bottom + 46;
          return (
            <g key={group.label}>
              <line x1={x1} y1={y} x2={x2} y2={y} stroke="var(--line-2)" strokeWidth={1} />
              <text
                x={(x1 + x2) / 2}
                y={y + 13}
                textAnchor="middle"
                fill="var(--muted)"
                className="type-num type-meta"
              >
                {group.label}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="type-meta mt-2 text-faint">
        {compact &&
          references.map((r) => (
            <span key={r.key} className="mr-3 inline-flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block h-0 w-4 border-t border-dashed"
                style={{ borderColor: r.line }}
              />
              <span style={{ color: r.text }}>{r.label}</span>
              <span>· {r.sub}</span>
            </span>
          ))}
        {compact && <br />}
        Execution accuracy on the fixed gate split, on a zero baseline. The dashed violet line is the
        human-expert execution accuracy reported by {PAPER_HUMAN.citation} on a{' '}
        {PAPER_HUMAN.evaluatedOn || 'sample of the paper’s test set'} — a published figure about a
        different question set, not a measurement of this system. The amber line is the incumbent
        pipeline champion measured here. A hatched bar was gated and rejected: it is what the loop
        tried, not what the runtime does.
      </figcaption>
    </figure>
  );
}

