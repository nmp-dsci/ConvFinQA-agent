import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { cn } from '@/lib/utils';
import { getCampaigns } from './api';
import type { CampaignExperiment, ChampionPoint } from './api';
import { AdminPage, Caveat, EmptyState, ErrorNote, LoadingRows, Panel, Verdict } from './ui';

/**
 * Campaigns: the optimisation loop as a series of experiments, and what moved.
 *
 * The Experiments tab beside this one lists *runs* — everything the tracking
 * store recorded, in time order. This page is the other view of the same data:
 * experiments grouped into the campaigns they belong to, each one a single
 * subagent's prompt rewritten and gated. That grouping is what makes the chart
 * legible, because every step on it has exactly one named cause.
 *
 * Both this page and the published write-up read `evaluation/story.json`, so
 * they cannot disagree. Rebuild both with `convfinqa-evalloop story`.
 */

const AGENTS = ['triage', 'preprocess', 'retriever', 'calculator'] as const;
type Agent = (typeof AGENTS)[number];

const AGENT_COLOR: Record<Agent, string> = {
  triage: '#7aa2f7',
  preprocess: '#bb9af7',
  retriever: '#e0af68',
  calculator: '#73daca',
};

function pct(value: number | null | undefined, digits = 1) {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`;
}

function pp(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : `${(value * 100 >= 0 ? '+' : '')}${(value * 100).toFixed(2)}pp`;
}

/**
 * Overall accuracy plus each subagent's own gold-derived metric, at every point
 * the champion moved. Hand-drawn SVG rather than a charting dependency: five
 * series over a handful of points does not justify one, and this way the shape
 * is the same in the app and on the published page.
 */
function ChampionChart({ track }: { track: ChampionPoint[] }) {
  const points = track.filter((p) => p.accuracy !== null && p.accuracy !== undefined);
  if (points.length < 2) return null;

  const w = 900;
  const h = 300;
  const [left, right, top, bottom] = [58, 132, 24, 48];
  const series: Array<{ name: string; colour: string; width: number; values: Array<number | null> }> = [
    { name: 'overall', colour: '#dbe3ee', width: 2.4, values: points.map((p) => p.accuracy ?? null) },
    ...AGENTS.map((a) => ({
      name: a,
      colour: AGENT_COLOR[a],
      width: 1.4,
      values: points.map((p) => p.panel?.[a] ?? null),
    })),
  ];
  const all = series.flatMap((s) => s.values).filter((v): v is number => v !== null);
  const rawLo = Math.min(...all);
  const rawHi = Math.max(...all);
  const pad = Math.max(0.02, (rawHi - rawLo) * 0.25);
  const lo = Math.max(0, rawLo - pad);
  const hi = Math.min(1, rawHi + pad);
  const xOf = (i: number) => left + (i * (w - left - right)) / Math.max(1, points.length - 1);
  const yOf = (v: number) => top + ((hi - v) / (hi - lo)) * (h - top - bottom);

  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Champion accuracy and per-subagent metrics">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const v = lo + f * (hi - lo);
          return (
            <g key={f}>
              <line x1={left} y1={yOf(v)} x2={w - right} y2={yOf(v)} stroke="currentColor" className="text-line" strokeWidth={1} />
              <text x={left - 8} y={yOf(v) + 3} textAnchor="end" className="fill-faint font-mono text-[10px]">
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        {points.map((p, i) => (
          <g key={`${p.version}-${i}`}>
            <text x={xOf(i)} y={h - 26} textAnchor="middle" className="fill-text font-mono text-[10px]">
              {p.version}
            </text>
            {p.target_agent && (
              <text
                x={xOf(i)}
                y={h - 12}
                textAnchor="middle"
                className="font-mono text-[9px]"
                fill={AGENT_COLOR[p.target_agent as Agent] ?? '#6b7a90'}
              >
                ↑ {p.target_agent}
              </text>
            )}
          </g>
        ))}
        {series.map((s) => {
          const pts = s.values
            .map((v, i) => (v === null ? null : `${xOf(i)},${yOf(v)}`))
            .filter((v): v is string => v !== null);
          if (pts.length < 2) return null;
          const lastIdx = s.values.reduce<number>((acc, v, i) => (v !== null ? i : acc), -1);
          return (
            <g key={s.name}>
              <polyline points={pts.join(' ')} fill="none" stroke={s.colour} strokeWidth={s.width} strokeLinejoin="round" />
              {s.values.map((v, i) =>
                v === null ? null : <circle key={i} cx={xOf(i)} cy={yOf(v)} r={3} fill={s.colour} />,
              )}
              {lastIdx >= 0 && (
                <text x={w - right + 8} y={yOf(s.values[lastIdx] ?? 0) + 3} fill={s.colour} className="font-mono text-[10px]">
                  {s.name}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="type-meta mt-2 text-faint">
        Overall gate accuracy in white; each subagent&rsquo;s own gold-derived metric in colour. The arrow
        under a version names the single subagent that experiment rewrote — which is why a step in one
        coloured line can be read as the cause of the step in the white one.
      </figcaption>
    </figure>
  );
}

function ExperimentCard({ exp }: { exp: CampaignExperiment }) {
  const [open, setOpen] = useState(false);
  const p = exp.cluster_p_one_sided;
  return (
    <div className="rounded-[5px] border border-line bg-panel">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full flex-wrap items-center gap-2.5 px-3 py-2.5 text-left"
      >
        <span className="font-mono text-[12px] text-text">{exp.label || exp.candidate_version}</span>
        <Verdict ok={exp.promoted}>{exp.promoted ? 'promoted' : 'rejected'}</Verdict>
        <span
          className="rounded-[4px] border px-1.5 py-0.5 font-mono text-[10px]"
          style={{ color: AGENT_COLOR[exp.target_agent as Agent], borderColor: AGENT_COLOR[exp.target_agent as Agent] }}
        >
          {exp.target_agent}
        </span>
        <span className="ml-auto type-num text-[12px] text-muted">
          {pp(exp.accuracy_delta)} · p={p === null ? '—' : p.toFixed(3)}
        </span>
      </button>
      {open && (
        <div className="border-t border-line px-3 py-3">
          <p className="type-small mb-2 text-muted">
            <span className="font-mono text-text">
              {exp.baseline_version} → {exp.candidate_version}
            </span>{' '}
            — {exp.summary_of_changes || 'no summary recorded'}
          </p>
          {exp.rationale && <p className="type-small mb-3 text-muted">{exp.rationale}</p>}
          <div className="mb-3 grid grid-cols-2 gap-2 md:grid-cols-4">
            {AGENTS.map((a) => (
              <div key={a} className="rounded-[4px] border border-line-2 bg-panel-2 px-2 py-1.5">
                <div className="mono-caps text-faint">{a}</div>
                <div className="type-num text-[12px]">
                  {pct(exp.panel_baseline?.[a])} → {pct(exp.panel_candidate?.[a])}
                </div>
              </div>
            ))}
          </div>
          <div className="type-small text-faint">
            {exp.fixed ?? 0} fixed / {exp.broken ?? 0} broken of {exp.n_compared ?? 0} shared questions ·
            95% CI [{pp(exp.delta_ci_lo)}, {pp(exp.delta_ci_hi)}]
          </div>
          {exp.diff && (
            <pre className="mt-2 max-h-80 overflow-auto rounded-[4px] border border-line bg-bg p-2.5 font-mono text-[11px] leading-relaxed">
              {exp.diff.split('\n').map((line, i) => (
                <div
                  key={i}
                  className={cn(
                    line.startsWith('+') && !line.startsWith('+++') && 'text-good',
                    line.startsWith('-') && !line.startsWith('---') && 'text-bad',
                    line.startsWith('@@') && 'text-violet',
                  )}
                >
                  {line}
                </div>
              ))}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function Campaigns() {
  const [only, setOnly] = useState<string>('');
  const query = useQuery({
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const data = query.data;

  const shown = useMemo(
    () => (data?.experiments ?? []).filter((e) => !only || e.campaign === only),
    [data, only],
  );

  return (
    <AdminPage
      eyebrow="optimisation"
      title="Campaigns"
      sub="Each experiment rewrites exactly one subagent's prompt; a paired significance test on the fixed gate split decides whether it becomes the champion."
      testId="campaigns"
    >
      <Panel
        title="champion track"
        endpoint="/eval/campaigns"
        note={data?.rule || 'the promotion rule'}
        right={<span className="type-small text-faint">champion {data?.champion ?? '—'}</span>}
      >
        {query.isLoading ? (
          <LoadingRows rows={6} />
        ) : query.error ? (
          <ErrorNote error={query.error} />
        ) : !data?.champion_track?.length ? (
          <EmptyState>
            No campaign has been recorded yet. Run <code className="font-mono">convfinqa-evalloop cycle
            --campaign c01</code>, then <code className="font-mono">convfinqa-evalloop story</code>.
          </EmptyState>
        ) : (
          <>
            <ChampionChart track={data.champion_track} />
            <Caveat>
              Only promoted experiments move this line. Rejections are below and are most of the record —
              at this split size a real two- or three-point gain is not distinguishable from noise, so the
              gate refuses it rather than promoting on a hunch.
            </Caveat>
          </>
        )}
      </Panel>

      <Panel
        title="experiments"
        endpoint="/eval/campaigns"
        note="every challenger, promoted or not"
        right={
          <span className="type-small text-faint">
            {shown.length} experiments · {shown.filter((e) => e.promoted).length} promoted
          </span>
        }
      >
        {(data?.campaigns?.length ?? 0) > 1 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {['', ...(data?.campaigns ?? []).map((c) => c.name)].map((name) => (
              <button
                key={name || 'all'}
                type="button"
                onClick={() => setOnly(name)}
                className={cn(
                  'rounded-[4px] border px-2.5 py-1 mono-caps transition-colors',
                  name === only ? 'border-accent text-accent' : 'border-line text-muted hover:border-line-2',
                )}
              >
                {name || 'all'}
              </button>
            ))}
          </div>
        )}
        {shown.length === 0 ? (
          <EmptyState>Nothing gated yet.</EmptyState>
        ) : (
          <div className="flex flex-col gap-2">
            {shown.map((exp) => (
              <ExperimentCard key={`${exp.campaign}-${exp.label}-${exp.candidate_version}`} exp={exp} />
            ))}
          </div>
        )}
      </Panel>
    </AdminPage>
  );
}
