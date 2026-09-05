import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import type { BoardData } from './useBoardData';

type LampTone = 'good' | 'amber' | 'bad' | 'info' | 'idle';

const DOT_TONE: Record<LampTone, string> = {
  good: 'border-good bg-good shadow-[0_0_6px_var(--good-glow)]',
  amber: 'border-amber',
  bad: 'border-bad bg-bad',
  info: 'border-info bg-info',
  idle: 'border-line-2',
};

const TEXT_TONE: Record<LampTone, string> = {
  good: 'text-good',
  amber: 'text-amber',
  bad: 'text-bad',
  info: 'text-info',
  idle: 'text-faint',
};

interface LampProps {
  label: string;
  value: string;
  tone: LampTone;
  /**
   * Dashed means replayed or unverified, solid means measured. The shape, not
   * the colour, is what a colour-blind reader and a greyscale screenshot are
   * left with — so no lamp may distinguish itself by colour alone.
   */
  dashed?: boolean;
  tooltip: ReactNode;
  to?: string;
}

function Lamp({ label, value, tone, dashed = false, tooltip, to }: LampProps) {
  const body = (
    <span
      data-testid={`lamp-${label}`}
      data-tone={tone}
      data-shape={dashed ? 'dashed' : 'solid'}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-line bg-panel py-1 pl-2 pr-2.5',
        to && 'transition-colors hover:border-amber-line hover:bg-panel-2',
      )}
    >
      <span
        aria-hidden
        className={cn('size-2 shrink-0 rounded-full border', DOT_TONE[tone], dashed && 'border-dashed bg-transparent shadow-none')}
      />
      <span className="mono-caps text-faint">{label}</span>
      <span className={cn('type-num text-[11px]', TEXT_TONE[tone])}>{value}</span>
    </span>
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {to ? (
          <Link to={to} className="cursor-pointer">
            {body}
          </Link>
        ) : (
          <span className="cursor-default">{body}</span>
        )}
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-xs">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * Mode, champion, gate — the three facts that decide how to read every other
 * number on the board.
 *
 * Mode is first and is the only lamp whose *shape* changes: a solid green ring
 * means this deployment holds a key and answers with the model; a dashed amber
 * ring means it holds no key at all and chat is replayed from recordings.
 * Everything below the strip has to be read in that light, which is why it
 * sits above the tiles rather than beside them.
 */
export function LampStrip({ board }: { board: BoardData }) {
  const { health, isDemo, champion, campaigns } = board;

  /**
   * The gate lamp reports the campaign's most recent verdict.
   *
   * It used to run the legacy comparator over the 770-question corpus and
   * report *that* rule's answer, which since the campaign protocol is the wrong
   * question asked of the wrong population: it applied a net-positive rule the
   * loop retired, to a corpus the loop does not gate on, about versions the
   * loop rolled back. A lamp that reads "pass" under a rule nothing promotes on
   * is worse than no lamp.
   */
  const gateLamp = (() => {
    const experiments = campaigns?.experiments ?? [];
    const latest = experiments[experiments.length - 1];
    if (!latest) {
      return {
        value: 'no challenger',
        tone: 'idle' as LampTone,
        dashed: true,
        tooltip:
          'No experiment has been gated yet. The gate promotes a challenger only when it is net positive on the shared gate questions AND clears one-sided cluster-corrected McNemar at α = 0.05.',
      };
    }
    const p = latest.cluster_p_one_sided;
    const pText = p == null ? '—' : p.toFixed(3);
    const delta =
      latest.accuracy_delta == null
        ? '—'
        : `${latest.accuracy_delta >= 0 ? '+' : ''}${(latest.accuracy_delta * 100).toFixed(2)}pp`;
    if (latest.promoted) {
      return {
        value: `${latest.label} promoted`,
        tone: 'good' as LampTone,
        dashed: false,
        tooltip: `${latest.baseline_version} → ${latest.candidate_version} by rewriting ${latest.target_agent} alone: ${delta} on the gate split, one-sided clustered McNemar p = ${pText}.`,
      };
    }
    return {
      value: `${latest.label} refused`,
      tone: 'bad' as LampTone,
      dashed: false,
      tooltip: (
        <>
          <div className="mb-1 font-medium">The gate is working, not broken.</div>
          {latest.candidate_version} rewrote {latest.target_agent} and moved the gate split by{' '}
          {delta} — {latest.fixed ?? 0} questions fixed against {latest.broken ?? 0} broken — but at
          p = {pText} that is not distinguishable from noise at α = 0.05. Promotion needs net
          positive <em>and</em> significance; net positive alone promoted three versions whose
          confidence intervals contained zero.
        </>
      ),
    };
  })();

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Lamp
        label="mode"
        value={isDemo ? 'replay · keyless' : 'live'}
        tone={isDemo ? 'amber' : 'good'}
        dashed={isDemo}
        to="/admin/system"
        tooltip={
          isDemo
            ? 'This deployment holds no API key. Chat replays conversations recorded in development through the same events a live turn emits. Nothing on this page was measured against live production traffic.'
            : `Live against ${health?.bundle.lm_mini ?? 'the champion model'}. Turns are answered by the model, not replayed.`
        }
      />
      <Lamp
        label="champion"
        value={champion ?? campaigns?.champion ?? 'unset'}
        tone="info"
        to="/admin/experiments"
        tooltip={
          health
            ? `Bundle ${health.bundle_id} · prompts ${health.bundle.prompts_version} · ${health.bundle.lm_mini} · dataset ${health.bundle.dataset_hash} · code ${health.bundle.code_sha}.`
            : 'The version currently serving.'
        }
      />
      <Lamp
        label="gate"
        value={gateLamp.value}
        tone={gateLamp.tone}
        dashed={gateLamp.dashed}
        to="/admin/campaigns"
        tooltip={gateLamp.tooltip}
      />
    </div>
  );
}
