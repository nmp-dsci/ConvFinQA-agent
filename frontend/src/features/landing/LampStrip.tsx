import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { formatPercent } from './format';
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
  const { health, isDemo, champion, championVersion, gate, gateCandidate } = board;

  const gateLamp = (() => {
    if (!gateCandidate || !gate) {
      return {
        value: gateCandidate ? 'checking…' : 'no challenger',
        tone: 'idle' as LampTone,
        dashed: true,
        tooltip:
          'The promotion gate compares the newest version against the champion and refuses anything that loses accuracy or flips a passing question to failing. Nothing is waiting on it right now.',
      };
    }
    if (gate.promotable) {
      return {
        value: `${gateCandidate} pass`,
        tone: 'good' as LampTone,
        dashed: false,
        tooltip: `${gateCandidate} clears the promotion contract against ${gate.baseline_version}: accuracy ${formatPercent(gate.baseline_accuracy)} → ${formatPercent(gate.candidate_accuracy)} with no pass→fail flips.`,
      };
    }
    return {
      value: `${gateCandidate} refused`,
      tone: 'bad' as LampTone,
      dashed: false,
      tooltip: (
        <>
          <div className="mb-1 font-medium">The gate is working, not broken.</div>
          {gate.reason} · {gate.regressions.length} pass→fail flips against{' '}
          {gate.baseline_version}. Promotion needs accuracy ≥ champion <em>and</em> no flips;
          beating the average alone would let "fixed numbers, broke programs" through.
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
        value={champion ?? championVersion?.version ?? 'unset'}
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
        to="/admin/experiments"
        tooltip={gateLamp.tooltip}
      />
    </div>
  );
}
