import { useMode } from '../modeStore';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/**
 * Live vs demo, told as a shape before it is told as a colour — and now as a
 * sentence before either.
 *
 * A solid ring means this deployment holds a key and answers with the model.
 * A dashed amber ring means it holds no key at all and chat is replayed from
 * recordings. Dashed-vs-solid survives colour-blindness, greyscale printing
 * and a screenshot pasted into a doc — which matters, because "was this figure
 * live or replayed?" is the single question this app must never let a reader
 * get wrong. The words say it outright: `dev · live model` or
 * `demo · recorded replay`, at a size a visitor can read.
 */
export function ModeLamp() {
  const health = useMode((s) => s.health);
  const loading = useMode((s) => s.loading);

  if (!health) {
    return (
      <span
        data-testid="mode-lamp"
        data-mode={loading ? 'loading' : 'unknown'}
        className="type-meta inline-flex items-center gap-2 text-faint"
      >
        <span className="size-2.5 rounded-full border border-dashed border-line-2" />
        {loading ? 'checking…' : 'backend offline'}
      </span>
    );
  }

  const isDemo = health.mode === 'demo';

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          data-testid="mode-lamp"
          data-mode={health.mode}
          className={
            isDemo
              ? 'type-meta inline-flex cursor-default items-center gap-2 rounded-full border border-dashed border-amber-line px-2 py-0.5 text-amber'
              : 'type-meta inline-flex cursor-default items-center gap-2 rounded-full border border-good-line px-2 py-0.5 text-good'
          }
        >
          <span
            aria-hidden
            className={
              isDemo
                ? 'size-2.5 rounded-full border border-dashed border-amber'
                : 'size-2.5 rounded-full border border-good bg-good shadow-[0_0_6px_var(--good-glow)]'
            }
          />
          <span className="type-num">{isDemo ? 'demo · recorded replay' : 'dev · live model'}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-xs">
        {isDemo
          ? 'This deployment holds no API key. Chat replays conversations recorded in development, through the same events a live turn emits. Everything else reads the same committed artifacts the dev app does.'
          : `Live against ${health.runtime === 'agent_sdk' ? 'the Claude Agent SDK' : health.bundle.lm_mini}. Turns are answered by the model, not replayed.`}
      </TooltipContent>
    </Tooltip>
  );
}
