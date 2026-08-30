import { useMode } from '../modeStore';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

/**
 * Live vs demo, told as a shape before it is told as a colour.
 *
 * A solid ring means this deployment holds a key and answers with the model.
 * A dashed amber ring means it holds no key at all and chat is replayed from
 * recordings. Dashed-vs-solid survives colour-blindness, greyscale printing
 * and a screenshot pasted into a doc — which matters, because "was this figure
 * live or replayed?" is the single question this app must never let a reader
 * get wrong.
 */
export function ModeLamp() {
  const health = useMode((s) => s.health);
  const loading = useMode((s) => s.loading);

  if (!health) {
    return (
      <span
        data-testid="mode-lamp"
        data-mode={loading ? 'loading' : 'unknown'}
        className="inline-flex items-center gap-2 text-[11px] text-faint"
      >
        <span className="size-2.5 rounded-full border border-dashed border-line-2" />
        {loading ? 'checking…' : 'offline'}
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
          className="inline-flex items-center gap-2 text-[11px] cursor-default"
        >
          <span
            aria-hidden
            className={
              isDemo
                ? 'size-2.5 rounded-full border border-dashed border-amber'
                : 'size-2.5 rounded-full border border-good bg-good shadow-[0_0_6px_var(--good-glow)]'
            }
          />
          <span className={isDemo ? 'text-amber' : 'text-good'}>
            {isDemo ? 'demo · replay' : 'dev · live'}
          </span>
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-xs">
        {isDemo
          ? 'This deployment holds no API key. Chat replays conversations recorded in development, through the same events a live turn emits. Everything else reads the same committed artifacts the dev app does.'
          : `Live against ${health.bundle.lm_mini}. Turns are answered by the model, not replayed.`}
      </TooltipContent>
    </Tooltip>
  );
}
