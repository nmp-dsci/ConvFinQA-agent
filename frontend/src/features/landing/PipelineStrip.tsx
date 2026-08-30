import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

const STAGES = [
  {
    name: 'triage',
    hint: 'Look-up or calculation? Decides whether the middle two stages run at all.',
  },
  {
    name: 'preprocess',
    hint: 'Resolves "that" and "this change" against the conversation so far, and plans the program. Skipped on a look-up turn.',
  },
  {
    name: 'retriever',
    hint: 'Finds the supporting values in the filing text and table. Runs on every turn — it is the one stage a look-up cannot skip.',
  },
  {
    name: 'calculator',
    hint: 'Runs the six-operation tool loop — add, subtract, multiply, divide, exp, greater — and returns the final result.',
  },
] as const;

/**
 * The pipeline, as a strip rather than a diagram.
 *
 * It earns its place on the landing page for one reason: the routing is the
 * interesting part. `pipeline/runner.py` short-circuits a number-selection
 * turn straight from triage to the retriever and returns — preprocess and
 * calculator never run — which is why a look-up costs a fraction of what a
 * program turn costs, and why the latency and cost tiles above are an average
 * over two quite different shapes of turn.
 */
export function PipelineStrip() {
  return (
    <div className="rounded-md border border-line bg-panel p-3">
      <div className="mono-caps mb-2">pipeline</div>
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1.5">
        {STAGES.map((stage, i) => (
          <span key={stage.name} className="flex items-center gap-1.5">
            {i > 0 && (
              <span aria-hidden className="type-num text-[11px] text-faint">
                →
              </span>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="type-num cursor-default rounded-[3px] border border-line-2 bg-panel-2 px-1.5 py-0.5 text-[11px] text-text">
                  {stage.name}
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-xs">
                {stage.hint}
              </TooltipContent>
            </Tooltip>
          </span>
        ))}
      </div>
      <p className="type-meta mt-2 text-faint">
        A look-up turn goes triage → retriever and stops. Only a program turn runs all four.
      </p>
    </div>
  );
}
