import { useMode } from '../modeStore';
import { Badge, Mono, Spinner } from './ui';

const SURFACES = [
  {
    key: 'chat',
    title: 'Chat',
    blurb:
      'Pick a filing, then ask in your own words or step through the dataset’s own questions — watching all four stages resolve as they go.',
    demo: 'replayed from recordings',
    live: 'live against the champion',
  },
  {
    key: 'data',
    title: 'Data & answers',
    blurb:
      'Which conversations the optimizer saw and which it never did, plus every question with gold beside each version’s answer — including the ones a version got wrong.',
    demo: 'fully live',
    live: 'fully live',
  },
  {
    key: 'traces',
    title: 'Traces',
    blurb:
      'Every turn the system has answered, stage by stage: what each agent saw, what it returned, which tools ran, and how long it took.',
    demo: 'fully live',
    live: 'fully live',
  },
  {
    key: 'experiments',
    title: 'Experiments',
    blurb:
      'Every eval, GEPA and auto-research run, with the champion/challenger registry and the full promotion history.',
    demo: 'fully live',
    live: 'fully live',
  },
  {
    key: 'research',
    title: 'Research console',
    blurb:
      'Launch a round of automated prompt research or a GEPA optimisation, and watch it stream. Results land as the next challenger.',
    demo: 'visible, launch disabled',
    live: 'launch enabled',
  },
] as const;

const PIPELINE = [
  { name: 'Triage', blurb: 'look-up or calculation?' },
  { name: 'Preprocess', blurb: 'resolve “that”, plan the program' },
  { name: 'Retriever', blurb: 'find the values in text + table' },
  { name: 'Calculator', blurb: 'run the six-op tool loop' },
];

export function LandingPage({ onEnter }: { onEnter: () => void }) {
  const health = useMode((s) => s.health);
  const loading = useMode((s) => s.loading);
  const isDemo = health?.mode === 'demo';

  if (loading && !health) return <Spinner label="Checking deployment mode…" />;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-14">
        <header className="mb-12">
          <div className="flex items-center gap-3 mb-5">
            <div className="size-10 rounded-full bg-accent flex items-center justify-center font-semibold">
              CF
            </div>
            {health && (
              <Badge tone={isDemo ? 'warn' : 'good'}>
                {isDemo ? 'public demo — read only' : 'dev — live model'}
              </Badge>
            )}
          </div>

          <h1 className="text-3xl font-semibold tracking-tight mb-3">
            Conversational financial QA, with its own evidence
          </h1>
          <p className="text-textMuted leading-relaxed max-w-2xl">
            A four-agent pipeline that answers multi-turn questions about SEC filings — where
            later questions depend on earlier answers, and “what was that as a percentage?”
            has to resolve against a number the system produced three turns ago. Every answer
            is traceable to the bundle that produced it and the eval run that scored it.
          </p>

          <div className="flex flex-wrap items-baseline gap-x-8 gap-y-3 mt-8">
            <div>
              <div className="text-2xl font-semibold tabular-nums">
                72.8% <span className="text-textMuted mx-1">→</span>
                <span className="text-accent2">77.7%</span>
              </div>
              <div className="text-xs text-textMuted mt-1">
                v1 → v2 on the 309 questions no optimizer saw
              </div>
            </div>
            <div>
              <div className="text-2xl font-semibold tabular-nums text-textMuted">
                73.0% <span className="mx-1">→</span> 77.1%
              </div>
              <div className="text-xs text-textMuted mt-1">
                the same versions across all 770 scored questions
              </div>
            </div>
            {health?.champion && (
              <div>
                <div className="text-2xl font-semibold">
                  <Mono className="text-base">{health.champion}</Mono>
                </div>
                <div className="text-xs text-textMuted mt-1">current champion bundle</div>
              </div>
            )}
          </div>
          <p className="text-xs text-textMuted mt-3 max-w-2xl">
            Two numbers, because only one of them supports a generalisation claim. The
            optimizer trained on 120 of the 200 scored conversations, so the 770-question
            figure mixes seen and unseen work; the 309-question figure is measured purely on
            conversations no prompt was ever tuned against. Both are shown rather than the
            flattering one, and you can check the split membership yourself under
            Data &amp; answers.
          </p>
        </header>

        <section className="mb-12">
          <h2 className="text-xs uppercase tracking-wider text-textMuted mb-4">
            How one question is answered
          </h2>
          <ol className="grid gap-2 sm:grid-cols-4">
            {PIPELINE.map((stage, i) => (
              <li key={stage.name} className="bg-panel border border-white/5 rounded-lg p-3">
                <div className="text-[11px] text-textMuted tabular-nums mb-1">
                  {String(i + 1).padStart(2, '0')}
                </div>
                <div className="text-sm font-medium">{stage.name}</div>
                <div className="text-xs text-textMuted mt-1 leading-snug">{stage.blurb}</div>
              </li>
            ))}
          </ol>
          <p className="text-xs text-textMuted mt-3">
            A look-up question stops after Triage and Retriever; only a calculation walks the
            full path. The trace viewer shows which route each turn actually took.
          </p>
        </section>

        <section className="mb-12">
          <h2 className="text-xs uppercase tracking-wider text-textMuted mb-4">What’s inside</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {SURFACES.map((surface) => (
              <div key={surface.key} className="bg-panel border border-white/5 rounded-lg p-4">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <h3 className="text-sm font-medium">{surface.title}</h3>
                  <Badge
                    tone={
                      (isDemo ? surface.demo : surface.live).startsWith('fully live')
                        ? 'good'
                        : 'neutral'
                    }
                  >
                    {isDemo ? surface.demo : surface.live}
                  </Badge>
                </div>
                <p className="text-xs text-textMuted leading-relaxed">{surface.blurb}</p>
              </div>
            ))}
          </div>
        </section>

        {isDemo && (
          <section className="mb-10 bg-panel border border-amber-400/30 rounded-lg p-5">
            <h2 className="text-sm font-medium mb-2">What this demo does and doesn’t do</h2>
            <p className="text-xs text-textMuted leading-relaxed mb-3">
              This deployment holds no API keys and makes no model calls — by construction, not
              by policy. Chat replays conversations recorded in development, streamed through the
              same events a live turn emits. Everything else is genuinely live: it reads the same
              committed evaluation artifacts the development app does.
            </p>
            <p className="text-xs text-textMuted leading-relaxed">
              Ask one of the suggested questions for a report and you’ll see the real recorded
              run. Ask something outside the recordings and it will tell you so rather than
              inventing a number.
            </p>
          </section>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={onEnter}
            className="px-5 py-2.5 rounded-md bg-accent2 text-bg font-medium hover:brightness-110 transition"
            data-testid="landing-enter"
          >
            {isDemo ? 'Enter demo' : 'Open app'}
          </button>
          {health && (
            <span className="text-[11px] text-textMuted">
              bundle <Mono>{health.bundle_id}</Mono> · prompts{' '}
              <Mono>{health.bundle.prompts_version}</Mono> · build{' '}
              <Mono>{health.bundle.code_sha}</Mono>
              {isDemo ? ` · ${health.demo_reports} recorded reports` : ''}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
