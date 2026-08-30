import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { PaperSection, BenchmarkSection } from './SectionPaper';
import { BundleSection, LlmSection, PipelineSection } from './SectionArchitecture';
import { EvaluationSection, OptimisationSection, PromotionSection } from './SectionContract';
import {
  DeploySection,
  NewVersionSection,
  ObservabilitySection,
  OpenWorkSection,
} from './SectionOperations';
import { Lamp, Live, Mono } from './ui';
import { useSystemData } from './useSystemData';

/**
 * The debrief.
 *
 * This is the page the whole redesign exists for: a stranger lands here and can
 * reconstruct the complete structure of what was built — the benchmark it is
 * measured against, the pipeline, the choke point, the versioning, the
 * evaluation discipline, the promotion contract, the optimisation, what is
 * observed, how it ships, how to make the next version, and what is broken.
 *
 * Three constraints hold the whole file together:
 *
 *  1. **Every figure is labelled by origin.** From the paper (with its table
 *     number), read live from this deployment, recomputed from a committed
 *     artefact, or read from the source. A page arguing for provenance that
 *     does not carry its own would be self-refuting.
 *  2. **It must read with the backend stopped.** The static content is the
 *     document; live values are decoration on top of it and degrade to em
 *     dashes with a stated reason. Someone who has just cloned the repository
 *     is the most likely reader.
 *  3. **The broken things are on it, first, in the reader's language.** The
 *     value of a debrief is entirely in whether it is true.
 */

interface TocEntry {
  id: string;
  label: string;
}

const TOC: TocEntry[] = [
  { id: 'paper', label: 'The ConvFinQA paper' },
  { id: 'benchmark', label: 'Benchmark vs the baselines' },
  { id: 'pipeline', label: 'Pipeline and routing' },
  { id: 'llm', label: 'The one place a model is built' },
  { id: 'bundle', label: 'The bundle fingerprint' },
  { id: 'evaluation', label: 'The two-number rule' },
  { id: 'promotion', label: 'Promotion contract & CI gate' },
  { id: 'optimisation', label: 'GEPA and the s7 harness' },
  { id: 'observability', label: 'What is captured, and what is not' },
  { id: 'deploy', label: 'Deploy' },
  { id: 'new-version', label: 'Make a new version' },
  { id: 'open-work', label: 'Open work' },
];

/**
 * Which section the reader is in.
 *
 * A throttled scroll read rather than an IntersectionObserver: these sections
 * are several screens tall, so an observer keyed on a narrow band at the top of
 * the viewport goes quiet for most of a long section and the highlight lags
 * behind the reader. Asking "which section heading did I last pass" answers the
 * question directly.
 *
 * Everything here is a convenience. It is guarded so that losing the highlight
 * can never lose the page.
 */
function useActiveSection(root: React.RefObject<HTMLElement | null>): string {
  const [active, setActive] = useState(TOC[0].id);

  useEffect(() => {
    const container = root.current;
    if (!container) return;

    let frame = 0;
    const read = () => {
      frame = 0;
      const top = container.getBoundingClientRect().top;
      let current = TOC[0].id;
      for (const entry of TOC) {
        const el = container.querySelector(`#${entry.id}`);
        if (!el) continue;
        if (el.getBoundingClientRect().top - top <= 120) current = entry.id;
      }
      setActive(current);
    };

    const onScroll = () => {
      if (frame) return;
      frame =
        typeof requestAnimationFrame === 'function' ? requestAnimationFrame(read) : (read(), 0);
    };

    read();
    container.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      container.removeEventListener('scroll', onScroll);
      if (frame && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(frame);
    };
  }, [root]);

  return active;
}

function Contents({
  active,
  onJump,
}: {
  active: string;
  onJump: (id: string) => void;
}) {
  return (
    <nav aria-label="Contents" className="min-w-0">
      <p className="mono-caps mb-2">contents</p>
      <ol className="space-y-0.5">
        {TOC.map((entry, i) => (
          <li key={entry.id}>
            <a
              href={`#${entry.id}`}
              onClick={(e) => {
                e.preventDefault();
                onJump(entry.id);
              }}
              className={cn(
                'flex items-baseline gap-2 rounded-sm px-1.5 py-1 transition-colors',
                'type-small hover:bg-panel hover:text-text',
                active === entry.id ? 'bg-panel text-amber' : 'text-muted',
              )}
            >
              <span className="type-num shrink-0 text-[10px] text-faint">
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="min-w-0">{entry.label}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}

export default function SystemRoute() {
  const data = useSystemData();
  const scrollRef = useRef<HTMLDivElement>(null);
  const active = useActiveSection(scrollRef);

  const jump = (id: string) => {
    const el = scrollRef.current?.querySelector(`#${id}`);
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const health = data.health;

  return (
    <div
      ref={scrollRef}
      data-testid="system-route"
      className="h-full overflow-y-auto overflow-x-hidden bg-ground"
    >
      <div className="mx-auto w-full max-w-[1180px] px-4 py-8 lg:px-8 lg:py-12">
        {/* ---------------------------------------------------------------- */}
        {/* Masthead                                                          */}
        {/* ---------------------------------------------------------------- */}
        <header className="min-w-0 border-b border-line pb-7">
          <p className="mono-caps">system · how it was built and how to operate it</p>
          <h1 className="type-display mt-3 max-w-[24ch]">
            Four agents, one choke point, one <span className="text-amber">contract</span>.
          </h1>
          <p className="type-lede mt-4 max-w-[64ch]">
            The paper this benchmark comes from and where this system lands among its baselines; the
            pipeline and the single string that routes it; the fingerprint that makes “a model
            version” mean something when every model is somebody else’s API; the contract that
            refused the last challenger; the keyless public deployment; and the steps to ship the
            next version — including the two things that do not work today.
          </p>

          <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2">
            <span className="type-meta inline-flex items-center gap-2">
              <Lamp state={data.isDemo ? 'replay' : health ? 'good' : 'idle'} />
              {health ? (
                data.isDemo ? (
                  'recorded demo · this container holds no API key'
                ) : (
                  'development deployment · live model calls'
                )
              ) : (
                'backend not reachable — the document below still stands'
              )}
            </span>
            <span className="type-meta">
              champion <Live value={health?.champion} reason="/healthz did not answer" />
            </span>
            <span className="type-meta break-all">
              bundle <Live value={health?.bundle_id} reason="/healthz did not answer" />
            </span>
            <span className="type-meta">
              code <Live value={health?.bundle.code_sha} reason="/healthz did not answer" />
            </span>
          </div>

          {!health && (
            <p className="type-meta mt-3 max-w-[70ch] rounded-md border border-dashed border-line-2 bg-panel/60 p-2.5">
              Nothing on this page has been substituted. Every live figure reads{' '}
              <Mono>—</Mono> until <Mono>/healthz</Mono> answers; every cited figure, diagram and
              procedure below is static and complete without it.
            </p>
          )}
        </header>

        {/* ---------------------------------------------------------------- */}
        {/* Contents + document                                               */}
        {/* ---------------------------------------------------------------- */}
        <div className="mt-8 grid min-w-0 gap-8 xl:grid-cols-[210px_minmax(0,1fr)] xl:gap-10">
          <div className="min-w-0 xl:sticky xl:top-4 xl:self-start">
            <Contents active={active} onJump={jump} />
          </div>

          <div className="min-w-0 space-y-10">
            <PaperSection />
            <BenchmarkSection data={data} />
            <PipelineSection />
            <LlmSection data={data} />
            <BundleSection data={data} />
            <EvaluationSection data={data} />
            <PromotionSection data={data} />
            <OptimisationSection data={data} />
            <ObservabilitySection data={data} />
            <DeploySection data={data} />
            <NewVersionSection />
            <OpenWorkSection data={data} />
          </div>
        </div>
      </div>
    </div>
  );
}
