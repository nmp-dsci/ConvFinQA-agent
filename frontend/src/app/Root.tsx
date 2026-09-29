import { useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/tooltip';
import { versionLabel } from '@/features/admin/lib';
import { Shell } from '@/kit/Shell';
import { ChipBar } from '@/kit/ui';
import { useMode } from '../modeStore';
import { useStore } from '../store';
import { SLOTS, SUBPAGES, slotOf, subpageOf } from './nav';

/**
 * The house shell with ConvFinQA's props: the brand, the eight slots, the mode pill and the
 * provenance footer. Health and the report list are read once per session, here, so switching
 * tabs never re-asks the server what deployment this is.
 */
export function Root() {
  const loadReports = useStore((s) => s.loadReports);
  const loadMode = useMode((s) => s.load);
  const health = useMode((s) => s.health);
  const loading = useMode((s) => s.loading);

  useEffect(() => {
    void loadMode();
    void loadReports();
  }, [loadMode, loadReports]);

  const serving = health?.serving_champion ?? health?.champion;
  const mode = health
    ? health.mode === 'demo'
      ? {
          label: 'demo · read only',
          title:
            'This deployment holds no API key. Chat replays conversations recorded in development, through the same events a live turn emits; every other page reads the same committed files the dev app does.',
        }
      : { label: 'live · dev', tone: 'ok' as const, title: 'This deployment holds a key and answers with the model.' }
    : loading
      ? null
      : { label: 'backend offline', title: '/healthz did not answer' };

  return (
    <TooltipProvider delayDuration={200}>
      <Shell
        brand={
          <>
            convfinqa<b>-agent</b>
          </>
        }
        tabs={SLOTS}
        mode={mode}
        footer={
          <>
            Every number on these pages is read from a committed file — <code>evaluation/story.json</code>,{' '}
            <code>evaluation/predictions/</code>, <code>evaluation/readiness.json</code>, the trace snapshot — or
            from this deployment&rsquo;s <code>/healthz</code>. Serving{' '}
            <code>{serving ? versionLabel(serving) : '—'}</code>
            {health?.runtime === 'agent_sdk' ? ' on the Agent SDK' : ''}.{' '}
            <a href="https://github.com/nmp-dsci/ConvFinQA-agent">Source</a> ·{' '}
            <a href="https://arxiv.org/abs/2210.03849">ConvFinQA (Chen et al., 2022)</a>.
          </>
        }
      />
    </TooltipProvider>
  );
}

/** A slot with several pages: its chip bar above whichever page is open. */
export function SlotLayout() {
  const { pathname } = useLocation();
  const slot = slotOf(pathname);
  const subs = slot ? SUBPAGES[slot.key] : undefined;
  return (
    <>
      {subs && (
        <ChipBar
          label={`${slot?.label} pages`}
          current={subpageOf(pathname)}
          items={subs.map((p) => ({ key: p.key, to: p.to, label: p.label }))}
        />
      )}
      <Outlet />
    </>
  );
}
