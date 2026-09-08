import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { TooltipProvider } from '../components/ui/tooltip';
import { NextFooter } from '../components/console/NextFooter';
import { PageHeader } from '../components/console/PageHeader';
import { NavRail } from './NavRail';
import { NAV, NAV_ITEMS, activeItem, nextItem, positionOf, prevItem } from './nav';

/**
 * The navigation is the story order, and three surfaces read it: the rail,
 * the page-header eyebrow and the next-step footer. These pin that they agree.
 */

function render(path: string, node: React.ReactNode): string {
  return renderToStaticMarkup(
    <MemoryRouter initialEntries={[path]}>
      <TooltipProvider>{node}</TooltipProvider>
    </MemoryRouter>,
  );
}

describe('the nav model', () => {
  it('has four labelled groups covering every route once', () => {
    expect(NAV.map((g) => g.label)).toEqual(['Product', 'Built', 'Evidence', 'Operations']);
    const paths = NAV_ITEMS.map((i) => i.to);
    expect(new Set(paths).size).toBe(paths.length);
    expect(paths).toContain('/admin/system');
    expect(paths).toContain('/admin/runtimes');
    expect(paths).toContain('/admin/readiness');
  });

  it('matches children of a route but never lets / or /admin swallow everything', () => {
    expect(activeItem('/admin/traces/abc')?.label).toBe('Traces');
    expect(activeItem('/chat/Double_MAR/2010/page_55.pdf')?.label).toBe('Chat');
    expect(activeItem('/admin/evaluations')?.label).toBe('Evaluations');
    expect(activeItem('/admin')?.label).toBe('Scoreboard');
    expect(activeItem('/debrief')?.label).toBe('Architecture');
  });

  it('prints the position as group · nn of nn', () => {
    expect(positionOf('/admin/campaigns')).toBe('Evidence · 04 of 06');
    expect(positionOf('/')).toBe('Product · 01 of 02');
    expect(positionOf('/nowhere')).toBeNull();
  });

  it('walks the story order forwards and back, with no dead end but the last', () => {
    expect(nextItem('/')?.label).toBe('Chat');
    expect(nextItem('/chat')?.label).toBe('Architecture');
    expect(nextItem('/admin/system')?.label).toBe('Readiness');
    expect(nextItem('/admin/readiness')?.label).toBe('Scoreboard');
    expect(prevItem('/')).toBeNull();
    expect(nextItem('/admin/research')).toBeNull();
    expect(prevItem('/admin/research')?.label).toBe('Traces');
  });
});

describe('the rail', () => {
  it('renders every item with a visible label and marks the current page', () => {
    const html = render('/admin/runtimes', <NavRail />);
    for (const item of NAV_ITEMS) expect(html).toContain(`>${item.label}<`);
    expect(html).toContain('aria-current="page"');
    expect(html).toContain('data-testid="nav-runtimes"');
    // Group headers are on the wide rail.
    expect(html).toContain('>Evidence<');
  });
});

describe('the page header and footer', () => {
  it('derive the eyebrow and the next step from the nav', () => {
    const html = render(
      '/admin/campaigns',
      <>
        <PageHeader title="Campaigns" verdict="Seven experiments, one promotion." />
        <NextFooter />
      </>,
    );
    expect(html).toContain('Evidence · 04 of 06');
    expect(html).toContain('Seven experiments, one promotion.');
    expect(html).toContain('href="/admin/runtimes"');
    expect(html).toContain('href="/admin/dataset"');
  });
});
