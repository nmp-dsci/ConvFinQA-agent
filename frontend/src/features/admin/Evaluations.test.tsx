import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { TooltipProvider } from '../../components/ui/tooltip';
import Evaluations from './Evaluations';

/**
 * The Campaigns page was folded into Experiments; Evaluations' sub-header used
 * to point readers at "Campaigns" for the pipeline track. This renders the
 * real page (queries left pending, since the header text does not depend on
 * their data) and asserts a reader sees "Experiments", never a dangling
 * "Campaigns" reference.
 */
describe('Evaluations sub-header', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('points readers at Experiments for the pipeline track, not the deleted Campaigns page', () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => {})),
    );
    const client = new QueryClient();
    const html = renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <TooltipProvider>
          <MemoryRouter>
            <Evaluations />
          </MemoryRouter>
        </TooltipProvider>
      </QueryClientProvider>,
    );

    expect(html).toMatch(/see Experiments for the pipeline track/i);
    expect(html).not.toMatch(/see Campaigns/i);
    expect(html).toMatch(/Runtimes for the single-session challenger/);
  });
});
