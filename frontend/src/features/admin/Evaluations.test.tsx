import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { TooltipProvider } from '../../components/ui/tooltip';
import Evaluations from './Evaluations';
import { SliceChart } from './SliceChart';

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

describe('SliceChart', () => {
  it('draws one bar per scored bucket, none for an empty one, each with its denominator', () => {
    const html = renderToStaticMarkup(
      <SliceChart
        testId="slices"
        series={[
          {
            version: 'v2',
            points: [
              { label: '0', accuracy: 0.6, n_correct: 3, n_total: 5 },
              { label: '1', accuracy: 0, n_correct: 0, n_total: 0 },
            ],
          },
          { version: 'v3_1', points: [{ label: '0', accuracy: 0.8, n_correct: 4, n_total: 5 }] },
        ]}
      />,
    );
    expect(html).toContain('data-testid="slices"');
    expect(html.match(/<rect/g)).toHaveLength(2);
    expect(html).toContain('3/5 questions');
    expect(html).toContain('llm-v3_1');
    expect(html).not.toMatch(/recharts/);
  });

  it('says so when there is no slice at all', () => {
    const html = renderToStaticMarkup(<SliceChart testId="slices" series={[]} />);
    expect(html).toContain('no slice in this payload');
  });
});
