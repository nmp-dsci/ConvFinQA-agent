import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import type { ReactElement } from 'react';
import { CAMPAIGNS, PASSES, STEPS, TRAIN } from './fixtures';
import { PassLedger } from './PassLedger';
import { RunsView, TRAIN_TOGGLE_LABEL } from './RunsPage';
import { StoryView } from './StoryPage';
import { TimelineChart } from './TimelineChart';
import { NO_FILTERS, ledgerRows, timelineModel } from './timeline';

function html(node: ReactElement): string {
  return renderToStaticMarkup(<MemoryRouter>{node}</MemoryRouter>);
}

function count(markup: string, needle: string): number {
  return markup.split(needle).length - 1;
}

describe('the timeline chart', () => {
  const markup = html(<TimelineChart passes={PASSES} steps={STEPS} gateQuestions={349} />);

  it('draws one point per complete pass and none for the incomplete one', () => {
    expect(count(markup, 'data-testid="timeline-point"')).toBe(PASSES.filter((p) => p.complete).length);
    expect(markup).not.toContain('data-run="sdk-evalloop-test100-sdk_v1·s1-20260905_132411"');
    expect(markup).toContain('data-testid="timeline-excluded"');
    expect(markup).toContain('Left off the axis: 1 incomplete pass');
  });

  it('bands the days, breaks the axis across the gap and says how long it was', () => {
    expect(count(markup, 'data-band=')).toBe(3);
    expect(markup).toContain('data-break="40"');
    expect(markup).toContain('>40d<');
  });

  it('draws the champion line and names the champion beside it', () => {
    expect(markup).toContain('data-testid="champion-line"');
    expect(markup).toContain('champion 90.5%');
  });

  it('marks each outcome with a shape and says it in words in the key', () => {
    expect(markup).toContain('data-outcome="falls_short"');
    expect(markup).toContain('<polygon');
    expect(markup).toContain('falls short');
    expect(markup).toContain('champion of the day — moves only on a promotion');
  });

  it('says why it is empty rather than drawing nothing', () => {
    const empty = html(<TimelineChart passes={[]} steps={[]} />);
    expect(empty).toContain('data-testid="timeline-empty"');
    expect(empty).toContain('no complete test pass is on record yet');
  });

  it('labels only the passes that moved or challenged the champion in its compact form', () => {
    const compact = html(<TimelineChart passes={PASSES} steps={STEPS} compact />);
    expect(count(compact, 'class="tl-label"')).toBeLessThan(count(markup, 'class="tl-label"'));
    expect(compact).toContain('data-compact="true"');
  });
});

describe('the pass ledger', () => {
  const rows = ledgerRows(timelineModel(PASSES, STEPS).passes);

  it('lists every pass, the incomplete one included, with its reason', () => {
    const markup = html(<PassLedger rows={rows} filters={NO_FILTERS} onFilters={() => {}} />);
    expect(count(markup, 'data-testid="pass-row"')).toBe(PASSES.length);
    expect(markup).toContain('never compared: 176 of 349 turns hit the session limit');
    expect(markup).toContain('the champion’s own pass');
    expect(markup).toContain('paired · p 0.043 toward worse');
  });

  it('offers a chip per value the rows hold and narrows to the one picked', () => {
    const markup = html(
      <PassLedger rows={rows} filters={{ runtime: 'pipeline', model: '', outcome: '' }} onFilters={() => {}} />,
    );
    expect(markup).toContain('data-filter="runtime"');
    expect(markup).toContain('data-filter="model"');
    expect(markup).toContain('data-filter="outcome"');
    expect(count(markup, 'data-testid="pass-row"')).toBe(3);
  });
});

describe('Optimise › Story', () => {
  const markup = html(<StoryView data={CAMPAIGNS} />);

  it('heads the page with who leads, since when, at what, and what has tried since', () => {
    expect(markup).toContain('sdk-v1 · sonnet-5 · high has led since 5 Sep 2026 at <em>90.5%</em>');
    expect(markup).toContain('2 challengers since, none has beaten it');
    expect(count(markup, '<em>')).toBe(1);
  });

  it('tells the champion line step by step, folding in the record that restated it', () => {
    expect(markup).toContain('The champion moved 2 times in 6 complete passes');
    expect(markup).toContain('Recorded as the one universal champion on 15 Oct 2026');
    expect(markup).toContain('c01-e03: +4.58pp over llm-v2');
  });

  it('lists the challengers since the champion with their verdicts, and every pass below', () => {
    expect(markup).toContain('Since 5 Sep 2026: sdk-v2 · sonnet-5 −2.87pp (rejected), sdk-v1 · haiku-5-5 −3.15pp (falls short)');
    expect(markup).toContain('data-testid="challenger-ledger"');
    expect(markup).toContain('Every test pass, 7 of them, oldest first');
  });
});

describe('Runs › Every test pass', () => {
  const props = { filters: NO_FILTERS, onFilters: () => {}, onTrain: () => {} };

  it('lists the test passes with the train toggle off by default', () => {
    const markup = html(<RunsView data={CAMPAIGNS} withTrain={false} {...props} />);
    expect(markup).toContain(TRAIN_TOGGLE_LABEL);
    expect(TRAIN_TOGGLE_LABEL).toBe('train: writes versions, never promotes them');
    expect(count(markup, 'data-testid="pass-row"')).toBe(PASSES.length);
    expect(count(markup, 'data-testid="train-row"')).toBe(0);
    expect(markup).toContain(`${TRAIN.length} train passes are hidden`);
    expect(markup).toContain('href="/runs/legacy"');
  });

  it('adds the train passes, marked as train, when the toggle is on', () => {
    const markup = html(<RunsView data={CAMPAIGNS} withTrain {...props} />);
    expect(count(markup, 'data-testid="train-row"')).toBe(TRAIN.length);
    expect(markup).toContain('seed 2027 · stops at first wrong');
    expect(markup).toContain('checked=""');
  });
});
