import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { getCampaigns } from '../admin/api';
import type { CampaignsResponse } from '../admin/api';
import { AdminPage, ErrorNote, LoadingRows, Panel } from '../admin/ui';
import { PassLedger } from './PassLedger';
import { ledgerRows, timelineModel } from './timeline';
import type { LedgerFilters } from './timeline';

/**
 * Runs › Every test pass — the slot's first page (s21, owner's D4).
 *
 * The Runs tab used to open on the legacy 770-question scoring of v1–v3_1: a different population,
 * a retired protocol, and no Agent SDK run at all. It now opens on every pass the gate split has
 * seen, oldest first, with the agent each ran as and what its record says about it. Train passes —
 * early-stopped, resampled every cycle, in-sample for the rewrite they fed — are behind a toggle,
 * off by default, because their accuracy is not a measurement of anything a reader compares.
 *
 * The lens is the URL: `?runtime=`, `?model=`, `?outcome=`, `?train=1`.
 */

export const TRAIN_TOGGLE_LABEL = 'train: writes versions, never promotes them';

export default function RunsPage() {
  const query = useQuery({
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const [params, setParams] = useSearchParams();
  const filters: LedgerFilters = {
    runtime: params.get('runtime') ?? '',
    model: params.get('model') ?? '',
    outcome: params.get('outcome') ?? '',
  };
  const withTrain = params.get('train') === '1';

  function update(next: Partial<LedgerFilters & { train: string }>) {
    const out = new URLSearchParams(params);
    for (const [key, value] of Object.entries(next)) {
      if (value) out.set(key, value);
      else out.delete(key);
    }
    setParams(out, { replace: true });
  }

  return (
    <RunsView
      data={query.data}
      loading={query.isLoading}
      error={query.error}
      filters={filters}
      withTrain={withTrain}
      onFilters={(f) => update(f)}
      onTrain={(on) => update({ train: on ? '1' : '' })}
    />
  );
}

/** The page for a payload — split from the route so a test can render it from a fixture. */
export function RunsView({
  data,
  loading = false,
  error = null,
  filters,
  withTrain,
  onFilters,
  onTrain,
}: {
  data: CampaignsResponse | undefined;
  loading?: boolean;
  error?: unknown;
  filters: LedgerFilters;
  withTrain: boolean;
  onFilters: (next: LedgerFilters) => void;
  onTrain: (on: boolean) => void;
}) {
  const model = timelineModel(data?.test_timeline, data?.champion_steps);
  const train = data?.train_passes ?? [];
  const rows = ledgerRows(model.passes, withTrain ? train : []);
  const gateQuestions = typeof data?.split?.gate_questions === 'number' ? (data.split.gate_questions as number) : null;
  const nVersions = new Set(model.passes.map((p) => p.version)).size;
  const nModels = new Set(model.passes.map((p) => p.model ?? '')).size;

  return (
    <AdminPage
      testId="runs-passes"
      eyebrow="runs · every test pass"
      title={
        model.passes.length ? (
          <>
            <em>{model.passes.length}</em> test passes of {nVersions} prompt versions on {nModels} models, every
            one on the same {gateQuestions ?? 'fixed'} gate questions
          </>
        ) : (
          <>
            No test pass is on <em>record</em> yet
          </>
        )
      }
      sub={
        <>
          Oldest first, each set against the champion of its day: what the gate or the challenger rule
          said, by how much, and with what p. Train passes write the versions and never promote them, so
          they stay behind the toggle. The early versions’ 770-question scoring is its own page,{' '}
          <Link to="/runs/legacy">Legacy v1–v3_1</Link> — a different population under a retired
          protocol.
        </>
      }
    >
      {error ? <ErrorNote error={error} /> : null}
      {loading && !data ? (
        <LoadingRows rows={6} />
      ) : (
        <Panel
          testId="runs-ledger"
          title={
            withTrain
              ? `${model.passes.length} test passes and ${train.length} train passes, oldest first`
              : `${model.passes.length} test passes, oldest first — the same rows the story is drawn from`
          }
          endpoint="/optimise"
          to="/optimise"
          right={
            <label className="train-toggle" data-testid="train-toggle">
              <input type="checkbox" checked={withTrain} onChange={(e) => onTrain(e.target.checked)} />
              {TRAIN_TOGGLE_LABEL}
            </label>
          }
          note={
            withTrain
              ? 'Train rows are shaded and marked train: each draw early-stops at the first wrong answer and is resampled per cycle, so its accuracy is in-sample for the rewrite it fed and never set against the champion.'
              : `${train.length} train passes are hidden. Turn the toggle on to list them beside the test passes.`
          }
        >
          <PassLedger rows={rows} filters={filters} onFilters={onFilters} showSplit={withTrain} />
        </Panel>
      )}
    </AdminPage>
  );
}
