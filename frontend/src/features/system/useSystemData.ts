import { useQuery } from '@tanstack/react-query';
import {
  compareVersions,
  getEvalSummary,
  getExperiments,
  getProductionMetrics,
  listEvalRuns,
  listVersions,
} from '../../lib/api';
import type { ProductionMetrics, VersionAccuracyRow } from '../../lib/api';
import { getCampaigns } from '../admin/api';
import type { RuntimeComparison } from '../admin/api';
import { qk } from '../../lib/queryClient';
import { useMode } from '../../modeStore';
import type { ComparisonResult, Health, ModelAccuracy, VersionAccuracy } from '../../types';

/**
 * Everything the debrief reads from the live deployment.
 *
 * The page is a document first and a dashboard second, so every field here is
 * optional and every consumer must render without it. That is not defensive
 * habit — it is a requirement: this page has to stay readable with the backend
 * stopped, because the most likely reader is someone who cloned the repo and
 * has not started a server yet. Static prose stands; live values print an em
 * dash and say which endpoint did not answer.
 *
 * It also means no figure describing this system is written in the frontend.
 * Accuracy, the champion, the bundle, the gate verdict and the slice table all
 * come from here, so promoting a challenger updates the debrief and nobody has
 * to remember to edit a paragraph.
 */
export interface SystemData {
  health: Health | null;
  isDemo: boolean;

  /** `/admin/versions` — execution and program accuracy per version. */
  versions: VersionAccuracyRow[] | undefined;
  /** The champion's row, or the newest version when there is no alias. */
  championVersion: VersionAccuracyRow | undefined;
  championName: string | undefined;

  /** `/admin/experiments` — the only place the never-seen split is computed. */
  championHoldout: VersionAccuracy | undefined;
  versionHoldouts: VersionAccuracy[] | undefined;

  /** `/eval/runs/<champion>/summary` — the per-slice figures for the chart. */
  championSlices: ModelAccuracy | undefined;
  evalRuns: string[] | undefined;

  /** The comparator's verdict on the newest version that is not the champion. */
  gate: ComparisonResult | undefined;
  gateCandidate: string | undefined;

  /**
   * `/metrics/production`, read only for the two facts the observability
   * section states: whether the trace store is capturing at all, and how many
   * turns it holds. The numbers themselves live on the landing board and the
   * traces page, which is where a reader can drill into them.
   */
  metrics: ProductionMetrics | null | undefined;

  /**
   * `/eval/campaigns` → `runtime_comparison`: the pipeline champion and the single-session
   * champion, each scored on the fixed gate split. What the Leaderboard puts beside the paper.
   * `undefined` while loading or when the read failed; an arm's `accuracy` is `null` until run.
   */
  runtimes: RuntimeComparison | null | undefined;
  /** Questions and reports in the gate split those two arms were scored on. */
  gateQuestions: number | null;
  gateReports: number | null;

  loading: boolean;
  /** Which reads failed, named, so the page can say what is missing and why. */
  failures: string[];
}

export function useSystemData(): SystemData {
  const health = useMode((s) => s.health);
  const isDemo = health?.mode === 'demo';

  const versionsQuery = useQuery({ queryKey: qk.versions, queryFn: listVersions });
  const experimentsQuery = useQuery({ queryKey: qk.experiments, queryFn: getExperiments });
  const runsQuery = useQuery({ queryKey: qk.evalRuns, queryFn: listEvalRuns });
  const metricsQuery = useQuery({
    queryKey: qk.productionMetrics,
    queryFn: getProductionMetrics,
  });

  const versions = versionsQuery.data;
  const championVersion =
    versions?.find((v) => v.version === (health?.pipeline_champion ?? health?.champion)) ??
    versions?.[versions.length - 1];
  const championName = championVersion?.version;

  // The slice table is the champion's, not "the latest run's". Asking for a
  // version that has no committed CSV returns a 404, which the query surfaces
  // as a failure rather than as an empty chart.
  const slicesQuery = useQuery({
    queryKey: qk.evalSummary(championName ?? ''),
    queryFn: () => getEvalSummary(championName as string),
    enabled: Boolean(championName),
  });

  const orderedNames = versions?.map((v) => v.version) ?? [];
  const gateCandidate = orderedNames.filter((v) => v !== championName).slice(-1)[0];
  const gateQuery = useQuery({
    queryKey: qk.compare(championName ?? '', gateCandidate ?? ''),
    queryFn: () => compareVersions(championName as string, gateCandidate as string),
    enabled: Boolean(championName && gateCandidate),
  });

  // The same key and function the Optimise pages use: one payload, one cache entry, so the
  // Leaderboard and the Runtimes page cannot show two different arms.
  const campaignsQuery = useQuery({
    queryKey: ['eval-campaigns'],
    queryFn: () => getCampaigns(),
    staleTime: 60_000,
  });
  const split = campaignsQuery.data?.split;
  const splitCount = (key: string): number | null => {
    const v = split?.[key];
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  };

  const failures: string[] = [];
  if (versionsQuery.error) failures.push('/admin/versions');
  if (experimentsQuery.error) failures.push('/admin/experiments');
  if (runsQuery.error) failures.push('/eval/runs');
  if (slicesQuery.error) failures.push(`/eval/runs/${championName}/summary`);
  if (gateQuery.error) failures.push('/admin/compare');
  if (metricsQuery.error) failures.push('/metrics/production');
  if (campaignsQuery.error) failures.push('/eval/campaigns');

  return {
    health,
    isDemo,
    versions,
    championVersion,
    championName,
    championHoldout: experimentsQuery.data?.versions?.find((v) => v.version === championName),
    versionHoldouts: experimentsQuery.data?.versions,
    championSlices: slicesQuery.data?.models?.pydantic,
    evalRuns: runsQuery.data,
    gate: gateQuery.data,
    gateCandidate,
    metrics: metricsQuery.data,
    runtimes: campaignsQuery.data ? (campaignsQuery.data.runtime_comparison ?? null) : undefined,
    gateQuestions: splitCount('gate_questions'),
    gateReports: splitCount('gate_reports'),
    loading: versionsQuery.isLoading || experimentsQuery.isLoading,
    failures,
  };
}
