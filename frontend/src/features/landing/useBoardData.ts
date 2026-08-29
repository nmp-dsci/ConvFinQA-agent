import { useQuery } from '@tanstack/react-query';
import {
  compareVersions,
  getDemoQuestions,
  getExperiments,
  getProductionMetrics,
  getQuestions,
  listDemoReports,
  listReports,
  listVersions,
} from '../../lib/api';
import type { MetricsSource, SourceMetrics, VersionAccuracyRow } from '../../lib/api';
import { qk } from '../../lib/queryClient';
import { useMode } from '../../modeStore';
import type { ComparisonResult, Health, VersionAccuracy } from '../../types';
import { formatFilingId } from './format';

/**
 * Everything the status board reads, in one hook.
 *
 * Two things are deliberate here. First, no figure is computed in a component:
 * the board can only render what one of these queries returned, so there is
 * nowhere for a hardcoded number to hide. Second, accuracy and holdout
 * accuracy come from two different endpoints and stay in two different fields
 * all the way to the tile — `/admin/versions` cannot tell you which
 * conversations the optimizer saw, and blending the two figures is the one
 * mistake this whole project exists to avoid.
 */

// ---------------------------------------------------------------------------
// Recorded conversations
// ---------------------------------------------------------------------------

export interface RecordedConversation {
  reportId: string;
  /** `MAR · 2010 · p.55`. */
  label: string;
  firstQuestion: string;
  nTurns: number;
  /**
   * Turns the recorded run got right, or `null` when this came from the plain
   * report list, which carries gold answers but no scored result.
   */
  nCorrect: number | null;
  /** True when the pack can replay this conversation without a model call. */
  recorded: boolean;
}

/**
 * The pack first, the dataset second.
 *
 * `/demo/reports` is mounted in both deployments and, when a pack is present,
 * is strictly the better source: it knows the turn count and whether each
 * recorded turn was right. Only when there is no pack at all does this fall
 * back to `/reports` — and then the "N ✓" half of the card is dropped rather
 * than guessed, because the dataset knows the gold answer but not what this
 * system answered.
 */
async function loadRecordedConversations(limit: number): Promise<RecordedConversation[]> {
  let packed: Array<{ reportId: string; nTurns: number }> = [];
  try {
    packed = (await listDemoReports())
      .slice(0, limit)
      .map((r) => ({ reportId: r.report_id, nTurns: r.n_questions }));
  } catch {
    // No pack on this deployment — fall through to the dataset.
    packed = [];
  }

  if (packed.length > 0) {
    return Promise.all(
      packed.map(async (p) => {
        const questions = await getDemoQuestions(p.reportId);
        return {
          reportId: p.reportId,
          label: formatFilingId(p.reportId),
          firstQuestion: questions[0]?.question ?? '',
          nTurns: questions.length || p.nTurns,
          nCorrect: questions.filter((q) => q.correct).length,
          recorded: true,
        };
      }),
    );
  }

  const ids = (await listReports('', limit)).slice(0, limit);
  return Promise.all(
    ids.map(async (reportId) => {
      const questions = await getQuestions(reportId);
      return {
        reportId,
        label: formatFilingId(reportId),
        firstQuestion: questions[0]?.question ?? '',
        nTurns: questions.length,
        nCorrect: null,
        recorded: false,
      };
    }),
  );
}

// ---------------------------------------------------------------------------
// The board
// ---------------------------------------------------------------------------

export interface BoardData {
  health: Health | null;
  isDemo: boolean;
  champion: string | null;

  /** `/admin/versions` — execution and program accuracy per version. */
  versions: VersionAccuracyRow[] | undefined;
  /** The champion's row, or the last version if there is no champion alias. */
  championVersion: VersionAccuracyRow | undefined;
  /** `/admin/experiments` — the only place holdout accuracy is computed. */
  championHoldout: VersionAccuracy | undefined;
  /** The same rows for every version, for the version-over-version delta. */
  versionHoldouts: VersionAccuracy[] | undefined;
  /** Previous version, for the "v1 → v2" delta. */
  previousVersion: VersionAccuracyRow | undefined;

  /**
   * The metrics group that describes *this* deployment: `serving` in dev,
   * `demo` in the replay deployment. Never a sum of the two — a turn recorded
   * at 6.7s and replayed in 2s is not a 2s turn.
   */
  metricsSource: MetricsSource;
  metrics: SourceMetrics | null;
  metricsGeneratedAt: string | undefined;
  metricsWindowHours: number | undefined;
  traceCaptureEnabled: boolean | undefined;
  metricsLoading: boolean;

  /** The comparator's verdict on the newest non-champion version. */
  gate: ComparisonResult | undefined;
  gateCandidate: string | undefined;

  recorded: RecordedConversation[] | undefined;
  recordedLoading: boolean;

  /** True while nothing has arrived yet, so the board can show one skeleton. */
  loading: boolean;
  /** Set when a read failed outright, so the board says so instead of blanking. */
  error: string | null;
}

export function useBoardData(recordedLimit = 3): BoardData {
  const health = useMode((s) => s.health);
  const isDemo = health?.mode === 'demo';
  const champion = health?.champion ?? null;

  const versionsQuery = useQuery({
    queryKey: qk.versions,
    queryFn: listVersions,
  });

  const experimentsQuery = useQuery({
    queryKey: qk.experiments,
    queryFn: getExperiments,
  });

  const metricsQuery = useQuery({
    queryKey: qk.productionMetrics,
    queryFn: getProductionMetrics,
    // The one live surface on the page: these are turns this process has
    // served, and they move while someone is looking at the board.
    staleTime: 15_000,
    refetchInterval: 60_000,
  });

  const recordedQuery = useQuery({
    queryKey: qk.recordedConversations(recordedLimit),
    queryFn: () => loadRecordedConversations(recordedLimit),
  });

  const versions = versionsQuery.data;
  const championVersion =
    versions?.find((v) => v.version === champion) ?? versions?.[versions.length - 1];
  const championName = championVersion?.version;

  const orderedNames = versions?.map((v) => v.version) ?? [];
  const championIndex = championName ? orderedNames.indexOf(championName) : -1;
  const previousVersion = championIndex > 0 ? versions?.[championIndex - 1] : undefined;

  // The gate lamp reports the comparator's verdict on the newest version that
  // is not the champion. That is the live promotion question — "is there
  // something waiting that we refused, and why" — rather than a static badge.
  const gateCandidate = orderedNames.filter((v) => v !== championName).slice(-1)[0];
  const gateQuery = useQuery({
    queryKey: qk.compare(championName ?? '', gateCandidate ?? ''),
    queryFn: () => compareVersions(championName as string, gateCandidate as string),
    enabled: Boolean(championName && gateCandidate),
  });

  const metricsSource: MetricsSource = isDemo ? 'demo' : 'serving';
  const metrics = metricsQuery.data?.sources?.[metricsSource] ?? null;

  const championHoldout = experimentsQuery.data?.versions?.find(
    (v) => v.version === championName,
  );

  const firstError =
    versionsQuery.error ?? experimentsQuery.error ?? metricsQuery.error ?? recordedQuery.error;

  return {
    health,
    isDemo,
    champion,
    versions,
    championVersion,
    championHoldout,
    versionHoldouts: experimentsQuery.data?.versions,
    previousVersion,
    metricsSource,
    metrics,
    metricsGeneratedAt: metricsQuery.data?.generated_at,
    metricsWindowHours: metricsQuery.data?.window_hours,
    traceCaptureEnabled: metricsQuery.data?.trace_capture_enabled,
    metricsLoading: metricsQuery.isLoading,
    gate: gateQuery.data,
    gateCandidate,
    recorded: recordedQuery.data,
    recordedLoading: recordedQuery.isLoading,
    loading: versionsQuery.isLoading && experimentsQuery.isLoading,
    error: firstError ? String((firstError as Error).message ?? firstError) : null,
  };
}
