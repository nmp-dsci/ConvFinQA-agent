import { useQuery } from '@tanstack/react-query';
import {
  compareVersions,
  getExperiments,
  getProductionMetrics,
  getRegistry,
  getResearchStatus,
  getRules,
  getSplits,
  getTraceStats,
  listRuleVariants,
  listVersions,
} from '../../lib/api';
import type { MetricsSource } from '../../lib/api';
import { qk } from '../../lib/queryClient';
import { useMode } from '../../modeStore';
import { joinVersionRows } from './lib';
import type { VersionRow } from './lib';

/**
 * The reads every admin page shares, behind one hook.
 *
 * They go through the app's single query client with the keys already declared
 * in `lib/queryClient.ts`, so opening Overview and then Evaluations re-uses the
 * same cached `/admin/versions` and `/admin/experiments` responses instead of
 * asking twice. Keys that only this feature needs are declared below rather
 * than added to the shared file, which Phase 5 is editing in parallel.
 */

/** Query keys owned by the admin feature alone. */
export const ak = {
  answers: (reportId: string, onlyDisagreements: boolean, limit: number) =>
    ['admin', 'answers', reportId, onlyDisagreements, limit] as const,
  traceList: (source: string, reportId: string, sessionId: string, limit: number) =>
    ['admin', 'traceList', source, reportId, sessionId, limit] as const,
  liveTrace: (id: string) => ['admin', 'liveTrace', id] as const,
  evalTrace: (version: string, reportId: string, turnIndex: number) =>
    ['admin', 'evalTrace', version, reportId, turnIndex] as const,
};

// ---------------------------------------------------------------------------
// Versions — the two-number rule, joined once
// ---------------------------------------------------------------------------

export function useVersionRows(): {
  rows: VersionRow[];
  champion: string | null;
  isLoading: boolean;
  error: unknown;
  /** Whether the payload came from live MLflow or the committed snapshot. */
  experimentsSource: string | undefined;
} {
  const champion = useMode((s) => s.health?.champion) ?? null;
  const versions = useQuery({ queryKey: qk.versions, queryFn: listVersions });
  const experiments = useQuery({ queryKey: qk.experiments, queryFn: getExperiments });

  return {
    rows: joinVersionRows(versions.data, experiments.data?.versions, champion),
    champion,
    isLoading: versions.isLoading || experiments.isLoading,
    error: versions.error ?? experiments.error,
    experimentsSource: experiments.data?.source,
  };
}

export function useExperiments() {
  return useQuery({ queryKey: qk.experiments, queryFn: getExperiments });
}

export function useRegistry() {
  return useQuery({ queryKey: qk.registry, queryFn: getRegistry });
}

export function useSplits() {
  return useQuery({ queryKey: qk.splits, queryFn: getSplits });
}

export function useTraceStats() {
  return useQuery({ queryKey: qk.traceStats, queryFn: getTraceStats });
}

/**
 * The comparator's verdict for one pair.
 *
 * Disabled rather than defaulted when either side is missing: comparing a
 * version against itself would print a promotable verdict that means nothing.
 */
export function useComparison(baseline: string | undefined, candidate: string | undefined) {
  return useQuery({
    queryKey: qk.compare(baseline ?? '', candidate ?? ''),
    queryFn: () => compareVersions(baseline as string, candidate as string),
    enabled: Boolean(baseline && candidate && baseline !== candidate),
  });
}

// ---------------------------------------------------------------------------
// Production metrics
// ---------------------------------------------------------------------------

export function useProductionMetrics() {
  return useQuery({
    queryKey: qk.productionMetrics,
    queryFn: getProductionMetrics,
    // The one live surface in admin: these are turns this process is serving
    // while an operator watches. Everything else is a committed artifact.
    staleTime: 15_000,
    refetchInterval: 60_000,
  });
}

/** Which metrics group describes *this* deployment. Never a sum of groups. */
export function useDeploymentSource(): MetricsSource {
  const isDemo = useMode((s) => s.health?.mode) === 'demo';
  return isDemo ? 'demo' : 'serving';
}

// ---------------------------------------------------------------------------
// Research and rules
// ---------------------------------------------------------------------------

export function useResearchStatus() {
  return useQuery({
    queryKey: qk.researchStatus,
    queryFn: getResearchStatus,
    staleTime: 5_000,
    // A running round produces log lines every second; an idle one produces
    // nothing, and polling it forever would be pure noise on a demo that can
    // never launch anything. So the interval follows the job, not the clock.
    refetchInterval: (query) => (query.state.data?.busy ? 3_000 : false),
  });
}

export function useRuleVariants() {
  return useQuery({ queryKey: qk.ruleVariants, queryFn: listRuleVariants });
}

/**
 * The s7 rule stores.
 *
 * An empty `variant` is meaningful, not a missing argument: `/admin/rules`
 * then answers with `settings.variant`, and the response says which one that
 * was. That is how the page opens on the variant this backend is actually
 * configured with rather than on a guess.
 */
export function useRules(variant: string) {
  return useQuery({
    queryKey: qk.rules(variant),
    queryFn: () => getRules(variant),
  });
}
