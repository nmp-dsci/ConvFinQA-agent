import { QueryClient } from '@tanstack/react-query';
import { ApiError } from '../api';

/**
 * One client for the whole app.
 *
 * The defaults are tuned for what this backend actually serves: almost every
 * read is a committed artifact (prediction CSVs, the registry, the MLflow
 * snapshot) that does not change between requests, so aggressive refetching
 * buys nothing and costs a round trip on every tab focus.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Committed artifacts do not move under us; a minute of staleness is
      // free. Live surfaces (research status, traces) override this locally.
      staleTime: 60_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      // Retrying a 4xx just repeats the same wrong request. A 404 in
      // particular is a real answer here — /metrics/production does not exist
      // until Phase 1 ships it, and the UI has to tolerate that rather than
      // hammer the endpoint three more times before showing an empty card.
      retry: (failureCount, error) => {
        if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false;
        return failureCount < 2;
      },
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    },
    mutations: {
      // A write that failed is a write the operator must decide about. Never
      // retry a promotion or a research launch behind their back.
      retry: false,
    },
  },
});

/** Namespaced query keys, so an invalidation cannot miss a caller by typo. */
export const qk = {
  health: ['health'] as const,
  reports: (q: string) => ['reports', q] as const,
  reportDocument: (rid: string) => ['report', rid, 'document'] as const,
  reportQuestions: (rid: string) => ['report', rid, 'questions'] as const,
  evalRuns: ['eval', 'runs'] as const,
  evalSummary: (run: string) => ['eval', 'runs', run, 'summary'] as const,
  evalPredictions: (run: string, model: string) =>
    ['eval', 'runs', run, 'predictions', model] as const,
  splits: ['eval', 'splits'] as const,
  answers: (rid: string, onlyDisagreements: boolean) =>
    ['eval', 'answers', rid, onlyDisagreements] as const,
  traces: (reportId: string, source: string) => ['traces', reportId, source] as const,
  trace: (id: string) => ['traces', id] as const,
  traceStats: ['traces', 'stats'] as const,
  experiments: ['admin', 'experiments'] as const,
  experimentRun: (id: string) => ['admin', 'experiments', id] as const,
  registry: ['admin', 'registry'] as const,
  versions: ['admin', 'versions'] as const,
  compare: (baseline: string, candidate: string) =>
    ['admin', 'compare', baseline, candidate] as const,
  rules: (variant: string) => ['admin', 'rules', variant] as const,
  ruleVariants: ['admin', 'rules', 'variants'] as const,
  researchStatus: ['admin', 'research', 'status'] as const,
  productionMetrics: ['metrics', 'production'] as const,
  /** The landing board's recorded-conversation cards (pack + first questions). */
  recordedConversations: (limit: number) => ['landing', 'recorded', limit] as const,
};
