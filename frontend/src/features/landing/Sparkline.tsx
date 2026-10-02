import { sparkPolylines } from './format';

export type SparkTone = 'amber' | 'good' | 'bad' | 'info';

interface SparklineProps {
  /** One value per bucket. `null` means "nothing measured then". */
  values: Array<number | null | undefined>;
  tone?: SparkTone;
  /** What to say when there is not enough measured data to draw a line. */
  emptyLabel?: string;
}

/**
 * A series, or an honest refusal to draw one: plotting nulls as zeros would turn a deployment
 * that served nothing into a flat line at the floor, which looks like a measurement. Below two
 * real points this renders a label. `amber` (the Console's accent) draws in the house accent.
 */
export function Sparkline({ values, tone = 'amber', emptyLabel = 'no series yet' }: SparklineProps) {
  const runs = sparkPolylines(values);
  if (runs.length === 0) {
    return (
      <span className="spark-empty" aria-label={emptyLabel}>
        — {emptyLabel}
      </span>
    );
  }
  return (
    <svg viewBox="0 0 100 18" preserveAspectRatio="none" role="img" aria-hidden className={`spark ${tone === 'amber' ? '' : tone}`}>
      {runs.map((points) => (
        <polyline key={points} points={points} />
      ))}
    </svg>
  );
}
