import { sparkPolylines } from './format';

export type SparkTone = 'amber' | 'good' | 'bad' | 'info';

const STROKE: Record<SparkTone, string> = {
  amber: 'var(--amber)',
  good: 'var(--good)',
  bad: 'var(--bad)',
  info: 'var(--info)',
};

interface SparklineProps {
  /** One value per hourly bucket. `null` means "nothing measured that hour". */
  values: Array<number | null | undefined>;
  tone?: SparkTone;
  /** What to say when there is not enough measured data to draw a line. */
  emptyLabel?: string;
}

/**
 * An hourly series, or an honest refusal to draw one.
 *
 * The refusal is the point. `/metrics/production` returns 24 buckets whether
 * or not anything happened in them, and plotting nulls as zeros would turn a
 * deployment that has served nothing into a flat line at the floor — which
 * looks like a measurement. Below two real points this renders a label instead
 * of a chart.
 */
export function Sparkline({ values, tone = 'amber', emptyLabel = 'no series yet' }: SparklineProps) {
  const runs = sparkPolylines(values);

  if (runs.length === 0) {
    return (
      <div className="mono-caps flex h-[18px] items-center" aria-label={emptyLabel}>
        <span className="mr-1.5 h-px w-4 border-t border-dashed border-line-2" aria-hidden />
        {emptyLabel}
      </div>
    );
  }

  return (
    <svg
      viewBox="0 0 100 18"
      preserveAspectRatio="none"
      role="img"
      aria-hidden
      className="block h-[18px] w-full"
    >
      {runs.map((points) => (
        <polyline
          key={points}
          points={points}
          fill="none"
          stroke={STROKE[tone]}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}
