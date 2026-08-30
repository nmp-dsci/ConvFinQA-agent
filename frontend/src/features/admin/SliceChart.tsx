import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatPercent } from '../landing/format';

/**
 * Accuracy by slice, one bar per label, one colour per version.
 *
 * Two things this chart refuses to do. It does not draw a bar for a slice with
 * no questions in it — an empty bucket is not zero accuracy — and it prints the
 * denominator in the tooltip, because "33% on q_order 6" is six questions and
 * reads very differently once you know that.
 */

export interface SlicePoint {
  label: string;
  accuracy: number;
  n_correct: number;
  n_total: number;
}

export interface SliceSeries {
  version: string;
  points: SlicePoint[];
}

/** Three versions is the whole history; a fourth falls back to amber. */
const SERIES_COLOR = ['var(--amber)', 'var(--info)', 'var(--violet)'];

interface ChartRow {
  label: string;
  [version: string]: string | number | null;
}

function buildRows(series: SliceSeries[]): { rows: ChartRow[]; totals: Record<string, number> } {
  const labels: string[] = [];
  for (const s of series) {
    for (const p of s.points) if (!labels.includes(p.label)) labels.push(p.label);
  }
  const totals: Record<string, number> = {};
  const rows = labels.map((label) => {
    const row: ChartRow = { label };
    for (const s of series) {
      const point = s.points.find((p) => p.label === label);
      // No questions in this bucket means no bar, not a bar at zero.
      row[s.version] = point && point.n_total > 0 ? point.accuracy * 100 : null;
      if (point) totals[`${s.version}::${label}`] = point.n_total;
    }
    return row;
  });
  return { rows, totals };
}

export function SliceChart({
  series,
  height = 190,
  testId,
}: {
  series: SliceSeries[];
  height?: number;
  testId?: string;
}) {
  const { rows, totals } = buildRows(series);

  if (rows.length === 0) {
    return (
      <div className="rounded-[5px] border border-line border-dashed px-3 py-5 text-center">
        <p className="type-small text-faint">no slice in this payload</p>
      </div>
    );
  }

  return (
    <div data-testid={testId} className="min-w-0">
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 4, right: 4, bottom: 0, left: -18 }} barGap={2}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fill: 'var(--faint)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={{ stroke: 'var(--line-2)' }}
            />
            <YAxis
              domain={[0, 100]}
              width={44}
              tick={{ fill: 'var(--faint)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              cursor={{ fill: 'var(--amber-soft)' }}
              contentStyle={{
                background: 'var(--panel-2)',
                border: '1px solid var(--line-2)',
                borderRadius: 5,
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: 'var(--text)',
              }}
              labelStyle={{ color: 'var(--faint)' }}
              formatter={(value: unknown, name: unknown, item: unknown) => {
                const label = (item as { payload?: ChartRow })?.payload?.label ?? '';
                const n = totals[`${String(name)}::${label}`];
                const pct = typeof value === 'number' ? formatPercent(value / 100) : '—';
                return [n ? `${pct} of ${n}` : pct, String(name)];
              }}
            />
            {series.map((s, i) => (
              <Bar key={s.version} dataKey={s.version} radius={[2, 2, 0, 0]} isAnimationActive={false}>
                {rows.map((row) => (
                  <Cell key={row.label} fill={SERIES_COLOR[i % SERIES_COLOR.length]} />
                ))}
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-3">
        {series.map((s, i) => (
          <span key={s.version} className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="size-2 rounded-[2px]"
              style={{ background: SERIES_COLOR[i % SERIES_COLOR.length] }}
            />
            <span className="mono-caps">{s.version}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
