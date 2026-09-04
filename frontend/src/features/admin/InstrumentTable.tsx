import type { ReactNode } from 'react';
import { useState } from 'react';
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import type { ColumnDef, RowData, SortingState } from '@tanstack/react-table';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * The mock's `table.tb`: a dense instrument table.
 *
 * Mono-caps column heads, tabular monospaced figures right-aligned, the first
 * column in the UI face because it is a name rather than a measurement, and a
 * champion row tinted `--amber-soft`. TanStack Table supplies sorting and the
 * row model; the look is entirely ours.
 *
 * Every table scrolls inside its own `overflow-x-auto` box. That is not
 * cosmetic: eleven columns of telemetry cannot fit a 420px phone, and a page
 * that scrolls sideways as a whole is the failure mode this avoids.
 */

/** Per-column display hints, read off `column.columnDef.meta`. */
export interface CellMeta {
  /** Defaults to `right` — these are measurements. Names opt into `left`. */
  align?: 'left' | 'right';
  /** Defaults to true for right-aligned columns. */
  mono?: boolean;
  /**
   * Column width. Under the default auto layout this is a *minimum*; under
   * `layout="fixed"` it is the width, so give every column one and prefer
   * percentages summing to 100.
   */
  width?: string;
  /** Wrap rather than truncate (questions, rule text). */
  wrap?: boolean;
}

declare module '@tanstack/react-table' {
  // The generics are required by the declaration being augmented even though
  // this table's hints do not vary with row or cell type.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ColumnMeta<TData extends RowData, TValue> extends CellMeta {}
}

export interface InstrumentTableProps<T> {
  data: T[];
  columns: Array<ColumnDef<T, unknown>>;
  /** Stable identity per row; used as the React key. */
  rowKey: (row: T, index: number) => string;
  /** Extra classes per row — the champion tint, a failed turn. */
  rowClass?: (row: T) => string | undefined;
  /** Below this the table scrolls horizontally inside its own box. */
  minWidth?: number;
  emptyLabel?: ReactNode;
  initialSorting?: SortingState;
  /** Caps the body height and scrolls, for the long tables. */
  maxHeight?: number;
  /**
   * `fixed` sizes columns purely from `meta.width`, ignoring content.
   *
   * Auto layout grows a column to avoid breaking an unbreakable token — a
   * report id like `Double_HIG/2016/page_253.pdf` — and pays for it by pushing
   * the last column off-screen, which no per-column minimum can prevent. Wide
   * tables of long strings want `fixed`; tables of short measurements do not.
   */
  layout?: 'auto' | 'fixed';
  testId?: string;
}

export function InstrumentTable<T>({
  data,
  columns,
  rowKey,
  rowClass,
  minWidth = 560,
  emptyLabel = 'no rows',
  initialSorting = [],
  maxHeight,
  layout = 'auto',
  testId,
}: InstrumentTableProps<T>) {
  const [sorting, setSorting] = useState<SortingState>(initialSorting);

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (data.length === 0) {
    return (
      <div className="rounded-[5px] border border-line border-dashed px-3 py-5 text-center">
        <p className="type-small text-faint">{emptyLabel}</p>
      </div>
    );
  }

  return (
    <div
      data-testid={testId}
      className="min-w-0 overflow-x-auto"
      style={maxHeight ? { maxHeight, overflowY: 'auto' } : undefined}
    >
      <table
        className={cn('w-full border-collapse text-[11px]', layout === 'fixed' && 'table-fixed')}
        style={{ minWidth }}
      >
        <thead className={cn(maxHeight && 'sticky top-0 z-10 bg-panel')}>
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => {
                const meta = (header.column.columnDef.meta ?? {}) as CellMeta;
                const align = meta.align ?? 'right';
                const sortable = header.column.getCanSort();
                const dir = header.column.getIsSorted();
                return (
                  <th
                    key={header.id}
                    scope="col"
                    style={
                      meta.width
                        ? layout === 'fixed'
                          ? { width: meta.width }
                          : { minWidth: meta.width }
                        : undefined
                    }
                    className={cn(
                      'mono-caps border-b border-line-2 px-1.5 py-1 font-medium whitespace-nowrap',
                      align === 'left' ? 'text-left' : 'text-right',
                    )}
                  >
                    {sortable ? (
                      <button
                        type="button"
                        onClick={header.column.getToggleSortingHandler()}
                        className={cn(
                          'mono-caps inline-flex items-center gap-0.5 hover:text-amber',
                          dir && 'text-amber',
                        )}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {dir === 'asc' && <ChevronUp aria-hidden className="size-2.5" />}
                        {dir === 'desc' && <ChevronDown aria-hidden className="size-2.5" />}
                      </button>
                    ) : (
                      flexRender(header.column.columnDef.header, header.getContext())
                    )}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row, index) => (
            <tr
              key={rowKey(row.original, index)}
              className={cn('border-b border-line last:border-0', rowClass?.(row.original))}
            >
              {row.getVisibleCells().map((cell) => {
                const meta = (cell.column.columnDef.meta ?? {}) as CellMeta;
                const align = meta.align ?? 'right';
                const mono = meta.mono ?? align === 'right';
                return (
                  <td
                    key={cell.id}
                    className={cn(
                      'px-1.5 py-1 align-top',
                      align === 'left' ? 'text-left' : 'text-right',
                      mono
                        ? 'type-num text-[10.5px] text-muted'
                        : 'font-sans text-[11.5px] text-text',
                      meta.wrap ? 'break-words' : 'whitespace-nowrap',
                    )}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The champion tint from the mock's `tr.ch`. */
export const CHAMPION_ROW = 'bg-amber-soft [&>td]:text-text';
