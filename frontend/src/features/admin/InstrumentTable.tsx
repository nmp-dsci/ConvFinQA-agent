import type { ReactNode } from 'react';
import { useState } from 'react';
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import type { ColumnDef, RowData, SortingState } from '@tanstack/react-table';

/**
 * A house table (ui/DESIGN.md §3 Tables) with TanStack's sorting underneath: a `.tw` scroll
 * wrapper, mono uppercase heads, `th.sortable` / `th.sorted`, `td.num` for measurements (right,
 * tabular), and the promoted row as `tr.pro`. Every table scrolls inside its own box so the page
 * never scrolls sideways.
 */

/** Per-column display hints, read off `column.columnDef.meta`. */
export interface CellMeta {
  /** Defaults to `right` — these are measurements. Names opt into `left`. */
  align?: 'left' | 'right';
  /** Defaults to true for right-aligned columns. */
  mono?: boolean;
  /** A minimum under the default layout; the width under `layout="fixed"`. */
  width?: string;
  /** Wrap rather than hold the cell on one line (questions, rule text). */
  wrap?: boolean;
}

declare module '@tanstack/react-table' {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ColumnMeta<TData extends RowData, TValue> extends CellMeta {}
}

export interface InstrumentTableProps<T> {
  data: T[];
  columns: Array<ColumnDef<T, unknown>>;
  rowKey: (row: T, index: number) => string;
  /** Row classes: `pro` for the champion, `dim`, `warnrow`, `click`. */
  rowClass?: (row: T) => string | undefined;
  minWidth?: number;
  emptyLabel?: ReactNode;
  initialSorting?: SortingState;
  /** Caps the body height and scrolls, heads sticky. */
  maxHeight?: number;
  /** `fixed` sizes columns purely from `meta.width` (wide tables of long ids). */
  layout?: 'auto' | 'fixed';
  caption?: ReactNode;
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
  caption,
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
      <p className="empty" data-testid={testId ? `${testId}-empty` : undefined}>
        {emptyLabel}
      </p>
    );
  }

  return (
    <div
      data-testid={testId}
      className={`tw${maxHeight ? ' scroll-y' : ''}`}
      style={maxHeight ? { maxHeight, overflowY: 'auto' } : undefined}
    >
      <table style={{ minWidth, tableLayout: layout }}>
        {caption && <caption>{caption}</caption>}
        <thead>
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => {
                const meta = (header.column.columnDef.meta ?? {}) as CellMeta;
                const align = meta.align ?? 'right';
                const sortable = header.column.getCanSort();
                const dir = header.column.getIsSorted();
                const cls = [
                  align === 'right' ? 'num' : 'left',
                  sortable ? 'sortable' : '',
                  dir ? 'sorted' : '',
                ]
                  .filter(Boolean)
                  .join(' ');
                return (
                  <th
                    key={header.id}
                    scope="col"
                    className={cls}
                    aria-sort={dir === 'asc' ? 'ascending' : dir === 'desc' ? 'descending' : undefined}
                    style={
                      meta.width ? (layout === 'fixed' ? { width: meta.width } : { minWidth: meta.width }) : undefined
                    }
                  >
                    {sortable ? (
                      <button type="button" className="sort" onClick={header.column.getToggleSortingHandler()}>
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {dir === 'asc' ? ' ▲' : dir === 'desc' ? ' ▼' : ''}
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
            <tr key={rowKey(row.original, index)} className={rowClass?.(row.original)}>
              {row.getVisibleCells().map((cell, i) => {
                const meta = (cell.column.columnDef.meta ?? {}) as CellMeta;
                const align = meta.align ?? 'right';
                const mono = meta.mono ?? align === 'right';
                const cls = [
                  align === 'right' ? (mono ? 'num' : 'num sans') : i === 0 ? 'sub left' : 'left',
                  mono && align === 'left' ? 'mono' : '',
                  meta.wrap ? 'wrap-cell' : 'nowrap',
                ]
                  .filter(Boolean)
                  .join(' ');
                return (
                  <td key={cell.id} className={cls}>
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

/** The promoted row, as the house marks it. */
export const CHAMPION_ROW = 'pro';
