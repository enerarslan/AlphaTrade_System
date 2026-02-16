import { useState, useRef, useCallback } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type ColumnFiltersState,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowUpDown, ArrowUp, ArrowDown, Download, Search } from "lucide-react";

interface DataGridProps<T> {
  data: T[];
  columns: ColumnDef<T, unknown>[];
  /** Max visible height before scrolling */
  maxHeight?: number;
  /** Enable row click handler */
  onRowClick?: (row: T) => void;
  /** Show global search filter */
  searchable?: boolean;
  /** Enable CSV export */
  exportable?: boolean;
  /** Filename for CSV export (no extension) */
  exportFilename?: string;
}

export default function DataGrid<T>({
  data,
  columns,
  maxHeight = 500,
  onRowClick,
  searchable = true,
  exportable = true,
  exportFilename = "export",
}: DataGridProps<T>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const parentRef = useRef<HTMLDivElement>(null);

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter, columnFilters },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  const { rows } = table.getRowModel();

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 36,
    overscan: 10,
  });

  const exportCsv = useCallback(() => {
    const headers = columns.map((c) => {
      const header = c.header;
      return typeof header === "string" ? header : String((c as { accessorKey?: string }).accessorKey ?? "");
    });
    const csvRows = [headers.join(",")];
    for (const row of data) {
      const vals = columns.map((c) => {
        const key = (c as { accessorKey?: string }).accessorKey;
        if (!key) return "";
        const val = (row as Record<string, unknown>)[key];
        return val != null ? `"${String(val).replace(/"/g, '""')}"` : "";
      });
      csvRows.push(vals.join(","));
    }
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${exportFilename}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [data, columns, exportFilename]);

  const virtualItems = virtualizer.getVirtualItems();

  return (
    <div className="overflow-hidden rounded-xl border border-white/[0.08] bg-white/[0.02]">
      {/* Toolbar */}
      {(searchable || exportable) && (
        <div className="flex items-center justify-between border-b border-white/[0.06] px-3 py-2">
          {searchable ? (
            <div className="flex items-center gap-2">
              <Search className="h-3.5 w-3.5 text-slate-500" />
              <input
                type="text"
                placeholder="Filter..."
                value={globalFilter}
                onChange={(e) => setGlobalFilter(e.target.value)}
                className="bg-transparent text-xs text-slate-300 outline-none placeholder:text-slate-600"
              />
            </div>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500">
              {rows.length} row{rows.length !== 1 ? "s" : ""}
            </span>
            {exportable && (
              <button
                onClick={exportCsv}
                className="flex items-center gap-1 rounded-md px-2 py-1 text-[10px] text-slate-500 transition-colors hover:bg-white/5 hover:text-slate-300"
              >
                <Download className="h-3 w-3" />
                CSV
              </button>
            )}
          </div>
        </div>
      )}

      {/* Table */}
      <div ref={parentRef} className="overflow-auto" style={{ maxHeight }}>
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id} className="border-b border-white/[0.06] bg-slate-900/80 backdrop-blur">
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    className="whitespace-nowrap px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-widest text-slate-500"
                    style={{ width: header.getSize() }}
                  >
                    {header.isPlaceholder ? null : (
                      <button
                        className="flex items-center gap-1"
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getIsSorted() === "asc" ? (
                          <ArrowUp className="h-3 w-3 text-cyan-400" />
                        ) : header.column.getIsSorted() === "desc" ? (
                          <ArrowDown className="h-3 w-3 text-cyan-400" />
                        ) : (
                          <ArrowUpDown className="h-3 w-3 text-slate-600" />
                        )}
                      </button>
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody
            style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}
          >
            {virtualItems.map((vi) => {
              const row = rows[vi.index];
              return (
                <tr
                  key={row.id}
                  onClick={() => onRowClick?.(row.original)}
                  className={`absolute left-0 w-full border-b border-white/[0.03] transition-colors ${
                    onRowClick ? "cursor-pointer" : ""
                  } hover:bg-white/[0.04]`}
                  style={{
                    height: `${vi.size}px`,
                    transform: `translateY(${vi.start}px)`,
                    display: "table-row",
                  }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td
                      key={cell.id}
                      className="whitespace-nowrap px-3 py-2 font-mono text-slate-300"
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
