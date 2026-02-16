import { useMemo, useRef, useEffect, useState, useCallback } from "react";

interface CorrelationHeatmapProps {
  /** Symmetric correlation matrix: { "AAPL": { "MSFT": 0.82, "GOOG": 0.65, ... }, ... } */
  matrix: Record<string, Record<string, number>>;
  /** Override cell size */
  cellSize?: number;
}

function colorForCorr(value: number): string {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped >= 0) {
    // 0 → gray, 1 → cyan
    const r = Math.round(30 + (1 - clamped) * 100);
    const g = Math.round(200 * clamped + 40 * (1 - clamped));
    const b = Math.round(220 * clamped + 60 * (1 - clamped));
    return `rgb(${r},${g},${b})`;
  } else {
    // 0 → gray, -1 → red
    const intensity = Math.abs(clamped);
    const r = Math.round(220 * intensity + 60 * (1 - intensity));
    const g = Math.round(40 + (1 - intensity) * 100);
    const b = Math.round(40 + (1 - intensity) * 80);
    return `rgb(${r},${g},${b})`;
  }
}

export default function CorrelationHeatmap({ matrix, cellSize = 40 }: CorrelationHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  const symbols = useMemo(() => Object.keys(matrix).sort(), [matrix]);
  const labelWidth = 60;
  const headerHeight = 60;

  const width = labelWidth + symbols.length * cellSize;
  const height = headerHeight + symbols.length * cellSize;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Draw cells
    symbols.forEach((rowSym, ri) => {
      symbols.forEach((colSym, ci) => {
        const val = matrix[rowSym]?.[colSym] ?? 0;
        const x = labelWidth + ci * cellSize;
        const y = headerHeight + ri * cellSize;

        // Cell background
        ctx.fillStyle = colorForCorr(val);
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
        ctx.globalAlpha = 1;

        // Cell text
        ctx.fillStyle = Math.abs(val) > 0.5 ? "#ffffff" : "#94a3b8";
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(val.toFixed(2), x + cellSize / 2, y + cellSize / 2);
      });
    });

    // Row labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    symbols.forEach((sym, i) => {
      ctx.fillText(sym.slice(0, 6), labelWidth - 4, headerHeight + i * cellSize + cellSize / 2);
    });

    // Column labels (rotated)
    ctx.save();
    symbols.forEach((sym, i) => {
      const x = labelWidth + i * cellSize + cellSize / 2;
      ctx.save();
      ctx.translate(x, headerHeight - 4);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(sym.slice(0, 6), 0, 0);
      ctx.restore();
    });
    ctx.restore();
  }, [symbols, matrix, cellSize, width, height]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleMouse = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const col = Math.floor((mx - labelWidth) / cellSize);
    const row = Math.floor((my - headerHeight) / cellSize);

    if (col >= 0 && col < symbols.length && row >= 0 && row < symbols.length) {
      const val = matrix[symbols[row]]?.[symbols[col]] ?? 0;
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        text: `${symbols[row]} × ${symbols[col]}: ${val.toFixed(3)}`,
      });
    } else {
      setTooltip(null);
    }
  };

  if (symbols.length === 0) {
    return <p className="text-sm text-slate-500">No correlation data available.</p>;
  }

  return (
    <div className="relative overflow-auto rounded-xl border border-white/[0.06] bg-white/[0.02] p-2">
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouse}
        onMouseLeave={() => setTooltip(null)}
        className="cursor-crosshair"
      />
      {tooltip && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-white/10 bg-slate-900/95 px-2 py-1 font-mono text-xs text-slate-200 shadow-lg backdrop-blur"
          style={{ left: tooltip.x + 12, top: tooltip.y - 24 }}
        >
          {tooltip.text}
        </div>
      )}
      {/* Color legend */}
      <div className="mt-2 flex items-center justify-center gap-2">
        <span className="text-[10px] text-rose-400">-1.0</span>
        <div className="h-2 w-32 rounded-full" style={{
          background: "linear-gradient(to right, rgb(220,40,40), rgb(60,100,80), rgb(30,200,220))",
        }} />
        <span className="text-[10px] text-cyan-400">+1.0</span>
      </div>
    </div>
  );
}
