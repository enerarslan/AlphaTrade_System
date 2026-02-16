import { useMemo } from "react";

interface SparklineProps {
  /** Array of numeric values */
  data: number[];
  /** Width in pixels */
  width?: number;
  /** Height in pixels */
  height?: number;
  /** Override color, otherwise uses trend direction */
  color?: string;
  /** Line width */
  strokeWidth?: number;
}

export default function Sparkline({
  data,
  width = 60,
  height = 20,
  color,
  strokeWidth = 1.5,
}: SparklineProps) {
  const path = useMemo(() => {
    if (data.length < 2) return "";
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const points = data.map((v, i) => ({
      x: (i / (data.length - 1)) * width,
      y: height - ((v - min) / range) * height,
    }));
    return points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  }, [data, width, height]);

  const isUp = data.length >= 2 && data[data.length - 1] >= data[0];
  const lineColor = color ?? (isUp ? "#34d399" : "#f87171");

  if (data.length < 2) return <div style={{ width, height }} />;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="inline-block"
    >
      <defs>
        <linearGradient id={`sg-${lineColor}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={lineColor} stopOpacity={0.3} />
          <stop offset="100%" stopColor={lineColor} stopOpacity={0} />
        </linearGradient>
      </defs>
      {/* Fill area */}
      <path
        d={`${path} L ${width} ${height} L 0 ${height} Z`}
        fill={`url(#sg-${lineColor})`}
      />
      {/* Line */}
      <path d={path} fill="none" stroke={lineColor} strokeWidth={strokeWidth} />
      {/* Current value dot */}
      <circle
        cx={width}
        cy={parseFloat(path.split(" ").slice(-1)[0]) || height / 2}
        r={2}
        fill={lineColor}
      />
    </svg>
  );
}
