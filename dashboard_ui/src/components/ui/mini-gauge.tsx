import { useMemo } from "react";

interface MiniGaugeProps {
  /** 0 to 1 */
  value: number;
  /** Size in px */
  size?: number;
  /** Label text (center) */
  label?: string;
  /** Thresholds: [warningAt, dangerAt] as 0-1 */
  thresholds?: [number, number];
}

export default function MiniGauge({
  value,
  size = 48,
  label,
  thresholds = [0.6, 0.85],
}: MiniGaugeProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const radius = (size - 6) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - clamped);

  const color = useMemo(() => {
    if (clamped >= thresholds[1]) return "#f87171"; // red
    if (clamped >= thresholds[0]) return "#fbbf24"; // amber
    return "#34d399"; // green
  }, [clamped, thresholds]);

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={3}
        />
        {/* Progress arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          className="transition-all duration-700 ease-out"
          style={{ filter: `drop-shadow(0 0 4px ${color})` }}
        />
      </svg>
      <span
        className="absolute text-center font-mono leading-none"
        style={{ fontSize: size * 0.22, color }}
      >
        {label ?? `${Math.round(clamped * 100)}%`}
      </span>
    </div>
  );
}
