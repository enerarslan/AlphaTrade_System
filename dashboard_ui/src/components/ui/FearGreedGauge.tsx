import { useMemo } from "react";
import { motion } from "framer-motion";
import { useMarketData } from "@/lib/marketData";

export default function FearGreedGauge() {
  const fearGreed = useMarketData((state) => state.fearGreed);
  const value = fearGreed?.value ?? 50;

  const { color, bgGlow } = useMemo(() => {
    if (value <= 25) return { color: "#f43f5e", bgGlow: "rgba(244,63,94,0.15)" };
    if (value <= 45) return { color: "#f59e0b", bgGlow: "rgba(245,158,11,0.12)" };
    if (value <= 55) return { color: "#94a3b8", bgGlow: "rgba(148,163,184,0.08)" };
    if (value <= 75) return { color: "#10b981", bgGlow: "rgba(16,185,129,0.12)" };
    return { color: "#06b6d4", bgGlow: "rgba(6,182,212,0.15)" };
  }, [value]);

  // SVG arc geometry
  const cx = 70, cy = 70, r = 56;
  const startAngle = -210;
  const endAngle = 30;
  const totalAngle = endAngle - startAngle; // 240 degrees
  const needleAngle = startAngle + (value / 100) * totalAngle;

  function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }

  function arcPath(startA: number, endA: number) {
    const s = polarToCartesian(cx, cy, r, startA);
    const e = polarToCartesian(cx, cy, r, endA);
    const largeArc = Math.abs(endA - startA) > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${largeArc} 1 ${e.x} ${e.y}`;
  }

  const needleTip = polarToCartesian(cx, cy, r - 8, needleAngle);

  return (
    <div className="flex flex-col items-center gap-1 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3" style={{ boxShadow: `0 0 30px ${bgGlow}` }}>
      <svg width="140" height="90" viewBox="0 0 140 90">
        {/* Track */}
        <path d={arcPath(startAngle, endAngle)} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" strokeLinecap="round" />
        {/* Value arc */}
        <motion.path
          d={arcPath(startAngle, startAngle + (value / 100) * totalAngle)}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.5, ease: "easeOut" }}

        />
        {/* Needle */}
        <motion.line
          x1={cx}
          y1={cy}
          x2={needleTip.x}
          y2={needleTip.y}
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}

        />
        {/* Center dot */}
        <circle cx={cx} cy={cy} r="3" fill={color} />
        {/* Value text */}
        <text x={cx} y={cy + 22} textAnchor="middle" fill={color} fontSize="18" fontWeight="700" fontFamily="IBM Plex Mono, monospace">
          {value}
        </text>
      </svg>
      <span className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
        {fearGreed?.classification ?? "Loading..."}
      </span>
    </div>
  );
}
