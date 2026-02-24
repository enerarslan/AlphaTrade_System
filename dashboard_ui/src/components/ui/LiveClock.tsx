import { useEffect, useState } from "react";

export default function LiveClock() {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const local = now.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const utc = now.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

  // NYSE market hours: 9:30-16:00 ET (Mon-Fri)
  const etHour = Number(now.toLocaleString("en-US", { hour: "2-digit", hour12: false, timeZone: "America/New_York" }));
  const etMinute = now.getMinutes();
  const day = now.getDay(); // 0=Sun, 6=Sat
  const isWeekday = day >= 1 && day <= 5;
  const etMins = etHour * 60 + etMinute;
  const isOpen = isWeekday && etMins >= 570 && etMins < 960; // 9:30-16:00
  const isPremarket = isWeekday && etMins >= 240 && etMins < 570; // 4:00-9:30
  const isAfterHours = isWeekday && etMins >= 960 && etMins < 1200; // 16:00-20:00

  const marketStatus = isOpen
    ? { label: "MARKET OPEN", color: "text-emerald-400", dot: "bg-emerald-400" }
    : isPremarket
      ? { label: "PRE-MARKET", color: "text-amber-400", dot: "bg-amber-400" }
      : isAfterHours
        ? { label: "AFTER-HOURS", color: "text-amber-400", dot: "bg-amber-400" }
        : { label: "MARKET CLOSED", color: "text-rose-400", dot: "bg-rose-400" };

  return (
    <div className="space-y-2 rounded-xl border border-white/[0.06] bg-white/[0.02] px-3 py-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-semibold uppercase tracking-[0.2em] text-slate-500">Local</span>
        <span className="font-mono text-sm font-semibold text-cyan-300 text-glow-cyan">{local}</span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-semibold uppercase tracking-[0.2em] text-slate-500">UTC</span>
        <span className="font-mono text-xs text-slate-400">{utc}</span>
      </div>
      <div className="flex items-center gap-2 pt-0.5">
        <span className={`inline-block h-1.5 w-1.5 rounded-full pulse-dot ${marketStatus.dot}`} />
        <span className={`text-[9px] font-bold uppercase tracking-[0.15em] ${marketStatus.color}`}>{marketStatus.label}</span>
      </div>
    </div>
  );
}
