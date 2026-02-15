type SignalItem = {
  signal_id: string;
  timestamp: string;
  symbol: string;
  direction: string;
  strength: number;
  confidence: number;
  model_source: string;
};

function SignalChip({ signal }: { signal: SignalItem }) {
  const side = String(signal.direction || "").toUpperCase();
  const positive = side.includes("LONG") || side.includes("BUY");
  return (
    <div
      className={`mx-1 inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs ${
        positive
          ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
          : "border-rose-500/30 bg-rose-500/10 text-rose-400"
      }`}
    >
      <span className="font-semibold">{signal.symbol || "--"}</span>
      <span className={positive ? "text-emerald-500" : "text-rose-500"}>{side || "N/A"}</span>
      <span className="font-mono text-slate-400">S {Number(signal.strength ?? 0).toFixed(2)}</span>
      <span className="font-mono text-slate-400">C {Number(signal.confidence ?? 0).toFixed(2)}</span>
      <span className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[10px] text-slate-500">
        {signal.model_source || "model"}
      </span>
    </div>
  );
}

export function SignalTape({ signals }: { signals: SignalItem[] }) {
  if (!signals.length) {
    return <p className="text-sm text-slate-500">No live signals yet. Stream will populate automatically.</p>;
  }

  const tape = signals.slice(0, 20);
  const loop = [...tape, ...tape];

  return (
    <div className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-white/[0.02] py-2.5">
      <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-16 bg-gradient-to-r from-slate-950/90 to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-16 bg-gradient-to-l from-slate-950/90 to-transparent" />
      <div className="flex w-max animate-signal-tape whitespace-nowrap pr-4">
        {loop.map((signal, idx) => (
          <SignalChip key={`${signal.signal_id || signal.timestamp}-${idx}`} signal={signal} />
        ))}
      </div>
    </div>
  );
}
