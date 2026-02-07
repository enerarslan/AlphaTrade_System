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
      className={`mx-1 inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs ${
        positive
          ? "border-emerald-300 bg-emerald-50 text-emerald-800"
          : "border-rose-300 bg-rose-50 text-rose-800"
      }`}
    >
      <span className="font-semibold">{signal.symbol || "--"}</span>
      <span>{side || "N/A"}</span>
      <span className="font-mono">S {Number(signal.strength ?? 0).toFixed(2)}</span>
      <span className="font-mono">C {Number(signal.confidence ?? 0).toFixed(2)}</span>
      <span className="rounded bg-white/70 px-1.5 py-0.5 font-mono text-[10px]">
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
    <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-50/80 py-2">
      <div className="pointer-events-none absolute inset-y-0 left-0 w-16 bg-gradient-to-r from-slate-50 to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 w-16 bg-gradient-to-l from-slate-50 to-transparent" />
      <div className="flex w-max animate-signal-tape whitespace-nowrap pr-4">
        {loop.map((signal, idx) => (
          <SignalChip key={`${signal.signal_id || signal.timestamp}-${idx}`} signal={signal} />
        ))}
      </div>
    </div>
  );
}

