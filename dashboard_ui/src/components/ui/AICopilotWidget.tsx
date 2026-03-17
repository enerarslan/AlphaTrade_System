import { useState, useRef, useEffect } from "react";
import { Bot, Send, User, Minimize2, Maximize2, X, BrainCircuit, Terminal, Database, Trash2, ChevronDown, ChevronRight, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "@/lib/store";
import { api } from "@/lib/api";

type Message = {
  id: string;
  sender: "user" | "ai";
  text: string;
  timestamp: Date;
  sources?: string[];
  thinking?: string | null;
};

const QUICK_PROMPTS = [
  "Risk durumum nasıl?",
  "Portfolio summary",
  "Any active alerts?",
  "Model health check",
  "Execution quality",
  "Full system diagnostic",
  "Pozisyon analizi",
];

function ThinkingBlock({ thinking }: { thinking: string }) {
  const [expanded, setExpanded] = useState(false);
  const lines = thinking.split("\n").filter(Boolean);
  const preview = lines.slice(0, 2).join(" ").slice(0, 80);

  return (
    <div className="mt-2 rounded-lg border border-fuchsia-500/10 bg-fuchsia-500/[0.03]">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-[10px] text-fuchsia-400/70 hover:text-fuchsia-300 transition-colors"
      >
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        <Sparkles size={9} />
        <span className="font-mono font-medium">Deep Analysis</span>
        {!expanded && <span className="ml-1 text-fuchsia-400/40 truncate">{preview}...</span>}
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-2.5 pb-2 text-[10px] text-fuchsia-300/50 font-mono leading-relaxed whitespace-pre-wrap max-h-[200px] overflow-y-auto scrollbar-thin">
              {thinking}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function AICopilotWidget({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "ai",
      text: "AlphaTrade Copilot online — JPMorgan-grade analysis powered by Qwen 3 LLM with live system access. I perform deep reasoning on your portfolio, risk metrics, signals, models, and more. Ask me anything.",
      timestamp: new Date(),
      sources: ["system"],
    }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);

  const bottomRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const portfolio = useStore(state => state.portfolio);
  const riskMetrics = useStore(state => state.riskMetrics);
  const alerts = useStore(state => state.alerts);
  const positions = useStore(state => state.positions);
  const performance = useStore(state => state.performance);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping, isOpen]);

  // Timer for showing elapsed time during inference
  useEffect(() => {
    if (isTyping) {
      setElapsedMs(0);
      timerRef.current = setInterval(() => setElapsedMs(prev => prev + 100), 100);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isTyping]);

  const handleSend = async (overrideText?: string) => {
    const text = overrideText ?? input;
    if (!text.trim() || isTyping) return;

    const userMsg: Message = { id: Date.now().toString(), sender: "user", text, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await api.post<{ response: string; sources: string[]; thinking?: string | null }>("/copilot/chat", {
        message: text,
        context: {
          portfolio: {
            equity: portfolio?.equity,
            cash: portfolio?.cash,
            daily_pnl: portfolio?.daily_pnl,
            total_pnl: portfolio?.total_pnl,
            positions_count: portfolio?.positions_count ?? positions.length,
            gross_exposure: portfolio?.gross_exposure,
            net_exposure: portfolio?.net_exposure,
          },
          risk: {
            current_drawdown: riskMetrics?.current_drawdown,
            portfolio_var_95: riskMetrics?.portfolio_var_95,
            portfolio_var_99: riskMetrics?.portfolio_var_99,
            max_drawdown_30d: riskMetrics?.max_drawdown_30d,
            beta_exposure: riskMetrics?.beta_exposure,
          },
          performance: {
            sharpe_ratio_30d: performance?.sharpe_ratio_30d,
            sortino_ratio_30d: performance?.sortino_ratio_30d,
            win_rate_30d: performance?.win_rate_30d,
            profit_factor: performance?.profit_factor,
          },
          alerts_count: alerts.filter(a => a.status !== "RESOLVED").length,
          positions_count: positions.length,
        },
      }, { timeout: 120000 });

      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: "ai",
        text: response.data.response,
        timestamp: new Date(),
        sources: response.data.sources,
        thinking: response.data.thinking,
      }]);
    } catch {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: "ai",
        text: "Ollama AI engine unreachable. Ensure Ollama is running (`ollama serve`) with the qwen3:4b model loaded.",
        timestamp: new Date(),
        sources: ["error"],
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const clearChat = () => {
    setMessages([{
      id: "welcome-" + Date.now(),
      sender: "ai",
      text: "Chat cleared. How can I help?",
      timestamp: new Date(),
      sources: ["system"],
    }]);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95, transition: { duration: 0.2 } }}
          className={`fixed bottom-4 right-4 z-50 flex flex-col overflow-hidden rounded-2xl border border-fuchsia-500/30 bg-slate-950/95 shadow-[0_0_40px_rgba(217,70,239,0.15)] transition-all duration-300 backdrop-blur-xl ${isExpanded ? 'w-[650px] h-[75vh]' : 'w-[400px] h-[520px]'}`}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-fuchsia-500/20 bg-fuchsia-500/10 px-4 py-3">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-fuchsia-500/20 text-fuchsia-400">
                <BrainCircuit size={18} />
              </div>
              <div>
                <h3 className="font-bold text-slate-100 text-sm">AlphaTrade Copilot</h3>
                <p className="text-[10px] text-fuchsia-400 font-mono tracking-wider flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-400 animate-pulse"></span>
                  QWEN 3 REASONING + LIVE DATA
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button onClick={clearChat} className="p-1.5 text-slate-400 hover:text-slate-200 transition-colors rounded-md hover:bg-white/[0.1]" title="Clear chat">
                <Trash2 size={14} />
              </button>
              <button onClick={() => setIsExpanded(!isExpanded)} className="p-1.5 text-slate-400 hover:text-slate-200 transition-colors rounded-md hover:bg-white/[0.1]">
                {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
              </button>
               <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-rose-400 transition-colors rounded-md hover:bg-white/[0.1]">
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Chat Feed */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 text-sm scrollbar-thin">
            {messages.map((msg) => (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                key={msg.id}
                className={`flex gap-3 ${msg.sender === "user" ? "flex-row-reverse" : "flex-row"}`}
              >
                <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${msg.sender === "user" ? "bg-cyan-500/20 text-cyan-400" : "bg-fuchsia-500/20 text-fuchsia-400"}`}>
                  {msg.sender === "user" ? <User size={14} /> : <Bot size={14} />}
                </div>
                <div className={`max-w-[85%] rounded-xl px-4 py-2.5 ${msg.sender === "user" ? "bg-cyan-500/10 border border-cyan-500/20 text-cyan-100 rounded-tr-none" : "bg-white/[0.04] border border-white/[0.08] text-slate-300 rounded-tl-none"}`}>
                  <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                  {/* Thinking/Reasoning collapsible block */}
                  {msg.thinking && <ThinkingBlock thinking={msg.thinking} />}
                  <div className="flex items-center justify-between mt-1.5 gap-2">
                    <span className={`text-[9px] ${msg.sender === "user" ? "text-cyan-500/60" : "text-slate-500"}`}>
                      {msg.timestamp.toLocaleTimeString()}
                    </span>
                    {msg.sources && msg.sources.length > 0 && msg.sources[0] !== "system" && msg.sources[0] !== "error" && (
                      <div className="flex items-center gap-1">
                        <Database size={8} className="text-fuchsia-500/50" />
                        <span className="text-[8px] text-fuchsia-400/60 font-mono">
                          {msg.sources.filter(s => s !== "ollama" && s !== "fallback").join(", ")}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}

            {isTyping && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-fuchsia-500/20 text-fuchsia-400">
                  <Bot size={14} />
                </div>
                <div className="bg-white/[0.04] border border-white/[0.08] rounded-xl rounded-tl-none px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      <span className="w-1.5 h-1.5 bg-fuchsia-500/50 rounded-full animate-bounce"></span>
                      <span className="w-1.5 h-1.5 bg-fuchsia-500/50 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                      <span className="w-1.5 h-1.5 bg-fuchsia-500/50 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                    </div>
                    <span className="text-[10px] text-fuchsia-400/50 font-mono">
                      deep reasoning... {(elapsedMs / 1000).toFixed(1)}s
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Quick Prompts */}
          {messages.length <= 2 && !isTyping && (
            <div className="px-3 pb-2 flex flex-wrap gap-1.5">
              {QUICK_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => void handleSend(prompt)}
                  className="rounded-full border border-fuchsia-500/20 bg-fuchsia-500/[0.05] px-3 py-1 text-[10px] text-fuchsia-300 hover:bg-fuchsia-500/10 hover:border-fuchsia-500/30 transition-all"
                >
                  {prompt}
                </button>
              ))}
            </div>
          )}

          {/* Input Area */}
          <div className="p-3 bg-slate-950/80 border-t border-white/[0.06]">
             <div className="relative flex items-center">
                <Terminal size={16} className="absolute left-3 text-fuchsia-500/50" />
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && void handleSend()}
                  placeholder="Ask anything about your trading system..."
                  className="w-full bg-white/[0.03] border border-white/[0.1] rounded-lg py-2.5 pl-9 pr-12 text-sm text-slate-200 placeholder-slate-500 outline-none focus:border-fuchsia-500/50 focus:bg-white/[0.06] transition-all font-mono"
                />
                <button
                  onClick={() => void handleSend()}
                  disabled={!input.trim() || isTyping}
                  className="absolute right-2 p-1.5 text-fuchsia-400 hover:text-fuchsia-300 disabled:opacity-50 disabled:hover:text-fuchsia-400 transition-colors"
                >
                  <Send size={16} />
                </button>
             </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
