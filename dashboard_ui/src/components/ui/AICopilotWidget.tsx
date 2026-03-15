import { useState, useRef, useEffect } from "react";
import { Bot, Send, User, Minimize2, Maximize2, X, BrainCircuit, Terminal } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "@/lib/store";

type Message = {
  id: string;
  sender: "user" | "ai";
  text: string;
  timestamp: Date;
};

export default function AICopilotWidget({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "ai",
      text: "System Whisperer Copilot online. I am monitoring execution latency, market depth, and portfolio risks. How can I assist?",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  
  const bottomRef = useRef<HTMLDivElement>(null);
  
  // Quick access to store metrics for AI context
  const portfolio = useStore(state => state.portfolio);
  const riskMetrics = useStore(state => state.riskMetrics);
  const alerts = useStore(state => state.alerts);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping, isOpen]);

  const handleSend = () => {
    if (!input.trim()) return;
    
    const userMsg: Message = { id: Date.now().toString(), sender: "user", text: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsTyping(true);

    // Simulate AI Processing and Contextual Response
    setTimeout(() => {
      let responseText = "Processing request...";
      const query = userMsg.text.toLowerCase();

      if (query.includes("drawdown") || query.includes("risk")) {
        responseText = `Current portfolio drawdown is ${((riskMetrics?.current_drawdown ?? 0)*100).toFixed(2)}%. Value at Risk (95%) is $${(riskMetrics?.portfolio_var_95 ?? 0).toLocaleString()}. Risk boundaries are currently holding steady.`;
      } else if (query.includes("pnl") || query.includes("profit") || query.includes("loss")) {
         const pnl = portfolio?.total_pnl ?? 0;
         responseText = `Total PnL stands at ${pnl >= 0 ? "+" : ""}$${pnl.toLocaleString()}. The algorithmic execution seems stable across major pairs.`;
      } else if (query.includes("alert") || query.includes("incident")) {
         const activeCount = alerts.filter(a => a.status !== "RESOLVED").length;
         responseText = `There are currently ${activeCount} active alerts in the command center. I recommend verifying the Operations terminal for infrastructure health.`;
      } else if (query.includes("flatten") || query.includes("kill")) {
         responseText = `WARNING: If you need to flatten the portfolio, use the 'F' hotkey on the Execution Terminal, or engage the Kill Switch if this is a systemic failure.`;
      } else {
         const responses = [
           "I've cross-referenced that with the live tape. Algorithmic flow is normal.",
           "Analyzing order book density... the resistance level above appears heavy.",
           "Logging this query. SRE telemetry shows no immediate anomalies.",
           "My NLP models suggest market sentiment is slightly leaning bullish right now based on alternative data streams."
         ];
         responseText = responses[Math.floor(Math.random() * responses.length)];
      }

      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: "ai",
        text: responseText,
        timestamp: new Date()
      }]);
      setIsTyping(false);
    }, 1200 + Math.random() * 1000);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95, transition: { duration: 0.2 } }}
          className={`fixed bottom-4 right-4 z-50 flex flex-col overflow-hidden rounded-2xl border border-fuchsia-500/30 bg-slate-950/95 shadow-[0_0_40px_rgba(217,70,239,0.15)] transition-all duration-300 backdrop-blur-xl ${isExpanded ? 'w-[600px] h-[70vh]' : 'w-[380px] h-[500px]'}`}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-fuchsia-500/20 bg-fuchsia-500/10 px-4 py-3">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-fuchsia-500/20 text-fuchsia-400">
                <BrainCircuit size={18} />
              </div>
              <div>
                <h3 className="font-bold text-slate-100 text-sm">System Whisperer</h3>
                <p className="text-[10px] text-fuchsia-400 font-mono tracking-wider flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-400 animate-pulse"></span>
                  AI COPILOT ONLINE
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button onClick={() => setIsExpanded(!isExpanded)} className="p-1.5 text-slate-400 hover:text-slate-200 transition-colors rounded-md hover:bg-white/[0.1]">
                {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
              </button>
               <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-rose-400 transition-colors rounded-md hover:bg-white/[0.1]">
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Chat Feed */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-sm scrollbar-thin">
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
                <div className={`max-w-[80%] rounded-xl px-4 py-2 ${msg.sender === "user" ? "bg-cyan-500/10 border border-cyan-500/20 text-cyan-100 rounded-tr-none" : "bg-white/[0.04] border border-white/[0.08] text-slate-300 rounded-tl-none"}`}>
                  <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                  <span className={`block mt-1 text-[9px] ${msg.sender === "user" ? "text-cyan-500/60 text-right" : "text-slate-500 text-left"}`}>
                    {msg.timestamp.toLocaleTimeString()}
                  </span>
                </div>
              </motion.div>
            ))}
            
            {isTyping && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-fuchsia-500/20 text-fuchsia-400">
                  <Bot size={14} />
                </div>
                <div className="bg-white/[0.04] border border-white/[0.08] rounded-xl rounded-tl-none px-4 py-3 flex items-center gap-1">
                   <span className="w-1.5 h-1.5 bg-fuchsia-500/50 rounded-full animate-bounce"></span>
                   <span className="w-1.5 h-1.5 bg-fuchsia-500/50 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                   <span className="w-1.5 h-1.5 bg-fuchsia-500/50 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                </div>
              </motion.div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input Area */}
          <div className="p-3 bg-slate-950/80 border-t border-white/[0.06]">
             <div className="relative flex items-center">
                <Terminal size={16} className="absolute left-3 text-fuchsia-500/50" />
                <input 
                  type="text" 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                  placeholder="Ask the system..."
                  className="w-full bg-white/[0.03] border border-white/[0.1] rounded-lg py-2.5 pl-9 pr-12 text-sm text-slate-200 placeholder-slate-500 outline-none focus:border-fuchsia-500/50 focus:bg-white/[0.06] transition-all font-mono"
                />
                <button 
                  onClick={handleSend}
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
