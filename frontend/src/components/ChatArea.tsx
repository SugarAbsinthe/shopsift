import { useState, useRef, useEffect } from "react";
import { ArrowUp, Sparkles, Square } from "lucide-react";
import MessageBubble from "./MessageBubble";

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatAreaProps {
  messages: DisplayMessage[];
  isLoading: boolean;
  status?: string | null;
  onSend: (question: string) => void;
  onCancel?: () => void;
  showQuickChips?: boolean;
}

const QUICK_CHIPS = [
  "预算8000元，办公和编程，推荐笔记本",
  "预算4000元，推荐256GB存储的手机",
];

export default function ChatArea({
  messages,
  isLoading,
  status,
  onSend,
  onCancel,
  showQuickChips = true,
}: ChatAreaProps) {
  const [input, setInput] = useState("");
  const [quickChips, setQuickChips] = useState(QUICK_CHIPS);
  const [catalogHint, setCatalogHint] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/catalog/categories", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((data) => {
        if (!controller.signal.aborted && data?.categories?.length) {
          setQuickChips(data.categories.slice(0, 2).map((category: { example: string }) => category.example));
          if (data.categories.some((category: { status: string }) => category.status === "trial")) {
            setCatalogHint("目录处于试点阶段，价格为采集时快照；部分商品缺少体验资料。");
          }
        } else if (!controller.signal.aborted && data?.available === false) {
          setQuickChips([]);
        }
      }).catch(() => {});
    return () => controller.abort();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = () => {
    if (!input.trim() || isLoading) return;
    onSend(input.trim());
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className={`flex-1 overflow-y-auto ${isEmpty ? "flex items-center justify-center" : "px-4 py-5"}`}>
        {isEmpty && showQuickChips && !isLoading ? (
          <div className="w-full max-w-xl mx-auto text-center px-6">
            <div className="w-14 h-14 rounded-2xl bg-indigo-100 flex items-center justify-center mx-auto mb-5">
              <Sparkles size={26} className="text-indigo-500" />
            </div>
            <h2 className="text-lg font-semibold text-slate-700 mb-1">需要我帮你找什么？</h2>
            <p className="text-sm text-slate-400 mb-6">告诉我你的需求和预算，我帮你找到最合适的</p>
            {catalogHint && <p className="text-xs text-slate-400 mb-4">{catalogHint}</p>}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {quickChips.map((chip, i) => (
                <button
                  key={i}
                  onClick={() => onSend(chip)}
                  disabled={isLoading}
                  className="text-left text-sm px-4 py-3 rounded-xl border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/50 text-slate-600 hover:text-indigo-700 transition-all disabled:opacity-50 bg-white shadow-sm"
                >
                  {chip}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble key={i} role={msg.role} content={msg.content} />
            ))}

            {isLoading && (
              <div role="status" className="flex items-center gap-2.5 text-slate-400 text-sm py-2 px-1 animate-in">
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                {status || "正在思考..."}
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input bar */}
      <div className="border-t border-slate-200 px-4 py-3.5 bg-white">
        <div className="max-w-3xl mx-auto flex items-end gap-2.5 bg-slate-50 rounded-2xl border border-slate-200 px-4 py-2 focus-within:border-indigo-300 focus-within:ring-2 focus-within:ring-indigo-100 focus-within:bg-white transition-all">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="说说你的需求..."
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none bg-transparent text-sm outline-none py-1.5 placeholder:text-slate-400 disabled:opacity-50"
          />
          {isLoading ? (
            <button
              onClick={onCancel}
              className="shrink-0 w-9 h-9 rounded-xl bg-red-500 text-white hover:bg-red-600 transition-all flex items-center justify-center shadow-sm"
              title="停止生成"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              aria-label="发送消息"
              disabled={!input.trim()}
              className="shrink-0 w-9 h-9 rounded-xl bg-indigo-500 text-white hover:bg-indigo-600 disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center justify-center shadow-sm"
            >
              <ArrowUp size={17} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
