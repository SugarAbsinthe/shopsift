import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface DebugPanelProps {
  stage: string;
  toolRounds: number;
  productContext: string;
  userProfile: string;
  visible: boolean;
  onClose: () => void;
}

const STAGE_LABELS: Record<string, string> = {
  discovery: "👋 发现需求",
  needs_elicitation: "🔍 需求挖掘",
  search: "🛒 产品搜索",
  comparison: "⚖️ 产品对比",
  objection_handling: "💬 异议处理",
  recommendation: "🎯 最终推荐",
  summary: "✅ 总结收尾",
};

type Tab = "profile" | "products" | "tools";

export default function DebugPanel({
  stage,
  toolRounds,
  productContext,
  userProfile,
  visible,
  onClose,
}: DebugPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>("profile");
  const [height, setHeight] = useState(240);
  const [limits, setLimits] = useState({ min: 140, max: 480 });
  const panelRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerId: number; y: number; height: number } | null>(null);

  useEffect(() => {
    if (!visible) return;
    const container = panelRef.current?.parentElement;
    if (!container) return;
    const updateLimits = () => {
      // Reserve chat space, including its input, even in a short window.
      const available = container.clientHeight;
      const chatSpace = Math.max(96, Math.min(200, available / 2));
      const max = Math.max(12, Math.floor(Math.min(available * 0.65, available - chatSpace)));
      const min = Math.min(140, max);
      setLimits({ min, max });
      setHeight(current => Math.min(max, Math.max(min, current)));
    };
    updateLimits();
    const observer = new ResizeObserver(updateLimits);
    observer.observe(container);
    return () => {
      observer.disconnect();
      dragRef.current = null;
    };
  }, [visible]);

  const resize = (value: number) => setHeight(Math.min(limits.max, Math.max(limits.min, value)));
  if (!visible) return null;

  const tabs: { key: Tab; label: string }[] = [
    { key: "profile", label: "画像" },
    { key: "products", label: "检索" },
    { key: "tools", label: "工具" },
  ];

  return (
    <div
      ref={panelRef}
      role="region"
      aria-label="调试面板"
      className="flex flex-col shrink-0 min-h-0 overflow-hidden border-t border-slate-200 bg-white"
      style={{ height }}
    >
      <div
        role="separator"
        aria-label="调整调试面板高度"
        aria-orientation="horizontal"
        aria-controls="debug-panel-content"
        aria-valuemin={limits.min}
        aria-valuemax={limits.max}
        aria-valuenow={height}
        tabIndex={0}
        title="上下拖动调整高度，也可使用方向键"
        className="group h-3 shrink-0 flex items-center justify-center cursor-row-resize touch-none select-none hover:bg-indigo-50 focus-visible:outline-2 focus-visible:outline-indigo-400 focus-visible:-outline-offset-2"
        onPointerDown={event => {
          if (event.button !== 0) return;
          event.preventDefault();
          event.currentTarget.focus();
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { pointerId: event.pointerId, y: event.clientY, height };
        }}
        onPointerMove={event => {
          const drag = dragRef.current;
          if (drag?.pointerId === event.pointerId) resize(drag.height + drag.y - event.clientY);
        }}
        onPointerUp={() => { dragRef.current = null; }}
        onPointerCancel={() => { dragRef.current = null; }}
        onLostPointerCapture={() => { dragRef.current = null; }}
        onKeyDown={event => {
          const value = { ArrowUp: height + 24, ArrowDown: height - 24, Home: limits.min, End: limits.max }[event.key];
          if (value === undefined) return;
          event.preventDefault();
          resize(value);
        }}
      >
        <span className="w-10 h-1 rounded-full bg-slate-200 group-hover:bg-indigo-300" />
      </div>
      {/* Metrics */}
      <div className="flex shrink-0 items-center gap-5 px-5 py-2 border-b border-slate-100">
        <span className="text-xs text-slate-500">
          阶段 <strong className="text-slate-700">{STAGE_LABELS[stage] || stage}</strong>
        </span>
        <span className="text-xs text-slate-500">
          工具轮次 <strong className="text-slate-700">{toolRounds}</strong>
        </span>
        <span className="text-xs text-slate-500">
          检索 <strong className={productContext ? "text-emerald-600" : "text-slate-400"}>
            {productContext ? "已触发" : "未触发"}
          </strong>
        </span>
        <div className="flex-1" />
        <button onClick={onClose} aria-label="关闭调试面板" className="text-slate-300 hover:text-slate-500 p-1">
          <X size={14} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex shrink-0 px-5 pt-2 gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-3 py-1.5 text-xs font-medium rounded-t-lg transition-all ${
              activeTab === tab.key
                ? "bg-slate-50 text-slate-800"
                : "text-slate-400 hover:text-slate-600"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div id="debug-panel-content" className="flex-1 min-h-0 px-5 py-3 bg-slate-50 overflow-auto">
        {activeTab === "profile" && (
          <pre className="text-xs text-slate-600 whitespace-pre-wrap font-mono leading-relaxed">
            {userProfile || "暂无画像数据"}
          </pre>
        )}
        {activeTab === "products" && (
          productContext ? (
            <div className="markdown-body text-xs">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{productContext}</ReactMarkdown>
            </div>
          ) : (
            <p className="text-xs text-slate-400">本轮未触发产品检索</p>
          )
        )}
        {activeTab === "tools" && (
          <div className="text-xs text-slate-500 space-y-1">
            <p>🔄 本轮工具调用 <strong>{toolRounds}</strong> 轮</p>
          </div>
        )}
      </div>
    </div>
  );
}
