import { useState } from "react";
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
  if (!visible) return null;

  const tabs: { key: Tab; label: string }[] = [
    { key: "profile", label: "画像" },
    { key: "products", label: "检索" },
    { key: "tools", label: "工具" },
  ];

  return (
    <div className="border-t border-slate-200 bg-white">
      {/* Metrics */}
      <div className="flex items-center gap-5 px-5 py-2.5 border-b border-slate-100">
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
        <button onClick={onClose} className="text-slate-300 hover:text-slate-500 p-1">
          <X size={14} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex px-5 pt-2.5 gap-1">
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
      <div className="px-5 py-3 bg-slate-50 max-h-56 overflow-y-auto">
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
