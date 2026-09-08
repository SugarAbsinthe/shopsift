import { useEffect, useRef } from "react";
import { Bug, Sparkles } from "lucide-react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import DebugPanel from "./components/DebugPanel";
import { useChat } from "./hooks/useChat";

export default function App() {
  const {
    state,
    dispatch,
    selectConversation,
    newConversation,
    removeConversation,
    sendMessage,
    cancel,
  } = useChat();

  const didInit = useRef(false);
  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    if (!state.currentConvId && state.conversations.length === 0) {
      newConversation();
    }
  }, [state.conversations, state.currentConvId]);

  return (
    <div className="h-screen flex flex-col bg-white">
      {/* Header */}
      <header className="h-13 border-b border-slate-200 flex items-center px-5 shrink-0 bg-gradient-to-r from-indigo-50 to-white">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center">
            <Sparkles size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-800 leading-tight">ShopSift</h1>
            <p className="text-[11px] text-slate-400 leading-tight">智能购物助手 · 随时在线</p>
          </div>
        </div>
        <div className="flex-1" />
        <button
          onClick={() => dispatch({ type: "TOGGLE_DEBUG" })}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full transition-all ${
            state.showDebug
              ? "bg-indigo-100 text-indigo-700 shadow-sm"
              : "text-slate-400 hover:text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Bug size={13} />
          调试
        </button>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          conversations={state.conversations}
          currentConvId={state.currentConvId}
          stage={state.lastStage}
          profile={state.lastProfile}
          onNewChat={newConversation}
          onSelect={selectConversation}
          onDelete={removeConversation}
        />

        <main className="flex-1 flex flex-col min-w-0 bg-slate-50/50">
          {state.error && (
            <div className="bg-red-50 border-b border-red-200 px-5 py-2.5 text-sm text-red-700 flex items-center animate-in">
              <span className="flex-1">{state.error}</span>
              <button
                onClick={() => dispatch({ type: "SET_ERROR", payload: null })}
                className="text-xs text-red-400 hover:text-red-600 underline ml-3"
              >
                关闭
              </button>
            </div>
          )}

          <ChatArea
            messages={state.messages}
            isLoading={state.isLoading}
            onSend={sendMessage}
            onCancel={cancel}
          />

          <DebugPanel
            stage={state.lastStage}
            toolRounds={state.lastToolRounds}
            productContext={state.lastProductContext}
            userProfile={state.lastProfile}
            visible={state.showDebug}
            onClose={() => dispatch({ type: "TOGGLE_DEBUG" })}
          />
        </main>
      </div>
    </div>
  );
}
