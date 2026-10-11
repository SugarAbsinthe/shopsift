import { useReducer, useCallback, useEffect, useRef } from "react";
import type { Conversation, ChatMessage, MessageRecord } from "../types";
import * as api from "../api/client";

/* ---- State ---- */

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
  details?: {
    stage?: string;
    tool_rounds?: number;
    has_product_context?: boolean;
  } | null;
}

interface AppState {
  conversations: Conversation[];
  currentConvId: string | null;
  messages: DisplayMessage[];
  lastStage: string;
  lastProfile: string;
  lastProductContext: string;
  lastToolRounds: number;
  isLoading: boolean;
  showDebug: boolean;
  error: string | null;
}

const initialState: AppState = {
  conversations: [],
  currentConvId: null,
  messages: [],
  lastStage: "discovery",
  lastProfile: "",
  lastProductContext: "",
  lastToolRounds: 0,
  isLoading: false,
  showDebug: false,
  error: null,
};

/* ---- Actions ---- */

type Action =
  | { type: "SET_CONVERSATIONS"; payload: Conversation[] }
  | { type: "ADD_CONVERSATION"; payload: Conversation }
  | { type: "SET_CURRENT_CONV"; payload: string }
  | { type: "SET_MESSAGES"; payload: DisplayMessage[] }
  | { type: "ADD_MESSAGE"; payload: DisplayMessage }
  | { type: "APPEND_TOKEN"; payload: string }
  | { type: "SET_ANSWER"; payload: string }
  | { type: "REMOVE_EMPTY_ASSISTANT" }
  | { type: "UPDATE_LAST_DETAILS"; payload: Record<string, unknown> }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_RESPONSE"; payload: { stage: string; user_profile: string; product_context: string; tool_rounds: number } }
  | { type: "TOGGLE_DEBUG" }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_STATUS"; payload: string | null }
  | { type: "REMOVE_CONVERSATION"; payload: string };

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "SET_CONVERSATIONS":
      return { ...state, conversations: action.payload };
    case "ADD_CONVERSATION":
      return { ...state, conversations: [action.payload, ...state.conversations.filter(c => c.id !== action.payload.id)] };
    case "SET_CURRENT_CONV":
      return { ...initialState, conversations: state.conversations, showDebug: state.showDebug, currentConvId: action.payload };
    case "SET_MESSAGES":
      return { ...state, messages: action.payload };
    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload] };
    case "APPEND_TOKEN": {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content: last.content + action.payload };
      }
      return { ...state, messages: msgs };
    }
    case "SET_ANSWER": {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content: action.payload };
      }
      return { ...state, messages: msgs };
    }
    case "REMOVE_EMPTY_ASSISTANT": {
      const last = state.messages[state.messages.length - 1];
      if (last?.role === "assistant" && !last.content) {
        return { ...state, messages: state.messages.slice(0, -1) };
      }
      return state;
    }
    case "UPDATE_LAST_DETAILS": {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, details: { ...last.details, ...action.payload } };
      }
      return { ...state, messages: msgs };
    }
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_RESPONSE":
      return {
        ...state,
        lastStage: action.payload.stage,
        lastProfile: action.payload.user_profile,
        lastProductContext: action.payload.product_context,
        lastToolRounds: action.payload.tool_rounds,
      };
    case "TOGGLE_DEBUG":
      return { ...state, showDebug: !state.showDebug };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "SET_STATUS":
      return { ...state, error: null };  // status messages are transient, just clear errors
    case "REMOVE_CONVERSATION":
      return {
        ...(state.currentConvId === action.payload ? { ...initialState, showDebug: state.showDebug } : state),
        conversations: state.conversations.filter((c) => c.id !== action.payload),
        currentConvId: state.currentConvId === action.payload ? null : state.currentConvId,
      };
    default:
      return state;
  }
}

/* ---- Hook ---- */

export function useChat() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const abortRef = useRef<AbortController | null>(null);
  const activeConvRef = useRef<string | null>(null);
  const selectionVersionRef = useRef(0);
  const requestVersionRef = useRef(0);

  const invalidateStream = useCallback(() => {
    requestVersionRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  useEffect(() => () => {
    selectionVersionRef.current += 1;
    activeConvRef.current = null;
    invalidateStream();
  }, [invalidateStream]);

  /* Load conversation list on mount */
  useEffect(() => {
    api.listConversations().then((convs) => {
      dispatch({ type: "SET_CONVERSATIONS", payload: convs });
    }).catch(() => {});
  }, []);

  /* Select a conversation */
  const selectConversation = useCallback(async (convId: string) => {
    invalidateStream();
    activeConvRef.current = convId;
    const selectionVersion = ++selectionVersionRef.current;
    dispatch({ type: "SET_CURRENT_CONV", payload: convId });
    try {
      const msgs = await api.getMessages(convId);
      if (selectionVersion !== selectionVersionRef.current || activeConvRef.current !== convId) return;
      dispatch({
        type: "SET_MESSAGES",
        payload: msgs.map((m: MessageRecord) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
          details: m.details ? JSON.parse(m.details) : null,
        })),
      });
    } catch {
      if (selectionVersion !== selectionVersionRef.current || activeConvRef.current !== convId) return;
      dispatch({ type: "SET_MESSAGES", payload: [] });
    }
  }, [invalidateStream]);

  /* Create new conversation */
  const newConversation = useCallback(async () => {
    const selectionVersion = ++selectionVersionRef.current;
    try {
      const conv = await api.createConversation();
      dispatch({ type: "ADD_CONVERSATION", payload: conv });
      if (selectionVersion === selectionVersionRef.current) await selectConversation(conv.id);
    } catch (e: any) {
      if (selectionVersion === selectionVersionRef.current) dispatch({ type: "SET_ERROR", payload: e.message });
    }
  }, [selectConversation]);

  /* Delete conversation */
  const removeConversation = useCallback(async (convId: string) => {
    try {
      await api.deleteConversation(convId);
      if (activeConvRef.current === convId) {
        invalidateStream();
        selectionVersionRef.current += 1;
        activeConvRef.current = null;
      }
      dispatch({ type: "REMOVE_CONVERSATION", payload: convId });
    } catch (e: any) {
      if (activeConvRef.current === convId) dispatch({ type: "SET_ERROR", payload: e.message });
    }
  }, [invalidateStream]);

  /* Send a message (streaming) */
  const sendMessage = useCallback(async (question: string) => {
    const convId = state.currentConvId;
    if (!convId || activeConvRef.current !== convId || !question.trim()) return;

    /* Cancel any in-flight request */
    invalidateStream();
    // A pending history response must not replace messages added by this send.
    selectionVersionRef.current += 1;
    const requestVersion = requestVersionRef.current;
    const controller = new AbortController();
    abortRef.current = controller;
    const isCurrent = () => activeConvRef.current === convId
      && requestVersionRef.current === requestVersion && !controller.signal.aborted;

    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });
    dispatch({ type: "ADD_MESSAGE", payload: { role: "user", content: question } });
    api.saveMessage(convId, "user", question).catch(() => {});

    /* Add empty assistant placeholder for streaming */
    dispatch({ type: "ADD_MESSAGE", payload: { role: "assistant", content: "" } });

    const chatHistory: ChatMessage[] = state.messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({ role: m.role, content: m.content }));

    let answerText = "";

    try {
      await api.sendMessageStream(convId, question, chatHistory, {
        onToken(content) {
          if (!isCurrent()) return;
          answerText += content;
          dispatch({ type: "APPEND_TOKEN", payload: content });
        },
        onStatus(message) {
          if (!isCurrent()) return;
          dispatch({ type: "SET_STATUS", payload: message });
        },
        onStage(stage) {
          if (!isCurrent()) return;
          dispatch({ type: "UPDATE_LAST_DETAILS", payload: { stage } });
        },
        onToolStart(tools) {
          if (!isCurrent()) return;
          dispatch({ type: "SET_STATUS", payload: `正在调用：${tools.join(", ")}` });
        },
        onToolEnd(_tools, statuses) {
          if (!isCurrent()) return;
          const failed = statuses.some((status) => status === "error");
          dispatch({
            type: "SET_STATUS",
            payload: failed ? "部分工具执行失败，正在整理已有结果" : "工具执行完成，正在整理结果",
          });
        },
        onDone(meta) {
          if (!isCurrent()) return;
          answerText = meta.answer;
          dispatch({ type: "SET_ANSWER", payload: answerText });
          if (answerText) {
            api.saveMessage(convId, "assistant", answerText, {
              run_id: meta.run_id,
              stage: meta.stage,
              tool_rounds: meta.tool_rounds,
              agent_rounds: meta.agent_rounds,
              stop_reason: meta.stop_reason,
              latency_ms: meta.latency_ms,
              request_id: meta.request_id,
              llm_calls: meta.llm_calls,
              llm_latency_ms: meta.llm_latency_ms,
              llm_retries: meta.llm_retries,
              input_tokens: meta.input_tokens,
              output_tokens: meta.output_tokens,
              total_tokens: meta.total_tokens,
              retrieval_triggered: meta.retrieval_triggered,
              cache_hit: meta.cache_hit,
              requested_tools: meta.requested_tools,
              executed_tools: meta.executed_tools,
              tool_errors: meta.tool_errors,
            }).catch(() => {});
          } else {
            dispatch({ type: "REMOVE_EMPTY_ASSISTANT" });
          }
          dispatch({
            type: "UPDATE_LAST_DETAILS",
            payload: { stage: meta.stage, tool_rounds: meta.tool_rounds },
          });
          dispatch({
            type: "SET_RESPONSE",
            payload: {
              stage: meta.stage,
              user_profile: meta.user_profile,
              product_context: meta.product_context,
              tool_rounds: meta.tool_rounds,
            },
          });
        },
        onError(message) {
          if (!isCurrent()) return;
          if (answerText === "") {
            dispatch({ type: "REMOVE_EMPTY_ASSISTANT" });
          }
          dispatch({ type: "SET_ERROR", payload: message });
        },
      }, controller.signal);

      /* Refresh conversation list (title may have changed) */
      if (!isCurrent()) return;
      const convs = await api.listConversations();
      if (isCurrent()) dispatch({ type: "SET_CONVERSATIONS", payload: convs });
    } catch (e: any) {
      if (!isCurrent()) return;
      if (answerText === "") {
        dispatch({ type: "REMOVE_EMPTY_ASSISTANT" });
      }
      if (e?.name !== "AbortError") {
        dispatch({ type: "SET_ERROR", payload: e.message });
      }
    } finally {
      if (isCurrent()) {
        abortRef.current = null;
        requestVersionRef.current += 1;
        dispatch({ type: "SET_LOADING", payload: false });
      }
    }
  }, [state.currentConvId, state.messages, invalidateStream]);

  const cancel = useCallback(() => {
    invalidateStream();
    dispatch({ type: "REMOVE_EMPTY_ASSISTANT" });
    dispatch({ type: "SET_LOADING", payload: false });
  }, [invalidateStream]);

  return {
    state,
    dispatch,
    selectConversation,
    newConversation,
    removeConversation,
    sendMessage,
    cancel,
  };
}
