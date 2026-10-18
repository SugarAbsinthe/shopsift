/* ---- Thin fetch wrapper for ShopSift API ---- */

const BASE = "/api";
const DEFAULT_TIMEOUT = 90_000; // 90s — agent init + LLM call can be slow

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);

  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/* ---- Chat ---- */

export async function sendMessage(
  convId: string,
  question: string,
  chatHistory: { role: string; content: string }[],
) {
  return request<import("../types").ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      conv_id: convId,
      question,
      chat_history: chatHistory,
    }),
  });
}

export interface StreamCallbacks {
  onToken: (content: string) => void;
  onStatus: (message: string) => void;
  onStage: (stage: string) => void;
  onToolStart?: (tools: string[]) => void;
  onToolEnd?: (tools: string[], statuses: string[]) => void;
  onDone: (meta: import("../types").StreamDoneMeta) => void;
  onError: (message: string) => void;
}

export async function sendMessageStream(
  convId: string,
  question: string,
  chatHistory: { role: string; content: string }[],
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });
  if (signal?.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, DEFAULT_TIMEOUT);
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conv_id: convId, question, chat_history: chatHistory }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    if (!res.headers.get("content-type")?.includes("text/event-stream")) {
      throw new Error("服务未返回对话流，请检查后端连接后重试");
    }
    reader = res.body?.getReader();
    if (!reader) throw new Error("服务没有返回响应内容");

    const dispatchEvent = (block: string): boolean => {
      let eventType = "";
      const lines: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        if (line.startsWith("data:")) lines.push(line.slice(5).replace(/^ /, ""));
      }
      if (!lines.length) return false;
      if (!["token", "status", "stage", "tool_start", "tool_end", "done", "error"].includes(eventType)) return false;
      let data;
      try {
        data = JSON.parse(lines.join("\n"));
        if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error();
      } catch {
        throw new Error("对话流数据格式错误，请重试");
      }
      switch (eventType) {
        case "token":
          if (typeof data.content !== "string") throw new Error("对话流内容格式错误");
          callbacks.onToken(data.content);
          break;
        case "status": callbacks.onStatus(data.message); break;
        case "stage": callbacks.onStage(data.stage); break;
        case "tool_start": callbacks.onToolStart?.(data.tools || []); break;
        case "tool_end": callbacks.onToolEnd?.(data.tools || [], data.statuses || []); break;
        case "done":
          if (typeof data.answer !== "string") throw new Error("对话结果格式错误");
          callbacks.onDone(data);
          return true;
        case "error":
          callbacks.onError(data.message || "对话执行失败，请重试");
          return true;
      }
      return false;
    };

    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        if (dispatchEvent(block)) return;
      }
      if (done) {
        if (buffer.trim() && dispatchEvent(buffer)) return;
        throw new Error("对话连接提前中断，回复尚未完成，请重试");
      }
    }
  } catch (error) {
    if (timedOut && !signal?.aborted) throw new Error("对话响应超时，请稍后重试");
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
    if (reader) {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }
}

/* ---- Conversations ---- */

export async function listConversations() {
  return request<import("../types").Conversation[]>("/conversations");
}

export async function createConversation(title = "新对话", model = "") {
  return request<import("../types").Conversation>("/conversations", {
    method: "POST",
    body: JSON.stringify({ title, model }),
  });
}

export async function deleteConversation(convId: string) {
  return request<{ deleted: string }>(`/conversations/${convId}`, {
    method: "DELETE",
  });
}

/* ---- Messages ---- */

export async function getMessages(convId: string) {
  return request<import("../types").MessageRecord[]>(
    `/conversations/${convId}/messages`,
  );
}

export async function saveMessage(
  convId: string,
  role: "user" | "assistant",
  content: string,
  details?: Record<string, unknown>,
) {
  return request<import("../types").MessageRecord>(
    `/conversations/${convId}/messages`,
    {
      method: "POST",
      body: JSON.stringify({ role, content, details }),
    },
  );
}

/* ---- Health ---- */

export async function healthCheck() {
  return request<import("../types").HealthResponse>("/health");
}
