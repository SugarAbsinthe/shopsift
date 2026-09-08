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
    return res.json();
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
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conv_id: convId,
      question,
      chat_history: chatHistory,
    }),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6));
        switch (eventType) {
          case "token":
            callbacks.onToken(data.content);
            break;
          case "status":
            callbacks.onStatus(data.message);
            break;
          case "stage":
            callbacks.onStage(data.stage);
            break;
          case "tool_start":
            callbacks.onToolStart?.(data.tools || []);
            break;
          case "tool_end":
            callbacks.onToolEnd?.(data.tools || [], data.statuses || []);
            break;
          case "done":
            callbacks.onDone(data);
            break;
          case "error":
            callbacks.onError(data.message);
            break;
        }
        eventType = "";
      }
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
