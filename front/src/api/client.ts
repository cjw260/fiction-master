import type {
  Book,
  ChatEvent,
  ChatScope,
  Conversation,
  ConversationDetail,
  ConversationUpdate,
  HealthStatus,
  IngestionJob,
  ModelStatus,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}/api/v1${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: response.statusText }));
    throw new Error(error.message ?? `API request failed (${response.status})`);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export const fictionApi = {
  health: () => api<HealthStatus>("/health"),
  models: () => api<ModelStatus>("/system/models"),
  books: () => api<Book[]>("/books"),
  syncLibrary: () =>
    api<{ job_id: string; status: "queued" }>("/library/sync", { method: "POST" }),
  reindexBook: (bookId: string) =>
    api<{ job_id: string; status: "queued" }>(`/books/${bookId}/reindex`, { method: "POST" }),
  job: (jobId: string) => api<IngestionJob>(`/jobs/${jobId}`),
  conversations: () => api<Conversation[]>("/conversations"),
  conversation: (id: string) => api<ConversationDetail>(`/conversations/${id}`),
  createConversation: (scope: ChatScope = { mode: "auto" }) =>
    api<Conversation>("/conversations", { method: "POST", body: JSON.stringify({ scope }) }),
  updateConversation: (id: string, update: ConversationUpdate) =>
    api<Conversation>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify(update),
    }),
  deleteConversation: (id: string) =>
    api<void>(`/conversations/${id}`, { method: "DELETE" }),
};

export async function* streamMessage(
  conversationId: string,
  content: string,
  scope?: ChatScope,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(
    `${API_BASE}/api/v1/conversations/${conversationId}/messages/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ content, scope }),
      signal,
    },
  );
  if (!response.ok || !response.body) {
    const error = await response.json().catch(() => ({ message: response.statusText }));
    throw new Error(error.message ?? `Chat stream failed (${response.status})`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += value ?? "";
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = block.match(/^event:\s*(.+)$/m)?.[1];
      const data = block.match(/^data:\s*(.+)$/m)?.[1];
      if (event && data) yield { event, data: JSON.parse(data) } as ChatEvent;
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
}
