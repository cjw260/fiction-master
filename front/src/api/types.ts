export type BookStatus =
  | "discovered"
  | "indexing"
  | "ready"
  | "error"
  | "missing"
  | "duplicate";

export interface Book {
  id: string;
  relative_path: string;
  file_format: string;
  title: string;
  author: string | null;
  file_size: number;
  word_count: number;
  chapter_count: number;
  chunk_count: number;
  status: BookStatus;
  error: string | null;
  duplicate_of: string | null;
  updated_at: string;
  graph_status: "disabled" | "pending" | "queued" | "indexing" | "ready" | "error" | "paused" | "superseded" | "removing";
  graph_error: string | null;
}

export interface IngestionJob {
  id: string;
  job_type: "sync" | "reindex";
  target_book_id: string | null;
  status: "queued" | "running" | "succeeded" | "succeeded_with_errors" | "failed";
  total_files: number;
  completed_files: number;
  current_file: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Citation {
  id: string;
  ordinal: number;
  book_id: string | null;
  book_title: string;
  chapter_title: string;
  chunk_id: string;
  excerpt: string;
  start_offset: number;
  end_offset: number;
}

export interface AnswerMetrics {
  sources: {
    books: number;
    chapters: number;
    evidence: number;
  };
  retrieval: {
    rounds: number;
    dense: boolean;
    bm25: boolean;
    rerank: boolean;
    graph: boolean;
    graph_mode: string | null;
    graph_queries: number;
    graph_fallback: boolean;
  };
  calls: {
    chat: number;
    embedding: number;
    rerank: number;
    graph: number;
  };
  timing: {
    first_token_ms: number | null;
    total_ms: number;
  };
  tokens: {
    input: number;
    output: number;
  };
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  status: "streaming" | "completed" | "failed" | "cancelled";
  content: string;
  model: string | null;
  usage: Record<string, unknown>;
  metrics: AnswerMetrics | null;
  latency_ms: number | null;
  error: string | null;
  created_at: string;
  citations: Citation[];
}

export interface Conversation {
  id: string;
  title: string;
  scope_mode: "auto" | "books";
  book_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface HealthStatus {
  status: "ok" | "degraded";
  database: boolean;
  vector_store: boolean;
  models_configured: boolean;
  graph_enabled: boolean;
  graph_available: boolean;
  version: string;
}

export interface ModelStatus {
  chat_model: string;
  embedding_model: string;
  rerank_model: string;
  chat_configured: boolean;
  embedding_configured: boolean;
  rerank_configured: boolean;
}

export type ChatScope =
  | { mode: "auto"; book_ids?: never }
  | { mode: "books"; book_ids: string[] };

export interface ConversationUpdate {
  title?: string;
  scope?: ChatScope;
}

export type ChatEvent =
  | { event: "status"; data: { run_id: string; stage: string; detail: string; [key: string]: unknown } }
  | { event: "delta"; data: { run_id: string; text: string } }
  | { event: "citation"; data: Citation }
  | { event: "done"; data: { run_id: string; message: Message } }
  | {
      event: "error";
      data: { run_id: string; code: string; message: string; retryable: boolean };
    };
