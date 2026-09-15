import React, { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  BookOpen,
  Bot,
  Check,
  Library,
  LoaderCircle,
  MessageSquare,
  Paperclip,
  Pencil,
  Plus,
  RefreshCw,
  RotateCcw,
  Send,
  Settings,
  Square,
  Trash2,
  X,
} from "lucide-react";

import { fictionApi, streamMessage } from "./api/client";
import type {
  AnswerMetrics,
  Book,
  BookStatus,
  ChatScope,
  Citation,
  Conversation,
  HealthStatus,
  IngestionJob,
  Message,
  ModelStatus,
} from "./api/types";

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  conversation_id: "",
  role: "assistant",
  status: "completed",
  content:
    "你好，我是**小说大师**。我会依据 fiction 目录中已经完成索引的小说原文，回答剧情、人物和世界观问题，并标注可核对的原文出处。今天想聊点什么？",
  model: null,
  usage: {},
  metrics: null,
  latency_ms: null,
  error: null,
  created_at: new Date(0).toISOString(),
  citations: [],
};

const BOOK_STATUS: Record<BookStatus, { label: string; className: string }> = {
  discovered: { label: "待索引", className: "bg-[#e3dcd1] text-[#6b655b]" },
  indexing: { label: "索引中", className: "bg-amber-100 text-amber-800" },
  ready: { label: "已索引", className: "bg-emerald-100 text-emerald-800" },
  error: { label: "失败", className: "bg-red-100 text-red-700" },
  missing: { label: "已缺失", className: "bg-slate-200 text-slate-600" },
  duplicate: { label: "重复", className: "bg-blue-100 text-blue-700" },
};

const GRAPH_STATUS: Record<Book["graph_status"], string> = {
  disabled: "图谱关闭",
  pending: "图谱待构建",
  queued: "图谱排队中",
  indexing: "图谱构建中",
  ready: "关系图谱就绪",
  error: "图谱降级",
  paused: "图谱已暂停",
  superseded: "图谱已更新",
  removing: "图谱清理中",
};

const RUNNING_JOB_STATUSES = new Set(["queued", "running"]);

function formatWordCount(value: number): string {
  if (!value) return "—";
  if (value < 10_000) return `${value.toLocaleString("zh-CN")}字`;
  const digits = value >= 100_000 ? 0 : 1;
  return `${(value / 10_000).toFixed(digits)}万字`;
}

function friendlyError(message: string): string {
  const normalized = message.toLowerCase();
  if (normalized.includes("allocationquota.freetieronly") || normalized.includes("free quota exhausted")) {
    return "百炼免费额度已用完，请充值或关闭“仅使用免费额度”后重试。";
  }
  if (normalized.includes("api key is not configured")) {
    return "模型 API Key 尚未配置，请填写根目录 .env 后重启后端。";
  }
  if (normalized.includes("failed to fetch") || normalized.includes("networkerror")) {
    return "无法连接小说大师后端，请确认后端已启动。";
  }
  if (message.length > 220) return `${message.slice(0, 217)}...`;
  return message;
}

function messageDraft(
  id: string,
  conversationId: string,
  role: "user" | "assistant",
  content: string,
  status: Message["status"],
): Message {
  return {
    id,
    conversation_id: conversationId,
    role,
    status,
    content,
    model: null,
    usage: {},
    metrics: null,
    latency_ms: null,
    error: null,
    created_at: new Date().toISOString(),
    citations: [],
  };
}

function RichText({
  content,
  role,
  citations,
  onCitation,
}: {
  content: string;
  role: Message["role"];
  citations: Citation[];
  onCitation: (citation: Citation) => void;
}) {
  const paragraphs = content.split("\n");
  return paragraphs.map((paragraph, paragraphIndex) => (
    <React.Fragment key={`${paragraphIndex}-${paragraph.slice(0, 12)}`}>
      {paragraph.split(/(\*\*.*?\*\*|\[\d+\])/g).map((part, partIndex) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return (
            <strong
              key={partIndex}
              className={role === "user" ? "font-semibold text-white" : "font-bold text-foreground"}
            >
              {part.slice(2, -2)}
            </strong>
          );
        }
        const ordinal = part.match(/^\[(\d+)\]$/)?.[1];
        const citation = ordinal
          ? citations.find((item) => item.ordinal === Number(ordinal))
          : undefined;
        if (citation) {
          return (
            <button
              key={partIndex}
              type="button"
              onClick={() => onCitation(citation)}
              className="mx-0.5 align-super text-[11px] font-sans font-bold text-primary hover:underline"
              aria-label={`查看引用 ${citation.ordinal}`}
            >
              [{citation.ordinal}]
            </button>
          );
        }
        return <React.Fragment key={partIndex}>{part}</React.Fragment>;
      })}
      {paragraphIndex < paragraphs.length - 1 && <br className="my-2" />}
    </React.Fragment>
  ));
}

function configurationLabel(configured: boolean): string {
  return configured ? "已配置" : "未配置";
}

function formatDuration(milliseconds: number | null): string {
  if (milliseconds === null) return "—";
  if (milliseconds < 1000) return `${milliseconds}毫秒`;
  return `${(milliseconds / 1000).toFixed(milliseconds < 10_000 ? 1 : 0)}秒`;
}

function AnswerMetricsSummary({ metrics }: { metrics: AnswerMetrics }) {
  const retrievalStages = [
    metrics.retrieval.dense ? "Dense" : null,
    metrics.retrieval.bm25 ? "BM25" : null,
    metrics.retrieval.rerank ? "Rerank" : null,
    metrics.retrieval.graph
      ? `LightRAG${metrics.retrieval.graph_mode ? `(${metrics.retrieval.graph_mode})` : ""}`
      : null,
  ].filter(Boolean);
  const retrieval = metrics.retrieval.rounds
    ? `${metrics.retrieval.rounds}轮 · ${retrievalStages.join(" + ") || "无可用检索器"}`
    : "未检索";
  const rows = [
    [
      "来源",
      `${metrics.sources.books}本小说 · ${metrics.sources.chapters}个章节 · ${metrics.sources.evidence}条证据`,
    ],
    ["检索", retrieval],
    [
      "模型调用",
      `Chat ${metrics.calls.chat}次 · Embedding ${metrics.calls.embedding}次 · Rerank ${metrics.calls.rerank}次 · Graph ${metrics.calls.graph}次`,
    ],
    [
      "响应时间",
      `首字 ${formatDuration(metrics.timing.first_token_ms)} · 总计 ${formatDuration(metrics.timing.total_ms)}`,
    ],
    [
      "Chat Token",
      `输入 ${metrics.tokens.input.toLocaleString("zh-CN")} · 输出 ${metrics.tokens.output.toLocaleString("zh-CN")}`,
    ],
  ];

  return (
    <dl className="mt-4 space-y-1.5 border-t border-border/70 pt-3 font-sans text-[11px] leading-relaxed text-muted-foreground">
      {rows.map(([label, value]) => (
        <div key={label} className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-2">
          <dt className="font-medium text-sidebar-foreground/75">{label}</dt>
          <dd className="min-w-0 break-words">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function App() {
  const [books, setBooks] = useState<Book[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [scopeMode, setScopeMode] = useState<"auto" | "books">("auto");
  const [selectedBookIds, setSelectedBookIds] = useState<string[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [sending, setSending] = useState(false);
  const [streamDetail, setStreamDetail] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<IngestionJob | null>(null);
  const [scopeOpen, setScopeOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<ModelStatus | null>(null);
  const [expandedCitationId, setExpandedCitationId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const streamControllerRef = useRef<AbortController | null>(null);
  const conversationRequestRef = useRef(0);
  const jobPollRef = useRef(0);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const scopePickerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const [bookData, conversationData] = await Promise.all([
          fictionApi.books(),
          fictionApi.conversations(),
        ]);
        if (cancelled) return;
        setBooks(bookData);
        setConversations(conversationData);
        if (conversationData[0]) {
          const detail = await fictionApi.conversation(conversationData[0].id);
          if (cancelled) return;
          setActiveConversationId(detail.id);
          setMessages(detail.messages);
          setScopeMode(detail.scope_mode);
          setSelectedBookIds(detail.book_ids);
        }
      } catch (error) {
        if (!cancelled) {
          setNotice(error instanceof Error ? error.message : "无法连接小说大师后端");
        }
      } finally {
        if (!cancelled) setInitialLoading(false);
      }

      try {
        const [healthData, modelData] = await Promise.all([
          fictionApi.health(),
          fictionApi.models(),
        ]);
        if (!cancelled) {
          setHealth(healthData);
          setModels(modelData);
        }
      } catch {
        // The primary API error is already surfaced above; settings can retry independently.
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
      streamControllerRef.current?.abort();
      jobPollRef.current += 1;
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: sending ? "smooth" : "auto" });
  }, [messages, sending]);

  useEffect(() => {
    function closePicker(event: MouseEvent) {
      if (
        scopePickerRef.current &&
        event.target instanceof Node &&
        !scopePickerRef.current.contains(event.target)
      ) {
        setScopeOpen(false);
      }
    }
    document.addEventListener("mousedown", closePicker);
    return () => document.removeEventListener("mousedown", closePicker);
  }, []);

  const readyBooks = books.filter((book) => book.status === "ready");
  const selectedReadyBooks = readyBooks.filter((book) => selectedBookIds.includes(book.id));
  const displayedMessages = messages.length ? messages : [WELCOME_MESSAGE];
  const jobRunning = activeJob ? RUNNING_JOB_STATUSES.has(activeJob.status) : false;

  async function refreshBooks() {
    const nextBooks = await fictionApi.books();
    setBooks(nextBooks);
    return nextBooks;
  }

  async function refreshConversations() {
    const nextConversations = await fictionApi.conversations();
    setConversations(nextConversations);
    return nextConversations;
  }

  async function refreshSystemStatus() {
    try {
      const [healthData, modelData] = await Promise.all([
        fictionApi.health(),
        fictionApi.models(),
      ]);
      setHealth(healthData);
      setModels(modelData);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "无法读取后端状态");
    }
  }

  async function openConversation(id: string) {
    streamControllerRef.current?.abort();
    const requestId = ++conversationRequestRef.current;
    setActiveConversationId(id);
    setConversationLoading(true);
    setNotice(null);
    setExpandedCitationId(null);
    try {
      const detail = await fictionApi.conversation(id);
      if (requestId !== conversationRequestRef.current) return;
      setMessages(detail.messages);
      setScopeMode(detail.scope_mode);
      setSelectedBookIds(detail.book_ids);
    } catch (error) {
      if (requestId === conversationRequestRef.current) {
        setNotice(error instanceof Error ? error.message : "读取对话失败");
      }
    } finally {
      if (requestId === conversationRequestRef.current) setConversationLoading(false);
    }
  }

  async function createConversation() {
    if (creatingConversation) return null;
    streamControllerRef.current?.abort();
    conversationRequestRef.current += 1;
    setCreatingConversation(true);
    setNotice(null);
    try {
      const conversation = await fictionApi.createConversation({ mode: "auto" });
      setConversations((current) => [conversation, ...current]);
      setActiveConversationId(conversation.id);
      setMessages([]);
      setScopeMode("auto");
      setSelectedBookIds([]);
      setExpandedCitationId(null);
      return conversation;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "创建对话失败");
      return null;
    } finally {
      setCreatingConversation(false);
    }
  }

  async function saveRename(id: string) {
    const title = renameValue.trim();
    if (!title) return;
    try {
      const updated = await fictionApi.updateConversation(id, { title });
      setConversations((current) =>
        current.map((conversation) => (conversation.id === id ? updated : conversation)),
      );
      setRenamingId(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "重命名失败");
    }
  }

  async function deleteConversation(id: string) {
    if (!window.confirm("确定删除这个对话及其全部消息吗？")) return;
    if (id === activeConversationId) streamControllerRef.current?.abort();
    try {
      await fictionApi.deleteConversation(id);
      const remaining = conversations.filter((conversation) => conversation.id !== id);
      setConversations(remaining);
      if (id === activeConversationId) {
        setActiveConversationId(null);
        setMessages([]);
        setScopeMode("auto");
        setSelectedBookIds([]);
        if (remaining[0]) await openConversation(remaining[0].id);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "删除对话失败");
    }
  }

  async function applyScope(nextMode: "auto" | "books", nextBookIds: string[]) {
    const normalizedMode = nextMode === "books" && nextBookIds.length ? "books" : "auto";
    const normalizedIds = normalizedMode === "books" ? nextBookIds : [];
    const previousMode = scopeMode;
    const previousIds = selectedBookIds;
    setScopeMode(normalizedMode);
    setSelectedBookIds(normalizedIds);
    if (!activeConversationId) return;
    const scope: ChatScope =
      normalizedMode === "books"
        ? { mode: "books", book_ids: normalizedIds }
        : { mode: "auto" };
    try {
      const updated = await fictionApi.updateConversation(activeConversationId, { scope });
      setConversations((current) =>
        current.map((conversation) =>
          conversation.id === activeConversationId ? updated : conversation,
        ),
      );
    } catch (error) {
      setScopeMode(previousMode);
      setSelectedBookIds(previousIds);
      setNotice(error instanceof Error ? error.message : "保存检索范围失败");
    }
  }

  function toggleBookScope(bookId: string) {
    const nextIds = selectedBookIds.includes(bookId)
      ? selectedBookIds.filter((id) => id !== bookId)
      : [...selectedBookIds, bookId];
    void applyScope(nextIds.length ? "books" : "auto", nextIds);
  }

  async function pollJob(jobId: string) {
    const pollId = ++jobPollRef.current;
    while (pollId === jobPollRef.current) {
      try {
        const job = await fictionApi.job(jobId);
        if (pollId !== jobPollRef.current) return;
        setActiveJob(job);
        await refreshBooks();
        if (!RUNNING_JOB_STATUSES.has(job.status)) {
          if (job.error) setNotice(job.error);
          window.setTimeout(() => {
            if (pollId === jobPollRef.current) setActiveJob(null);
          }, 5000);
          return;
        }
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "读取索引任务失败");
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
  }

  async function syncLibrary() {
    if (jobRunning) return;
    setNotice(null);
    try {
      const accepted = await fictionApi.syncLibrary();
      setActiveJob({
        id: accepted.job_id,
        job_type: "sync",
        target_book_id: null,
        status: accepted.status,
        total_files: 0,
        completed_files: 0,
        current_file: null,
        error: null,
        created_at: new Date().toISOString(),
        started_at: null,
        finished_at: null,
      });
      void pollJob(accepted.job_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "启动同步失败");
    }
  }

  async function reindexBook(bookId: string) {
    if (jobRunning) return;
    setNotice(null);
    try {
      const accepted = await fictionApi.reindexBook(bookId);
      setActiveJob({
        id: accepted.job_id,
        job_type: "reindex",
        target_book_id: bookId,
        status: accepted.status,
        total_files: 1,
        completed_files: 0,
        current_file: null,
        error: null,
        created_at: new Date().toISOString(),
        started_at: null,
        finished_at: null,
      });
      void pollJob(accepted.job_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "启动重建失败");
    }
  }

  function currentScope(): ChatScope {
    return scopeMode === "books" && selectedBookIds.length
      ? { mode: "books", book_ids: selectedBookIds }
      : { mode: "auto" };
  }

  async function sendContent(rawContent: string) {
    const content = rawContent.trim();
    if (!content || sending) return;
    setSending(true);
    setNotice(null);
    setStreamDetail("正在创建回答...");

    let conversationId = activeConversationId;
    if (!conversationId) {
      try {
        const conversation = await fictionApi.createConversation(currentScope());
        conversationId = conversation.id;
        setActiveConversationId(conversation.id);
        setConversations((current) => [conversation, ...current]);
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "创建对话失败");
        setSending(false);
        setStreamDetail("");
        return;
      }
    }

    setInputValue("");
    const localSuffix = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const userMessage = messageDraft(
      `local-user-${localSuffix}`,
      conversationId,
      "user",
      content,
      "completed",
    );
    const assistantId = `local-assistant-${localSuffix}`;
    const assistantMessage = messageDraft(
      assistantId,
      conversationId,
      "assistant",
      "",
      "streaming",
    );
    setMessages((current) => [...current, userMessage, assistantMessage]);

    const controller = new AbortController();
    streamControllerRef.current = controller;
    try {
      for await (const event of streamMessage(
        conversationId,
        content,
        currentScope(),
        controller.signal,
      )) {
        if (event.event === "status") {
          setStreamDetail(event.data.detail);
        } else if (event.event === "delta") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + event.data.text }
                : message,
            ),
          );
        } else if (event.event === "citation") {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, citations: [...message.citations, event.data] }
                : message,
            ),
          );
        } else if (event.event === "done") {
          setMessages((current) =>
            current.map((message) => (message.id === assistantId ? event.data.message : message)),
          );
        } else if (event.event === "error") {
          setNotice(event.data.message);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    id: event.data.run_id,
                    status: "failed",
                    content: message.content || "回答生成失败。",
                    error: event.data.message,
                  }
                : message,
            ),
          );
        }
      }
      await refreshConversations();
    } catch (error) {
      const aborted = error instanceof DOMException && error.name === "AbortError";
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                status: aborted ? "cancelled" : "failed",
                content: message.content || (aborted ? "已停止生成。" : "回答生成失败。"),
                error: aborted
                  ? "已由用户停止"
                  : error instanceof Error
                    ? error.message
                    : "回答生成失败",
              }
            : message,
        ),
      );
      if (!aborted) setNotice(error instanceof Error ? error.message : "回答生成失败");
    } finally {
      if (streamControllerRef.current === controller) streamControllerRef.current = null;
      setSending(false);
      setStreamDetail("");
    }
  }

  function handleSend(event?: React.FormEvent) {
    event?.preventDefault();
    void sendContent(inputValue);
  }

  function retryMessage(index: number) {
    const previousUser = [...messages.slice(0, index)]
      .reverse()
      .find((message) => message.role === "user");
    if (previousUser) void sendContent(previousUser.content);
  }

  function stopGeneration() {
    setStreamDetail("正在停止生成...");
    streamControllerRef.current?.abort();
  }

  return (
    <div className="flex h-screen w-full bg-background text-foreground overflow-hidden">
      <aside className="relative w-[280px] sm:w-[320px] bg-sidebar border-r border-border flex flex-col shrink-0 transition-all">
        <div className="p-4 sm:p-6 pb-2">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center text-primary-foreground shadow-sm">
              <BookOpen size={18} />
            </div>
            <h1 className="font-serif font-bold text-xl tracking-wide text-sidebar-foreground">
              小说大师
            </h1>
          </div>

          <button
            type="button"
            onClick={() => void createConversation()}
            disabled={creatingConversation}
            className="w-full flex items-center gap-2 bg-background hover:bg-muted/50 border border-border text-sidebar-foreground px-4 py-2.5 rounded-lg transition-colors text-sm font-medium shadow-sm disabled:opacity-60"
          >
            {creatingConversation ? (
              <LoaderCircle size={16} className="text-primary animate-spin" />
            ) : (
              <Plus size={16} className="text-primary" />
            )}
            开启新对话
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between gap-2 mb-3 px-1 text-muted-foreground">
            <div className="flex items-center gap-2 min-w-0">
              <Library size={16} />
              <h2 className="text-xs font-semibold uppercase tracking-wider truncate">
                数据库藏书 (RAG)
              </h2>
            </div>
            <button
              type="button"
              onClick={() => void syncLibrary()}
              disabled={jobRunning}
              className="p-1 rounded hover:text-primary hover:bg-muted/60 disabled:opacity-50"
              title="同步 fiction 目录"
              aria-label="同步 fiction 目录"
            >
              <RefreshCw size={15} className={jobRunning ? "animate-spin" : ""} />
            </button>
          </div>

          {activeJob && (
            <div className="mb-3 px-3 py-2 rounded-lg border border-border bg-background/70 text-xs text-muted-foreground">
              <div className="flex items-center gap-2 text-sidebar-foreground font-medium">
                {jobRunning && <LoaderCircle size={13} className="animate-spin text-primary" />}
                <span>
                  {jobRunning
                    ? activeJob.job_type === "sync"
                      ? "正在同步藏书"
                      : "正在重建索引"
                    : activeJob.status.startsWith("succeeded")
                      ? "索引任务已完成"
                      : "索引任务失败"}
                </span>
              </div>
              {jobRunning && (
                <div className="mt-1 truncate" title={activeJob.current_file ?? undefined}>
                  {activeJob.current_file || "准备中"}
                  {activeJob.total_files > 0 &&
                    ` · ${activeJob.completed_files}/${activeJob.total_files}`}
                </div>
              )}
            </div>
          )}

          <div className="space-y-1.5">
            {initialLoading ? (
              <div className="flex items-center gap-2 px-3 py-4 text-sm text-muted-foreground">
                <LoaderCircle size={15} className="animate-spin" /> 正在读取藏书
              </div>
            ) : books.length === 0 ? (
              <button
                type="button"
                onClick={() => void syncLibrary()}
                className="w-full px-3 py-4 rounded-lg border border-dashed border-border text-sm text-muted-foreground hover:text-primary hover:border-primary/40"
              >
                fiction 中暂无藏书，点击同步
              </button>
            ) : (
              books.map((novel) => {
                const status = BOOK_STATUS[novel.status];
                const canReindex = !["indexing", "missing", "duplicate"].includes(novel.status);
                return (
                  <div
                    key={novel.id}
                    className="group flex flex-col p-3 rounded-lg hover:bg-muted/50 transition-colors border border-transparent hover:border-border"
                    title={
                      novel.error
                        ? friendlyError(novel.error)
                        : novel.graph_error
                          ? friendlyError(novel.graph_error)
                          : novel.relative_path
                    }
                  >
                    <div className="flex justify-between items-start gap-2 mb-1">
                      <span className="font-serif font-semibold text-sidebar-foreground group-hover:text-primary transition-colors truncate">
                        {novel.title}
                      </span>
                      <div className="flex items-center gap-1 shrink-0">
                        {canReindex && (
                          <button
                            type="button"
                            onClick={() => void reindexBook(novel.id)}
                            disabled={jobRunning}
                            className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-0.5 text-muted-foreground hover:text-primary disabled:opacity-30"
                            title={`重建《${novel.title}》索引`}
                            aria-label={`重建《${novel.title}》索引`}
                          >
                            <RotateCcw size={12} />
                          </button>
                        )}
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${status.className}`}>
                          {status.label}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-xs text-muted-foreground gap-2">
                      <span className="truncate">{novel.author || "作者未知"}</span>
                      <span className="shrink-0">{formatWordCount(novel.word_count)}</span>
                    </div>
                    {novel.graph_status !== "disabled" && (
                      <div
                        className={`mt-1 text-[10px] ${
                          novel.graph_status === "ready"
                            ? "text-emerald-700"
                            : novel.graph_status === "error"
                              ? "text-amber-700"
                              : "text-muted-foreground"
                        }`}
                      >
                        {GRAPH_STATUS[novel.graph_status]}
                      </div>
                    )}
                    {novel.error && novel.status !== "ready" && (
                      <div className="mt-1.5 text-[11px] text-red-700 truncate">
                        {friendlyError(novel.error)}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>

          <div className="mt-8 mb-3 px-1 text-muted-foreground flex items-center gap-2">
            <MessageSquare size={16} />
            <h2 className="text-xs font-semibold uppercase tracking-wider">最近对话</h2>
          </div>
          <div className="space-y-1">
            {!initialLoading && conversations.length === 0 && (
              <div className="px-3 py-2 text-sm text-muted-foreground">还没有历史对话</div>
            )}
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`group flex items-center rounded-md transition-colors ${
                  activeConversationId === conversation.id ? "bg-muted/70" : "hover:bg-muted/50"
                }`}
              >
                {renamingId === conversation.id ? (
                  <div className="flex items-center gap-1 w-full px-2 py-1">
                    <input
                      value={renameValue}
                      onChange={(event) => setRenameValue(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") void saveRename(conversation.id);
                        if (event.key === "Escape") setRenamingId(null);
                      }}
                      autoFocus
                      maxLength={256}
                      className="min-w-0 flex-1 bg-background border border-border rounded px-2 py-1 text-sm outline-none focus:border-primary"
                    />
                    <button
                      type="button"
                      onClick={() => void saveRename(conversation.id)}
                      className="p-1 text-primary"
                      aria-label="保存名称"
                    >
                      <Check size={14} />
                    </button>
                  </div>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={() => void openConversation(conversation.id)}
                      className="min-w-0 flex-1 text-left truncate px-3 py-2 text-sm text-sidebar-foreground"
                    >
                      {conversation.title}
                    </button>
                    <div className="flex items-center pr-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100">
                      <button
                        type="button"
                        onClick={() => {
                          setRenamingId(conversation.id);
                          setRenameValue(conversation.title);
                        }}
                        className="p-1.5 text-muted-foreground hover:text-primary"
                        aria-label={`重命名${conversation.title}`}
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        type="button"
                        onClick={() => void deleteConversation(conversation.id)}
                        className="p-1.5 text-muted-foreground hover:text-red-700"
                        aria-label={`删除${conversation.title}`}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>

        {settingsOpen && (
          <div className="absolute z-30 left-4 right-4 bottom-[76px] rounded-xl border border-border bg-background p-4 shadow-xl text-xs">
            <div className="flex items-center justify-between mb-3">
              <span className="font-semibold text-sidebar-foreground">后端与模型状态</span>
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className="text-muted-foreground hover:text-primary"
                aria-label="关闭设置"
              >
                <X size={15} />
              </button>
            </div>
            <div className="space-y-2 text-muted-foreground">
              <div className="flex justify-between">
                <span>服务</span>
                <span className={health?.status === "ok" ? "text-emerald-700" : "text-amber-700"}>
                  {health ? (health.status === "ok" ? "正常" : "配置不完整") : "未知"}
                </span>
              </div>
              {health?.graph_enabled && (
                <div className="flex justify-between">
                  <span>LightRAG</span>
                  <span className={health.graph_available ? "text-emerald-700" : "text-amber-700"}>
                    {health.graph_available ? "可用" : "不可用（已降级）"}
                  </span>
                </div>
              )}
              {models && (
                <>
                  <div className="flex justify-between gap-2">
                    <span className="truncate">{models.chat_model}</span>
                    <span>{configurationLabel(models.chat_configured)}</span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="truncate">{models.embedding_model}</span>
                    <span>{configurationLabel(models.embedding_configured)}</span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="truncate">{models.rerank_model}</span>
                    <span>{configurationLabel(models.rerank_configured)}</span>
                  </div>
                </>
              )}
            </div>
            <button
              type="button"
              onClick={() => void refreshSystemStatus()}
              className="mt-3 w-full flex items-center justify-center gap-1.5 rounded-md border border-border py-1.5 text-muted-foreground hover:text-primary"
            >
              <RefreshCw size={12} /> 刷新状态
            </button>
          </div>
        )}

        <div className="p-4 sm:p-6 border-t border-border">
          <button
            type="button"
            onClick={() => {
              const opening = !settingsOpen;
              setSettingsOpen(opening);
              if (opening) void refreshSystemStatus();
            }}
            className="flex items-center gap-3 text-sm text-sidebar-foreground hover:text-primary transition-colors w-full px-2 py-2 rounded-md hover:bg-muted/50"
          >
            <Settings size={18} />
            偏好设置
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col min-w-0 bg-background relative">
        <header className="h-14 sm:h-0 border-b border-border sm:border-none flex items-center px-4 shrink-0 sm:hidden">
          <h1 className="font-serif font-bold text-lg">小说大师</h1>
        </header>

        {notice && (
          <div className="absolute z-20 top-4 left-1/2 -translate-x-1/2 max-w-[calc(100%-2rem)] flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 shadow-md">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span className="line-clamp-3">{friendlyError(notice)}</span>
            <button
              type="button"
              onClick={() => setNotice(null)}
              className="shrink-0 hover:text-red-950"
              aria-label="关闭错误提示"
            >
              <X size={15} />
            </button>
          </div>
        )}

        <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-8 sm:py-10 scroll-smooth">
          <div className="max-w-3xl mx-auto space-y-8 sm:space-y-12 pb-28">
            {conversationLoading ? (
              <div className="flex justify-center items-center gap-2 py-20 text-sm text-muted-foreground">
                <LoaderCircle size={18} className="animate-spin" /> 正在读取对话
              </div>
            ) : (
              displayedMessages.map((message, messageIndex) => (
                <div
                  key={message.id}
                  className={`flex gap-4 sm:gap-6 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <div className="w-8 h-8 sm:w-10 sm:h-10 shrink-0 rounded-full bg-sidebar border border-border flex items-center justify-center text-primary mt-1">
                      <Bot size={20} />
                    </div>
                  )}

                  <div
                    className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-5 py-4 ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground rounded-tr-sm shadow-md"
                        : "bg-transparent text-foreground"
                    }`}
                  >
                    <div
                      className={`prose prose-sm sm:prose-base leading-relaxed ${
                        message.role === "assistant"
                          ? "font-serif text-[15px] sm:text-[17px]"
                          : "font-sans"
                      }`}
                    >
                      {message.status === "streaming" && !message.content ? (
                        <span className="inline-flex items-center gap-2 text-muted-foreground font-sans text-sm">
                          <LoaderCircle size={15} className="animate-spin" /> 正在查阅原文
                        </span>
                      ) : (
                        <RichText
                          content={message.content}
                          role={message.role}
                          citations={message.citations}
                          onCitation={(citation) =>
                            setExpandedCitationId((current) =>
                              current === citation.id ? null : citation.id,
                            )
                          }
                        />
                      )}
                    </div>

                    {message.role === "assistant" && message.citations.length > 0 && (
                      <div className="mt-4 space-y-2 font-sans">
                        <div className="flex flex-wrap gap-1.5">
                          {message.citations.map((citation) => (
                            <button
                              key={citation.id}
                              type="button"
                              onClick={() =>
                                setExpandedCitationId((current) =>
                                  current === citation.id ? null : citation.id,
                                )
                              }
                              className={`max-w-full truncate rounded-full border px-2.5 py-1 text-xs transition-colors ${
                                expandedCitationId === citation.id
                                  ? "border-primary bg-primary/10 text-primary"
                                  : "border-border text-muted-foreground hover:border-primary/40 hover:text-primary"
                              }`}
                            >
                              [{citation.ordinal}] {citation.book_title} · {citation.chapter_title}
                            </button>
                          ))}
                        </div>
                        {message.citations.map(
                          (citation) =>
                            expandedCitationId === citation.id && (
                              <div
                                key={`excerpt-${citation.id}`}
                                className="rounded-lg border border-border bg-sidebar/70 p-3 text-xs leading-relaxed text-muted-foreground"
                              >
                                <div className="mb-1 font-semibold text-sidebar-foreground">
                                  《{citation.book_title}》 · {citation.chapter_title}
                                </div>
                                {citation.excerpt}
                              </div>
                            ),
                        )}
                      </div>
                    )}

                    {message.role === "assistant" &&
                      message.status === "completed" &&
                      message.metrics && <AnswerMetricsSummary metrics={message.metrics} />}

                    {message.role === "assistant" &&
                      ["failed", "cancelled"].includes(message.status) && (
                        <div className="mt-3 flex items-center gap-2 font-sans text-xs text-muted-foreground">
                          <span>
                            {message.status === "cancelled"
                              ? "生成已停止"
                              : friendlyError(message.error || "回答生成失败")}
                          </span>
                          {message.status === "failed" && message.id !== "welcome" && (
                            <button
                              type="button"
                              disabled={sending}
                              onClick={() => retryMessage(messageIndex)}
                              className="inline-flex items-center gap-1 text-primary hover:underline disabled:opacity-50"
                            >
                              <RotateCcw size={11} /> 重试
                            </button>
                          )}
                        </div>
                      )}
                  </div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        <div className="absolute bottom-0 w-full bg-gradient-to-t from-background via-background to-transparent pt-10 pb-6 px-4 sm:px-8">
          <div className="max-w-3xl mx-auto">
            <form
              onSubmit={handleSend}
              className="relative flex items-end gap-2 bg-sidebar border border-border rounded-2xl p-2 shadow-sm focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all"
            >
              <div ref={scopePickerRef} className="relative shrink-0">
                {scopeOpen && (
                  <div className="absolute z-30 bottom-full left-0 mb-3 w-72 max-h-80 overflow-y-auto rounded-xl border border-border bg-background p-2 shadow-xl">
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                      本次检索范围
                    </div>
                    <button
                      type="button"
                      onClick={() => void applyScope("auto", [])}
                      className="w-full flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-left text-sm hover:bg-muted/50"
                    >
                      <span>
                        <span className="block font-medium">自动识别小说</span>
                        <span className="block text-xs text-muted-foreground">根据问题自动选择藏书</span>
                      </span>
                      {scopeMode === "auto" && <Check size={15} className="text-primary" />}
                    </button>
                    <div className="my-1 border-t border-border" />
                    {readyBooks.length === 0 ? (
                      <div className="px-3 py-3 text-xs text-muted-foreground">暂无可选的已索引小说</div>
                    ) : (
                      readyBooks.map((book) => {
                        const selected = scopeMode === "books" && selectedBookIds.includes(book.id);
                        return (
                          <button
                            key={book.id}
                            type="button"
                            onClick={() => toggleBookScope(book.id)}
                            className="w-full flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-left text-sm hover:bg-muted/50"
                          >
                            <span className="min-w-0">
                              <span className="block truncate font-serif font-medium">{book.title}</span>
                              <span className="block truncate text-xs text-muted-foreground">
                                {book.author || "作者未知"}
                              </span>
                            </span>
                            <span
                              className={`w-4 h-4 shrink-0 rounded border flex items-center justify-center ${
                                selected ? "border-primary bg-primary text-white" : "border-border"
                              }`}
                            >
                              {selected && <Check size={11} />}
                            </span>
                          </button>
                        );
                      })
                    )}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => setScopeOpen((open) => !open)}
                  className="relative p-2.5 sm:p-3 text-muted-foreground hover:text-primary transition-colors rounded-xl hover:bg-muted/50"
                  title={
                    scopeMode === "auto"
                      ? "检索范围：自动识别"
                      : `检索范围：${selectedReadyBooks.map((book) => book.title).join("、")}`
                  }
                  aria-label="选择检索书籍范围"
                  aria-expanded={scopeOpen}
                >
                  <Paperclip size={20} />
                  {scopeMode === "books" && (
                    <span className="absolute top-1.5 right-1.5 min-w-3 h-3 px-0.5 rounded-full bg-primary text-[8px] leading-3 text-white text-center">
                      {selectedBookIds.length}
                    </span>
                  )}
                </button>
              </div>

              <textarea
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSend();
                  }
                }}
                disabled={conversationLoading}
                placeholder="询问关于数据库中小说的任何问题..."
                className="flex-1 max-h-40 min-h-[44px] bg-transparent resize-none outline-none py-3 px-2 text-sm sm:text-base placeholder:text-muted-foreground/70 disabled:opacity-60"
                rows={1}
                maxLength={4000}
              />

              {sending ? (
                <button
                  type="button"
                  onClick={stopGeneration}
                  className="p-2.5 sm:p-3 shrink-0 rounded-xl bg-primary text-primary-foreground shadow-md hover:bg-primary/90"
                  title="停止生成"
                  aria-label="停止生成"
                >
                  <Square size={16} fill="currentColor" />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!inputValue.trim() || conversationLoading}
                  className={`p-2.5 sm:p-3 shrink-0 rounded-xl transition-all ${
                    inputValue.trim() && !conversationLoading
                      ? "bg-primary text-primary-foreground shadow-md hover:bg-primary/90"
                      : "bg-muted text-muted-foreground cursor-not-allowed"
                  }`}
                  aria-label="发送消息"
                >
                  <Send size={18} className={inputValue.trim() ? "ml-0.5" : ""} />
                </button>
              )}
            </form>
            <div className="text-center mt-3 text-xs text-muted-foreground font-medium truncate px-4">
              {sending && streamDetail
                ? streamDetail
                : scopeMode === "auto"
                  ? "自动选择小说 · 回答可能有误，请核实引用原文"
                  : `仅检索：${selectedReadyBooks.map((book) => book.title).join("、")}`}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
