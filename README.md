# 小说大师

“小说大师”是一个从本地小说素材构建知识库、回答剧情/人物/世界观问题并给出原文依据的 RAG Agent。

当前仓库包含：

- React 19 + Vite + Tailwind 前端，已接通藏书、任务、对话、流式回答与引用。
- FastAPI 后端和 OpenAPI 文档。
- TXT、Markdown、EPUB 章节解析与增量同步。
- Qdrant dense + BM25 混合检索，Qwen 重排。
- LangGraph 有界查询改写、书目路由、证据检查和一次重检。
- SQLite 会话、消息、索引任务与引用持久化。
- POST SSE 流式回答和前端 TypeScript 客户端。

## 快速开始

需要 Python 3.12/3.13、[uv](https://docs.astral.sh/uv/) 和 Node.js 24 + pnpm 10。

```bash
cp .env.example .env
```

在 `.env` 中至少填写：

```dotenv
DASHSCOPE_API_KEY=你的百炼APIKey
```

将 `.txt`、`.md` 或 `.epub` 小说放入 `fiction/`，然后启动后端：

```bash
cd backend
uv sync
uv run fiction-master serve
```

API 默认运行在 `http://127.0.0.1:8000`，Swagger 文档位于
`http://127.0.0.1:8000/docs`。服务启动后会增量扫描 `fiction/`；也可以调用：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/library/sync
```

开发前端：

```bash
pnpm install
pnpm dev
```

前端默认是 `http://localhost:5173`。Vite 开发环境可配置 `VITE_API_BASE_URL=http://127.0.0.1:8000`，或由前端代理 `/api`。

## 模型配置

默认使用阿里云百炼：

| 能力 | 默认模型 | 用途 |
|---|---|---|
| Chat | `qwen-plus` | 改写、路由、证据判断和回答 |
| Embedding | `text-embedding-v4`，1024 维 | 语义召回 |
| Rerank | `qwen3-rerank` | 候选原文二次排序 |

Chat 与 Embedding 使用 OpenAI-compatible 客户端，可以分别设置 `*_BASE_URL`、`*_API_KEY` 和 `*_MODEL` 切换供应商。Rerank 使用独立的 DashScope 适配器。

`GET /api/v1/system/models` 只返回模型名和密钥是否配置，不会泄露密钥。

## 素材和索引规则

- `fiction/` 是唯一原始素材入口；后端不会向里面写文件。
- 每个文件视为一本小说，可使用子目录整理。
- 完整小说被 `.gitignore` 排除，不会进入公开 Git。
- SQLite、Qdrant、索引版本和缓存都位于 `data/`。
- 文件 SHA-256 不变时不会重新调用向量 API。
- 重建先写新索引版本，完成后原子切换；失败时继续保留旧索引。
- 删除源文件并同步后，该书会标记为 `missing` 并停止参与检索。

TXT 会依次尝试 UTF-8 BOM、UTF-8、GB18030。切片严格限制在单章内，约 900 字符，最大 1200 字符，重叠约 150 字符。

## API

主要接口：

- `GET /api/v1/health`
- `GET /api/v1/books`
- `POST /api/v1/library/sync`
- `POST /api/v1/books/{book_id}/reindex`
- `GET /api/v1/jobs/{job_id}`
- `GET|POST /api/v1/conversations`
- `GET|PATCH|DELETE /api/v1/conversations/{id}`
- `POST /api/v1/conversations/{id}/messages/stream`

流式请求和事件格式见 [前端联调说明](docs/FRONTEND_INTEGRATION.md)。
技术选型理由、完整数据流与后续阶段见 [架构与推进路线](docs/ARCHITECTURE.md)。

## 验证

```bash
cd backend
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest

cd ..
pnpm lint
pnpm build
```

测试使用合成短篇素材，不会把本地完整小说带入 CI。

## 单容器部署

```bash
cp .env.example .env
# 填写 DASHSCOPE_API_KEY
docker compose up --build -d
```

容器只运行一个 Uvicorn worker，这是嵌入式 Qdrant 的约束。`fiction/` 以只读方式挂载，`data/` 持久化。默认只监听服务器本机的 `127.0.0.1:8000`，公网部署请使用带认证的 HTTPS 反向代理。完整步骤见 [服务器部署指南](docs/DEPLOYMENT.md)。需要多用户或多实例时，应迁移到 PostgreSQL、独立 Qdrant 和持久任务队列。
