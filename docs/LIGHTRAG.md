# LightRAG 关系图谱检索

LightRAG 是可选的旁路检索器。它负责发现人物、事件、地点和势力之间的关系，
但不会直接生成最终答案，也不会成为引用来源。所有图谱线索都会再次查询主 Qdrant
索引，最终回答仍然只引用可定位到书籍、章节和字符偏移的原文。

## 启用

先在 `.env` 中设置一个随机 API Key 并启用功能：

```dotenv
LIGHTRAG_ENABLED=true
LIGHTRAG_API_KEY=请替换为足够长的随机值
```

默认 Compose 配置复用项目现有的百炼 Chat 和 Embedding 配置，并将 LightRAG 固定为
`v1.5.5`。图谱扩展后的候选统一使用小说大师现有的 Rerank，sidecar 内部不重复重排。
Compose 文件使用可选依赖的 `required: false`，需要 Docker Compose 2.20 或更高版本。
启动应用和图谱 sidecar：

```bash
docker compose --profile graph up --build -d
docker compose ps
curl -H "X-API-Key: $LIGHTRAG_API_KEY" http://127.0.0.1:9621/health
curl http://127.0.0.1:8000/api/v1/health
```

本地直接运行 Python 后端时，保持
`LIGHTRAG_BASE_URL=http://127.0.0.1:9621`。Compose 会把应用容器中的地址覆盖为
`http://lightrag:9621`。

然后从前端点击“同步 fiction”，或者调用：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/library/sync
```

主索引完成后书籍立即可以问答；图谱索引在独立后台任务中继续执行。`GET
/api/v1/books` 的 `graph_status` 会依次显示 `queued`、`indexing`、`ready`。已经完成
主索引的旧书也会在下一次同步时补建图谱，不会重复生成主向量。
持久化为 `paused` 的图谱版本保持暂停，普通同步与自动补建不会重新提交它；
显式重建书籍会生成新版本并按当前配置重新建图。

## 查询数据流

只有关系、因果、变化过程、势力和全书级问题会触发图谱检索。第一轮主检索和
LightRAG `/query/data` 并行执行；LightRAG 返回的实体与关系被转换成最多三条补充
查询，再回查 Qdrant。主检索的 RRF 权重为 `1.0`，图谱扩展默认是 `0.7`，之后统一
经过 Qwen Rerank、证据审查和现有引用生成管线。

LightRAG 超时、不可用、书籍图谱未就绪或没有返回当前书籍版本的来源时，系统会
自动退回原有 Dense + BM25 检索。图谱故障不会把书籍状态改成不可用。

回答指标包含：

- `retrieval.graph`：是否实际采用图谱线索；
- `retrieval.graph_mode`：本次使用的 `mix` 或 `global` 模式；
- `retrieval.graph_queries`：转换出的补充查询数量；
- `retrieval.graph_fallback`：图谱调用是否失败并降级；
- `calls.graph`：LightRAG 查询次数。

## 索引与来源隔离

每一章会投影成一个独立文本，来源格式为：

```text
fiction-master--book-<book_id>--version-<index_version>--chapter-000001.md
```

LightRAG v1.5.5 会把 `file_source` 规范化为 basename，因此书 ID 和版本必须编码在
文件名本身，不能依赖目录层级隔离。

`graph_indices` 表保存图谱版本、LightRAG Track ID、Document ID 和所有来源。Agent
只接受与当前 Qdrant active version 匹配的来源；旧版本图谱即使远端清理暂时失败，
也不能成为最终证据。源文件删除或变成重复文件时，后台会请求删除对应 LightRAG
文档。

当前使用一个 `fiction_master` workspace。LightRAG Server 尚不提供稳定的、按每个
请求任意切换 workspace 的契约，因此本项目不依赖 workspace header。书目隔离由
版本化 `file_source` 过滤和最终 Qdrant `active_versions` 过滤共同保证。

## 配置调优

常用配置：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `LIGHTRAG_MODE` | `mix` | 关系问题默认模式；全局模式由路由器自动选择 |
| `LIGHTRAG_TOP_K` | `10` | 图谱实体/关系候选数 |
| `LIGHTRAG_CHUNK_TOP_K` | `12` | LightRAG 文本候选数 |
| `LIGHTRAG_ENABLE_RERANK` | `false` | 是否让 sidecar 内部额外重排；默认由主应用统一重排 |
| `LIGHTRAG_EXPANDED_QUERY_LIMIT` | `3` | 回查 Qdrant 的关系查询上限 |
| `LIGHTRAG_RRF_WEIGHT` | `0.7` | 全部图谱扩展相对主检索的总融合权重 |
| `LIGHTRAG_QUERY_TIMEOUT_SECONDS` | `10` | 图谱查询总时限（含重试），超时即降级 |
| `LIGHTRAG_INDEX_BATCH_SIZE` | `20` | 每次提交的章节数 |
| `LIGHTRAG_INDEX_MAX_WAIT_SECONDS` | `3600` | 单本书图谱构建最长等待时间 |
| `LIGHTRAG_REQUEST_RETRIES` | `2` | 写入冲突、限流或暂时故障的重试次数，可通过环境变量调高 |
| `LIGHTRAG_INDEX_BOOK_TITLES` | 空 | 逗号分隔的建图书名白名单；留空表示全部 |
| `MAX_ASYNC_LLM` | `8` | LightRAG 的 LLM 请求并发上限 |
| `MAX_PARALLEL_INSERT` | `3` | LightRAG 内部同时处理的文档数 |
| `EMBEDDING_FUNC_MAX_ASYNC` | `16` | Embedding 请求并发上限 |
| `EMBEDDING_BATCH_NUM` | `10` | 单次 Embedding 请求的文本数量；DashScope `text-embedding-v4` 不可超过 10 |

首次建图会额外调用 LLM 抽取实体和关系，耗时与费用通常明显高于只生成 Embedding。
建议先用一至三本书完成关系题评测，再调大并发和候选数量。

## 安全与排查

- Compose 只把 `9621` 绑定到 `127.0.0.1`，不要改成公网监听；
- 必须设置 `LIGHTRAG_API_KEY`，应用和 sidecar 使用同一个值；
- 不要设置以 `/api` 开头的 `LIGHTRAG_API_PREFIX`；
- 更新 LightRAG 版本前先查阅其安全公告和 API 变更，并运行本项目完整测试；
- 图谱日志：`docker compose --profile graph logs -f lightrag`；
- 应用状态：`GET /api/v1/health` 中检查 `graph_enabled` 和 `graph_available`；
- 单书状态：`GET /api/v1/books` 中检查 `graph_status`。

关闭功能只需将 `LIGHTRAG_ENABLED=false` 并重启应用。已有图谱数据会保留在
`data/lightrag/`，但不会参与索引或查询。
