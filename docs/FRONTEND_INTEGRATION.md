# 前端联调契约

前端已经通过 [types.ts](../front/src/api/types.ts) 和 [client.ts](../front/src/api/client.ts) 接通后端，并保留了原有视觉设计。本文件同时作为后续维护接口时的联调契约。

## 已接入的 UI 功能

1. `GET /api/v1/books` 驱动藏书、字数、索引状态和错误提示。
2. “同步 fiction”与单书重建按钮轮询并显示任务进度。
3. conversations CRUD 驱动新建、切换、重命名和删除对话。
4. 输入区纸夹按钮用于“自动识别/指定书籍”范围选择。
5. `streamMessage()` 消费 SSE，按 `delta` 累加回答并支持停止生成。
6. `[n]` 可点击，引用卡展示书名、章节与 `excerpt`。
7. 偏好设置展示后端健康状态和三个模型的配置状态。

Vite 开发服务器默认将 `/api` 代理到 `http://127.0.0.1:8000`，无需单独配置 CORS；可用根目录环境变量 `VITE_DEV_API_TARGET` 覆盖目标地址。

## 发起流式问答

自动路由：

```json
{
  "content": "唐三为什么要跳下鬼见愁？",
  "scope": { "mode": "auto" }
}
```

手动锁定一本或多本书：

```json
{
  "content": "比较两本书对成长的描写",
  "scope": {
    "mode": "books",
    "book_ids": ["book-id-1", "book-id-2"]
  }
}
```

## SSE 事件

每个事件都是标准的 `event:` + JSON `data:`：

```text
event: status
data: {"run_id":"...","stage":"retrieving","detail":"正在进行语义与关键词混合检索"}

event: delta
data: {"run_id":"...","text":"唐三选择跳下鬼见愁……[1]"}

event: citation
data: {"ordinal":1,"book_title":"斗罗大陆","chapter_title":"第一集 ...","excerpt":"..."}

event: done
data: {"run_id":"...","message":{...}}
```

`stage` 可能为：`accepted`、`routing`、`retrieving`、`reranking`、`writing`。

错误事件：

```text
event: error
data: {"run_id":"...","code":"MODEL_NOT_CONFIGURED","message":"...","retryable":false}
```

用户点击停止时中止 fetch 的 `AbortController`；后端会把对应助手消息标记为 `cancelled`。

## 藏书状态

- `discovered`：已发现文件。
- `indexing`：正在解析或生成索引。
- `ready`：可以问答；若 `error` 非空，表示最近一次重建失败但旧索引仍可用。
- `error`：没有可用索引，显示错误和重试操作。
- `missing`：源文件已删除。
- `duplicate`：内容与另一文件完全相同，不重复建索引。
