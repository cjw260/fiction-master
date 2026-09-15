# 完整数据快照

2026-09-15 保存的本地数据，共 18 个文件，原始大小 678,298,268 字节；
压缩后 289,940,146 字节，分为 6 卷，每卷不超过 48 MiB。
普通 `git clone` 即可取得全部分卷，无需 Git LFS 或外部下载地址。

## 包含内容

- 两本小说原文：`fiction/1.txt`（斗罗大陆）、`fiction/2.txt`（斗破苍穹）。
- SQLite 应用数据库：2 本书、1,984 章、索引版本、图谱任务，以及现有 3 个对话、36 条消息和引用。
- Qdrant 主向量索引及元数据，10,330 个原文切片。
- LightRAG 的 GraphML 实体关系图、实体/关系/切片向量、全部 KV 存储、文档状态、LLM 响应缓存。
- LightRAG 的 tokenizer 缓存。

保存的是**全部现有数据**：图谱任务为 1 份 `ready`、1 份 `paused`，并不表示两本书
的全部章节都已完成图谱构建。暂停状态会保留，普通同步不会自动重启该版本的图谱任务。

快照时同时暂停应用与 LightRAG 容器，复制完成后恢复，暂停约 1.3 秒。
两个 SQLite 数据库均通过 `PRAGMA integrity_check`；全部 JSON 与 GraphML 均可解析。
`.env` 密钥、进程锁、系统元数据和已单独提交的目录说明未打包。

## 校验与恢复

需要 Python 3.12 或 3.13；脚本仅使用标准库。临时解压与恢复需要约 2 GB 可用空间。
在仓库根目录执行：

```bash
# 校验所有分卷、整体压缩包、解压后的每一个文件
python3 scripts/restore_dataset.py --verify-only

# 恢复到当前仓库（首次启动服务前执行）
python3 scripts/restore_dataset.py

# 或恢复到一个新的目录
python3 scripts/restore_dataset.py --destination /path/to/empty-directory
```

脚本会先校验再恢复；遇到同名已有数据文件会拒绝覆盖。
已有安装请先停止相关服务、另行备份现有 `data/` 与小说文件，再恢复到空目录。
不要向运行中的服务数据目录恢复备份。

随后按根目录 README 配置 `.env` 并启动应用。要使用已有图谱，再设置
`LIGHTRAG_ENABLED=true` 和自己的 `LIGHTRAG_API_KEY`，启动 `graph` profile。
保持 Embedding 模型 `text-embedding-v4`、维度 `1024`，以及 LightRAG
workspace `fiction_master`，以匹配已保存的向量。

`manifest.json` 记录快照时间、文件清单、字节数，以及每个文件/分卷/整体压缩包的
SHA-256。恢复过程中不会调用模型 API；启动后的新提问或显式重建仍按应用配置执行。
