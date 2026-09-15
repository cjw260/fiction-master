# 单机服务器部署

以下方案适用于一台 Linux 服务器、一个使用者和嵌入式 Qdrant。生产环境只运行一个应用容器实例。

## 1. 准备服务器

安装 Git、Docker Engine 和 Docker Compose Plugin。建议至少准备 2 核 CPU、4 GB 内存，以及能容纳小说原文和向量索引的磁盘空间。

只在防火墙或云安全组开放：

- `22/tcp`：SSH，最好限制来源 IP。
- `80/tcp` 和 `443/tcp`：有域名并使用 HTTPS 时开放。
- 不要向公网开放 `8000/tcp`。

## 2. 获取项目和小说素材

```bash
sudo install -d -o "$USER" -g "$USER" /opt/fiction-master
git clone https://github.com/cjw260/fiction-master.git /opt/fiction-master
cd /opt/fiction-master
cp .env.example .env
```

编辑 `.env`，至少设置：

```dotenv
APP_ENV=production
DASHSCOPE_API_KEY=你的百炼APIKey
AUTO_SYNC_ON_STARTUP=true
```

`.env` 和小说正文不会进入 Git。需要单独把素材传到服务器：

```bash
rsync -av --progress ./fiction/ user@server:/opt/fiction-master/fiction/
```

也可以直接在服务器的项目 `fiction/` 目录放入 TXT、Markdown 或 EPUB。

## 3. 启动

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f fiction-master
```

容器启动后会扫描 `fiction/`。长篇小说第一次生成 Embedding 需要一些时间和模型额度。检查状态：

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Compose 默认只把服务绑定到服务器本机 `127.0.0.1:8000`，不能从公网直接访问。

如需关系图谱检索，在 `.env` 设置 `LIGHTRAG_ENABLED=true` 和一个随机的
`LIGHTRAG_API_KEY`，然后使用：

```bash
docker compose --profile graph up --build -d
docker compose --profile graph logs -f lightrag
```

LightRAG 的 `9621` 同样只绑定到 `127.0.0.1`，不得通过反向代理向公网暴露。
图谱数据位于 `data/lightrag/`，会随本节的 `data` 备份一起保存。完整说明见
[LightRAG 关系图谱检索](LIGHTRAG.md)。

## 4. 配置域名、HTTPS 和访问密码

推荐在宿主机安装 Caddy。先将域名的 A/AAAA 记录指向服务器，再生成密码哈希：

```bash
caddy hash-password
```

将输出的哈希写入 `/etc/caddy/Caddyfile`：

```caddyfile
fiction.example.com {
    encode zstd gzip

    basic_auth {
        admin 这里替换为密码哈希
    }

    reverse_proxy 127.0.0.1:8000
}
```

然后校验并重载：

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy 会在域名解析正确且 80/443 可访问时自动申请和续期 HTTPS 证书。当前应用没有自己的用户系统，因此不要移除反向代理认证。

如果暂时没有域名，优先通过 SSH 隧道访问：

```bash
ssh -L 8000:127.0.0.1:8000 user@server
```

然后在本机打开 `http://127.0.0.1:8000`。

## 5. 更新、备份和恢复

更新代码：

```bash
git pull --ff-only
docker compose up --build -d
docker compose ps
```

备份前先停止容器，确保 SQLite 和嵌入式 Qdrant 文件一致：

```bash
docker compose stop
tar -czf fiction-master-data-$(date +%F).tar.gz data fiction
docker compose start
```

备份文件应存放到另一台机器或对象存储。`.env` 包含密钥，如需备份必须加密保存。

恢复时停止容器，将备份中的 `data/` 和 `fiction/` 放回项目目录，再重新启动。不要同时运行两个共享同一 `data/` 目录的容器，否则嵌入式 Qdrant 会拒绝启动。

## 6. 常用排查命令

```bash
docker compose ps
docker compose logs --tail=200 fiction-master
docker compose restart fiction-master
curl http://127.0.0.1:8000/api/v1/health
```

常见状态：

- `models_configured: false`：检查容器中的 `.env`，然后重建或重启。
- Qdrant 目录被锁：确认只有一个应用容器在运行。
- 小说状态为 `error`：查看任务错误；模型额度恢复后，从 UI 点击单书重建。
- SSE 经代理不流式：确认代理没有缓存响应，且没有设置过短的上游响应超时。
- `graph_enabled: true` 但 `graph_available: false`：检查 `lightrag` 容器日志、API Key 和模型配置；普通问答会自动降级。
