# 容器镜像发布和拉取

仓库里的 GitHub Action 会把 `Dockerfile` 构建成 GitHub Container Registry 镜像：

```text
ghcr.io/jsdfhasuh/kaggle_relay
```

触发规则：

- push 到 `main`：发布 `latest`、`main` 和 `sha-<commit>` 标签。
- push `v*` tag：发布同名版本标签，例如 `v1.0.0`。
- Pull Request：只构建验证，不推送镜像。
- 每周一 04:17（Asia/Shanghai）：定时重建并推送 `latest`。
- 手动运行 workflow：可以从 GitHub Actions 页面触发。

服务器直接拉仓库镜像：

```bash
docker compose -f docker-compose.ghcr.yml pull kaggle-relay
docker compose -f docker-compose.ghcr.yml up -d --force-recreate
```

如果当前服务器还在用本地 `Dockerfile` build 版，第一次切换到仓库镜像也用上面两条命令。服务名仍然是 `kaggle-relay`，`./relay-data:/data` 数据卷不变。

只想确认远端有没有新镜像，可以先执行：

```bash
docker compose -f docker-compose.ghcr.yml pull kaggle-relay
docker compose -f docker-compose.ghcr.yml images
```

如果要让服务器也自动拉取并重启，可以在服务器 crontab 加一条，例如每天 04:30 执行：

```cron
30 4 * * * cd /docker_volume/kaggle_relay && docker compose -f docker-compose.ghcr.yml pull kaggle-relay && docker compose -f docker-compose.ghcr.yml up -d --force-recreate
```

这个服务器是 `linux/arm64` 架构，Action 同时发布 `linux/amd64` 和 `linux/arm64`，Docker 会自动拉匹配架构的镜像。

如果 GHCR 包是 private，需要先登录：

```bash
docker login ghcr.io -u jsdfhasuh
```

密码使用有 `read:packages` 权限的 GitHub token。包设为 public 后，拉取通常不需要登录。

发布后可以检查：

```bash
docker compose -f docker-compose.ghcr.yml exec -T kaggle-relay kaggle --version
curl -i http://127.0.0.1:8000/v1/health
```

`/v1/health` 如果没有带认证 token，返回 `401` 是正常的；说明服务已经在响应。
