# 容器镜像和服务器部署

## Live 服务器部署

恢复逻辑、worker 异常保护、前端和 API 代码必须来自同一份源码。服务器现场部署优先使用本地 checkout 构建：

```bash
cd /docker_volume/kaggle_relay
docker compose up -d --build --force-recreate kaggle-relay
```

这条命令会使用仓库里的 `docker-compose.yml` 和 `Dockerfile`，并保留现有的 `./relay-data:/data` 数据卷。替换容器前必须做 SQLite 一致性备份。数据库使用 WAL 模式，不能只在宿主机复制 `relay.db`，否则可能遗漏 `relay.db-wal` 中尚未 checkpoint 的事务：

```bash
cd /docker_volume/kaggle_relay
mkdir -p relay-data/backups
BACKUP_NAME="pre-deploy-$(date -u +%Y%m%dT%H%M%SZ).db"
docker compose exec -T kaggle-relay python -c "import sqlite3; src=sqlite3.connect('/data/relay.db'); dst=sqlite3.connect('/data/backups/$BACKUP_NAME'); src.backup(dst); dst.close(); src.close()"
test -s "relay-data/backups/$BACKUP_NAME"
sha256sum "relay-data/backups/$BACKUP_NAME"
```

保留输出的文件名和 SHA-256。部署失败需要回滚时，先停止 Relay，再恢复该备份；不要删除整个 `relay-data` 目录。

部署后至少确认：

```bash
docker compose exec -T kaggle-relay grep -n "recover_incomplete_jobs" /app/app/main.py
docker compose exec -T kaggle-relay grep -n "resume_kernel_job" /app/app/worker.py
curl -i http://127.0.0.1:8000/v1/health
```

`/v1/health` 如果没有带认证 token，返回 `401` 是正常的；说明服务已经在响应。

## GHCR 镜像

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

Action 会先执行完整测试，再构建镜像。定时和手动运行会禁用 Docker
构建缓存，以便重新解析未锁死的依赖；当前镜像要求 Kaggle CLI
`>=2.2.4,<3`。

## Watchtower 自动更新

只有明确接受 mutable tag 自动漂移的服务器才使用
`docker-compose.watchtower.yml`。该文件固定跟踪 GHCR `latest`，并带有
Watchtower enable label；当前服务器的 Watchtower 会扫描全部容器，因此
不需要修改其全局配置，也不会影响其他容器的更新策略。

首次从本地 build 容器切换前，必须确认本地预期提交已经 push，且对应
GitHub Action 已成功发布。然后备份数据库并在仓库目录执行：

```bash
cd /docker_volume/kaggle_relay
docker compose -f docker-compose.watchtower.yml pull kaggle-relay
docker compose -f docker-compose.watchtower.yml up -d --pull always --force-recreate kaggle-relay
```

默认项目名和服务名与 `docker-compose.yml` 相同，因此 Compose 会替换现有
`kaggle_relay-kaggle-relay-1`，不会创建并行写同一 SQLite 数据目录的第二个
Relay。`./relay-data:/data` 保持不变。

部署后确认镜像来源、Kaggle CLI 和服务响应：

```bash
docker inspect -f '{{.Config.Image}}' kaggle_relay-kaggle-relay-1
docker compose -f docker-compose.watchtower.yml exec -T kaggle-relay kaggle --version
curl -i http://127.0.0.1:8000/v1/health
docker logs --since 5m watchtower
```

无认证访问 `/v1/health` 返回 `401` 是正常的。后续 GHCR `latest` digest
变化时，Watchtower 会拉取镜像并重建 Relay；重启恢复逻辑会接管未结束
任务。需要回滚时，使用下一节记录过的 `sha-<commit>` 或 digest，通过
`docker-compose.ghcr.yml` 重建；不要回滚或删除 `relay-data`。

## GHCR 固定版本部署

不要在 live 恢复场景直接使用 `latest`。只有确认远端镜像包含当前要部署的代码后，才使用 `docker-compose.ghcr.yml`，并显式指定 tag 或 digest：

```bash
cd /docker_volume/kaggle_relay
export KAGGLE_RELAY_IMAGE=ghcr.io/jsdfhasuh/kaggle_relay:sha-<commit>
docker compose -f docker-compose.ghcr.yml pull kaggle-relay
docker compose -f docker-compose.ghcr.yml up -d --force-recreate kaggle-relay
```

也可以使用 digest：

```bash
export KAGGLE_RELAY_IMAGE=ghcr.io/jsdfhasuh/kaggle_relay@sha256:<digest>
```

如果未设置 `KAGGLE_RELAY_IMAGE`，`docker-compose.ghcr.yml` 会直接失败，避免误跑旧的 `latest`。

只想确认远端镜像信息，可以先执行：

```bash
export KAGGLE_RELAY_IMAGE=ghcr.io/jsdfhasuh/kaggle_relay:sha-<commit>
docker compose -f docker-compose.ghcr.yml pull kaggle-relay
docker compose -f docker-compose.ghcr.yml images
```

这个服务器是 `linux/arm64` 架构，Action 同时发布 `linux/amd64` 和 `linux/arm64`，Docker 会自动拉匹配架构的镜像。

如果 GHCR 包是 private，需要先登录：

```bash
docker login ghcr.io -u jsdfhasuh
```

密码使用有 `read:packages` 权限的 GitHub token。包设为 public 后，拉取通常不需要登录。
