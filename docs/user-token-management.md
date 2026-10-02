# 用户删除与 Token 复看

管理员在“用户与 Token”中查看、复制普通用户 Token，或删除已没有未结束任务的用户。普通用户通过“我的 Token”查看自己的凭据，无需查看 Key 权限。管理密钥与 legacy 单凭据不提供查看或删除操作。

- `POST /v1/auth/relay-tokens/{id}/reveal`：管理员或本人按需读取普通用户 Token。成功响应为 `{"id": "...", "token": "..."}`，带 `Cache-Control: no-store`。Cookie 请求要求同源。其他用户 403，管理员查询不存在用户 404。
- `DELETE /v1/auth/relay-tokens/{id}`：仅管理员可调用，成功 204。管理身份 403，不存在 404，有非终态任务 409；错误 detail 含 `code=user_has_active_jobs`、`active_job_count` 和可读 message。
- `GET /v1/jobs?owner={id}`：管理员专用精确用户筛选，供删除冲突后的任务入口使用，可与原有筛选组合。

仅 `complete`、`failed`、`canceled` 属于终态。删除保留历史任务、结果文件及 Kaggle 凭据；原 Bearer 和 UI Cookie 立即失效。可选配置字段 `retired_relay_token_ids` 保存不可重用的用户 ID，不保留其 Token 原文。配置加载和用户创建均拒绝复用已删除 ID。有独立管理密钥时允许普通用户列表为空。

配置持久化和内存状态发布使用同一临界区；锁顺序为 admission/storage budget → auth config。任务创建中的远程查询保持在锁外，进入 admission 锁后重新检查身份和认证配置版本；身份失效返回 401，期间配置改变则返回 409 要求重试。删除和任务创建不会产生孤立新任务。

页面原文只在按需响应和临时输入框中出现，60 秒自动清除；隐藏、页面切换、刷新、退出和 pagehide 清除。晚到响应不能重新填入已清除的展示。复制失败提供手动选择；默认列表、URL、浏览器持久存储和日志不包含 Token。

## 验证

Windows 使用 Relay `.venv` 运行服务端测试；客户端使用本机实际存在的 `emo-vision-train` 环境。跨仓库合同测试需指定 `KAGGLE_RELAY_SOURCE` 和 `KAGGLE_RELAY_PYTHON`，避免误将缺省跳过当作通过。

本次测试和页面截图见 `docs/evidence/user-token-management/`。截图使用本地独立的虚构用户，原文保持隐藏。

## 部署与回滚

部署保持原 Compose 配置、持久数据挂载和环境变量；新镜像基于线上旧镜像更新应用源码。切换前核对没有非终态任务，保留一致性数据库备份、配置、上传意图、旧镜像和 Compose 文件。

回滚不得还原旧 auth.json 或数据库，否则会复活被删除用户或覆盖新增任务。旧版解析器不能加载零普通用户的配置，因此回滚镜像在原旧镜像上仅携带新的 auth_config.py 兼容解析器，保留已删除 ID 校验与零普通用户支持。原旧镜像同时保留。回滚脚本只切换镜像与对应源码，不回滚任何业务数据或认证配置。
