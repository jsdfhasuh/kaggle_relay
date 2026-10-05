# 本地验证记录

- 实现基线：线上 cf6d710bdaf5e08f2b2ca067f8e8b9b60b232c24。
- Relay 完整 pytest：467 passed / 3 skipped（Windows 缺少 POSIX process-group、Linux cleanup、Linux fork）。之后只补充 6 个边界测试，专项重新运行为 27 passed，包含 26 个用户管理测试和 Node UI 测试入口。
- Node 页面行为测试覆盖：隐藏/显示、复制成功和失败、60 秒自动清除、晚到响应、删除取消与冲突、精确用户任务筛选及权限导航。
- 桌面合同/上传恢复/任务分配：48 passed，指定 KAGGLE_RELAY_SOURCE 为本隔离工作区、KAGGLE_RELAY_PYTHON 为服务端虚拟环境，无缺省跳过。
- 本地真实浏览器：无 Key 查看权限用户可访问“我的 Token”、显示自身测试凭据并退出；管理员用户列表有查看、复制、删除操作，管理密钥无对应操作。
- admin-ui.jpg 为隔离本地页面截图，只有虚构用户与账号，Token 隐藏。
- 这些测试不提交 Kaggle 训练，不替代物理设备或生产训练验收。
