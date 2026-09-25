# A3 Kaggle 2.2.4 legacy Key 签名兼容修复

基点：`2529a12d1caa5ca77e2eb1313ee15a1522f346b8`。
分支：`codex/a3-transport-identity-fix`，开始时工作树干净，无额外 AGENTS.md。

## 核对与最小改动

核对已有 Relay venv 中 Kaggle **2.2.4** 的
`kaggle/api/kaggle_api_extended.py:4525`：`dataset_list` 的参数为
`self, sort_by, size, file_type, license_name, tag_ids, search, user, mine=False, page=1, max_size, min_size`，
没有 `page_size`，也没有接收任意关键字的参数。

唯一生产代码改动为 `_authenticated_identity()` 的 legacy_api_key 分支：

```diff
-api.dataset_list(mine=True, page_size=1)
+api.dataset_list(mine=True, page=1)
```

仍执行真实认证请求后比较 owner；没有把 `authenticate()`、配额成功或配置用户名当作身份验证。
Token/OAuth、上传意图、运输核验和任务身份逻辑不变。

## 严格回归

身份测试与隔离 SDK 子进程测试的 `dataset_list` 替身均改为
`def dataset_list(self, mine=False, page=1)`，没有 `**kwargs`，并断言 `mine is True and page == 1`。
其余 SDK 方法的原测试替身保留，不用来放宽此方法签名。

- 合法 Key：先证明认证请求已执行且尚无上传，再证明上传前重复校验后可取得原候选 7。
- 无效 Key/401、403：真实 worker 入口收到认证请求错误，不能进入 Dataset 查询、create/version 或 Kernel push。
- 错误 owner：认证请求成功仍拒绝；任务 refs、数据字节、冻结脚本未改，没有上传意图或远端写操作。
- 保留 Token/OAuth、三种运输布局、业务错误、响应丢失、未知意图、候选冲突、硬退出及同任务恢复回归。

使用已有解释器 `C:/Users/jsdfhasuh/my_scripts/kaggle_relay/.venv/Scripts/python.exe`，
Python 3.11.16，没有安装或更换环境。命令在 Relay 功能工作树执行：

```powershell
# 先新增严格测试，生产代码仍为 2529a12
& 'C:/Users/jsdfhasuh/my_scripts/kaggle_relay/.venv/Scripts/python.exe' -m pytest tests/test_a3_content_identity.py -k 'legacy_key or identity_reports' -q --junitxml=D:/kaggle-a3-repair-evidence/sdk-signature-before.xml
# 修复唯一调用参数后
& 'C:/Users/jsdfhasuh/my_scripts/kaggle_relay/.venv/Scripts/python.exe' -m pytest tests -q --junitxml=D:/kaggle-a3-repair-evidence/sdk-signature-after.xml
```

| 验证 | 结果 | 退出码 |
| --- | --- | --- |
| 修改前严格测试 | 5 FAIL、2 PASS、34 deselected；均复现 page_size 参数错误提前阻断 legacy 分支 | 1 |
| 修改后完整回归 | **336 PASS、2 SKIPPED、0 REGRESSION**，58.40 秒 | 0 |

修改前失败是严格替身揭示的已有兼容缺陷，前一轮宽松 `**kwargs` 未能发现它。
两个跳过项仍为 Windows 不支持的 POSIX 进程组与 fork 验证。
机器可读结果、失败/跳过用例和 JUnit SHA256 见 [sdk-signature-validation.json](sdk-signature-validation.json)。
原日志和 JUnit 保留在 `D:/kaggle-a3-repair-evidence/sdk-signature-{before,after}.{log,xml}`。
所有 SDK 请求均为离线测试，不宣称实际 Key 或云作业验收。

## 保护与边界

原 Relay 工作区仅有原 README 修改，其 SHA256 保持
`50f3903e06f216f396e043ee0f1552a71ba3c0d50201ec42ff33fa3a15e1623a`。
应用工作树保持干净、HEAD 为 `3bbb1006eb1fc58b117bb7056ac531823bd1c5b6`；
没有改应用冻结源码或旧任务身份。
未合 main、强推、部署、修改生产配置或启动新云作业；未推进 ONNX/DINO。
真实云端验收为 NOT_RUN，独立提交并推送后停在 A3 修复审查。
