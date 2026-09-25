# A3 transport / identity repair（2026-09-25）

基点 `4d7723373d5d49482a054e480f8f0db986374581`，独立分支
`codex/a3-transport-identity-fix`。应用配套分支 `feat/kaggle-job-build-foundation`，
基点 `ef2cfa17250525a3439a4921b67051e6bad11260`。**只交付代码供审查，未部署。**

## 行为与兼容

1. 原 `dataset_upload_inventory()` 已支持外层 ZIP 展开。真正缺口是内层 runtime ZIP
   被递归展开。本次新增 `payload_contract.py`，核验完整原包，或精确展开后的文件集合及
   每个文件的 SHA。拒绝等长篡改、混包、额外/重复成员；与应用模块保持相同内容。
   生产 worker 的新回执和缓存回执都绑定源码目录及内容摘要，下载原候选编号版本核验，
   不用 latest 替代。核验期间本地比较源变化也拒绝。
2. 动态调度、固定池自动选账号先检查实际身份，之后才检查配额。上传前重新核验配置 owner、
   任务 Dataset/Kernel owner 与凭据身份，不改写错误绑定的旧任务。
   access_token 采用 SDK authenticate 的 token introspection 结果；OAuth 再 introspect token，
   不信任缓存用户名；legacy API key 使用认证请求确认 username/key 主体后比较 owner。
   不支持的认证来源或不匹配均明确阻止，认证/配额成功不能代替身份门禁。
   显式固定账号的旧任务保持绑定，由 worker 上传前门禁拒绝错误身份。
3. 新验证 API 签名上传前检查；HTTP 403 不当作不存在。create/version 必须检查结构化
   error/errorMessage/invalidTags/status，业务错误没有成功回执。
   业务拒绝保存 rejected；响应缺失、传输异常或进程硬退出保持 unknown。
4. `<storage_dir>/dataset_upload_intents/` 在远端修改前原子写入原候选版本、用途、ref、
   原目录与内容摘要。落盘失败不上传；null/非对象/格式错拒绝。
   重试只读回原候选，不创建新版本。已有意图优先于共享缓存，缓存覆盖不能替换候选。
   意图独立于可清理 staging，清理不能把未知上传变成从未上传。
5. worker、服务重启恢复及 `/v1/jobs/{id}/complete` 的显式重试保持同一 job/ref/candidate。
   重启使用状态比较更新到 queued，满足真实 worker 队列入口的状态门禁。
   内容冲突拒绝 Kernel 提交；无意图的旧不确定上传仍按旧逻辑失败关闭，不推测候选。
6. 日志原因变化立即记录，同原因最多每 60 秒记录一次；轮询 SDK/CLI 原始重复日志抑制。
   旧文件清单路径和新字节验证路径都覆盖先 403、后缺文件；缺文件原因不再被吞掉。

保留旧 `DatasetUploadReceipt` 默认字段及旧无源码回执的兼容轮询入口；生产新上传/缓存
均走绑定目录的字节验证。缓存原源码若已被清理，拒绝复用，不能用名称/长度替代字节证据，
也不自动另建 Dataset 版本。意图状态 unknown 可以表示“已落盘但尚未来得及调用远端”，
该情况也需核验原候选或人工处置，不自动重传。

原上传包、任务身份、训练源码、RunPlan、模型支持模块和依赖策略未改。
应用已冻结的旧 bootstrap 不被原地升级；本次不宣称旧脚本的真实云端恢复通过。
现有 DINO 测试只补测试替身的身份接口/缓存来源目录，未修改 DINO 产品逻辑。

## 测试与证据

使用已有 `C:/Users/jsdfhasuh/my_scripts/kaggle_relay/.venv/Scripts/python.exe`；没有安装依赖。
在未修改基点及本分支分别运行：

```text
python -m pytest tests -q
python -m pytest tests/test_a3_content_identity.py -q
```

- 未修改基点：295 PASS、2 SKIPPED，退出 0。
- 分支完整回归：329 PASS、2 SKIPPED，退出 0。
- 最后局部补测：37 PASS，退出 0。与完整回归按用例去重为
  **332 PASS、0 BASELINE_FAILURE、0 REGRESSION、2 SKIPPED**。
- 跳过项为 Windows 上的 POSIX 进程组和 fork 用例，详见 [validation.json](validation.json)。
- 新测试覆盖三种包布局、等长篡改、身份来源/错 owner/OAuth 假缓存身份、固定池/动态选择、
  create/version HTTP 成功业务错误、403、意图落盘失败、null＋缓存覆盖、原候选冲突、
  响应丢失、真实子进程 `os._exit(71)`、worker 与 HTTP complete/重启恢复、清理保留意图。
  SDK 使用明确离线替身，不称为真实 Kaggle 验收。
- 应用使用保留的真实失败包（336 个成员）重放当时远端布局（338 个成员），
  本模块、应用模块和 bootstrap 都通过；原包 SHA 不变。它不是新的远端字节下载证据。
- 配套应用实际 Relay AST hook 改写/参数转发 2 PASS；应用回归、main golden 与更新器保护
  证据同在 validation.json；原始日志/JUnit 留在 `D:/kaggle-a3-repair-evidence`。

原 Relay 工作区未切换/清理，README SHA256 保持
`50f3903e06f216f396e043ee0f1552a71ba3c0d50201ec42ff33fa3a15e1623a`。
未知意图没有被清理。没有更改生产配置、部署服务、启动新云作业、合 main 或强推。
修复后真实训练/校准/结果接回与 workers=0/2、冻结产品验收为 **NOT_RUN**，停在 A3 审查。
