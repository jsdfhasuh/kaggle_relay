const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../app/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page() {
  const elements = new Map();
  const context = vm.createContext({
    URLSearchParams, console, setTimeout, clearTimeout,
    document: {getElementById(id) {
      if (!elements.has(id)) elements.set(id, {value: '', textContent: '', innerHTML: '', hidden: false});
      return elements.get(id);
    }},
  });
  vm.runInContext(script.slice(0, script.indexOf('    document.addEventListener("click"')), context);
  return context;
}

test('Relay 60% is never presented as training progress without a valid report', () => {
  const ui = page();
  for (const kernel_status of ['', 'running', 'KernelWorkerStatus.RUNNING', '{broken', 'null', '[]', '{}']) {
    const job = {status: 'waiting_kernel', progress: 60, kernel_status};
    assert.equal(ui.jobProgress(job).percent, null);
    assert.match(ui.renderProgress(job), /训练进度未知/);
    assert.doesNotMatch(ui.renderProgress(job), /60%|progressbar/);
  }
});

test('valid epochs drive the training bar independently of Relay progress', () => {
  const ui = page();
  const job = {status: 'waiting_kernel', progress: 62.4, kernel_status: JSON.stringify({epoch: 12, epochs: 100})};
  assert.equal(ui.jobProgress(job).stage, '训练中');
  assert.equal(ui.jobProgress(job).percent, 12);
  assert.match(ui.renderProgress(job), /12 \/ 100 轮/);
  assert.match(ui.renderProgress(job), /aria-valuenow="12"/);
  assert.equal(ui.jobProgress({...job, kernel_status: '{"epoch":100,"epochs":100}'}).stage, '等待 Kaggle 收尾');
});

test('invalid numeric values never create fake training progress', () => {
  const ui = page();
  for (const data of [
    {epoch: null, epochs: 100}, {epoch: '', epochs: 100}, {epoch: 2, epochs: 0},
    {epoch: 200, epochs: 100, remote_progress: 100}, {epoch: -1, epochs: 100},
    {epoch: 1.5, epochs: 100}, {epoch: Infinity, epochs: 100}, {remote_progress: null},
    {remote_progress: false}, {remote_progress: NaN}, {remote_progress: 101},
  ]) assert.equal(ui.trainingReport({kernel_status: data}), null);
});

test('PatchCore phase percentages are labeled as phase progress', () => {
  const ui = page();
  const payload = {backend: 'patchcore', phase: 'coreset', phase_label: 'Coreset', phase_index: 3,
    phase_count: 7, phase_current: 500, phase_total: 1000, phase_progress: 50, overall_progress: 80};
  const job = {status: 'waiting_kernel', progress: 76, kernel_status: JSON.stringify(payload)};
  assert.equal(ui.jobProgress(job).percent, 50);
  assert.match(ui.renderProgress(job), /阶段 3\/7：Coreset/);
  assert.match(ui.renderProgress(job), /该阶段最近回报进度/);
  for (const broken of [{phase_label: ''}, {phase_progress: Infinity}, {phase_index: 8}, {phase_current: 1001}]) {
    assert.equal(ui.trainingReport({kernel_status: {...payload, ...broken}}), null);
  }
});

test('stale training reports cannot override terminal or download stages', () => {
  const ui = page();
  for (const status of ['failed', 'canceled', 'complete', 'cancel_requested', 'downloading_output']) {
    const job = {status, kernel_status: '{"epoch":12,"epochs":100}'};
    assert.equal(ui.jobProgress(job).stage, ui.statusLabel(status));
    assert.equal(ui.jobProgress(job).percent, null);
    assert.match(ui.renderProgress(job), /上次回报/);
  }
  assert.doesNotMatch(ui.renderProgress({status: 'queued', kernel_status: '{"epoch":12,"epochs":100}'}), /12|progressbar/);
});

test('only explicit remote status distinguishes Kaggle startup from execution', () => {
  const ui = page();
  assert.equal(ui.jobProgress({status: 'waiting_kernel', kernel_status: 'queued'}).stage, '等待 Kaggle 启动');
  assert.equal(ui.jobProgress({status: 'waiting_kernel', kernel_status: 'running'}).stage, 'Kaggle 执行中');
  assert.equal(ui.jobProgress({status: 'waiting_kernel', kernel_status: ''}).stage, '等待 Kaggle 状态');
});

test('error summaries are conservative and preserve unrecognized errors', () => {
  const ui = page();
  for (const [error, expected] of [
    ['torch.cuda.OutOfMemoryError: CUDA out of memory', 'GPU 显存不足'],
    ['GPU quota exhausted', 'GPU 配额不足'],
    ['Dataset upload failed: HTTP 500', '数据集上传失败'],
    ['No space left on device', '磁盘空间不足'],
    ['403 Client Error: Forbidden', '认证或权限检查失败'],
    ['Kernel wait timed out: running', '等待 Kaggle 状态超时'],
  ]) assert.equal(ui.failureSummary({status: 'failed', error}).title, expected);
  for (const error of ['GPU quota authentication lookup timed out', 'Dataset status failed', 'training crashed: custom reason']) {
    assert.equal(ui.failureSummary({status: 'failed', error}).title, error);
    assert.equal(ui.failureSummary({status: 'failed', error}).recognized, false);
  }
});

test('detail shows original error, escaped HTML, and explicitly labeled Relay progress', () => {
  const ui = page();
  const error = 'CUDA out of memory <img src=x onerror=alert(1)>';
  ui.renderDetail({job_id: 'test', status: 'failed', error, progress: 60, recent_logs: []});
  assert.match(ui.qs('detailError').innerHTML, /GPU 显存不足/);
  assert.match(ui.qs('detailError').innerHTML, /CUDA out of memory &lt;img/);
  assert.doesNotMatch(ui.qs('detailError').innerHTML, /<img/);
  assert.match(ui.qs('detailSummary').innerHTML, /Relay 流程进度：60%（非训练完成比例）/);
});

test('in-progress filter excludes queued jobs, and search remains server-side', () => {
  const ui = page();
  ui.qs('limitSelect').value = '20';
  ui.qs('statusSelect').value = 'in_progress';
  ui.qs('jobSearch').value = 'CUDA';
  const params = ui.jobListParams();
  assert.equal(params.get('q'), 'CUDA');
  assert.equal(params.get('limit'), '20');
  assert.ok(params.get('status').includes('waiting_kernel'));
  assert.ok(!params.get('status').includes('queued'));
  ui.qs('statusSelect').value = 'queued';
  assert.equal(ui.jobListParams().get('status'), 'queued');
});

test('stop details explain early stopping and preserve separate artifact errors', () => {
  const ui = page();
  ui.renderDetail({job_id: 'early-stop', status: 'failed', error: 'artifact file is missing',
    stop_details: {reason: 'relay_failed', training_reason: 'early_stopping', source: 'training_log',
      confidence: 'confirmed', message: 'No improvement for 200 epochs', provider_status: 'COMPLETE',
      epoch: 473, epochs: 500, best_epoch: 273, patience: 200, elapsed_seconds: 42000}});
  const stop = ui.qs('detailStop');
  assert.equal(stop.hidden, false);
  for (const value of ['训练停止原因：提前停止（非训练失败）', '不控制最佳权重保存',
    '依据来源：训练日志', '确定程度：已确认', 'Kaggle 原始状态：COMPLETE',
    '已回报轮次：473', '设定总轮数：500', '最佳轮次：273', '早停等待轮数：200',
    '已用时间：42000 秒', 'No improvement for 200 epochs']) assert.ok(stop.innerHTML.includes(value), value);
  assert.equal(ui.qs('detailError').hidden, false);
  assert.match(ui.qs('detailError').innerHTML, /artifact file is missing/);
  assert.doesNotMatch(stop.innerHTML, /artifact file is missing/);
  assert.match(ui.qs('detailSummary').innerHTML, /失败/);
});

test('all stop reasons have conservative Chinese labels in details and list', () => {
  const ui = page();
  const expected = {
    early_stopping: '提前停止（非训练失败）', epochs_completed: '达到设定训练轮数',
    time_budget: '达到训练时间预算', completed_unknown: '已完成，具体停止原因未知',
    provider_canceled: 'Kaggle 已取消运行', user_canceled: '用户取消', training_failed: '训练失败',
    provider_failed: 'Kaggle 执行失败（训练停止原因另列）',
    monitoring_timeout: 'Relay 监控等待超时（训练结果未确认）',
    status_unavailable: '暂时无法获取状态（训练结果未确认）', relay_failed: 'Relay 处理失败（不代表训练失败）',
  };
  for (const [reason, label] of Object.entries(expected)) {
    const job = {job_id: reason, status: 'complete', stop_details: {reason}};
    ui.renderDetail(job);
    ui.renderJobs([job]);
    assert.ok(ui.qs('detailStop').innerHTML.includes(`停止原因：${label}`));
    assert.ok(ui.qs('jobsBody').innerHTML.includes(`停止记录：${label}`));
    assert.equal(ui.qs('detailError').hidden, true);
  }
});

test('time limit is only a suspicion and cancellation alone supplies no cause', () => {
  const ui = page();
  const job = {job_id: 'canceled', status: 'failed', stop_details: {reason: 'provider_canceled',
    source: 'provider_status', confidence: 'inferred', provider_status: 'CANCEL_ACKNOWLEDGED',
    suspected_reason: 'time_limit'}};
  ui.renderDetail(job);
  assert.match(ui.qs('detailStop').innerHTML, /疑似原因：达到 Kaggle 运行时限（推测，未确认）/);
  assert.match(ui.qs('detailStop').innerHTML, /确定程度：推测，未确认/);
  assert.match(ui.qs('detailStop').innerHTML, /平台取消状态本身不能确定取消原因/);
  delete job.stop_details.suspected_reason;
  ui.renderDetail(job);
  assert.doesNotMatch(ui.qs('detailStop').innerHTML, /运行时限/);
  for (const reason of ['monitoring_timeout', 'status_unavailable']) {
    ui.renderDetail({...job, stop_details: {reason}});
    assert.match(ui.qs('detailStop').innerHTML, /不能据此判断训练已停止或失败/);
    assert.match(ui.qs('detailStop').innerHTML, /观测层级：Relay 监控结果/);
  }
});

test('provider failure after training retains the training stop reason and export error', () => {
  const ui = page();
  ui.renderDetail({job_id: 'export-failed', status: 'failed', error: 'ONNX export failed',
    stop_details: {reason: 'provider_failed', training_reason: 'epochs_completed',
      provider_status: 'ERROR', confidence: 'confirmed', source: 'provider_status'}});
  const content = ui.qs('detailStop').innerHTML;
  assert.match(content, /停止原因：Kaggle 执行失败（训练停止原因另列）/);
  assert.match(content, /训练停止原因：达到设定训练轮数/);
  assert.match(content, /观测层级：Kaggle 平台状态/);
  assert.match(content, /平台执行失败不等于训练阶段失败/);
  assert.match(content, /Kaggle 原始状态：ERROR/);
  assert.match(ui.qs('detailError').innerHTML, /ONNX export failed/);
  assert.doesNotMatch(content, /ONNX export failed|训练停止原因：训练失败/);
});

test('missing or malformed stop details clear previous content without inventing reasons', () => {
  const ui = page();
  for (const stop_details of [undefined, null, {}, [], 'early_stopping', 1]) {
    ui.renderDetail({job_id: 'previous', stop_details: {reason: 'early_stopping'}});
    ui.renderDetail({job_id: 'older', status: 'complete', stop_details});
    assert.equal(ui.qs('detailStop').hidden, true);
    assert.equal(ui.qs('detailStop').innerHTML, '');
    ui.renderJobs([{job_id: 'older', status: 'complete', stop_details}]);
    assert.doesNotMatch(ui.qs('jobsBody').innerHTML, /停止记录/);
  }
  for (const reason of ['new_reason', '__proto__', 'constructor', undefined]) {
    ui.renderDetail({job_id: 'unknown', stop_details: {reason, confidence: 'new_confidence',
      source: 'constructor', epoch: null, epochs: '500', best_epoch: -1, patience: false, elapsed_seconds: Infinity}});
    const content = ui.qs('detailStop').innerHTML;
    assert.match(content, /停止原因：停止原因未知/);
    assert.match(content, /确定程度：未知/);
    assert.doesNotMatch(content, /正常结束|训练失败|运行时限|最佳轮次|已回报轮次|设定总轮数|早停等待轮数|已用时间/);
  }
});

test('stop details escape every supplied text field and preserve zero numeric values', () => {
  const ui = page();
  const attack = '<img src=x onerror="alert(1)"><script>alert(1)</script>&';
  const stop_details = Object.fromEntries(['reason', 'training_reason', 'source', 'confidence',
    'message', 'provider_status', 'suspected_reason'].map(key => [key, attack]));
  Object.assign(stop_details, {epoch: 0, best_epoch: 0, patience: 0, elapsed_seconds: 0});
  ui.renderDetail({job_id: 'unsafe', status: 'failed', error: 'independent error', stop_details});
  ui.renderJobs([{job_id: 'unsafe', status: 'failed', error: 'independent error', stop_details}]);
  const content = ui.qs('detailStop').innerHTML;
  assert.doesNotMatch(content + ui.qs('jobsBody').innerHTML, /<img|<script/);
  assert.match(content, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
  assert.match(content, /已回报轮次：0/);
  assert.match(content, /早停等待轮数：0/);
  assert.match(content, /已用时间：0 秒/);
  assert.match(ui.qs('jobsBody').innerHTML, /independent error/);
});

test('overview loads uncapped counts and clears stale values on error', async () => {
  const ui = page();
  let url;
  ui.api = async path => { url = path; return {total: 310, in_progress: 2, queued: 5, failed: 300, complete: 2, canceled: 1}; };
  await ui.loadJobOverview(0);
  assert.equal(url, '/v1/jobs/summary');
  assert.equal(ui.qs('failedCount').textContent, '300');
  assert.match(ui.qs('jobOverviewScope').textContent, /310/);
  ui.api = async () => { throw new Error('offline'); };
  await ui.loadJobOverview(0);
  assert.equal(ui.qs('failedCount').textContent, '—');
  assert.match(ui.qs('jobOverviewScope').textContent, /暂时无法加载/);
});

test('key permission hides navigation and prevents direct page selection', () => {
  const ui = page();
  const buttons = ['runtime', 'accounts', 'users'].map(pageNav => ({dataset: {pageNav}}));
  ui.document.querySelectorAll = () => buttons;
  ui.applySessionPermissions({can_view_keys: false});
  assert.equal(buttons[0].hidden, false);
  assert.equal(buttons[1].hidden, true);
  assert.equal(buttons[2].hidden, false);
  assert.equal(buttons[2].textContent, '我的 Token');
  assert.equal(ui.normalizePage('accounts'), 'runtime');
  assert.equal(ui.normalizePage('users'), 'users');
  ui.applySessionPermissions({can_view_keys: true});
  assert.equal(buttons[1].hidden, false);
  assert.equal(ui.normalizePage('accounts'), 'accounts');
});

test('receiving deadline errors have a clear failure summary', () => {
  const ui = page();
  assert.equal(ui.failureSummary({status: 'failed', error: 'upload timed out: incomplete after 3 hours from job creation; submit a new job'}).title, '上传接收超时');
});

function tokenPage() {
  const ui = page();
  vm.runInContext('activePage = "users"', ui);
  const timers = [];
  ui.setTimeout = (fn, delay) => { timers.push({fn, delay}); return timers.length; };
  ui.clearTimeout = () => {};
  ui.document.createElement = () => ({dataset: {}});
  ui.qs('userActionMessage').appendChild = item => { ui.actionButton = item; };
  ui.qs('revealedTokenValue').focus = () => {};
  ui.qs('revealedTokenValue').select = () => { ui.selected = true; };
  return {ui, timers};
}

test('reveal and copy are explicit and clear after 60 seconds', async () => {
  const {ui, timers} = tokenPage();
  let copied, request;
  ui.navigator = {clipboard: {writeText: async value => { copied = value; }}};
  ui.api = async (url, options) => { request = {url, options}; return {token: 'fixture-secret'}; };
  await ui.revealUserToken('user/a', true);
  assert.equal(request.url, '/v1/auth/relay-tokens/user%2Fa/reveal');
  assert.equal(request.options.method, 'POST');
  assert.equal(request.options.cache, 'no-store');
  assert.equal(copied, 'fixture-secret');
  assert.equal(ui.qs('revealedTokenValue').value, 'fixture-secret');
  assert.equal(timers[0].delay, 60000);
  timers[0].fn();
  assert.equal(ui.qs('revealedTokenValue').value, '');
  assert.equal(ui.qs('revealedTokenBox').hidden, true);
});

test('failed clipboard offers manual selection; hiding clears the secret', async () => {
  const {ui} = tokenPage();
  ui.navigator = {};
  ui.api = async () => ({token: 'manual-secret'});
  await ui.revealUserToken('user-a', true);
  assert.equal(ui.selected, true);
  assert.match(ui.qs('userActionMessage').textContent, /手动复制/);
  ui.clearRevealedToken();
  assert.equal(ui.qs('revealedTokenValue').value, '');
});

test('late reveal responses cannot repopulate after navigation, hide, or logout', async () => {
  const {ui} = tokenPage();
  let finish;
  ui.api = () => new Promise(resolve => { finish = resolve; });
  const pending = ui.revealUserToken('user-a');
  ui.clearRevealedToken();
  finish({token: 'late-secret'});
  await pending;
  assert.equal(ui.qs('revealedTokenValue').value, '');
  assert.equal(ui.qs('revealedTokenBox').hidden, true);
  assert.match(html, /async function logout\(\) \{\s+clearRevealedToken\(\)/);
  assert.match(html, /function setActivePage[^]*?clearRevealedToken\(\)/);
  assert.match(html, /pagehide", clearRevealedToken/);
});

test('cancel deletion makes no request; conflict exposes exact owner task link', async () => {
  const {ui} = tokenPage();
  let calls = 0;
  ui.confirm = () => false;
  ui.api = async () => { ++calls; };
  await ui.deleteRelayUser('user-a');
  assert.equal(calls, 0);
  ui.confirm = () => true;
  ui.api = async () => { throw {detail: {code: 'user_has_active_jobs', active_job_count: 3}}; };
  await ui.deleteRelayUser('user-a');
  assert.match(ui.qs('userActionMessage').textContent, /3 个未结束任务/);
  assert.equal(ui.actionButton.dataset.userJobs, 'user-a');
  ui.setActivePage = () => {};
  ui.loadRuntimeData = async () => {};
  await ui.filterUserJobs('user-a');
  assert.equal(ui.jobListParams().get('owner'), 'user-a');
  await ui.filterUserJobs(null);
  assert.equal(ui.jobListParams().has('owner'), false);
});

test('successful deletion reloads users and rendering contains no raw token', async () => {
  const {ui} = tokenPage();
  ui.confirm = () => true;
  let refreshed = false;
  ui.api = async () => {};
  ui.loadAuthConfig = async () => { refreshed = true; };
  await ui.deleteRelayUser('user-a');
  assert.equal(refreshed, true);
  assert.match(ui.qs('userActionMessage').textContent, /已删除/);
  ui.qs('authConfig').appendChild = () => {};
  ui.renderAuthConfig({mode: 'multi_key', can_manage_auth: false, relay_tokens: [{id: 'user-a', current: true, token: 'never-render'}]});
  assert.match(ui.qs('authConfig').innerHTML, /查看 Token/);
  assert.doesNotMatch(ui.qs('authConfig').innerHTML, /never-render|data-delete-user/);
  assert.equal(ui.qs('revealedTokenValue').value, '');
});
