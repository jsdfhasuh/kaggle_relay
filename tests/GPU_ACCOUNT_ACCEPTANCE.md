# Live GPU account verification

`acceptance_gpu_accounts.py` runs one private Kaggle Notebook per enabled real
account, grouping Key aliases by username. It requests Tesla T4 GPUs and checks
an actual CUDA matrix multiplication using Kaggle's preinstalled PyTorch.
It consumes GPU quota and creates retained private Notebooks. Run it explicitly
with production authorization, outside other training acceptance phases.

Copy the script into a private directory mounted in the Relay container, then
run it with `PYTHONPATH=/app`, a fresh `--run-id`, and a private `--output`
directory. The container supplies its existing configuration; do not pass
credentials on the command line.

```sh
python acceptance_gpu_accounts.py --run-id check-001 --output /data/private-gpu-check/check-001 --timeout 900
```

Each account has a durable `state.json` written before submission. Running the
same ID and output again observes the recorded Notebook without another push.
If a submission response is uncertain, inspect that Notebook before choosing a
new run. Reports contain account IDs and runtime evidence, but no credentials.

- `gpu_passed`: CUDA is available and GPU computation succeeded.
- `no_cuda_device`: CUDA is unavailable and the runtime reports zero devices.
- `unverified`: API, download, runtime or observation failure; do not infer that
  the account lacks GPU access.

This script does not disable Keys or cancel Notebooks. A timeout does not prove
the remote GPU stopped. Verify official terminal status before starting another
phase. A GPU probe is not full dataset/training/concurrency acceptance.

After confirming a bad account, use the existing admin Key settings or
`PATCH /v1/auth/kaggle-keys/{id}` with `enabled=false` and a factual
`disabled_reason`. Disable every alias of that account. Back up `auth.json`
privately first, and verify credentials and token permissions are unchanged.
Disabled Keys cannot receive new fixed/dynamic work; history remains accessible.
Already bound jobs keep their binding. Pending jobs wait if no enabled authorized
account remains. After a successful new GPU probe, explicitly reenable the Key.

Reported quota and successful authentication alone do not prove GPU access.
Do not attribute missing CUDA to phone verification without separate evidence.
