# YOLO small-dataset GPU policy

The gateway inserts a guard around YOLO `.train(data=...)` calls in GPU-enabled
Python submissions. It runs on Kaggle after the client's extraction and YAML
path repair, immediately before training. The guard uses Ultralytics' own
dataset resolver and image/label validation to count usable train/val samples.

If either split has fewer samples than the requested GPU count, the guard uses
the first requested GPU. For two T4 GPUs, a one-image validation set uses one
GPU; 220 train / 27 val keeps two GPUs. Explicit CPU and single-GPU requests
remain unchanged. Empty or invalid datasets raise the normal dataset error.
The test split does not affect this decision.

This prevents the known empty validation shard hang, without disabling multi-GPU
training for adequate datasets. The preflight adds a dataset validation pass;
existing valid Ultralytics label caches can be reused. The actual counts and
decision are printed with `[RELAY GPU POLICY]` in the Kaggle log.

The policy covers old generated client scripts, fixed/dynamic scheduling,
fresh uploads and dataset cache hits. Ordinary Python notebook cells are also
supported; IPython magic cells are preserved and are not rewritten. This is an
application execution policy, not a sandbox for arbitrary user code. Custom
calls without an explicit `data` keyword or unsupported tasks are not wrapped.

Only extracted submission code changes. Original ZIPs and their hashes,
dataset versions, callbacks, permissions, PatchCore contracts and cross-account
concurrency are preserved. Already-submitted Kaggle runs keep their old code;
cancel and confirm remote termination before resubmitting a stalled job.
This change adds no heartbeat or timeout and makes no GPU billing guarantees.
