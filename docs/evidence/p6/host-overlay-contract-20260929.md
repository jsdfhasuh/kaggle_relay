# Host-overlay DINO GPU contract

Baseline `f5e7c18`, runtime previously `3975d15`. Adds explicit capability `dino_cuda_host_torch210_cu128_cpu_coreset_v2` with `torch_index=host_cu128` and `environment=verified_host_overlay_v1`. Preserves the original cu126 GPU policy and old CPU/YOLO/CNN/DINO contracts. Strictly requires P6 v2, Kernel GPU enabled and CPU deployment execution; no uploaded snapshot is rewritten.

Application uses a verified host GPU stack with private supplementary packages, rather than installing another Torch/CUDA stack. Host Torch 2.10.0+cu128 / Torchvision 0.25.0+cu128 / NumPy 2.0.2 and real GPU operations must pass before installation. Independent ORT remains free of host training packages. New capability cannot inherit the old GPU qualification.

Tests: existing Relay interpreter, `pytest tests/test_dinov2_training_policy.py tests/test_dinov2_p6.py tests/test_dinov2_artifacts.py tests/test_relay_api.py -q --disable-warnings`. Policy/health tests cover both profiles and immutable request preservation. Real GPU training remains user acceptance.

Result: **176 passed, 2 warnings, exit 0, 31.79 seconds**. No cloud job or production mutation performed. The application also rejects a service advertising only the older cu126 GPU capability before preparing uploads.

Production update is pending explicit authorization for this new commit. Concrete rollout: recheck clean `/docker_volume/kaggle_relay`, no active jobs, back up configuration/auth/consistent SQLite/intents; retain actual old image; build source-only image from current image with no dependency installation; use existing Compose config/mounts with a fixed image override; verify all running source hashes, dependency inventory, authenticated public health and both GPU profiles. Retain rollback script/image and never restore an old DB over new jobs. No training submission, business data upload or EXE build.
