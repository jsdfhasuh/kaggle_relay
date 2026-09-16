# PatchCore dual output transport

2026-09-16. Branch: codex/patchcore-dual-artifacts.

Allow the two additional canonical root outputs deployment_model.pt and pt_export.json
in the PatchCore Kaggle download whitelist. Preserve required model.ckpt and existing
metadata; old CKPT-only jobs remain valid. Do not download duplicate nested checkpoints
or unrelated PT files. The desktop continues strict manifest/RunPlan/hash validation.

Verification: full pytest tests suite, 128 passed, 1 warning. The archive tests use
fixture bytes to verify transport only, never native model or EXE compatibility.

Production deployment: NOT_RUN. This branch must be deployed through the normal Relay
workflow after separate authorization before clients can retrieve both cloud outputs.
No production service, container, credentials, database or artifact retention changes.
