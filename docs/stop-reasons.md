# Training Stop Observations

Relay stores `stop_details` separately from the job status, `kernel_status`
progress, and artifact-processing `error`. It is exposed as an object by the
existing job APIs and displayed in the job list and details. The additive SQLite
migration defaults older records to `{}`. Retention cleanup keeps this field.

The record includes `reason`, `source`, `confidence`, `message`, and an allowlisted
`provider_status`. Available evidence adds `epoch`, `epochs`, `best_epoch`,
`patience`, and `elapsed_seconds`. Epoch numbers are one-based. A runtime time
budget may change the trainer's epoch threshold, so `time_budget` can have an
observed epoch larger than that dynamic threshold.

| Reason | Meaning |
| --- | --- |
| `training_stopped` | A runtime callback recorded training ending; Kaggle/artifact processing is still pending. |
| `early_stopping` | Runtime state or the standard EarlyStopping log confirms no improvement within patience. |
| `epochs_completed` | Runtime/final-progress evidence identifies the configured final epoch and Kaggle completed. |
| `time_budget` | Runtime state confirms its configured training budget caused stopping. |
| `completed_unknown` | Kaggle completed, but its training stop reason is unavailable. |
| `provider_canceled` | Kaggle confirmed cancellation; its cause is not established. |
| `user_canceled` | Relay cancellation was recorded, with provider confirmation when available. |
| `provider_failed` | Kaggle reported kernel failure; the failing stage may be unknown. |
| `monitoring_timeout` / `status_unavailable` | Monitoring is uncertain; this does not finalize a running job. |
| `relay_failed` | Relay processing failed without an authoritative training stop observation. |

A last provider log near twelve hours may add `suspected_reason: time_limit`.
This remains a suspicion, not a confirmed cancellation cause. Notebook names and
arbitrary error strings cannot be interpreted as provider status tokens.

The desktop YOLO runtime emits `TRAINING_PLATFORM_STOP {json}` before downstream
artifact processing, and sends an authenticated `training_stop` callback. The
callback preserves progress and does not finalize the job. Auto-DDP does not
propagate parent callbacks to child trainers: its parent reports only supported
saved-row evidence, while Relay can also parse the child's standard early-stop
log. Unknown DDP causes remain unknown. Frozen older submissions remain unchanged.

Later monitoring/log failures cannot erase previously recorded training evidence.
When the provider fails after training ends, `training_reason` retains the training
outcome while `reason` records the provider failure. A late runtime callback cannot
replace an already-observed terminal provider result. Stop-reporting failures do
not replace the original worker or training outcome.

No speed, batch, patience, model, or GPU policy is changed by this feature. A
`patience=200` run saves improving best weights normally; it only waits longer
before early stopping. This record is not proof of checkpoint download integrity.

Deployment must preserve the current image/dependencies, back up configuration
and a consistent database, and verify active jobs before recreating the service.
Rollback restores source/image only; never replace live data with an old snapshot.
