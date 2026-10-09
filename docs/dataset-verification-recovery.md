# Dataset verification recovery

Exact-version verification distinguishes typed transport failures from content
integrity failures. Connection/read timeouts, broken streams, and HTTP
408/409/429/500/502/503/504 retry the same frozen candidate up to five times.
Backoff starts at five seconds, doubles to sixty seconds, and includes up to
three seconds of jitter. A valid Retry-After (seconds or HTTP date) takes
precedence when longer. If it exceeds the remaining budget, verification pauses
without sending an early retry.

The recovery window is at most ten minutes from the first transient failure,
including subsequent content transfers. The existing publication and transfer
budgets still apply. The dedicated verification SDK subprocess caps Requests
connection/read timeouts at 15/60 seconds; it preserves shorter existing values.
Cancellation and shutdown remain effective during backoff and child execution.
This does not retry dataset creation or Kernel submission.

When retries are exhausted, the worker retains the original accepted/unknown
upload intent and reports `waiting_dataset` with the existing
`dataset_upload_outcome_unknown:` prefix. The original job's recovery action
rechecks that candidate. A newer Dataset version or shared cache never replaces
it. Actual hash/identity failures still prevent Kernel submission.

## Server-side automatic rechecks

The server scans retained `waiting_dataset` jobs every 15 seconds, independently
of desktop clients. After a classified publication/transport failure or an unknown
upload response, the next check is persisted in SQLite. Consecutive failure
delays are 5, 10, 20, then at most 30 **minutes**; a longer Kaggle Retry-After
always takes precedence. A successful partial-file batch continues after 15
seconds and resets consecutive failure backoff, not the original recovery window.
After 24 hours from the first scheduled recovery, automatic checks stop explicitly
with `exhausted`; the original candidate and manual recovery action remain.
Authentication, identity and content failures are not automatically retried.

`dataset_recheck_state`, `dataset_recheck_at`, `dataset_recheck_started_at` and
`dataset_recheck_count` are returned by the job API. `dataset_recheck_failures`
separately tracks consecutive failures so progress batches do not increase
exponential backoff. The web UI shows the next
check in local time, or an explicit manual-inspection/recovery requirement.
The original error remains visible while scheduled; every scheduled/claimed
check is logged. Restart preserves the schedule rather than resetting backoff.

Background checks never call dataset creation/version upload. They validate the
retained intent, ref, directory and content digest, and check only that exact
version. Whole-archive verification is preferred. Only Kaggle's explicit 404
`No gcs url found` allows an exact-version per-file fallback. After all original
content is verified, the
normal guarded Kernel submission path continues. Active, canceled and terminal
jobs are excluded; compare-and-set claims and the existing worker guard prevent
duplicate processing. Manual recovery uses the same retained-candidate path.

Ordinary `POST /v1/jobs/{job_id}/complete` calls are idempotent while Dataset
recovery is scheduled/checking: they return the current job without shortening
cooldown, resetting backoff/window or queueing another worker. An exhausted
24-hour window requires an explicit
`POST /v1/jobs/{job_id}/complete?restart_recheck=true`, also available through
the web UI's confirmed "重新开始核验" action. This starts a new recovery window
for the same candidate, not a new upload or training job. A blocked integrity,
identity or authorization condition requires manual inspection and cannot be
overridden by this flag.

## Persistent file verification and account pacing

`/data/dataset-verification.sqlite3` persists expected file sizes/digests and
completed verification evidence. A checkpoint binds the Dataset ref, exact
version, original job directory and frozen source content SHA-256. A different
binding/inventory or malformed completion evidence is rejected, never silently
discarded or substituted. Each file is committed only after its complete stream
matches both byte count and SHA-256; interrupted files are re-read, while
completed files survive process/container restart. Owner identity, frozen local
content, and the full remote inventory are still rechecked before completion.

Background fallback is limited to 64 new files and a 120-second batch budget.
Cancellation remains checked during pacing and streamed file reads. Batch
yielding preserves checkpoints and releases the worker slot. These limits do
not permit partial content to reach Kernel submission.

Account request reservations in the same SQLite store space Kaggle API calls by
1.5 seconds across jobs, credential aliases and verification child processes.
A Kaggle 429 persists an account-wide cooldown from Retry-After (60 seconds
when unavailable). Later shorter cooldowns cannot shorten it; queued reservations
and the scheduler respect it after restart. Google Storage transfers are not
charged as Kaggle API requests. This rate is a conservative request policy, not
a guarantee that Kaggle will never rate-limit.

The API exposes `relay_input_received`, `dataset_upload_state` and safe scoped
`dataset_verification` counters. The web and desktop distinguish received
inputs, accepted upload, pending verification, and completed verification. The
existing `dataset_upload_required` cache/transfer contract is unchanged; it is
not interpreted as an instruction to upload an already accepted candidate again.

Desktop clients show verification pause/retry separately from training failure.
Detailed errors remain in the diagnostic output. Installing server code does
not update an already running desktop process or a packaged desktop release.

Regression coverage includes checkpoint interruption/resume, exact-version
identity/content guards, account cooldown persistence, bounded background
batches, idempotent recovery and desktop/web presentation. Production Linux
child process cleanup is checked separately during deployment. Neither mocked
recovery tests nor health checks
constitute a completed real Kaggle training run.
