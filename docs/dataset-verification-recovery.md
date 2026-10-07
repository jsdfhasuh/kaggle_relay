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
upload response, the next check is persisted in SQLite. Delays are 5, 10, 20,
then at most 30 minutes; a longer Kaggle Retry-After always takes precedence.
After 24 hours from the first scheduled recovery, automatic checks stop explicitly
with `exhausted`; the original candidate and manual recovery action remain.
Authentication, identity and content failures are not automatically retried.

`dataset_recheck_state`, `dataset_recheck_at`, `dataset_recheck_started_at` and
`dataset_recheck_count` are returned by the job API. The web UI shows the next
check in local time, or an explicit manual-inspection/recovery requirement.
The original error remains visible while scheduled; every scheduled/claimed
check is logged. Restart preserves the schedule rather than resetting backoff.

Background checks never call dataset creation/version upload. They validate the
retained intent, ref, directory and content digest, and check only that exact
version's archive. A missing archive defers to the next scheduled check instead
of downloading thousands of files. Once the archive content is verified, the
normal guarded Kernel submission path continues. Active, canceled and terminal
jobs are excluded; compare-and-set claims and the existing worker guard prevent
duplicate processing. Manual recovery uses the same retained-candidate path.

Desktop clients show verification pause/retry separately from training failure.
Detailed errors remain in the diagnostic output. Installing server code does
not update an already running desktop process or a packaged desktop release.

This change does not add persistent per-file verification checkpoints or
account-wide request throttling. Whole-archive verification remains preferred;
the initial foreground verification fallback is unchanged. Scheduled background
rechecks wait for the complete archive instead of restarting per-file checks.

Validation on Windows: full Relay suite 503 passed, 3 skipped; desktop recovery
and monitor tests 59 passed. Production Linux child process cleanup is checked
separately during deployment. Neither mocked recovery tests nor health checks
constitute a completed real Kaggle training run.
