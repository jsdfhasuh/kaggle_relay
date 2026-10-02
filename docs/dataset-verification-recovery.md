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

Desktop clients show verification pause/retry separately from training failure.
Detailed errors remain in the diagnostic output. Installing server code does
not update an already running desktop process or a packaged desktop release.

This change does not add persistent per-file verification checkpoints or
account-wide request throttling. Whole-archive verification remains preferred;
the existing exact-version file fallback is unchanged.

Validation on Windows: full Relay suite 503 passed, 3 skipped; desktop recovery
and monitor tests 59 passed. Production Linux child process cleanup is checked
separately during deployment. Neither mocked recovery tests nor health checks
constitute a completed real Kaggle training run.
