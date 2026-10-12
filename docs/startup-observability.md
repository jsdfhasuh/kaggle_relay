# Startup Progress Ordering

Startup observations are diagnostics, not epochs or training percentages. Both
runtime callbacks and Kaggle log polling validate the same startup fields.
Callback authentication and frozen job identity checks are unchanged.

- Startup log events use `(startup_id, startup_sequence)` for deduplication.
- PatchCore phase heartbeats include `reported_at` in their deduplication key.
- Older startup sequences and delayed restart-parent observations are ignored.
- Once an epoch or a post-preparation PatchCore phase is stored, late startup
  observations cannot replace it. CNN `prepare` and DINO `input` may coexist
  with startup diagnostics before handoff.
- Startup observations cannot replace output-download or terminal job state.
- The callback and log-poll writers compare both job status and the exact previous
  `kernel_status` in their SQL update. A concurrent update triggers a fresh read
  and ordering check rather than accepting a stale decision.

No database migration is required. Existing jobs, Dataset versions, upload
intents and artifact bindings are retained. Runtime HTTP socket timeouts do not
bound DNS resolution; correctness must not rely on a bounded thread join having
stopped every in-flight request.

The service cannot create logs for an already submitted script that never
emitted startup observations. New runtime instrumentation requires a newly
frozen task from the updated desktop source or a rebuilt desktop package.
