# Upload concurrency

`RELAY_MAX_PARALLEL_UPLOADS_PER_USER` defaults to **8**. All jobs belonging to
one authenticated Relay principal share that budget. The existing global
`RELAY_MAX_PARALLEL_UPLOADS` budget remains **40** by default.

Job responses advertise `max_parallel_uploads` as the smaller of these two
configured limits. Excess requests receive HTTP 429 with `Retry-After: 1`;
disconnects, failed checksums and successful receipts release the occupied slot.
No database migration or change to the chunk receipt contract is required.

Compatible clients choose the minimum of the user preference, advertised
limit and missing chunk count. Existing clients capped at four still work.
Client transfer-speed telemetry is not a durable chunk acknowledgement.

Validation on the production-compatible `770db9c` base: full suite **384 passed,
2 skipped** before adding the cross-user regression; the subsequent focused
limit suite (including that regression) passed **4 tests**. It verifies eight
held uploads, cross-job sharing, distinct principals sharing the global budget,
429 backoff and slot release. Production deployment verification is recorded
separately; source tests alone do not establish live acceptance.
