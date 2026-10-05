# A3 Dataset create preflight fix — review before deployment

Base: `5ee0ba98bec5d7b1b1fd135c915c6cdcac9bfdcf`. The live CNN job stopped on status403 **before** create/version. This was not evidence of a delayed response to that job's create.

The upload path now verifies the real owner, then reads authenticated personal Dataset inventory with `dataset_list(mine=True, page=N)`, no search filter. An exact existing ref uses the current-version check. A completed inventory without the ref allows only a create attempt for candidate1, preceded by a durable scoped unknown intent with the frozen content digest and selection basis.

Inventory absence is not treated as infallible proof against indexing delays/races. The create business response decides acceptance. A conflict/rejection never becomes a version operation. None/malformed/repeated/incomplete pages and HTTP403 fail before intent/mutation; 1000 pages is a conservative upper bound. Existing-ref status403 still fails; `dataset_exists()` is unchanged.

Existing accepted/unknown intent recovery bypasses enumeration and mutation, reads back only the original candidate, and retains exact-content checks. Creation visibility delay retries only the same candidate `/1`; it does not create again or query latest. Intent write failure never uploads. No permissions, credentials, source snapshots, job identities or cloud dependency policy changed.

Strict mine/page fakes cover new refs, later-page existing refs, malformed/403 inventory, creation conflict, dropped response, delayed403 readback and intent persistence failure. Previous token/OAuth/key, transport tamper/mixed layout, candidate conflict, unknown recovery and hard-exit tests remain.
Installed Kaggle2.2.4 source signature was read without importing credentials; autospec accepts mine/page and rejects page_size. No SDK installation or upgrade.

Commands from this worktree with `C:/Users/jsdfhasuh/my_scripts/kaggle_relay/.venv/Scripts/python.exe`:

- Baseline `-m pytest tests/test_a3_content_identity.py tests/test_concurrency.py tests/test_relay_api.py -q`: exit0,157 passed,1 skipped.
- After first eight new cases, same selection: exit0,165 passed,1 skipped.
- Final after two further visibility/persistence cases, `-m pytest tests -q`: exit0,346 passed,2 Windows platform skips. No new failures.

Related application fix preserves canonical artifacts over same-name test diagnostics and validates the received file manifest. Its regression selection has153 passed, the same2 baseline failures,3 skips; other store/audit/resume/monitor/Relay checks103 passed. Full details and actual cloud-output receive replay are in the application branch's `docs/evidence/kaggle-build-a3/live-20260926/fix-review.md`.

This repair is **not deployed**. Production remains reviewed code33e361c. No new Dataset/Kernel, retry of the failed historical job, config update, dependency install, main merge or later-phase work occurred. The old live report remains unchanged; new-create cloud acceptance and outstanding A3 matrix remain PENDING/NOT_RUN. Stop for A3 review.
