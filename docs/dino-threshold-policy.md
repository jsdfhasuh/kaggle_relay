# DINO gateway threshold policy

New DINO v3 submissions using base runtime
`d31e1303ff94dd205ea5a9438cfb3f5247aa4002a0af876b2bc06c6f5fe5a266`
receive an explicit calibration overlay in the extracted Python entrypoint before
Kaggle submission, including dataset cache hits. Existing Kaggle runs are not
rewritten or resubmitted. YOLO and ordinary PatchCore submissions are unaffected.

The overlay lowers the F1 validation threshold by
`2 * (1e-5 + 1e-4 * abs(original_threshold))`. If another calibration score is
immediately below that threshold, the reduction is capped at half that gap.
Every calibration label must remain unchanged. Nonfinite values and a gap with
no representable lower threshold are rejected. Normal-only quantile calibration
retains its existing behavior. This addresses numerical label flips near the
inclusive `score >= threshold` boundary; it is not a guarantee of accuracy on
unseen data or of successful training for every dataset.

The policy wraps calibration before metrics, threshold receipts and PT export
are generated. `calibration.gateway_threshold_policy` records the policy ID,
policy source digest, base runtime digest, original/effective thresholds, and
requested/applied margins. The policy is separate from the frozen base runtime:
original upload ZIPs, wheel/source bytes, frozen dataset/run identities and their
hash validation are retained. Runtime and calibration source integrity checks
execute before the overlay. Native PT tensor/score/label verification and artifact
hash verification remain mandatory. No failed report is changed to PASS.

The supported entrypoint imports `run` from
`patchcore_dino_runtime.kaggle_bootstrap` and launches the schema-3 isolated cloud
worker. Unknown base runtime digests run unchanged with a skip message; other
entrypoint layouts are rejected rather than implicitly patched. The policy digest
is logged before submission, and the exact threshold reduction is written to
worker stderr and the bound calibration receipt.

## Opaque RGBA compatibility

The gateway also installs the `dino_opaque_rgba_rgb_v1` image reader in the
supported isolated worker. Before training, it verifies and decodes every frozen
sample. Only uint8 four-channel images whose alpha values are all 255 are read
as RGB, with the base decoder's IMREAD_COLOR orientation. RGB and grayscale8
retain the original decoder. Partial transparency, higher bit depths, corrupt
images and hash mismatches fail with the sample filename. Original image bytes,
archives and frozen identities are never rewritten.

The same reader serves training, calibration, test scoring and native verification
reference inputs. The independent target probe retains its original PIL RGB
conversion and integrity checks. Multiprocess loading requires the supported
Linux fork mode; another start method fails explicitly instead of losing the
overlay in child processes. Normal-only calibration also carries the image
receipt without changing its threshold calculation.

`calibration.gateway_image_policy` records the image policy source digest, base
runtime, converted sample count and SHA-256 of the sorted `(sample_id, original
image SHA-256)` pairs serialized as compact ASCII JSON. The receipt propagates
into the native PT source-threshold receipt. Exact previous gateway overlays are
authenticated and replaced on a submission retry; current rewriting is idempotent.
Unknown base runtimes keep their original behavior. Already submitted kernels
are not changed or retried.

Image test dependencies are in `requirements-test.txt`; the gateway production
container does not need OpenCV, NumPy or Pillow. Image decoding runs on Kaggle.

Dynamic account callbacks continue authenticating with their original kernel
alias. Progress responses expose the actual bound kernel/dataset references, so
older desktop clients cannot overwrite the assigned reference with that alias.

## Verification

Tests cover the observed boundary flip, preservation of calibration labels,
narrow/unrepresentable gaps, nonfinite rejection, an actual isolated Python
launcher, integrity failure propagation, repeatable entrypoint rewriting, and
submission with/without dataset cache while preserving upload hashes and run
identity.

The original uploaded wheel and saved scores for the September 22 incident were
replayed locally. The threshold decreased from `0.6087480783462524` to
`0.6086063287305832`. Single-image label mismatches decreased from one to zero
across 17 samples; all 17 batch comparisons also agreed. The original wheel's
input, runtime, threshold and metric validators passed. This is recorded-score
replay and submission validation, not a new cloud GPU training acceptance run.
