"""Transport fixtures only: these bytes are not executable models."""

import hashlib
import io
import json
import re
import shutil
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.archive import ArchiveError
from app.database import RelayDb
from app.dinov2_artifacts import (ARTIFACT_CONTRACT, ARTIFACT_FORMAT, IDENTITY_FIELDS,
                                 MANIFEST_NAME, REQUIRED_FILES, package_artifacts)
from app.kaggle_adapter import KaggleAdapter
from app.worker import finish_kernel_job
from test_relay_api import auth_headers, create_app, job_request_body, make_settings

IDENTITY = dict(zip(IDENTITY_FIELDS, ("a" * 64, "b" * 64, "run-dino", "c" * 64)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "download" / "artifacts"
    root.mkdir(parents=True)
    rows = []
    for name in sorted(REQUIRED_FILES):
        raw = ("fixture " + name).encode("ascii")
        (root / name).write_bytes(raw)
        rows.append({"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    document = {**IDENTITY, "artifact_format": ARTIFACT_FORMAT, "schema_version": 3,
                "backend": "patchcore", "task": "anomaly", "implementation": "anomalib",
                "anomalib_version": "2.2.0", "training_status": "completed", "export_status": "PASS",
                "pt_model_path": "com_dinov2_small.pt", "pt_export_path": "pt_export.json",
                "verification_path": "pt_verification.json",
                "source_artifact_manifest_path": "source_artifacts.json", "artifacts": rows,
                "artifact_set_sha256": hashlib.sha256(canonical(rows)).hexdigest()}
    (root / MANIFEST_NAME).write_bytes(canonical(document))
    return root, document, tmp_path / "artifacts.zip"


def test_download_and_package_only_exact_artifacts_files(bundle, tmp_path):
    root, document, target = bundle
    adapter = KaggleAdapter(make_settings(tmp_path), lambda _: None)
    adapter._run = Mock(return_value=SimpleNamespace(stdout="downloaded"))
    adapter.download_output("owner/kernel", tmp_path / "empty", ARTIFACT_CONTRACT)
    command = adapter._run.call_args.args[0]
    pattern = command[command.index("--file-pattern") + 1]
    assert all(re.fullmatch(pattern, "artifacts/" + name) for name in REQUIRED_FILES | {MANIFEST_NAME})
    assert re.fullmatch(pattern, "artifacts\\com_dinov2_small.pt")
    for name in ("private.txt", "deployment_model.pt", "x/com_dinov2_small.pt", "../model.ckpt",
                 "x\\model.ckpt", "Patchcore/weights/model.ckpt", "model.ckpt.bak",
                 "com_dinov2_small.pt", "artifacts/nested/model.ckpt", "diagnostics/artifacts/model.ckpt"):
        assert not re.fullmatch(pattern, name)
    (root / "private.txt").write_text("never copy")
    adapter.package_artifacts(root.parent, target, ARTIFACT_CONTRACT, expected_identity=IDENTITY)
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {"artifacts/" + name for name in REQUIRED_FILES | {MANIFEST_NAME}}
        for name in archive.namelist():
            assert archive.read(name) == (root.parent / name).read_bytes()
        assert archive.testzip() is None


@pytest.mark.parametrize("name", sorted(REQUIRED_FILES))
def test_every_required_file_is_mandatory(bundle, name):
    root, _, target = bundle
    (root / name).unlink()
    with pytest.raises(ArchiveError):
        package_artifacts(root, target, expected_identity=IDENTITY)
    assert not target.exists()


@pytest.mark.parametrize("fault", ["path", "duplicate", "unsorted", "hash", "size", "bool_size",
    "manifest_hash", "schema", "anomalib", "status", "export", "identity", "pt_path", "format"])
def test_manifest_failures_do_not_replace_existing_archive(bundle, fault):
    root, doc, target = bundle
    target.write_bytes(b"previous-complete-zip")
    if fault == "path":
        doc["artifacts"][0]["path"] = "../model.ckpt"
    elif fault == "duplicate":
        doc["artifacts"][0] = doc["artifacts"][1]
    elif fault == "unsorted":
        doc["artifacts"].reverse()
    elif fault in {"hash", "size", "bool_size"}:
        key, value = {"hash": ("sha256", "0" * 64), "size": ("size", 1), "bool_size": ("size", True)}[fault]
        doc["artifacts"][0][key] = value
        doc["artifact_set_sha256"] = hashlib.sha256(canonical(doc["artifacts"])).hexdigest()
    else:
        key, value = {"manifest_hash": ("artifact_set_sha256", "0" * 64), "schema": ("schema_version", 3.0),
            "anomalib": ("anomalib_version", "2.5.1"), "status": ("training_status", "failed"),
            "export": ("export_status", "NOT_REQUESTED"), "identity": ("run_id", "other-run"),
            "pt_path": ("pt_model_path", "deployment_model.pt"), "format": ("artifact_format", "future")}[fault]
        doc[key] = value
    (root / MANIFEST_NAME).write_bytes(canonical(doc))
    with pytest.raises(ArchiveError):
        package_artifacts(root, target, expected_identity=IDENTITY)
    assert target.read_bytes() == b"previous-complete-zip"
    assert not list(target.parent.glob(".dino-artifacts-*"))


def test_hashes_the_actual_archived_bytes(bundle):
    root, doc, target = bundle
    row = doc["artifacts"][0]
    path = root / row["path"]
    path.write_bytes(b"x" * row["size"])
    with pytest.raises(ArchiveError, match="hash/size"):
        package_artifacts(root, target, expected_identity=IDENTITY)
    assert not target.exists()
    assert not list(target.parent.glob(".dino-artifacts-*"))


def test_linked_artifact_is_rejected(bundle, monkeypatch):
    from pathlib import Path
    root, _, target = bundle
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == root / "model.ckpt" or original(path))
    with pytest.raises(ArchiveError, match="unsafe"):
        package_artifacts(root, target, expected_identity=IDENTITY)
    assert not target.exists()


def test_new_contract_does_not_accept_legacy_root_layout(bundle, tmp_path):
    root, _, target = bundle
    adapter = KaggleAdapter(make_settings(tmp_path), lambda _: None)
    with pytest.raises(ArchiveError):
        adapter.package_artifacts(root, target, ARTIFACT_CONTRACT, expected_identity=IDENTITY)
    assert not target.exists()


@pytest.mark.parametrize("fault", ["disk", "write", "replace"])
def test_io_failure_cleans_temporary_archive(bundle, monkeypatch, fault):
    root, _, target = bundle
    target.write_bytes(b"complete")
    budget = Mock()
    if fault == "disk":
        budget.check_free.side_effect = OSError("disk full")
    elif fault == "write":
        monkeypatch.setattr(zipfile.ZipFile, "open", Mock(side_effect=OSError("disk write failed")))
    else:
        monkeypatch.setattr("app.dinov2_artifacts.os.replace", Mock(side_effect=OSError("replace failed")))
    with pytest.raises(OSError):
        package_artifacts(root, target, expected_identity=IDENTITY, storage_budget=budget)
    assert target.read_bytes() == b"complete"
    assert not list(target.parent.glob(".dino-artifacts-*"))


@pytest.mark.parametrize("raw", [b"[1]", b"{", b"x" * (1024 * 1024 + 1), b'{"schema_version":3,"schema_version":3}'],
                         ids=["array", "invalid-json", "oversized", "duplicate-keys"])
def test_malformed_or_oversized_manifest(bundle, raw):
    root, _, target = bundle
    (root / MANIFEST_NAME).write_bytes(raw)
    with pytest.raises(ArchiveError):
        package_artifacts(root, target, expected_identity=IDENTITY)
    assert not target.exists()


def test_capability_create_status_and_restart_keep_v3_identity(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 401
        health = client.get("/v1/health", headers=auth_headers()).json()
        assert ARTIFACT_CONTRACT in health["artifact_contracts"]
        body = job_request_body(b"data", b"kernel", identity=IDENTITY, artifact_contract=ARTIFACT_CONTRACT)
        response = client.post("/v1/jobs", headers=auth_headers(), json=body)
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        for data in (response.json(), client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json(),
                     client.get("/v1/jobs", headers=auth_headers()).json()[0]):
            assert data["artifact_contract"] == ARTIFACT_CONTRACT
            assert {key: data[key] for key in IDENTITY_FIELDS} == IDENTITY
    reopened = RelayDb(settings.db_path).get_job(job_id)
    assert reopened["artifact_contract"] == ARTIFACT_CONTRACT


@pytest.mark.parametrize("missing", IDENTITY_FIELDS)
def test_v3_create_requires_all_identity_fields(tmp_path, missing):
    app = create_app(make_settings(tmp_path))
    body = job_request_body(b"data", b"kernel", identity=IDENTITY, artifact_contract=ARTIFACT_CONTRACT)
    del body[missing]
    with TestClient(app) as client:
        assert client.post("/v1/jobs", headers=auth_headers(), json=body).status_code == 422
        assert client.get("/v1/jobs", headers=auth_headers()).json() == []


@pytest.mark.parametrize("damaged", [False, True])
def test_worker_only_marks_complete_after_verified_package(bundle, tmp_path, monkeypatch, damaged):
    root, _, _ = bundle
    settings = make_settings(tmp_path / "service")
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/v1/jobs", headers=auth_headers(), json=job_request_body(
            b"data", b"kernel", identity=IDENTITY, artifact_contract=ARTIFACT_CONTRACT))
        job_id = response.json()["job_id"]
        adapter = KaggleAdapter(settings, lambda _: None)
        adapter.wait_kernel = Mock(return_value="complete")
        def download(_kernel_ref, output_dir, artifact_contract):
            assert artifact_contract == ARTIFACT_CONTRACT
            shutil.copytree(root, output_dir / "artifacts")
            if damaged:
                (output_dir / "artifacts/com_dinov2_small.pt").unlink()
            return "downloaded"
        adapter.download_output = download
        if damaged:
            with pytest.raises(ArchiveError):
                finish_kernel_job(settings, app.state.db, job_id, adapter)
            status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()
            assert status["status"] != "complete" and not status["can_download"]
        else:
            finish_kernel_job(settings, app.state.db, job_id, adapter)
            status = client.get(f"/v1/jobs/{job_id}", headers=auth_headers()).json()
            assert status["status"] == "complete" and status["can_download"]
            response = client.get(f"/v1/jobs/{job_id}/artifacts.zip", headers=auth_headers())
            assert response.status_code == 200
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                assert set(archive.namelist()) == {"artifacts/" + name for name in REQUIRED_FILES | {MANIFEST_NAME}}
