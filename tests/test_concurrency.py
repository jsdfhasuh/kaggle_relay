import asyncio
import hashlib
import importlib
import json
import os
import subprocess
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from test_relay_api import auth_headers, build_zip, create_job, job_request_body, seed_job, upload_all
from app.auth_config import KaggleCredentials
from app.archive import assemble_archive, safe_extract_zip
from app.capacity import CapacityError
from app.config import Settings
from app.database import RelayDb
from app.kaggle_adapter import KaggleAdapter, KaggleAdapterInterrupted
from app.main import cleanup_expired_jobs, create_app

main = importlib.import_module("app.main")


def pool_settings(tmp_path, count=10, shared=False, **overrides):
    keys = [dict(id=f"key{i}", username=f"user{i}", key=f"fake-account-key-{i}") for i in range(count)]
    config = {"kaggle_keys": keys, "relay_tokens": [
        dict(id=f"user{i}", token=f"fake-relay-token-{i}",
             allowed_kaggle_key_ids=[key["id"] for key in keys] if shared else [f"key{i}"])
        for i in range(count)
    ]}
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return Settings(api_token="", storage_dir=tmp_path / "data", auth_config_path=path, **overrides)


async def until(condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(.01)


def test_ten_users_keep_uploading_while_all_training_workers_are_busy(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path))
    release = threading.Event()
    started = set()

    def train(settings, db, job_id, auth_store=None):
        db.update_job(job_id, status="waiting_kernel")
        started.add(job_id)
        assert release.wait(20)
        db.finalize_job(job_id, "complete")

    monkeypatch.setattr(main, "process_job", train)

    async def scenario():
        # A smaller I/O pool than the production server makes starvation deterministic.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def submit(user):
                    headers = auth_headers(token=f"fake-relay-token-{user}")
                    dataset = build_zip({"dataset-metadata.json": json.dumps({"id": f"user{user}/data"}).encode(),
                                         "data.bin": bytes([user]) * (4 * 65536)})
                    kernel = build_zip({"kernel-metadata.json": json.dumps({"id": f"user{user}/kernel", "code_file": "train.py"}).encode(),
                                        "train.py": b"print(1)"})
                    body = job_request_body(dataset, kernel, dataset_ref=f"user{user}/data", kernel_ref=f"user{user}/kernel")
                    body["chunk_size"] = 65536
                    response = await client.post("/v1/jobs", headers=headers, json=body)
                    assert response.status_code == 200, response.text
                    job = response.json()
                    semaphore = asyncio.Semaphore(4)
                    async def send(kind, index, content):
                        async with semaphore:
                            async def stream():
                                for offset in range(0, len(content), 8192):
                                    await asyncio.sleep(0)
                                    yield content[offset:offset + 8192]
                            response = await client.put(f"/v1/jobs/{job['job_id']}/archives/{kind}/chunks/{index}",
                                headers={**headers, "X-Chunk-Size": str(len(content)),
                                         "X-Chunk-Sha256": hashlib.sha256(content).hexdigest()}, content=stream())
                            assert response.status_code == 200, response.text
                    for kind, data in (("dataset", dataset), ("kernel", kernel)):
                        await asyncio.gather(*(send(kind, offset // 65536, data[offset:offset+65536])
                                              for offset in range(0, len(data), 65536)))
                    response = await client.post(f"/v1/jobs/{job['job_id']}/complete", headers=headers)
                    assert response.status_code == 200, response.text
                    return job
                try:
                    jobs = await asyncio.wait_for(asyncio.gather(*(submit(i) for i in range(10))), 10)
                    await until(lambda: len(started) == 10)
                    # Another real upload and assembly must finish while all ten workers wait.
                    extra = await asyncio.wait_for(submit(0), 5)
                    assert app.state.db.get_job(extra["job_id"])["status"] == "queued"
                    for i, job in enumerate(jobs):
                        forbidden = await client.get(f"/v1/jobs/{job['job_id']}",
                            headers=auth_headers(token=f"fake-relay-token-{(i+1)%10}"))
                        assert forbidden.status_code == 404
                    assert app.state.upload_count == 0
                    assert app.state.db.total_reserved_bytes() == 0
                finally:
                    release.set()
                    await asyncio.wait_for(app.state.queue.join(), 5)
    asyncio.run(scenario())


def test_shared_account_pool_balances_atomic_reservations_and_reuses_quota(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, shared=True))
    calls = []
    def quota(adapter):
        calls.append(adapter.credentials.id)
        return {"available": True, "accelerators": [{"resource": "GPU", "remaining_hours": 30}]}
    monkeypatch.setattr(main.KaggleAdapter, "quota", quota)
    with TestClient(app) as client, ThreadPoolExecutor(10) as executor:
        def submit(i):
            return client.post("/v1/jobs", headers=auth_headers(token=f"fake-relay-token-{i}"),
                               json=job_request_body(b"a", b"b"))
        responses = list(executor.map(submit, range(10)))
    assert all(response.status_code == 200 for response in responses)
    assert {response.json()["kaggle_key_id"] for response in responses} == {f"key{i}" for i in range(10)}
    assert sorted(calls) == [f"key{i}" for i in range(10)]


def test_busy_account_does_not_block_other_accounts_and_canceled_queue_is_skipped(tmp_path, monkeypatch):
    app = create_app(pool_settings(tmp_path, count=2, worker_count=2))
    jobs = [seed_job(app, "queued", job_id=f"queued{i}") for i in range(3)]
    for job_id, key in zip(jobs, ("key0", "key0", "key1")):
        app.state.db.update_job(job_id, kaggle_key_id=key, relay_token_id=f"user{key[-1]}")
    release = threading.Event()
    started = []
    def train(settings, db, job_id, auth_store=None):
        started.append(job_id)
        db.update_job(job_id, status="waiting_kernel")
        assert release.wait(10)
        db.finalize_job(job_id, "complete")
    monkeypatch.setattr(main, "process_job", train)
    with TestClient(app) as client:
        try:
            client.portal.call(until, lambda: len(started) == 2)
            assert set(started) == {jobs[0], jobs[2]}
            waiting = app.state.db.get_job(jobs[1])
            assert waiting["status"] == "queued" and "account slot" in waiting["queue_reason"]
            response = client.post(f"/v1/jobs/{jobs[1]}/cancel", headers=auth_headers(token="fake-relay-token-0"))
            assert response.status_code == 200
        finally:
            release.set()
            client.portal.call(app.state.queue.join)
    assert jobs[1] not in started


def install_fake_sdk(tmp_path, monkeypatch):
    package = tmp_path / "sdk" / "kaggle" / "api"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "kaggle_api_extended.py").write_text('''
import os, time
from pathlib import Path
from types import SimpleNamespace
class KaggleApi:
    def authenticate(self):
        self.owner = os.environ['KAGGLE_USERNAME']
    def dataset_status(self, ref, format=None):
        return '{"status":"ready","current_version_number":3}' if format else 'ready'
    def dataset_create_version(self, *args, **kwargs):
        root = Path(os.environ['RELAY_TEST_CONTROL'])
        (root / (self.owner + '.started')).write_text(str(os.getpid()))
        deadline = time.monotonic() + 8
        while not (root / 'release').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('fake upload timed out')
            time.sleep(.01)
        assert os.environ['KAGGLE_USERNAME'] == self.owner
        print('credential=' + os.environ['KAGGLE_KEY'])
    def quota_view(self):
        return SimpleNamespace(gpu_quota=None,tpu_quota=None,quota_refresh_time=None)
''', encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(package.parent.parent))
    monkeypatch.setenv("RELAY_TEST_CONTROL", str(tmp_path))


def test_sdk_processes_isolate_credentials_and_do_not_block_quota(tmp_path, monkeypatch):
    install_fake_sdk(tmp_path, monkeypatch)
    settings = Settings(api_token="test", storage_dir=tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "file.txt").write_text("data")
    logs = []
    a = KaggleAdapter(settings, logs.append, credentials=KaggleCredentials("a", username="alice", key="fake-secret-alice"))
    b = KaggleAdapter(settings, logs.append, credentials=KaggleCredentials("b", username="bob", key="fake-secret-bob"))
    before = {key: os.environ.get(key) for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")}
    with ThreadPoolExecutor(3) as executor:
        first = executor.submit(a.upload_dataset, dataset, "alice/data", "test")
        second = executor.submit(b.upload_dataset, dataset, "bob/data", "test")
        try:
            deadline = time.monotonic() + 5
            while not all((tmp_path / f"{name}.started").exists() for name in ("alice", "bob")):
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert executor.submit(b.quota).result(timeout=3)["available"]
            pids = {(tmp_path / f"{name}.started").read_text() for name in ("alice", "bob")}
            assert len(pids) == 2
        finally:
            (tmp_path / "release").touch()
        assert first.result(5).expected_version_number == 4
        assert second.result(5).expected_files == (("file.txt", 4),)
    assert before == {key: os.environ.get(key) for key in before}
    assert not any("fake-secret-" in line for line in logs)


def test_command_timeout_terminates_process(tmp_path):
    adapter = KaggleAdapter(Settings(api_token="test", storage_dir=tmp_path, kaggle_cmd=sys.executable,
                                     command_timeout_seconds=1), lambda message: None)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        adapter._run(["-c", "import time; time.sleep(20)"])
    assert time.monotonic() - start < 4


@pytest.mark.parametrize("operation", ["assemble", "extract"])
def test_archive_writes_stop_when_free_space_drops(tmp_path, operation):
    data = build_zip({"data.bin": b"x" * (3 * 1024 * 1024)})
    source = tmp_path / "source.zip"
    source.write_bytes(data)
    calls = []
    def check_space(needed):
        calls.append(needed)
        if len(calls) == 2:
            raise CapacityError("test disk filled during copy")
    with pytest.raises(CapacityError):
        if operation == "assemble":
            chunks = tmp_path / "chunks"
            chunks.mkdir()
            (chunks / "0.part").write_bytes(data)
            target = tmp_path / "merged.zip"
            assemble_archive(chunks, target, len(data), len(data), hashlib.sha256(data).hexdigest(), check_space)
        else:
            target = tmp_path / "extracted" / "data.bin"
            safe_extract_zip(source, target.parent, len(data), check_space)
    assert target.stat().st_size == 1024 * 1024
    assert len(calls) == 2


def test_transfer_stops_child_when_disk_space_drops(tmp_path, monkeypatch):
    settings = Settings(api_token="test", storage_dir=tmp_path)
    class Budget:
        calls = 0
        def check_free(self):
            self.calls += 1
            if self.calls > 1:
                raise CapacityError("test disk filled during transfer")
    settings._storage_budget = Budget()
    real_popen = subprocess.Popen
    children = []
    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, "Popen", capture)
    adapter = KaggleAdapter(settings, lambda message: None)
    with pytest.raises(CapacityError):
        adapter._run_command([sys.executable, "-c", "import time; time.sleep(20)"], check_space=True)
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_timeout_kills_descendants_after_group_leader_exits(tmp_path):
    marker = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(20)"
    )
    parent_code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child_code!r}])"
    adapter = KaggleAdapter(Settings(api_token="test", storage_dir=tmp_path, command_timeout_seconds=1),
                            lambda message: None)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        adapter._run_command([sys.executable, "-c", parent_code])
    assert time.monotonic() - start < 9
    assert marker.exists()
    status = Path(f"/proc/{marker.read_text()}/status")
    if status.exists():
        assert "State:\tZ" in status.read_text()


def test_cleanup_is_idempotent_and_preserves_active_uploads(tmp_path):
    app = create_app(Settings(api_token="secret", storage_dir=tmp_path))
    stale = seed_job(app, "receiving", job_id="stale")
    busy = seed_job(app, "receiving", job_id="busy")
    for job_id in (stale, busy):
        app.state.db.update_job(job_id, upload_activity_at=1)
        directory = app.state.settings.jobs_dir / job_id
        directory.mkdir(parents=True)
        (directory / "0.part").write_bytes(b"saved")
    app.state.active_uploads[busy] = 1
    asyncio.run(cleanup_expired_jobs(app))
    asyncio.run(cleanup_expired_jobs(app))
    assert app.state.db.get_job(stale)["status"] == "failed"
    assert app.state.db.recent_logs(stale) == ["expired by relay retention cleanup"]
    assert (app.state.settings.jobs_dir / busy / "0.part").exists()
    app.state.active_uploads.clear()
    app.state.db.touch_upload(busy)
    asyncio.run(cleanup_expired_jobs(app))
    assert app.state.db.get_job(busy)["status"] == "receiving"


def test_log_bound_and_index_and_migration_upload_grace(tmp_path):
    app = create_app(Settings(api_token="secret", storage_dir=tmp_path, max_logs_per_job=3))
    job_id = seed_job(app, "receiving")
    app.state.db.update_job(job_id, upload_activity_at=None, updated_at=1)
    reopened = RelayDb(app.state.settings.db_path)
    assert reopened.get_job(job_id)["upload_activity_at"] > time.time() - 5
    for index in range(8):
        app.state.db.append_log(job_id, str(index))
    assert app.state.db.recent_logs(job_id) == ["5", "6", "7"]
    with app.state.db.connect() as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN SELECT message FROM logs WHERE job_id=? ORDER BY id DESC LIMIT 30", (job_id,)).fetchall()
    assert any("logs_job_id_id" in row[3] for row in plan)


def test_capacity_admission_is_atomic_and_preserves_upload_on_low_space(tmp_path, monkeypatch):
    settings = Settings(api_token="secret", storage_dir=tmp_path, chunk_size=8, min_free_bytes=10)
    app = create_app(settings)
    monkeypatch.setattr("app.capacity.shutil.disk_usage", lambda path: type("Disk", (), {"free": 40})())
    with TestClient(app) as client, ThreadPoolExecutor(2) as executor:
        body = job_request_body(b"12345678", b"x")
        responses = list(executor.map(lambda _: client.post("/v1/jobs", headers=auth_headers(), json=body), range(2)))
        assert sorted(response.status_code for response in responses) == [200, 503]
        job_id = next(response.json()["job_id"] for response in responses if response.status_code == 200)
        monkeypatch.setattr("app.capacity.shutil.disk_usage", lambda path: type("Disk", (), {"free": 10})())
        response = client.put(f"/v1/jobs/{job_id}/archives/dataset/chunks/0", content=b"12345678",
            headers=auth_headers({"X-Chunk-Size": "8", "X-Chunk-Sha256": hashlib.sha256(b"12345678").hexdigest()}))
        assert response.status_code == 503
        assert app.state.db.get_job(job_id)["status"] == "receiving"
        assert app.state.db.accepted_chunks(job_id)["dataset"] == []
        assert app.state.upload_count == 0


def test_shutdown_during_log_trim_waits_then_exits(tmp_path, monkeypatch):
    app = create_app(Settings(api_token="secret", storage_dir=tmp_path))
    started, release = threading.Event(), threading.Event()
    def trim():
        started.set()
        assert release.wait(5)
    monkeypatch.setattr(app.state.db, "trim_logs", trim)
    async def scenario():
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        await until(started.is_set)
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        await asyncio.sleep(.05)
        assert not shutdown.done()
        release.set()
        await asyncio.wait_for(shutdown, 3)
    asyncio.run(scenario())


def test_cleanup_retries_failed_removal_without_waiting_another_retention_period(tmp_path, monkeypatch):
    app = create_app(Settings(api_token="secret", storage_dir=tmp_path))
    job_id = seed_job(app, "receiving")
    app.state.db.update_job(job_id, upload_activity_at=1)
    real_remove = main.shutil.rmtree
    calls = []
    def fail_once(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("busy test directory")
        return real_remove(path, *args, **kwargs)
    monkeypatch.setattr(main.shutil, "rmtree", fail_once)
    asyncio.run(cleanup_expired_jobs(app))
    failed = app.state.db.get_job(job_id)
    assert failed["cleaned_at"] is None and failed["cleanup_due_at"] is not None
    asyncio.run(cleanup_expired_jobs(app))
    cleaned = app.state.db.get_job(job_id)
    assert cleaned["cleaned_at"] is not None and cleaned["cleanup_due_at"] is None


def test_idle_body_releases_slots_without_losing_confirmed_chunks(tmp_path):
    app = create_app(Settings(api_token="secret", storage_dir=tmp_path, upload_idle_seconds=1))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/jobs", headers=auth_headers(), json=job_request_body(b"12345678", b"x"))
            job_id = response.json()["job_id"]
            async def stream():
                yield b"1"
                await asyncio.sleep(5)
                yield b"2345678"
            response = await client.put(f"/v1/jobs/{job_id}/archives/dataset/chunks/0", content=stream(),
                headers=auth_headers({"X-Chunk-Size": "8", "X-Chunk-Sha256": hashlib.sha256(b"12345678").hexdigest()}))
            assert response.status_code == 408
            assert app.state.db.get_job(job_id)["status"] == "receiving"
            assert app.state.upload_count == 0 and not app.state.active_uploads
            assert not list(app.state.settings.jobs_dir.rglob("*.tmp"))
    asyncio.run(scenario())


def test_cancel_while_waiting_for_assembly_slot_restores_receiving(tmp_path, monkeypatch):
    app = create_app(Settings(api_token="secret", storage_dir=tmp_path, assembly_workers=1))
    job_id = seed_job(app, "receiving")
    monkeypatch.setattr(main, "expected_chunk_count", lambda size, chunk_size: 0)
    async def scenario():
        await app.state.assembly_slots.acquire()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                task = asyncio.create_task(client.post(f"/v1/jobs/{job_id}/complete", headers=auth_headers()))
                await until(lambda: app.state.db.get_job(job_id)["status"] == "assembling")
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert app.state.db.get_job(job_id)["status"] == "receiving"
        finally:
            app.state.assembly_slots.release()
    asyncio.run(scenario())
