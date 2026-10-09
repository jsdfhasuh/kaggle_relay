"""Durable exact-version checkpoints and cross-process account request pacing."""

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


FILE_BATCH_LIMIT = 64
FILE_BATCH_SECONDS = 120
REQUEST_INTERVAL_SECONDS = 1.5
PROGRESS_RECHECK_SECONDS = 15


class VerificationDeferred(RuntimeError):
    def __init__(self, detail, retry_after=PROGRESS_RECHECK_SECONDS, *, cooldown=False):
        super().__init__(detail)
        self.retry_after = retry_after
        self.cooldown = cooldown


def candidate_scope(dataset_ref, dataset_dir):
    value = json.dumps([dataset_ref, str(Path(dataset_dir).absolute())], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


class VerificationStore:
    def __init__(self, storage_dir, initialize=True):
        self.path = Path(storage_dir) / "dataset-verification.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("payload_checkpoint_unsafe_path")
        if not initialize:
            return
        with self.connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS candidates (
                scope TEXT PRIMARY KEY, binding TEXT NOT NULL, expected TEXT NOT NULL,
                verified TEXT NOT NULL, total_files INTEGER NOT NULL, total_bytes INTEGER NOT NULL,
                verified_files INTEGER NOT NULL DEFAULT 0, verified_bytes INTEGER NOT NULL DEFAULT 0,
                complete INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS accounts (
                owner TEXT PRIMARY KEY, next_request_at REAL NOT NULL DEFAULT 0,
                cooldown_until REAL NOT NULL DEFAULT 0)""")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def load(self, dataset_ref, version, dataset_dir, content_sha256, expected):
        scope = candidate_scope(dataset_ref, dataset_dir)
        binding = json.dumps([dataset_ref, version, str(Path(dataset_dir).absolute()), content_sha256],
                             separators=(",", ":"))
        encoded = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM candidates WHERE scope=?", (scope,)).fetchone()
            if row is None:
                conn.execute("""INSERT INTO candidates
                    (scope,binding,expected,verified,total_files,total_bytes,updated_at)
                    VALUES (?,?,?,'{}',?,?,?)""",
                    (scope, binding, encoded, len(expected), sum(v[0] for v in expected.values()), time.time()))
                verified = {}
            else:
                if row["binding"] != binding or row["expected"] != encoded:
                    raise ValueError("payload_checkpoint_scope_or_inventory_mismatch")
                verified = json.loads(row["verified"])
                if (not isinstance(verified, dict) or any(
                        name not in expected or value != list(expected[name]) for name, value in verified.items())
                        or row["verified_files"] != len(verified)
                        or row["verified_bytes"] != sum(v[0] for v in verified.values())
                        or row["total_files"] != len(expected)
                        or row["total_bytes"] != sum(v[0] for v in expected.values())
                        or row["complete"] not in (0, 1)
                        or row["complete"] and len(verified) != len(expected)):
                    raise ValueError("payload_checkpoint_invalid")
        return scope, verified

    def mark_file(self, scope, name, value):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT expected,verified FROM candidates WHERE scope=?", (scope,)).fetchone()
            expected, verified = json.loads(row["expected"]), json.loads(row["verified"])
            if expected.get(name) != list(value):
                raise ValueError("payload_checkpoint_file_mismatch")
            verified[name] = list(value)
            conn.execute("""UPDATE candidates SET verified=?,verified_files=?,verified_bytes=?,updated_at=?
                WHERE scope=?""", (json.dumps(verified, sort_keys=True, separators=(",", ":")),
                                  len(verified), sum(v[0] for v in verified.values()), time.time(), scope))

    def mark_complete(self, scope):
        with self.connect() as conn:
            result = conn.execute("""UPDATE candidates SET complete=1,updated_at=?
                WHERE scope=? AND verified_files=total_files AND verified_bytes=total_bytes""", (time.time(), scope))
            if result.rowcount != 1:
                raise ValueError("payload_checkpoint_incomplete")

    def snapshot(self, dataset_ref, version, dataset_dir, content_sha256):
        binding = json.dumps([dataset_ref, version, str(Path(dataset_dir).absolute()), content_sha256],
                             separators=(",", ":"))
        with self.connect() as conn:
            row = conn.execute("""SELECT binding,total_files,total_bytes,verified_files,verified_bytes,complete
                FROM candidates WHERE scope=?""", (candidate_scope(dataset_ref, dataset_dir),)).fetchone()
        if not row or row["binding"] != binding:
            return {}
        return {"state": "verified" if row["complete"] else "checking_files", "version_number": version,
                **{k: row[k] for k in ("total_files", "total_bytes", "verified_files", "verified_bytes")}}

    def cooldown(self, owner, seconds):
        until = time.time() + max(0, seconds)
        with self.connect() as conn:
            conn.execute("""INSERT INTO accounts(owner,cooldown_until) VALUES (?,?)
                ON CONFLICT(owner) DO UPDATE SET cooldown_until=MAX(cooldown_until,excluded.cooldown_until)""",
                (owner.casefold(), until))

    def cooldown_remaining(self, owner, now=None):
        with self.connect() as conn:
            row = conn.execute("SELECT cooldown_until FROM accounts WHERE owner=?", (owner.casefold(),)).fetchone()
        return max(0, row[0] - (time.time() if now is None else now)) if row else 0

    def reserve_request(self, owner):
        now = time.time()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO accounts(owner) VALUES (?)", (owner.casefold(),))
            row = conn.execute("SELECT * FROM accounts WHERE owner=?", (owner.casefold(),)).fetchone()
            if row["cooldown_until"] > now:
                raise VerificationDeferred("dataset_account_cooldown: original version retained",
                                           row["cooldown_until"] - now, cooldown=True)
            delay = max(0, row["next_request_at"] - now)
            if delay > 5:
                raise VerificationDeferred("dataset_account_pacing: yielding verification slot", delay)
            conn.execute("UPDATE accounts SET next_request_at=? WHERE owner=?",
                         (now + delay + REQUEST_INTERVAL_SECONDS, owner.casefold()))
        return delay


class DatasetRequestGate:
    def __init__(self, store, owner, check, deadline=None):
        self.store, self.owner, self.check, self.deadline = store, owner, check, deadline

    def before_request(self):
        self.check()
        delay = self.store.reserve_request(self.owner)
        if self.deadline is not None and time.monotonic() + delay >= self.deadline:
            raise VerificationDeferred("dataset_verification_batch_pending: time budget reached")
        until = time.monotonic() + delay
        while time.monotonic() < until:
            self.check()
            time.sleep(min(0.2, max(0, until - time.monotonic())))
        self.check()
        # A different worker may have received 429 after this reservation.
        cooldown = self.store.cooldown_remaining(self.owner)
        if cooldown:
            raise VerificationDeferred("dataset_account_cooldown: original version retained", cooldown, cooldown=True)
