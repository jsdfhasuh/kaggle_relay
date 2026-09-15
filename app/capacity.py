import shutil
import threading


class CapacityError(RuntimeError):
    pass


class StorageBudget:
    """Reservations cover future input writes; materialized bytes are already in disk usage."""

    def __init__(self, settings, db):
        self.settings = settings
        self.db = db
        self.lock = threading.RLock()

    def check_free(self, needed: int = 0) -> None:
        if shutil.disk_usage(self.settings.storage_dir).free < self.settings.min_free_bytes + needed:
            raise CapacityError("relay storage is full or below its reserved free-space limit; retry later")

    def check_admission(self, owner: str, reserved_bytes: int) -> None:
        with self.db.connect() as conn:
            total, owned = conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(relay_token_id=?), 0) FROM jobs
                   WHERE status NOT IN ('complete','failed','canceled')""", (owner,),
            ).fetchone()
        if total >= self.settings.max_active_jobs or owned >= self.settings.max_active_jobs_per_user:
            raise CapacityError("relay active job limit reached; finish or cancel an existing job")
        self.check_free(self.db.total_reserved_bytes() + reserved_bytes)

    def reserve(self, job_id: str, needed: int) -> None:
        with self.lock:
            self.check_free(self.db.total_reserved_bytes(excluding_job_id=job_id) + needed)
            self.db.update_job(job_id, reserved_bytes=needed)

    def consume(self, job_id: str, size: int) -> None:
        with self.lock, self.db.connect() as conn:
            conn.execute("UPDATE jobs SET reserved_bytes=MAX(0,reserved_bytes-?) WHERE job_id=?", (size, job_id))
