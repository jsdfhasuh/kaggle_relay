import threading
import time
from concurrent.futures import ThreadPoolExecutor


class QuotaCache:
    """Coalesce concurrent lookups for one credential without blocking other accounts."""

    def __init__(self, ttl_seconds: int = 30):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries = {}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="relay-quota")

    def submit(self, key, operation):
        with self._lock:
            now = time.monotonic()
            entry = self._entries.get(key)
            if entry:
                started, future = entry
                if not future.done() or (now - started < self.ttl_seconds and future.exception() is None):
                    return future
            self._entries = {k: item for k, item in self._entries.items()
                             if not item[1].done() or now - item[0] < self.ttl_seconds}
            future = self.executor.submit(operation)
            self._entries[key] = (now, future)
            return future

    def shutdown(self):
        self.executor.shutdown(wait=True)
