"""The launch heartbeat (rev 3a §3).

Jonny may check runs/arbitration/canonical/heartbeat.json from
anywhere; the dispatch ticks it at every config completion and the
unattended jobs (F) run an idle thread writing the elapsed markers at
least every ~10 minutes (60 s here).  Written atomically (tmp + rename)
under the module's lock; NEVER under data/.
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Heartbeat:
    """The current-step + progress marker, engine-owned."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else (
            REPO / "runs" / "arbitration" / "canonical" / "heartbeat.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.t0 = time.monotonic()
        self._thread = None

    def _write(self, payload, tmp_suffix=".tmp"):
        tmp = self.path.with_suffix(tmp_suffix)
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def tick(self, step, done, total, last_wall=None, note=None):
        """Config-granularity tick (called by the orchestrator thread)."""
        elapsed = time.monotonic() - self.t0
        with self.lock:
            eta = (elapsed * (total - done) / done) if (done and total) \
                else None
            payload = {
                "step": step, "done": int(done), "total": int(total),
                "frac": (round(done / total, 4) if total else None),
                "eta_s": (round(eta, 1) if eta is not None else None),
                "last_wall_s": last_wall,
                "elapsed_s": round(elapsed, 1),
                "note": note,
                "updated_utc": utcnow(),
            }
            self._write(payload)
        return payload

    def start_thread(self, step, note, interval_s=60.0):
        """The unattended-job thread (F): writes the elapsed markers at
        least every ~10 minutes (60 s) until stopped."""
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                self.tick(step, 0, 0, note=note)
                time.sleep(interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return stop

    def stop_thread(self):
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None