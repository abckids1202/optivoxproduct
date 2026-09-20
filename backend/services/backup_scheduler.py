"""Optional, isolated database backup scheduler.

The scheduler is disabled by default. When enabled explicitly, it performs
verified SQLite backups in a daemon thread and applies conservative retention.
It never blocks request handling, camera processing, or synchronization.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import BACKUP_INTERVAL_MINUTES, BACKUP_RETENTION_COUNT, BACKUP_RETENTION_DAYS, BACKUP_STATUS_PATH
from .storage_service import backup_database, purge_backups


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackupScheduler(threading.Thread):
    def __init__(
        self,
        *,
        interval_minutes: int = BACKUP_INTERVAL_MINUTES,
        retention_days: int = BACKUP_RETENTION_DAYS,
        retention_count: int = BACKUP_RETENTION_COUNT,
        status_path: Path = BACKUP_STATUS_PATH,
    ):
        super().__init__(name="optivox-backup-scheduler", daemon=True)
        self.interval_seconds = max(300, int(interval_minutes) * 60)
        self.retention_days = max(1, int(retention_days))
        self.retention_count = max(2, int(retention_count))
        self.status_path = Path(status_path)
        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        self._status: dict[str, Any] = {
            "enabled": True,
            "state": "waiting",
            "interval_minutes": self.interval_seconds // 60,
            "retention_days": self.retention_days,
            "retention_count": self.retention_count,
            "runs": 0,
            "failures": 0,
            "last_started_at": None,
            "last_success_at": None,
            "last_error": None,
            "last_backup_name": None,
            "last_deleted_count": 0,
        }
        self._publish()

    def _publish(self) -> None:
        with self._lock:
            payload = dict(self._status)
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, self.status_path)
        except OSError:
            # Backup status is observability, never a reason to stop backups.
            return

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def run_once(self) -> dict[str, Any]:
        started = _now()
        with self._lock:
            self._status.update({"state": "running", "last_started_at": started, "last_error": None})
            self._status["runs"] += 1
        self._publish()
        try:
            backup = backup_database()
            cleanup = purge_backups(
                self.retention_days,
                self.retention_count,
                execute_delete=True,
            )
            result = {
                "status": "success",
                "backup_name": Path(backup["path"]).name,
                "sha256": backup["sha256"],
                "deleted_count": int(cleanup.get("deleted_count", 0)),
            }
            with self._lock:
                self._status.update({
                    "state": "waiting",
                    "last_success_at": _now(),
                    "last_backup_name": result["backup_name"],
                    "last_deleted_count": result["deleted_count"],
                })
            self._publish()
            return result
        except Exception as exc:
            with self._lock:
                self._status.update({"state": "degraded", "last_error": str(exc)[:300]})
                self._status["failures"] += 1
            self._publish()
            return {"status": "failed", "error": str(exc)[:300]}

    def run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.run_once()
        with self._lock:
            self._status["state"] = "stopped"
        self._publish()

    def stop(self) -> None:
        self.stop_event.set()


def scheduler_status() -> dict[str, Any]:
    try:
        payload = json.loads(BACKUP_STATUS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"enabled": False, "state": "unavailable"}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"enabled": False, "state": "not_running"}
