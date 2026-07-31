"""Read-only SQLite access for the web API."""
import json
import os
import sqlite3
from datetime import date


class WebDB:
    def __init__(self, db_path):
        self.db_path = db_path

    def _conn(self):
        if not os.path.exists(self.db_path):
            return None
        try:
            uri = f"file:{os.path.abspath(self.db_path)}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5)
        except sqlite3.OperationalError:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _rows(self, sql, params=()):
        conn = self._conn()
        if conn is None:
            return []
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def _one(self, sql, params=()):
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    def stats(self, active_strangers=0):
        today = date.today().isoformat()
        faces = self._one("SELECT COUNT(*) AS c FROM people")
        events_today = self._one("SELECT COUNT(*) AS c FROM events WHERE date(timestamp)=?", (today,))
        clocked_in = self._one(
            "SELECT COUNT(*) AS c FROM attendance WHERE date=? AND clock_in IS NOT NULL AND clock_out IS NULL",
            (today,),
        )
        return {
            "enrolled_faces": faces["c"] if faces else 0,
            "events_today": events_today["c"] if events_today else 0,
            "clocked_in": clocked_in["c"] if clocked_in else 0,
            "active_strangers": active_strangers,
        }

    def recent_events(self, limit=20):
        rows = self._rows(
            "SELECT e.id, e.event_type, e.confidence, e.severity, e.location, e.camera_id, "
            "e.timestamp, e.details_json, p.name AS person_name "
            "FROM events e LEFT JOIN people p ON e.person_id = p.id "
            "ORDER BY e.timestamp DESC LIMIT ?",
            (limit,),
        )
        events = []
        for row in rows:
            try:
                details = json.loads(row["details_json"]) if row["details_json"] else {}
            except (json.JSONDecodeError, TypeError):
                details = {}
            events.append({
                "id": row["id"],
                "event_type": row["event_type"],
                "confidence": row["confidence"],
                "severity": row["severity"] or 0,
                "location": row["location"],
                "camera": row["camera_id"],
                "timestamp": row["timestamp"],
                "person": row["person_name"],
                "details": details,
            })
        return events

    def attendance_today(self):
        today = date.today().isoformat()
        rows = self._rows(
            "SELECT a.date, a.clock_in, a.clock_out, a.late_minutes, a.work_minutes, p.name "
            "FROM attendance a JOIN people p ON a.person_id = p.id WHERE a.date=? ORDER BY a.clock_in",
            (today,),
        )
        return [dict(row) for row in rows]

    def people(self, limit=200):
        rows = self._rows("SELECT id, name, role, created_at FROM people ORDER BY name LIMIT ?", (limit,))
        return [dict(row) for row in rows]