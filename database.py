import sqlite3
import json
import os
import threading
from datetime import datetime, timedelta
import numpy as np


def _utc_now() -> str:
    return datetime.utcnow().isoformat()


class EventDatabase:
    def __init__(self, db_path: str = "security.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30,
        )
        self.conn.row_factory = sqlite3.Row
        self._configure()

        self.retention_policies = {
            "events_days": 90,
            "archive_days": 365,
            "daily_stats_days": 730,
            "hourly_stats_days": 90,
            "snapshots_days": 30,
        }

    def _configure(self):
        pragmas = [
            "PRAGMA journal_mode=WAL;",
            "PRAGMA synchronous=NORMAL;",
            "PRAGMA foreign_keys=ON;",
            "PRAGMA temp_store=MEMORY;",
            "PRAGMA cache_size=-32000;",
        ]
        with self.lock:
            cur = self.conn.cursor()
            for p in pragmas:
                cur.execute(p)
            self.conn.commit()

    def _execute(self, sql: str, params: tuple = ()):
        with self.lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def _fetchall(self, sql: str, params: tuple = ()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params: tuple = ()):
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def setup_database(self):
        with self.lock:
            cur = self.conn.cursor()

            # 1. CREATE TABLES (Using IF NOT EXISTS preserves existing data/structures)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    action          TEXT    NOT NULL,
                    target          TEXT,
                    details_json    TEXT,
                    timestamp       TEXT    NOT NULL
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT    UNIQUE NOT NULL,
                    face_embedding  BLOB    NOT NULL,
                    thumbnail_path  TEXT,
                    created_at      TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id     INTEGER,
                    event_type    TEXT    NOT NULL,
                    confidence    REAL,
                    details_json  TEXT,
                    snapshot_path TEXT,
                    camera_id     TEXT    DEFAULT 'cam_0',
                    location      TEXT,
                    severity      INTEGER DEFAULT 0,
                    timestamp     TEXT    NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL
                )
            """)

            # ... (Include other table creations: event_stats_daily, behavior_profiles, etc. exactly as they are) ...
            cur.execute("""
                CREATE TABLE IF NOT EXISTS event_stats_daily (
                    date        TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    count       INTEGER DEFAULT 0,
                    PRIMARY KEY (date, event_type)
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS behavior_profiles (
                    person_id       INTEGER PRIMARY KEY,
                    profile_json    TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS event_stats_hourly (
                    hour         TEXT NOT NULL,
                    event_type   TEXT NOT NULL,
                    count        INTEGER DEFAULT 0,
                    PRIMARY KEY (hour, event_type)
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS zone_activity (
                    zone_id      TEXT NOT NULL,
                    timestamp    TEXT NOT NULL,
                    person_count INTEGER DEFAULT 0,
                    avg_dwell_sec REAL DEFAULT 0.0,
                    max_dwell_sec REAL DEFAULT 0.0
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date              TEXT PRIMARY KEY,
                    total_events      INTEGER DEFAULT 0,
                    unique_people     INTEGER DEFAULT 0,
                    top_event_type    TEXT,
                    top_event_count   INTEGER DEFAULT 0,
                    avg_confidence    REAL DEFAULT 0.0,
                    peak_hour         TEXT,
                    peak_hour_count   INTEGER DEFAULT 0
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS events_archive (
                    id            INTEGER PRIMARY KEY,
                    person_id     INTEGER,
                    event_type    TEXT    NOT NULL,
                    confidence    REAL,
                    details_json  TEXT,
                    snapshot_path TEXT,
                    camera_id     TEXT    DEFAULT 'cam_0',
                    location      TEXT,
                    severity      INTEGER DEFAULT 0,
                    timestamp     TEXT NOT NULL,
                    archived_at   TEXT NOT NULL
                )
            """)
            
            # 2. MIGRATE COLUMNS (Check for missing columns BEFORE creating indexes)
            # This ensures that if an old DB exists, we add necessary columns first.
            
            # Check 'events' table columns
            cur.execute("PRAGMA table_info(events)")
            event_cols = {r[1] for r in cur.fetchall()}
            
            # Add missing columns with default values if needed
            if "event_type" not in event_cols:
                # Note: SQLite doesn't support NOT NULL without DEFAULT on ALTER TABLE
                cur.execute("ALTER TABLE events ADD COLUMN event_type TEXT DEFAULT 'UNKNOWN'")
                
            if "thumbnail_path" not in event_cols:
                cur.execute("ALTER TABLE events ADD COLUMN thumbnail_path TEXT")
            
            if "severity" not in event_cols:
                cur.execute("ALTER TABLE events ADD COLUMN severity INTEGER DEFAULT 0")

            # Check 'people' table columns
            cur.execute("PRAGMA table_info(people)")
            people_cols = {r[1] for r in cur.fetchall()}
            if "thumbnail_path" not in people_cols:
                cur.execute("ALTER TABLE people ADD COLUMN thumbnail_path TEXT")

            # 3. CREATE INDEXES (Now that we are sure columns exist)
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_people_name    ON people(name)",
                "CREATE INDEX IF NOT EXISTS idx_events_time    ON events(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type)",
                "CREATE INDEX IF NOT EXISTS idx_events_person  ON events(person_id)",
                "CREATE INDEX IF NOT EXISTS idx_events_cam      ON events(camera_id)",
                "CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity)",
                "CREATE INDEX IF NOT EXISTS idx_events_location ON events(location)",
                "CREATE INDEX IF NOT EXISTS idx_stats_date      ON event_stats_daily(date)",
            ]:
                cur.execute(idx_sql)

            self.conn.commit()

    def enroll_person(self, name: str, embedding: np.ndarray,
                      thumbnail_path: str = None) -> int:
        emb_blob = np.asarray(embedding, dtype=np.float32).tobytes()
        now = _utc_now()

        with self.lock:
            cur = self.conn.execute("""
                INSERT INTO people (name, face_embedding, thumbnail_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    face_embedding  = excluded.face_embedding,
                    thumbnail_path  = COALESCE(excluded.thumbnail_path, thumbnail_path),
                    updated_at      = excluded.updated_at
            """, (name, emb_blob, thumbnail_path, now, now))
            self.conn.commit()
            return cur.lastrowid

    def get_known_face_names(self, limit=1000, offset=0):
        return self._fetchall(
            "SELECT id, name, thumbnail_path, created_at FROM people ORDER BY name LIMIT ? OFFSET ?",
            (limit, offset),
        )

    def get_face_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) as cnt FROM people")
        return row["cnt"] if row else 0

    def get_all_known_faces(self) -> dict:
        faces = {}
        rows = self._fetchall("SELECT name, face_embedding FROM people")
        for row in rows:
            faces[row["name"]] = np.frombuffer(
                row["face_embedding"], dtype=np.float32
            ).copy()
        return faces

    def log_audit(self, action: str, target: str = None, details: dict = None):
        with self.lock:
            self.conn.execute(
                "INSERT INTO audit_log (action, target, details_json, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (action, target, json.dumps(details or {}), _utc_now())
            )
            self.conn.commit()

    def get_audit_log(self, action: str = None, limit: int = 100):
        if action:
            return self._fetchall(
                "SELECT * FROM audit_log WHERE action = ? ORDER BY timestamp DESC LIMIT ?",
                (action, limit),
            )
        return self._fetchall(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        )

    def get_person_id(self, name: str):
        row = self._fetchone("SELECT id FROM people WHERE name = ?", (name,))
        return row["id"] if row else None

    def delete_person(self, name: str) -> bool:
        with self.lock:
            cur = self.conn.execute("DELETE FROM people WHERE name = ?", (name,))
            self.conn.commit()
            return cur.rowcount > 0

    def log_event(self, event_type: str, person_id: int = None,
                  confidence: float = None, details=None,
                  snapshot_path: str = None, camera_id: str = "cam_0",
                  location: str = None, severity: int = 0):
        if isinstance(details, dict):
            details_json = json.dumps(details)
        elif details is not None:
            details_json = json.dumps({"message": str(details)})
        else:
            details_json = json.dumps({})

        with self.lock:
            self.conn.execute("""
                INSERT INTO events (
                    person_id, event_type, confidence, details_json,
                    snapshot_path, camera_id, location, severity, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                person_id, event_type, confidence, details_json,
                snapshot_path, camera_id, location, severity, _utc_now(),
            ))
            self._update_daily_stats(event_type)
            self._update_hourly_stats(event_type)
            self.conn.commit()

    def _update_daily_stats(self, event_type: str):
        today = datetime.utcnow().date().isoformat()
        self.conn.execute("""
            INSERT INTO event_stats_daily (date, event_type, count)
            VALUES (?, ?, 1)
            ON CONFLICT(date, event_type) DO UPDATE SET count = count + 1
        """, (today, event_type))

    def _update_hourly_stats(self, event_type: str):
        hour = datetime.utcnow().strftime("%Y-%m-%d %H:00")
        self.conn.execute(
            "INSERT INTO event_stats_hourly (hour, event_type, count) VALUES (?, ?, 1) "
            "ON CONFLICT(hour, event_type) DO UPDATE SET count = count + 1",
            (hour, event_type)
        )

    def update_behavior_profile(self, person_id: int, profile: dict):
        with self.lock:
            self.conn.execute("""
                INSERT INTO behavior_profiles (person_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at   = excluded.updated_at
            """, (person_id, json.dumps(profile), _utc_now()))
            self.conn.commit()

    def get_behavior_profile(self, person_id: int) -> dict:
        row = self._fetchone(
            "SELECT profile_json FROM behavior_profiles WHERE person_id = ?",
            (person_id,),
        )
        return json.loads(row["profile_json"]) if row else {}

    def get_event_summary(self, days: int = 7) -> list:
        since = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        return self._fetchall("""
            SELECT event_type, SUM(count) as total
            FROM event_stats_daily
            WHERE date >= ?
            GROUP BY event_type
            ORDER BY total DESC
        """, (since,))

    def get_recent_events(self, limit: int = 100) -> list:
        return self._fetchall("""
            SELECT e.*, p.name AS person_name
            FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            ORDER BY e.timestamp DESC
            LIMIT ?
        """, (limit,))

    def get_events_by_type(self, event_type: str, limit: int = 50) -> list:
        return self._fetchall("""
            SELECT e.*, p.name AS person_name
            FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            WHERE e.event_type = ?
            ORDER BY e.timestamp DESC
            LIMIT ?
        """, (event_type, limit))

    def get_person_timeline(self, person_id: int, limit: int = 200) -> list:
        return self._fetchall("""
            SELECT * FROM events
            WHERE person_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """, (person_id, limit))

    def purge_old_events(self, days: int = 90):
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self.lock:
            self.conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            self.conn.commit()

    def export_events_to_json(self, output_dir: str = "logs") -> str:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(
            output_dir, f"events_{datetime.utcnow().date()}.json"
        )
        rows = self.get_recent_events(limit=10000)
        data = []
        for r in rows:
            data.append({key: r[key] for key in r.keys()})
        with open(fname, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[INFO] Exported {len(data)} events to {fname}")
        return fname

    def store_event_snapshot(self, event_id, snapshot_path=None, thumbnail_path=None):
        with self.lock:
            self.conn.execute(
                "UPDATE events SET snapshot_path = COALESCE(?, snapshot_path), "
                "thumbnail_path = COALESCE(?, thumbnail_path) WHERE id = ?",
                (snapshot_path, thumbnail_path, event_id)
            )
            self.conn.commit()

    def update_person_thumbnail(self, person_id, thumbnail_path):
        with self.lock:
            self.conn.execute(
                "UPDATE people SET thumbnail_path = ?, updated_at = ? WHERE id = ?",
                (thumbnail_path, _utc_now(), person_id)
            )
            self.conn.commit()

    def log_zone_activity(self, zone_id, person_count, avg_dwell=0.0, max_dwell=0.0):
        with self.lock:
            self.conn.execute("""
                INSERT INTO zone_activity (zone_id, timestamp, person_count, avg_dwell_sec, max_dwell_sec)
                VALUES (?, ?, ?, ?, ?)
            """, (zone_id, _utc_now(), person_count, avg_dwell, max_dwell))
            self.conn.commit()

    def archive_old_events(self, days=90):
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self.lock:
            self.conn.execute("""
                INSERT INTO events_archive
                    (id, person_id, event_type, confidence, details_json, snapshot_path,
                     camera_id, location, severity, timestamp, archived_at)
                SELECT id, person_id, event_type, confidence, details_json, snapshot_path,
                       camera_id, location, severity, timestamp, ?
                FROM events WHERE timestamp < ?
            """, (_utc_now(), cutoff))
            deleted = self.conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,)).rowcount
            self.conn.commit()
            return deleted

    def apply_retention_policies(self):
        results = {}

        archived = self.archive_old_events(days=self.retention_policies["events_days"])
        results["archived_events"] = archived

        archive_cutoff = (datetime.utcnow() - timedelta(days=self.retention_policies["archive_days"])).isoformat()
        with self.lock:
            cur = self.conn.execute("DELETE FROM events_archive WHERE timestamp < ?", (archive_cutoff,))
            results["purged_archived"] = cur.rowcount
            self.conn.commit()

        hourly_cutoff = (datetime.utcnow() - timedelta(days=self.retention_policies["hourly_stats_days"])).isoformat()
        with self.lock:
            cur = self.conn.execute("DELETE FROM event_stats_hourly WHERE hour < ?", (hourly_cutoff,))
            results["purged_hourly_stats"] = cur.rowcount
            self.conn.commit()

        daily_cutoff = (datetime.utcnow() - timedelta(days=self.retention_policies["daily_stats_days"])).isoformat()
        with self.lock:
            cur = self.conn.execute("DELETE FROM event_stats_daily WHERE date < ?", (daily_cutoff,))
            results["purged_daily_stats"] = cur.rowcount
            self.conn.commit()

        zone_cutoff = (datetime.utcnow() - timedelta(days=self.retention_policies["snapshots_days"])).isoformat()
        with self.lock:
            cur = self.conn.execute("DELETE FROM zone_activity WHERE timestamp < ?", (zone_cutoff,))
            results["purged_zone_activity"] = cur.rowcount
            self.conn.commit()

        return results

    def generate_daily_summary(self):
        today = datetime.utcnow().date().isoformat()
        with self.lock:
            row = self.conn.execute("""
                SELECT COUNT(*) as total, COUNT(DISTINCT person_id) as people,
                       AVG(confidence) as avg_conf
                FROM events WHERE date(timestamp) = ?
            """, (today,)).fetchone()

            top = self.conn.execute("""
                SELECT event_type, SUM(count) as cnt FROM event_stats_daily
                WHERE date = ? GROUP BY event_type ORDER BY cnt DESC LIMIT 1
            """, (today,)).fetchone()

            peak = self.conn.execute("""
                SELECT hour, SUM(count) as cnt FROM event_stats_hourly
                WHERE date(hour) = ? GROUP BY hour ORDER BY cnt DESC LIMIT 1
            """, (today,)).fetchone()

            self.conn.execute("""
                INSERT INTO daily_summary (date, total_events, unique_people, top_event_type,
                    top_event_count, avg_confidence, peak_hour, peak_hour_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_events = excluded.total_events,
                    unique_people = excluded.unique_people,
                    top_event_type = excluded.top_event_type,
                    top_event_count = excluded.top_event_count,
                    avg_confidence = excluded.avg_confidence,
                    peak_hour = excluded.top_hour,
                    peak_hour_count = excluded.top_hour_count
            """, (
                today,
                row["total"] if row else 0,
                row["people"] if row else 0,
                top["event_type"] if top else None,
                top["cnt"] if top else 0,
                row["avg_conf"] if row else 0.0,
                peak["hour"] if peak else None,
                peak["cnt"] if peak else 0,
            ))
            self.conn.commit()