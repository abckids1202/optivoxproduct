from __future__ import annotations

from .database import get_connection
from schema_migrations import ensure_schema_migrations


def ensure_platform_schema() -> None:
    """Apply small, idempotent backend-owned schema additions.

    The vision engine owns the original SQLite tables. These tables extend the
    platform contract without taking ownership of biometric storage or forcing
    a risky rewrite of the working local engine schema.
    """
    with get_connection() as con:
        # Allow a backend-only install to boot before the edge agent creates
        # its database. Existing engine tables are left untouched.
        con.executescript(
            """
            create table if not exists people (
                id integer primary key autoincrement,
                name text unique not null,
                role text,
                thumbnail_path text,
                metadata_json text,
                created_at text not null default (datetime('now')),
                updated_at text not null default (datetime('now'))
            );
            create table if not exists events (
                id integer primary key autoincrement,
                person_id integer,
                event_type text not null,
                confidence real,
                details_json text,
                snapshot_path text,
                evidence_checksum text,
                camera_id text default 'cam_0',
                location text,
                severity integer default 0,
                timestamp text not null,
                foreign key(person_id) references people(id) on delete set null
            );
            create table if not exists alert_log (
                id integer primary key autoincrement,
                channel text not null,
                event_type text not null,
                target text,
                status text,
                error text,
                timestamp text not null,
                source_event_id integer
            );
            create table if not exists attendance (
                id integer primary key autoincrement,
                person_id integer not null,
                date text not null,
                clock_in text,
                clock_out text,
                work_minutes integer default 0,
                late_minutes integer default 0,
                camera_id text,
                location text,
                notes text,
                attendance_status text not null default 'recorded',
                absence_type text,
                is_official integer not null default 1,
                subject text,
                schedule_key text,
                 expected_start text,
                 expected_end text,
                 early_departure_minutes integer default 0,
                 last_seen_at text,
                 clock_out_source text,
                 unique(person_id, date),
                foreign key(person_id) references people(id) on delete cascade
            );
            create table if not exists absence_records (
                id integer primary key autoincrement,
                person_id integer not null,
                absence_date text not null,
                subject text not null default '',
                status text not null default 'inferred',
                reason text,
                source text not null default 'system',
                is_official integer not null default 0,
                actor_id text,
                created_at text not null default (datetime('now')),
                updated_at text not null default (datetime('now')),
                unique(person_id, absence_date, subject),
                foreign key(person_id) references people(id) on delete cascade
            );
            create table if not exists attendance_schedules (
                id integer primary key autoincrement,
                class_name text not null,
                subject text,
                weekday integer not null,
                start_time text not null,
                end_time text,
                grace_minutes integer not null default 0,
                active integer not null default 1,
                unique(class_name, subject, weekday, start_time)
            );
            create table if not exists presence_sessions (
                id integer primary key autoincrement,
                entity_id text not null,
                track_id integer,
                track_generation integer default 1,
                person_id integer,
                label text not null default 'UNKNOWN',
                identity_state text not null default 'UNRESOLVED',
                liveness_status text,
                camera_id text not null default 'cam_0',
                started_at text not null,
                last_seen_at text not null,
                ended_at text,
                status text not null default 'active',
                 confidence real default 0.0,
                 first_frame_id integer,
                 last_frame_id integer,
                 closed_reason text,
                 foreign key(person_id) references people(id) on delete set null
            );
            create table if not exists recognition_evidence (
                id integer primary key autoincrement,
                entity_id text not null,
                presence_session_id integer,
                track_id integer,
                person_id integer,
                candidate_name text not null default 'UNKNOWN',
                decision text not null,
                similarity real default 0.0,
                quality_score real default 0.0,
                quality_ok integer default 0,
                liveness_status text,
                identity_state text,
                reason text,
                source_frame_id integer,
                observed_at text not null,
                details_json text,
                foreign key(presence_session_id) references presence_sessions(id) on delete set null,
                foreign key(person_id) references people(id) on delete set null
            );
            create table if not exists enrollment_operations (
                id integer primary key autoincrement,
                person_id integer,
                person_name text not null,
                operation text not null,
                status text not null,
                sample_count integer default 0,
                quality_json text,
                provenance_json text,
                actor_id text,
                created_at text not null default (datetime('now')),
                foreign key(person_id) references people(id) on delete set null
            );
            create table if not exists attendance_decisions (
                id integer primary key autoincrement,
                decision_key text not null unique,
                person_id integer,
                entity_id text,
                presence_session_id integer,
                recognition_evidence_id integer,
                decision text not null,
                reason text,
                identity_state text,
                liveness_status text,
                quality_score real,
                recognition_confidence real,
                source_frame_id integer,
                observed_at text not null,
                details_json text,
                created_at text not null default (datetime('now')),
                foreign key(person_id) references people(id) on delete set null,
                foreign key(presence_session_id) references presence_sessions(id) on delete set null,
                foreign key(recognition_evidence_id) references recognition_evidence(id) on delete set null
            );
            """
        )
        event_columns = {row[1] for row in con.execute("pragma table_info(events)").fetchall()}
        additions = {
            "entity_id": "text",
            "presence_session_id": "integer",
            "source_frame_id": "integer",
            "observation_type": "text",
            "evidence_path": "text",
            "review_status": "text not null default 'open'",
            "review_note": "text",
            "reviewed_at": "text",
            "reviewed_by": "text",
        }
        for name, definition in additions.items():
            if name not in event_columns:
                con.execute(f"alter table events add column {name} {definition}")

        alert_columns = {row[1] for row in con.execute("pragma table_info(alert_log)").fetchall()}
        if "source_event_id" not in alert_columns:
            con.execute("alter table alert_log add column source_event_id integer")

        attendance_columns = {row[1] for row in con.execute("pragma table_info(attendance)").fetchall()}
        attendance_additions = {
            "presence_session_id": "integer",
            "recognition_evidence_id": "integer",
            "decision_source": "text default 'automatic'",
            "identity_state": "text",
            "liveness_status": "text",
            "source_frame_id": "integer",
            "recognition_confidence": "real",
            "evidence_path": "text",
            "attendance_status": "text not null default 'recorded'",
            "absence_type": "text",
            "is_official": "integer not null default 1",
            "subject": "text",
            "schedule_key": "text",
            "expected_start": "text",
            "expected_end": "text",
            "early_departure_minutes": "integer default 0",
            "last_seen_at": "text",
            "clock_out_source": "text",
        }
        for name, definition in attendance_additions.items():
            if name not in attendance_columns:
                con.execute(f"alter table attendance add column {name} {definition}")

        presence_columns = {row[1] for row in con.execute(
            "pragma table_info(presence_sessions)").fetchall()}
        if "track_generation" not in presence_columns:
            con.execute("alter table presence_sessions add column track_generation integer default 1")
        if "closed_reason" not in presence_columns:
            con.execute("alter table presence_sessions add column closed_reason text")
        con.execute("update events set review_status='open' where review_status is null")
        con.execute("update attendance set decision_source='automatic' where decision_source is null")

        if con.execute("select 1 from sqlite_master where type='table' and name='incidents'").fetchone():
            incident_columns = {row[1] for row in con.execute("pragma table_info(incidents)").fetchall()}
            incident_additions = {
                "entity_id": "text",
                "camera_id": "text",
                "location": "text",
                "zone_id": "text",
                "presence_session_id": "integer",
                "assigned_to": "text",
                "false_positive": "integer not null default 0",
            }
            for name, definition in incident_additions.items():
                if name not in incident_columns:
                    con.execute(f"alter table incidents add column {name} {definition}")

        con.executescript(
            """
            create table if not exists platform_audit_log (
                id integer primary key autoincrement,
                action text not null,
                entity_type text not null,
                entity_id text,
                actor_type text not null default 'system',
                actor_id text,
                details_json text,
                created_at text not null default (datetime('now')),
                prev_hash text,
                record_hash text
            );
            create index if not exists idx_platform_audit_time
                on platform_audit_log(created_at);

            create table if not exists incidents (
                id integer primary key autoincrement,
                status text not null default 'open',
                category text not null,
                severity integer not null default 0,
                summary text not null,
                first_event_at text not null,
                last_event_at text not null,
                created_at text not null default (datetime('now')),
                updated_at text not null default (datetime('now')),
                resolution_note text,
                resolved_at text,
                resolved_by text,
                entity_id text,
                camera_id text,
                location text,
                zone_id text,
                presence_session_id integer,
                assigned_to text,
                false_positive integer not null default 0
            );
            create index if not exists idx_incidents_status on incidents(status);
            create index if not exists idx_incidents_updated on incidents(updated_at);

            create table if not exists incident_events (
                incident_id integer not null,
                event_id integer not null unique,
                created_at text not null default (datetime('now')),
                primary key (incident_id, event_id),
                foreign key (incident_id) references incidents(id) on delete cascade,
                foreign key (event_id) references events(id) on delete cascade
            );

            create table if not exists incident_review_actions (
                id integer primary key autoincrement,
                incident_id integer not null,
                action text not null,
                note text,
                actor_id text,
                created_at text not null default (datetime('now')),
                foreign key (incident_id) references incidents(id) on delete cascade
            );

            create table if not exists incident_alerts (
                id integer primary key autoincrement,
                incident_id integer not null,
                channel text not null,
                status text not null,
                attempted_at text not null default (datetime('now')),
                delivered_at text,
                error text,
                attempt_count integer not null default 1,
                source_alert_id integer,
                foreign key (incident_id) references incidents(id) on delete cascade
            );

            create table if not exists incident_evidence (
                id integer primary key autoincrement,
                incident_id integer not null,
                event_id integer,
                path text not null,
                evidence_type text not null default 'snapshot',
                source_frame_id integer,
                captured_at text not null,
                checksum text,
                status text not null default 'available',
                created_at text not null default (datetime('now')),
                unique(incident_id, event_id, path),
                foreign key (incident_id) references incidents(id) on delete cascade,
                foreign key (event_id) references events(id) on delete set null
            );

            create table if not exists platform_users (
                id integer primary key autoincrement,
                username text unique not null,
                password_hash text not null,
                role text not null default 'operator',
                active integer not null default 1,
                failed_login_count integer not null default 0,
                locked_until text,
                last_login_at text,
                last_login_ip text,
                created_at text not null default (datetime('now')),
                updated_at text not null default (datetime('now'))
            );

            create table if not exists platform_sessions (
                token_hash text primary key,
                user_id integer not null,
                expires_at text not null,
                csrf_token_hash text,
                rotated_from text,
                client_ip text,
                user_agent text,
                created_at text not null default (datetime('now')),
                last_seen_at text not null default (datetime('now')),
                foreign key (user_id) references platform_users(id) on delete cascade
            );

            create table if not exists command_idempotency (
                idempotency_key text primary key,
                command_id text not null,
                created_at text not null default (datetime('now'))
            );

            create table if not exists attendance_corrections (
                id integer primary key autoincrement,
                attendance_id integer,
                person_id integer not null,
                attendance_date text not null,
                before_json text not null,
                after_json text not null,
                reason text not null,
                actor_id text,
                created_at text not null default (datetime('now')),
                foreign key (attendance_id) references attendance(id) on delete set null,
                foreign key (person_id) references people(id) on delete cascade
            );
            create index if not exists idx_attendance_corrections_person_date
                on attendance_corrections(person_id, attendance_date);
            create index if not exists idx_events_entity on events(entity_id);
            create index if not exists idx_events_session on events(presence_session_id);
            create index if not exists idx_presence_entity on presence_sessions(entity_id, status);
            create index if not exists idx_presence_person on presence_sessions(person_id, last_seen_at);
            create index if not exists idx_evidence_entity on recognition_evidence(entity_id, observed_at);
            create index if not exists idx_evidence_person on recognition_evidence(person_id, observed_at);
            create index if not exists idx_attendance_decisions_entity
                on attendance_decisions(entity_id, observed_at);
            create index if not exists idx_attendance_decisions_person
                on attendance_decisions(person_id, observed_at);
            create index if not exists idx_absence_person_date on absence_records(person_id, absence_date);
            create index if not exists idx_absence_date on absence_records(absence_date, status);
            create index if not exists idx_incident_entity on incidents(entity_id, camera_id, location);
            create index if not exists idx_incident_session on incidents(presence_session_id);
            create index if not exists idx_incident_review on incident_review_actions(incident_id, created_at);
            create index if not exists idx_incident_evidence on incident_evidence(incident_id, captured_at);
            create index if not exists idx_session_expiry on platform_sessions(expires_at);
            """
        )
        user_columns = {row["name"] for row in con.execute("pragma table_info(platform_users)").fetchall()}
        for name, definition in (
            ("failed_login_count", "integer not null default 0"),
            ("locked_until", "text"),
            ("last_login_at", "text"),
            ("last_login_ip", "text"),
        ):
            if name not in user_columns:
                con.execute(f"alter table platform_users add column {name} {definition}")
        session_columns = {row["name"] for row in con.execute("pragma table_info(platform_sessions)").fetchall()}
        for name, definition in (
            ("csrf_token_hash", "text"),
            ("rotated_from", "text"),
            ("client_ip", "text"),
            ("user_agent", "text"),
        ):
            if name not in session_columns:
                con.execute(f"alter table platform_sessions add column {name} {definition}")
        event_columns = {row["name"] for row in con.execute("pragma table_info(events)").fetchall()}
        if "evidence_checksum" not in event_columns:
            con.execute("alter table events add column evidence_checksum text")
        ensure_schema_migrations(con)
        con.commit()
