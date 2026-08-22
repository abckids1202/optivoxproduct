from __future__ import annotations

from .database import get_connection


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
                camera_id text default 'cam_0',
                location text,
                severity integer default 0,
                timestamp text not null,
                foreign key(person_id) references people(id) on delete set null
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
                unique(person_id, date),
                foreign key(person_id) references people(id) on delete cascade
            );
            """
        )
        event_columns = {row[1] for row in con.execute("pragma table_info(events)").fetchall()}
        additions = {
            "review_status": "text not null default 'open'",
            "review_note": "text",
            "reviewed_at": "text",
            "reviewed_by": "text",
        }
        for name, definition in additions.items():
            if name not in event_columns:
                con.execute(f"alter table events add column {name} {definition}")

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
                created_at text not null default (datetime('now'))
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
                resolved_by text
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
            """
        )
        con.commit()
