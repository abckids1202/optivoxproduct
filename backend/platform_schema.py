from __future__ import annotations

import hashlib
import sqlite3

from .config import DEVICE_ID, ORGANIZATION_ID, SITE_ID
from .database import database_migration_lock, get_connection
from schema_migrations import (
    ensure_schema_migrations,
    finish_schema_migration_run,
    record_schema_state,
    start_schema_migration_run,
)


def _sql_literal(value: str) -> str:
    """Quote a configuration value for a SQLite DEFAULT expression."""
    return "'" + str(value).replace("'", "''") + "'"


def _add_column_if_missing(con, table: str, column: str, definition: str) -> None:
    """Apply one additive column migration safely under startup races."""
    columns = {row[1] for row in con.execute(f"pragma table_info({table})").fetchall()}
    if column in columns:
        return
    try:
        con.execute(f"alter table {table} add column {column} {definition}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _install_domain_integrity_guards(con, tables: tuple[str, ...]) -> None:
    """Reject impossible state transitions before they enter operational data."""
    enum_guards = (
        ("attendance", "attendance_status", ("recorded", "manual", "present", "late", "completed", "early_departure", "corrected", "excused", "absent"), "lower", False),
        ("presence_sessions", "identity_state", ("UNRESOLVED", "CANDIDATE", "CONFIRMED", "CONTRADICTED", "OCCLUDED", "EXPIRED", "SPOOF_SUSPECT"), "upper", True),
        ("presence_sessions", "liveness_status", ("REAL", "UNCERTAIN", "SPOOF_SUSPECT", "SUSPECT", "NOT_EVALUATED"), "upper", True),
        ("recognition_evidence", "identity_state", ("UNRESOLVED", "CANDIDATE", "CONFIRMED", "CONTRADICTED", "OCCLUDED", "EXPIRED", "SPOOF_SUSPECT"), "upper", True),
        ("recognition_evidence", "liveness_status", ("REAL", "UNCERTAIN", "SPOOF_SUSPECT", "SUSPECT", "NOT_EVALUATED"), "upper", True),
        ("attendance_decisions", "identity_state", ("UNRESOLVED", "CANDIDATE", "CONFIRMED", "CONTRADICTED", "OCCLUDED", "EXPIRED", "SPOOF_SUSPECT"), "upper", True),
        ("attendance_decisions", "liveness_status", ("REAL", "UNCERTAIN", "SPOOF_SUSPECT", "SUSPECT", "NOT_EVALUATED"), "upper", True),
        ("liveness_challenges", "liveness_status", ("REAL", "UNCERTAIN", "SPOOF_SUSPECT", "SUSPECT", "NOT_EVALUATED"), "upper", False),
        ("liveness_challenges", "challenge_state", ("IN_PROGRESS", "PASSED", "FAILED", "TIMED_OUT"), "upper", False),
        ("incidents", "status", ("open", "acknowledged", "assigned", "escalated", "confirmed", "dismissed", "resolved"), "lower", False),
        ("cybersecurity_incidents", "status", ("open", "acknowledged", "assigned", "escalated", "confirmed", "dismissed", "resolved"), "lower", False),
    )
    for table, column, allowed, normalizer, nullable in enum_guards:
        if table not in tables:
            continue
        columns = {row[1] for row in con.execute(f"pragma table_info({table})").fetchall()}
        if column not in columns:
            continue
        values = ", ".join(_sql_literal(value) for value in allowed)
        null_clause = "new.%s is not null and " % column if nullable else ""
        expression = f"{null_clause}{normalizer}(trim(coalesce(new.{column}, ''))) not in ({values})"
        for operation in ("insert", "update"):
            # Recreate these small guards so an existing database receives
            # changes to the domain vocabulary during a schema upgrade.
            con.execute(f"drop trigger if exists trg_domain_{table}_{column}_{operation}")
            con.execute(
                f"""create trigger if not exists trg_domain_{table}_{column}_{operation}
                    before {operation} on {table}
                    when {expression}
                    begin select raise(abort, 'invalid {table}.{column}'); end"""
            )

    chronology_guards = (
        (
            "attendance",
            "new.clock_in is not null and new.clock_out is not null and julianday(new.clock_out) < julianday(new.clock_in)",
            "attendance clock-out precedes clock-in",
            ("time_insert", "time_update"),
        ),
        (
            "presence_sessions",
            "new.ended_at is not null and julianday(new.ended_at) < julianday(new.started_at)",
            "presence session ended before it started",
            ("time_insert", "time_update"),
        ),
    )
    for table, expression, message, names in chronology_guards:
        if table not in tables:
            continue
        columns = {row[1] for row in con.execute(f"pragma table_info({table})").fetchall()}
        required = {"clock_in", "clock_out"} if table == "attendance" else {"started_at", "ended_at"}
        if not required.issubset(columns):
            continue
        for operation, name in (("insert", names[0]), ("update", names[1])):
            con.execute(
                f"""create trigger if not exists trg_domain_{table}_{name}
                    before {operation} on {table}
                    when {expression}
                    begin select raise(abort, '{message}'); end"""
            )

    if "attendance" in tables:
        for operation in ("insert", "update"):
            con.execute(
                f"""create trigger if not exists trg_domain_attendance_nonnegative_{operation}
                    before {operation} on attendance
                    when coalesce(new.work_minutes, 0) < 0
                      or coalesce(new.late_minutes, 0) < 0
                      or coalesce(new.early_departure_minutes, 0) < 0
                    begin select raise(abort, 'attendance durations cannot be negative'); end"""
            )


def _install_append_only_audit_guards(con) -> None:
    """Prevent ordinary UPDATE/DELETE operations on tamper-evident records."""
    for table in (
        "audit_log",
        "platform_audit_log",
        "audit_checkpoints",
        "attendance_corrections",
        "incident_review_actions",
        "incident_alerts",
    ):
        if not con.execute(
            "select 1 from sqlite_master where type='table' and name=?", (table,)
        ).fetchone():
            continue
        safe_table = table.replace('"', '""')
        safe_name = table.replace("_", "")
        update_condition = (
            "old.record_hash is not null or new.record_hash is null"
            if table in {"audit_log", "platform_audit_log"}
            else "1=1"
        )
        con.execute(
            f"create trigger if not exists trg_{safe_name}_append_only_update "
            f"before update on \"{safe_table}\" begin "
            f"select raise(abort, 'append-only audit record') where {update_condition}; end"
        )
        con.execute(
            f"create trigger if not exists trg_{safe_name}_append_only_delete "
            f"before delete on \"{safe_table}\" begin "
            "select raise(abort, 'append-only audit record'); end"
        )


def _ensure_platform_schema_unlocked(path=None) -> None:
    """Apply small, idempotent backend-owned schema additions.

    The vision engine owns the original SQLite tables. These tables extend the
    platform contract without taking ownership of biometric storage or forcing
    a risky rewrite of the working local engine schema.
    """
    with get_connection(path) as con:
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
                event_uid text,
                correlation_id text,
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
                 policy_version text,
                 unique(person_id, date),
                foreign key(person_id) references people(id) on delete cascade
            );
            create table if not exists policy_snapshots (
                policy_version text not null,
                declared_version text not null,
                organization_id text not null,
                site_id text not null,
                device_id text not null,
                document_json text not null,
                issues_json text not null default '[]',
                valid integer not null default 1,
                first_seen_at text not null default (datetime('now')),
                last_seen_at text not null default (datetime('now')),
                primary key(policy_version, organization_id, site_id, device_id)
            );
            create trigger if not exists trg_policy_snapshots_immutable_update
                before update on policy_snapshots
                when old.policy_version <> new.policy_version
                  or old.declared_version <> new.declared_version
                  or old.organization_id <> new.organization_id
                  or old.site_id <> new.site_id
                  or old.device_id <> new.device_id
                  or old.document_json <> new.document_json
                  or old.issues_json <> new.issues_json
                  or old.valid <> new.valid
                  or old.first_seen_at <> new.first_seen_at
                begin select raise(abort, 'policy snapshot content is immutable'); end;
            create trigger if not exists trg_policy_snapshots_immutable_delete
                before delete on policy_snapshots
                begin select raise(abort, 'policy snapshots are append-only'); end;
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
            create table if not exists liveness_challenges (
                id integer primary key autoincrement,
                entity_id text not null,
                presence_session_id integer,
                track_id integer,
                track_generation integer not null default 1,
                challenge_state text not null default 'IN_PROGRESS',
                phase text not null default 'CENTER',
                liveness_status text not null default 'UNCERTAIN',
                started_at text not null,
                updated_at text not null,
                completed_at text,
                attempt_number integer not null default 1,
                source_frame_id integer,
                failure_reason text,
                metrics_json text,
                foreign key(presence_session_id) references presence_sessions(id) on delete set null
            );
            """
        )
        # Keep the deployment boundary explicit across both database owners.
        # These columns are deliberately additive so existing local databases
        # remain readable while new backend writes receive stable defaults.
        scope_defaults = {
            "organization_id": ORGANIZATION_ID,
            "site_id": SITE_ID,
            "device_id": DEVICE_ID,
        }
        scoped_tables = (
            "people",
            "events",
            "attendance",
            "policy_snapshots",
            "absence_records",
            "attendance_schedules",
            "presence_sessions",
            "recognition_evidence",
            "enrollment_operations",
            "attendance_decisions",
            "liveness_challenges",
            "incidents",
            "incident_events",
            "incident_review_actions",
            "incident_alerts",
            "incident_evidence",
            "alert_log",
            "platform_audit_log",
            "attendance_corrections",
            "platform_outbox",
                "cybersecurity_events",
                "cybersecurity_incidents",
                "cybersecurity_incident_events",
                "cybersecurity_incident_alerts",
            "cybersecurity_reviews",
            "edge_sync_devices",
            "edge_sync_batches",
            "edge_sync_events",
        )
        for table in scoped_tables:
            if not con.execute(
                "select 1 from sqlite_master where type='table' and name=?", (table,)
            ).fetchone():
                continue
            columns = {row[1] for row in con.execute(f"pragma table_info({table})").fetchall()}
            for column, value in scope_defaults.items():
                if column not in columns:
                    try:
                        con.execute(
                            f"alter table {table} add column {column} text not null default {_sql_literal(value)}"
                        )
                    except sqlite3.OperationalError as exc:
                        # A second local process may have completed the same
                        # additive migration while this connection waited on
                        # SQLite's writer lock. Tolerate only that race.
                        if "duplicate column name" not in str(exc).lower():
                            raise
                con.execute(
                    f"update {table} set {column}=? where {column} is null or trim({column})=''",
                    (value,),
                )
        event_columns = {row[1] for row in con.execute("pragma table_info(events)").fetchall()}
        additions = {
            "entity_id": "text",
            "presence_session_id": "integer",
            "source_frame_id": "integer",
            "observation_type": "text",
            "evidence_path": "text",
            "correlation_id": "text",
            "review_status": "text not null default 'open'",
            "review_note": "text",
            "reviewed_at": "text",
            "reviewed_by": "text",
            "event_uid": "text",
        }
        for name, definition in additions.items():
            _add_column_if_missing(con, "events", name, definition)
        con.execute(
            "create unique index if not exists idx_events_event_uid "
            "on events(event_uid) where event_uid is not null"
        )

        alert_columns = {row[1] for row in con.execute("pragma table_info(alert_log)").fetchall()}
        if "source_event_id" not in alert_columns:
            _add_column_if_missing(con, "alert_log", "source_event_id", "integer")

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
            "policy_version": "text",
        }
        for name, definition in attendance_additions.items():
            _add_column_if_missing(con, "attendance", name, definition)

        presence_columns = {row[1] for row in con.execute(
            "pragma table_info(presence_sessions)").fetchall()}
        if "track_generation" not in presence_columns:
            _add_column_if_missing(con, "presence_sessions", "track_generation", "integer default 1")
        if "closed_reason" not in presence_columns:
            _add_column_if_missing(con, "presence_sessions", "closed_reason", "text")
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
                "correlation_id": "text",
                "assigned_to": "text",
                "false_positive": "integer not null default 0",
            }
            for name, definition in incident_additions.items():
                _add_column_if_missing(con, "incidents", name, definition)

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
                record_hash text,
                hash_version integer not null default 1
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
                correlation_id text,
                assigned_to text,
                false_positive integer not null default 0
            );
            create index if not exists idx_incidents_status on incidents(status);
            create index if not exists idx_incidents_updated on incidents(updated_at);
            create index if not exists idx_incidents_correlation on incidents(correlation_id);

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

            create table if not exists platform_outbox (
                id integer primary key autoincrement,
                event_id text not null unique,
                event_type text not null,
                aggregate_type text,
                aggregate_id text,
                payload_json text not null,
                payload_checksum text not null,
                status text not null default 'pending'
                    check (status in ('pending', 'processing', 'sent', 'failed', 'dead_letter')),
                attempts integer not null default 0,
                next_attempt_at text not null default (datetime('now')),
                locked_at text,
                locked_by text,
                sent_at text,
                last_error text,
                created_at text not null default (datetime('now')),
                updated_at text not null default (datetime('now')),
                organization_id text not null default 'local-organization',
                site_id text not null default 'local-site',
                device_id text not null default 'local-edge-cam-0'
            );
            create index if not exists idx_platform_outbox_due
                on platform_outbox(organization_id, site_id, status, next_attempt_at, id);
            create index if not exists idx_platform_outbox_aggregate
                on platform_outbox(organization_id, site_id, aggregate_type, aggregate_id);

            create table if not exists edge_sync_devices (
                device_id text not null,
                organization_id text not null,
                site_id text not null,
                last_sequence integer not null default 0,
                last_hash text not null default '',
                last_batch_id text,
                last_received_at text,
                status text not null default 'active',
                primary key (device_id, organization_id, site_id)
            );
            create table if not exists edge_sync_batches (
                id integer primary key autoincrement,
                batch_id text not null unique,
                device_id text not null,
                organization_id text not null,
                site_id text not null,
                first_sequence integer not null,
                last_sequence integer not null,
                last_hash text not null,
                record_count integer not null,
                body_checksum text not null,
                received_at text not null default (datetime('now')),
                unique(device_id, organization_id, site_id, last_sequence, last_hash)
            );
            create table if not exists edge_sync_events (
                id integer primary key autoincrement,
                event_id text not null,
                device_id text not null,
                organization_id text not null,
                site_id text not null,
                sequence integer not null,
                event_type text not null,
                occurred_at text,
                record_hash text not null,
                payload_json text not null,
                received_at text not null default (datetime('now')),
                unique(device_id, organization_id, site_id, sequence),
                unique(event_id, device_id, organization_id, site_id)
            );
            create index if not exists idx_edge_sync_events_scope_time
                on edge_sync_events(organization_id, site_id, device_id, received_at);

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
            create index if not exists idx_liveness_entity_time
                on liveness_challenges(entity_id, updated_at);
            create index if not exists idx_liveness_session_time
                on liveness_challenges(presence_session_id, updated_at);
            create index if not exists idx_absence_person_date on absence_records(person_id, absence_date);
            create index if not exists idx_absence_date on absence_records(absence_date, status);
            create index if not exists idx_incident_entity on incidents(entity_id, camera_id, location);
            create index if not exists idx_incident_session on incidents(presence_session_id);
            create index if not exists idx_incident_review on incident_review_actions(incident_id, created_at);
            create index if not exists idx_incident_evidence on incident_evidence(incident_id, captured_at);
            create index if not exists idx_session_expiry on platform_sessions(expires_at);
            """
        )
        # Application-security telemetry is intentionally separate from the
        # physical-vision incident tables above. Cyber events describe attacks
        # or failures of the application, edge device, or its trust boundary.
        con.executescript(
            """
            create table if not exists cybersecurity_events (
                id integer primary key autoincrement,
                correlation_id text not null unique,
                dedupe_key text unique,
                event_type text not null,
                category text not null,
                severity integer not null default 1,
                source text not null,
                actor_id text,
                actor_type text,
                device_id text,
                ip_address text,
                user_agent text,
                occurred_at text not null,
                details_json text,
                evidence_ref text,
                status text not null default 'open',
                incident_id integer,
                created_at text not null default (datetime('now')),
                foreign key(incident_id) references cybersecurity_incidents(id) on delete set null
            );
            create table if not exists cybersecurity_incidents (
                id integer primary key autoincrement,
                correlation_key text not null unique,
                category text not null,
                severity integer not null default 1,
                summary text not null,
                source text not null,
                actor_id text,
                device_id text,
                ip_address text,
                user_agent text,
                first_event_at text not null,
                last_event_at text not null,
                status text not null default 'open',
                assigned_to text,
                resolution_note text,
                resolved_at text,
                resolved_by text,
                created_at text not null default (datetime('now')),
                updated_at text not null default (datetime('now'))
            );
            create table if not exists cybersecurity_incident_events (
                incident_id integer not null,
                cyber_event_id integer not null unique,
                created_at text not null default (datetime('now')),
                primary key (incident_id, cyber_event_id),
                foreign key(incident_id) references cybersecurity_incidents(id) on delete cascade,
                foreign key(cyber_event_id) references cybersecurity_events(id) on delete cascade
            );
            create table if not exists cybersecurity_incident_alerts (
                id integer primary key autoincrement,
                incident_id integer not null,
                channel text not null,
                status text not null,
                correlation_id text,
                attempted_at text not null default (datetime('now')),
                delivered_at text,
                error text,
                attempt_count integer not null default 1,
                foreign key(incident_id) references cybersecurity_incidents(id) on delete cascade
            );
            create table if not exists cybersecurity_reviews (
                id integer primary key autoincrement,
                incident_id integer not null,
                action text not null,
                note text,
                actor_id text,
                created_at text not null default (datetime('now')),
                foreign key(incident_id) references cybersecurity_incidents(id) on delete cascade
            );
            create index if not exists idx_cyber_events_time on cybersecurity_events(occurred_at);
            create index if not exists idx_cyber_events_type on cybersecurity_events(event_type, occurred_at);
            create index if not exists idx_cyber_events_incident on cybersecurity_events(incident_id);
            create index if not exists idx_cyber_incidents_status on cybersecurity_incidents(status, updated_at);
            create index if not exists idx_cyber_incidents_category on cybersecurity_incidents(category, last_event_at);
            create index if not exists idx_cyber_alerts_status on cybersecurity_incident_alerts(status, attempted_at);
            create index if not exists idx_cyber_reviews_incident on cybersecurity_reviews(incident_id, created_at);
            """
        )
        # The incident, audit, and cybersecurity tables are created later in
        # this function. Re-run the additive scope pass after all DDL exists so
        # every backend-owned table receives the same deployment contract.
        for table in scoped_tables:
            if not con.execute(
                "select 1 from sqlite_master where type='table' and name=?", (table,)
            ).fetchone():
                continue
            columns = {row[1] for row in con.execute(f"pragma table_info({table})").fetchall()}
            for column, value in scope_defaults.items():
                _add_column_if_missing(
                    con, table, column, f"text not null default {_sql_literal(value)}"
                )
                con.execute(
                    f"update {table} set {column}=? where {column} is null or trim({column})=''",
                    (value,),
                )
        # SQLite has no native row-level security. These write-time guards
        # make the deployment boundary fail closed for cross-scope references;
        # the health checker still reports legacy corruption that predates the
        # triggers or was introduced by an offline import.
        con.executescript(
            """
            create trigger if not exists trg_scope_events_session
            before insert on events
            when new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )
            begin select raise(abort, 'event presence session scope mismatch'); end;

            create trigger if not exists trg_scope_attendance_person
            before insert on attendance
            when new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )
            begin select raise(abort, 'attendance person scope mismatch'); end;

            create trigger if not exists trg_scope_absence_person
            before insert on absence_records
            when new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )
            begin select raise(abort, 'absence person scope mismatch'); end;

            create trigger if not exists trg_scope_presence_person
            before insert on presence_sessions
            when new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )
            begin select raise(abort, 'presence person scope mismatch'); end;

            create trigger if not exists trg_scope_evidence_links
            before insert on recognition_evidence
            when (new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )) or (new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            ))
            begin select raise(abort, 'recognition evidence scope mismatch'); end;

            create trigger if not exists trg_scope_attendance_decision_links
            before insert on attendance_decisions
            when (new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )) or (new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )) or (new.recognition_evidence_id is not null and exists (
                select 1 from recognition_evidence e where e.id=new.recognition_evidence_id
                  and (e.organization_id<>new.organization_id or e.site_id<>new.site_id)
            ))
            begin select raise(abort, 'attendance decision scope mismatch'); end;

            create trigger if not exists trg_scope_liveness_session
            before insert on liveness_challenges
            when new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )
            begin select raise(abort, 'liveness session scope mismatch'); end;

            create trigger if not exists trg_scope_incident_session
            before insert on incidents
            when new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )
            begin select raise(abort, 'incident session scope mismatch'); end;

            create trigger if not exists trg_scope_incident_event
            before insert on incident_events
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
              or exists (select 1 from events e where e.id=new.event_id
                         and (e.organization_id<>new.organization_id or e.site_id<>new.site_id))
            begin select raise(abort, 'incident event scope mismatch'); end;

            create trigger if not exists trg_scope_incident_evidence
            before insert on incident_evidence
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
              or (new.event_id is not null and exists (select 1 from events e where e.id=new.event_id
                         and (e.organization_id<>new.organization_id or e.site_id<>new.site_id)))
            begin select raise(abort, 'incident evidence scope mismatch'); end;

            create trigger if not exists trg_scope_incident_review
            before insert on incident_review_actions
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'incident review scope mismatch'); end;

            create trigger if not exists trg_scope_incident_alert
            before insert on incident_alerts
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'incident alert scope mismatch'); end;

            create trigger if not exists trg_scope_cyber_event_incident
            before insert on cybersecurity_events
            when new.incident_id is not null and exists (
                select 1 from cybersecurity_incidents i where i.id=new.incident_id
                  and (i.organization_id<>new.organization_id or i.site_id<>new.site_id)
            )
            begin select raise(abort, 'cyber event incident scope mismatch'); end;

            create trigger if not exists trg_scope_cyber_incident_event
            before insert on cybersecurity_incident_events
            when exists (select 1 from cybersecurity_incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
              or exists (select 1 from cybersecurity_events e where e.id=new.cyber_event_id
                         and (e.organization_id<>new.organization_id or e.site_id<>new.site_id))
            begin select raise(abort, 'cyber incident event scope mismatch'); end;

            create trigger if not exists trg_scope_cyber_alert
            before insert on cybersecurity_incident_alerts
            when exists (select 1 from cybersecurity_incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'cyber alert scope mismatch'); end;

            create trigger if not exists trg_scope_cyber_review
            before insert on cybersecurity_reviews
            when exists (select 1 from cybersecurity_incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'cyber review scope mismatch'); end;
            """
        )
        # Deployment scope is an ownership boundary, not mutable business
        # data. Prevent re-parenting a row after it has acquired references;
        # migrations must create a new scoped record and preserve provenance.
        for table in scoped_tables:
            if not con.execute(
                "select 1 from sqlite_master where type='table' and name=?", (table,)
            ).fetchone():
                continue
            con.execute(
                f"""create trigger if not exists trg_scope_immutable_{table}
                    before update on {table}
                    when coalesce(old.organization_id, '')<>coalesce(new.organization_id, '')
                      or coalesce(old.site_id, '')<>coalesce(new.site_id, '')
                      or coalesce(old.device_id, '')<>coalesce(new.device_id, '')
                    begin select raise(abort, 'deployment scope is immutable'); end"""
            )
        _install_domain_integrity_guards(con, scoped_tables)
        con.executescript(
            """
            create trigger if not exists trg_scope_events_session_update
            before update on events
            when new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )
            begin select raise(abort, 'event presence session scope mismatch'); end;
            create trigger if not exists trg_scope_attendance_person_update
            before update on attendance
            when new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )
            begin select raise(abort, 'attendance person scope mismatch'); end;
            create trigger if not exists trg_scope_absence_person_update
            before update on absence_records
            when new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )
            begin select raise(abort, 'absence person scope mismatch'); end;
            create trigger if not exists trg_scope_presence_person_update
            before update on presence_sessions
            when new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )
            begin select raise(abort, 'presence person scope mismatch'); end;
            create trigger if not exists trg_scope_evidence_links_update
            before update on recognition_evidence
            when (new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )) or (new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            ))
            begin select raise(abort, 'recognition evidence scope mismatch'); end;
            create trigger if not exists trg_scope_attendance_decision_links_update
            before update on attendance_decisions
            when (new.person_id is not null and exists (
                select 1 from people p where p.id=new.person_id
                  and (p.organization_id<>new.organization_id or p.site_id<>new.site_id)
            )) or (new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )) or (new.recognition_evidence_id is not null and exists (
                select 1 from recognition_evidence e where e.id=new.recognition_evidence_id
                  and (e.organization_id<>new.organization_id or e.site_id<>new.site_id)
            ))
            begin select raise(abort, 'attendance decision scope mismatch'); end;
            create trigger if not exists trg_scope_liveness_session_update
            before update on liveness_challenges
            when new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )
            begin select raise(abort, 'liveness session scope mismatch'); end;
            create trigger if not exists trg_scope_incident_session_update
            before update on incidents
            when new.presence_session_id is not null and exists (
                select 1 from presence_sessions s where s.id=new.presence_session_id
                  and (s.organization_id<>new.organization_id or s.site_id<>new.site_id)
            )
            begin select raise(abort, 'incident session scope mismatch'); end;
            create trigger if not exists trg_scope_incident_event_update
            before update on incident_events
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
              or exists (select 1 from events e where e.id=new.event_id
                         and (e.organization_id<>new.organization_id or e.site_id<>new.site_id))
            begin select raise(abort, 'incident event scope mismatch'); end;
            create trigger if not exists trg_scope_incident_evidence_update
            before update on incident_evidence
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
              or (new.event_id is not null and exists (select 1 from events e where e.id=new.event_id
                         and (e.organization_id<>new.organization_id or e.site_id<>new.site_id)))
            begin select raise(abort, 'incident evidence scope mismatch'); end;
            create trigger if not exists trg_scope_incident_review_update
            before update on incident_review_actions
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'incident review scope mismatch'); end;
            create trigger if not exists trg_scope_incident_alert_update
            before update on incident_alerts
            when exists (select 1 from incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'incident alert scope mismatch'); end;
            create trigger if not exists trg_scope_cyber_event_incident_update
            before update on cybersecurity_events
            when new.incident_id is not null and exists (
                select 1 from cybersecurity_incidents i where i.id=new.incident_id
                  and (i.organization_id<>new.organization_id or i.site_id<>new.site_id)
            )
            begin select raise(abort, 'cyber event incident scope mismatch'); end;
            create trigger if not exists trg_scope_cyber_incident_event_update
            before update on cybersecurity_incident_events
            when exists (select 1 from cybersecurity_incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
              or exists (select 1 from cybersecurity_events e where e.id=new.cyber_event_id
                         and (e.organization_id<>new.organization_id or e.site_id<>new.site_id))
            begin select raise(abort, 'cyber incident event scope mismatch'); end;
            create trigger if not exists trg_scope_cyber_alert_update
            before update on cybersecurity_incident_alerts
            when exists (select 1 from cybersecurity_incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'cyber alert scope mismatch'); end;
            create trigger if not exists trg_scope_cyber_review_update
            before update on cybersecurity_reviews
            when exists (select 1 from cybersecurity_incidents i where i.id=new.incident_id
                         and (i.organization_id<>new.organization_id or i.site_id<>new.site_id))
            begin select raise(abort, 'cyber review scope mismatch'); end;
            """
        )
        con.executescript(
            """
            create index if not exists idx_events_scope_time
                on events(organization_id, site_id, timestamp);
            create index if not exists idx_attendance_scope_date
                on attendance(organization_id, site_id, date);
            create index if not exists idx_presence_scope_status
                on presence_sessions(organization_id, site_id, status);
            create unique index if not exists idx_presence_one_open_per_entity
                on presence_sessions(
                    coalesce(organization_id, ''),
                    coalesce(site_id, ''),
                    coalesce(device_id, ''),
                    coalesce(camera_id, ''),
                    entity_id
                )
                where status in ('active', 'occluded');
            create index if not exists idx_evidence_scope_time
                on recognition_evidence(organization_id, site_id, observed_at);
            create index if not exists idx_decisions_scope_time
                on attendance_decisions(organization_id, site_id, observed_at);
            create index if not exists idx_incidents_scope_status
                on incidents(organization_id, site_id, status, updated_at);
            create index if not exists idx_cyber_events_scope_time
                on cybersecurity_events(organization_id, site_id, occurred_at);
            create index if not exists idx_outbox_scope_status
                on platform_outbox(organization_id, site_id, status, next_attempt_at);
            create index if not exists idx_people_scope_name
                on people(organization_id, site_id, name);
            create index if not exists idx_alert_scope_time
                on alert_log(organization_id, site_id, timestamp);
            create index if not exists idx_absence_scope_date
                on absence_records(organization_id, site_id, absence_date);
            create index if not exists idx_schedule_scope_time
                on attendance_schedules(organization_id, site_id, weekday, start_time);
            create index if not exists idx_policy_snapshots_scope_time
                on policy_snapshots(organization_id, site_id, last_seen_at);
            create index if not exists idx_audit_scope_time
                on platform_audit_log(organization_id, site_id, created_at);
            create index if not exists idx_incident_evidence_scope_time
                on incident_evidence(organization_id, site_id, captured_at);
            create index if not exists idx_cyber_alert_scope_time
                on cybersecurity_incident_alerts(organization_id, site_id, attempted_at);
            create index if not exists idx_sync_batch_scope_time
                on edge_sync_batches(organization_id, site_id, device_id, received_at);
            create index if not exists idx_sync_event_scope_sequence
                on edge_sync_events(organization_id, site_id, device_id, sequence);
            """
        )
        user_columns = {row["name"] for row in con.execute("pragma table_info(platform_users)").fetchall()}
        for name, definition in (
            ("failed_login_count", "integer not null default 0"),
            ("locked_until", "text"),
            ("last_login_at", "text"),
            ("last_login_ip", "text"),
        ):
            _add_column_if_missing(con, "platform_users", name, definition)
        session_columns = {row["name"] for row in con.execute("pragma table_info(platform_sessions)").fetchall()}
        for name, definition in (
            ("csrf_token_hash", "text"),
            ("rotated_from", "text"),
            ("client_ip", "text"),
            ("user_agent", "text"),
        ):
            _add_column_if_missing(con, "platform_sessions", name, definition)
        event_columns = {row["name"] for row in con.execute("pragma table_info(events)").fetchall()}
        if "evidence_checksum" not in event_columns:
            _add_column_if_missing(con, "events", "evidence_checksum", "text")
        con.execute(
            "create unique index if not exists idx_events_event_uid "
            "on events(event_uid) where event_uid is not null"
        )
        ensure_schema_migrations(con)
        _install_append_only_audit_guards(con)
        record_schema_state(
            con,
            "platform",
            17,
            "deployment-scoped schema with attendance policy snapshot registry",
            hashlib.sha256(b"optivox-platform-schema-v17-policy-snapshot-registry").hexdigest(),
        )
        record_schema_state(
            con,
            "outbox",
            3,
            "transactional minimized operational event outbox with immutable scope",
            hashlib.sha256(b"optivox-outbox-schema-v3-immutable-deployment-scope").hexdigest(),
        )
        con.commit()


def ensure_platform_schema(path=None) -> None:
    """Apply the platform schema under the shared cross-process lock."""
    with database_migration_lock(path):
        run_id = None
        try:
            # Commit the start marker before the large additive operation so a
            # hard process failure remains visible to the next health check.
            with get_connection(path) as con:
                ensure_schema_migrations(con)
                run_id = start_schema_migration_run(con, "platform", 17)
                con.commit()
            _ensure_platform_schema_unlocked(path)
            with get_connection(path) as con:
                finish_schema_migration_run(con, run_id, "platform", 17, "applied")
                con.commit()
        except Exception as exc:
            if run_id:
                try:
                    with get_connection(path) as con:
                        finish_schema_migration_run(
                            con, run_id, "platform", 17, "failed", type(exc).__name__
                        )
                        con.commit()
                except Exception:
                    # Preserve the original failure; absence of a terminal
                    # marker is itself visible as a migration health issue.
                    pass
            raise
