"""Explicit safety policy for the OptiVox AI copilot.

The assistant is a read-only observer. Its natural-language suggestions are
never an authorization context for attendance, biometric, incident, or
permission changes.
"""

from __future__ import annotations


ASSISTANT_READ_ONLY_TOOLS = frozenset({
    "get_system_status",
    "get_enrolled_faces",
    "get_recent_events",
    "get_event_summary",
    "get_events_by_type",
    "get_person_events",
    "get_person_behavior_profile",
    "get_alert_history",
    "get_audit_log",
    "search_events",
    "get_attendance_report",
    "get_today_attendance",
    "get_active_strangers",
})

ASSISTANT_FORBIDDEN_ACTIONS = frozenset({
    "confirm_identity",
    "create_official_attendance",
    "clock_in",
    "clock_out",
    "dismiss_incident",
    "escalate_incident",
    "resolve_incident",
    "delete_biometric",
    "change_permissions",
    "enroll_person",
    "merge_people",
    "retrain_person",
})


def assistant_tool_allowed(name: str) -> bool:
    """Return true only for explicitly read-only database tools."""
    return str(name or "").strip() in ASSISTANT_READ_ONLY_TOOLS


def assistant_tool_policy(name: str) -> dict:
    """Describe how a requested assistant tool must be handled."""
    tool_name = str(name or "").strip()
    return {
        "tool": tool_name,
        "allowed": assistant_tool_allowed(tool_name),
        "mode": "read_only" if assistant_tool_allowed(tool_name) else "blocked",
        "requires_authorized_operator": tool_name in ASSISTANT_FORBIDDEN_ACTIONS,
    }
