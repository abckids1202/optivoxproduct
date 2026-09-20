from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from ..config import ORGANIZATION_ID, SITE_ID
from ..database import fetch_all, fetch_one


def _details(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {"raw": str(value)}


def _normalize_session(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status") or "active"
    identity_state = row.get("identity_state") or "UNRESOLVED"
    label = row.get("label") or "UNKNOWN"
    if str(identity_state).upper() != "CONFIRMED":
        label = "UNKNOWN"
    return {
        "id": row.get("id"),
        "entityId": row.get("entity_id"),
        "trackId": row.get("track_id"),
        "personId": row.get("person_id"),
        "label": label,
        "identityState": identity_state,
        "livenessStatus": row.get("liveness_status"),
        "cameraId": row.get("camera_id"),
        "startedAt": row.get("started_at"),
        "lastSeenAt": row.get("last_seen_at"),
        "endedAt": row.get("ended_at"),
        "closedReason": row.get("closed_reason"),
        "status": status,
        "active": status == "active",
        "confidence": row.get("confidence") or 0.0,
        "firstFrameId": row.get("first_frame_id"),
        "lastFrameId": row.get("last_frame_id"),
        "evidenceCount": int(row.get("evidence_count") or 0),
    }


def _normalize_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "entityId": row.get("entity_id"),
        "presenceSessionId": row.get("presence_session_id"),
        "trackId": row.get("track_id"),
        "personId": row.get("person_id"),
        "personName": row.get("person_name"),
        "candidateName": row.get("candidate_name") or "UNKNOWN",
        "decision": row.get("decision") or "unresolved",
        "similarity": row.get("similarity") or 0.0,
        "qualityScore": row.get("quality_score") or 0.0,
        "qualityOk": bool(row.get("quality_ok")),
        "livenessStatus": row.get("liveness_status"),
        "identityState": row.get("identity_state"),
        "reason": row.get("reason"),
        "sourceFrameId": row.get("source_frame_id"),
        "observedAt": row.get("observed_at"),
        "details": _details(row.get("details_json")),
    }


def _normalize_decision(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "decisionKey": row.get("decision_key"),
        "personId": row.get("person_id"),
        "entityId": row.get("entity_id"),
        "presenceSessionId": row.get("presence_session_id"),
        "recognitionEvidenceId": row.get("recognition_evidence_id"),
        "decision": row.get("decision"),
        "reason": row.get("reason"),
        "identityState": row.get("identity_state"),
        "livenessStatus": row.get("liveness_status"),
        "qualityScore": row.get("quality_score"),
        "recognitionConfidence": row.get("recognition_confidence"),
        "sourceFrameId": row.get("source_frame_id"),
        "observedAt": row.get("observed_at"),
        "details": _details(row.get("details_json")),
        "createdAt": row.get("created_at"),
    }


def _safe_liveness_metrics(value: Any) -> dict[str, Any] | None:
    """Expose only bounded scalar challenge metrics, never biometric material."""
    parsed = _details(value)
    if not isinstance(parsed, dict):
        return None
    allowed = {
        "forward_frames", "left_frames", "right_frames", "return_frames",
        "yaw", "quality", "duration_ms", "retry_count",
    }
    result: dict[str, Any] = {}
    for key in allowed:
        item = parsed.get(key)
        if isinstance(item, (str, int, float, bool)):
            result[key] = item
    return result or None


def _normalize_liveness(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "entityId": row.get("entity_id"),
        "presenceSessionId": row.get("presence_session_id"),
        "trackId": row.get("track_id"),
        "trackGeneration": row.get("track_generation") or 1,
        "challengeState": row.get("challenge_state") or "IN_PROGRESS",
        "phase": row.get("phase") or "CENTER",
        "livenessStatus": row.get("liveness_status") or "NOT_EVALUATED",
        "startedAt": row.get("started_at"),
        "updatedAt": row.get("updated_at"),
        "completedAt": row.get("completed_at"),
        "attemptNumber": row.get("attempt_number") or 1,
        "sourceFrameId": row.get("source_frame_id"),
        "failureReason": row.get("failure_reason"),
        "metrics": _safe_liveness_metrics(row.get("metrics_json")),
    }


def summary() -> dict[str, Any]:
    sessions = fetch_one(
        """select count(*) as total,
                  sum(case when status='active' then 1 else 0 end) as active,
                  sum(case when status='active' and identity_state='CONFIRMED' then 1 else 0 end) as confirmed,
                  sum(case when status='active' and (identity_state<>'CONFIRMED' or label='UNKNOWN') then 1 else 0 end) as unresolved
           from presence_sessions
           where organization_id=? and site_id=?""",
        [ORGANIZATION_ID, SITE_ID],
    ) or {}
    evidence = fetch_one(
        """select count(*) as total,
                  sum(case when decision='confirmed' then 1 else 0 end) as confirmed,
                  sum(case when decision in ('spoof_or_uncertain','contradicted') then 1 else 0 end) as rejected
           from recognition_evidence
           where organization_id=? and site_id=?""",
        [ORGANIZATION_ID, SITE_ID],
    ) or {}
    challenges = fetch_one(
        """select count(*) as total,
                  sum(case when challenge_state='PASSED' then 1 else 0 end) as passed,
                  sum(case when challenge_state in ('FAILED','TIMED_OUT') then 1 else 0 end) as failed,
                  sum(case when challenge_state='IN_PROGRESS' then 1 else 0 end) as active
           from liveness_challenges
           where organization_id=? and site_id=?""",
        [ORGANIZATION_ID, SITE_ID],
    ) or {}
    return {
        "presenceSessions": int(sessions.get("total") or 0),
        "activePresenceSessions": int(sessions.get("active") or 0),
        "activeConfirmedEntities": int(sessions.get("confirmed") or 0),
        "activeUnresolvedEntities": int(sessions.get("unresolved") or 0),
        "recognitionEvidence": int(evidence.get("total") or 0),
        "confirmedEvidence": int(evidence.get("confirmed") or 0),
        "rejectedEvidence": int(evidence.get("rejected") or 0),
        "livenessChallenges": int(challenges.get("total") or 0),
        "passedLivenessChallenges": int(challenges.get("passed") or 0),
        "failedLivenessChallenges": int(challenges.get("failed") or 0),
        "activeLivenessChallenges": int(challenges.get("active") or 0),
    }


def list_liveness_challenges(
    limit: int = 100,
    entity_id: str | None = None,
    session_id: int | None = None,
    challenge_state: str | None = None,
) -> list[dict[str, Any]]:
    """Return the operational liveness trail without biometric content."""
    clauses = ["l.organization_id=?", "l.site_id=?"]
    params: list[Any] = [ORGANIZATION_ID, SITE_ID]
    if entity_id:
        clauses.append("l.entity_id=?")
        params.append(entity_id)
    if session_id is not None:
        clauses.append("l.presence_session_id=?")
        params.append(session_id)
    if challenge_state:
        clauses.append("l.challenge_state=?")
        params.append(challenge_state.upper())
    where = f"where {' and '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    return [
        _normalize_liveness(row)
        for row in fetch_all(
            f"""select l.* from liveness_challenges l {where}
                order by l.updated_at desc, l.id desc limit ?""",
            params,
        )
    ]


def list_presence_sessions(
    limit: int = 100,
    status: str | None = None,
    person_id: int | None = None,
) -> list[dict[str, Any]]:
    clauses = ["s.organization_id=?", "s.site_id=?"]
    params: list[Any] = [ORGANIZATION_ID, SITE_ID]
    if status:
        clauses.append("s.status=?")
        params.append(status)
    if person_id is not None:
        clauses.append("s.person_id=?")
        params.append(person_id)
    where = f"where {' and '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    return [
        _normalize_session(row)
        for row in fetch_all(
            f"""select s.*, count(r.id) as evidence_count
                from presence_sessions s
                left join recognition_evidence r on r.presence_session_id=s.id
                {where}
                group by s.id
                order by s.last_seen_at desc, s.id desc limit ?""",
            params,
        )
    ]


def get_presence_session(session_id: int) -> dict[str, Any]:
    row = fetch_one(
        """select s.*, count(r.id) as evidence_count
           from presence_sessions s
           left join recognition_evidence r on r.presence_session_id=s.id
           where s.id=? and s.organization_id=? and s.site_id=? group by s.id""",
        [session_id, ORGANIZATION_ID, SITE_ID],
    )
    if not row:
        raise HTTPException(status_code=404, detail={"code": "PRESENCE_SESSION_NOT_FOUND", "message": "Presence session was not found."})
    result = _normalize_session(row)
    result["evidence"] = list_recognition_evidence(session_id=session_id, limit=200)
    result["events"] = fetch_all(
        """select id, event_type, confidence, details_json, snapshot_path,
                  camera_id, location, severity, timestamp, entity_id,
           presence_session_id, source_frame_id, observation_type,
                  evidence_path
           from events where presence_session_id=? and organization_id=? and site_id=? order by timestamp desc limit 200""",
        [session_id, ORGANIZATION_ID, SITE_ID],
    )
    for event in result["events"]:
        event_id = event.get("id")
        has_evidence = bool(event.get("snapshot_path") or event.get("evidence_path"))
        event["snapshot_available"] = has_evidence
        event["snapshot_url"] = f"/api/events/{event_id}/snapshot" if has_evidence else None
        event.pop("snapshot_path", None)
        event.pop("evidence_path", None)
    return result


def list_recognition_evidence(
    limit: int = 100,
    session_id: int | None = None,
    entity_id: str | None = None,
    person_id: int | None = None,
    decision: str | None = None,
) -> list[dict[str, Any]]:
    clauses = ["r.organization_id=?", "r.site_id=?"]
    params: list[Any] = [ORGANIZATION_ID, SITE_ID]
    if session_id is not None:
        clauses.append("r.presence_session_id=?")
        params.append(session_id)
    if entity_id:
        clauses.append("r.entity_id=?")
        params.append(entity_id)
    if person_id is not None:
        clauses.append("r.person_id=?")
        params.append(person_id)
    if decision:
        clauses.append("r.decision=?")
        params.append(decision)
    where = f"where {' and '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    return [
        _normalize_evidence(row)
        for row in fetch_all(
            f"""select r.*, p.name as person_name
                from recognition_evidence r
                left join people p on p.id=r.person_id
                {where}
                order by r.observed_at desc, r.id desc limit ?""",
            params,
        )
    ]


def list_attendance_decisions(
    limit: int = 100,
    entity_id: str | None = None,
    person_id: int | None = None,
    decision: str | None = None,
) -> list[dict[str, Any]]:
    """Expose the automatic gate audit trail without exposing biometrics."""
    clauses = ["d.organization_id=?", "d.site_id=?"]
    params: list[Any] = [ORGANIZATION_ID, SITE_ID]
    if entity_id:
        clauses.append("d.entity_id=?")
        params.append(entity_id)
    if person_id is not None:
        clauses.append("d.person_id=?")
        params.append(person_id)
    if decision:
        clauses.append("d.decision=?")
        params.append(decision)
    where = f"where {' and '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    return [
        _normalize_decision(row)
        for row in fetch_all(
            f"""select d.* from attendance_decisions d {where}
                order by d.observed_at desc, d.id desc limit ?""",
            params,
        )
    ]
