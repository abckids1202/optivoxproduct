"""Deterministic replay harness for trusted OptiVox decisions.

The harness consumes normalized frame observations rather than opening a
camera. It is intentionally model-free: replay data represents the outputs of
the perception layer, while this module verifies entity, attendance, presence,
and intrusion policy behavior.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, Iterable, Optional

from .correlation import CorrelationCore
from .policy_engine import PolicyEngine
from .security import SecuritySignalEngine


class ReplayValidationError(ValueError):
    """Raised when replay input is malformed or violates the input contract."""


def _tracks(frame: dict) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for item in frame.get("tracks") or []:
        if not isinstance(item, dict):
            raise ReplayValidationError("Each track must be an object.")
        try:
            track_id = int(item["id"])
            center = item["center"]
            result[track_id] = (float(center[0]), float(center[1]))
        except (KeyError, TypeError, ValueError, IndexError):
            raise ReplayValidationError("Track requires id and a two-value center.")
    return result


def _faces(frame: dict) -> list[dict]:
    faces = frame.get("faces") or []
    if not isinstance(faces, list) or any(not isinstance(face, dict) for face in faces):
        raise ReplayValidationError("faces must be a list of objects.")
    return [dict(face) for face in faces]


def _canonical_faces(state: dict, faces: Iterable[dict]) -> list[dict]:
    by_track = {
        int(entity["track_id"]): entity
        for entity in state.get("entities") or []
        if entity.get("track_id") is not None
    }
    result = []
    for original in faces:
        face = dict(original)
        try:
            track_id = int(face.get("oid"))
        except (TypeError, ValueError):
            continue
        entity = by_track.get(track_id) or {}
        identity = entity.get("identity") or {}
        liveness = entity.get("liveness") or {}
        face.update({
            "oid": track_id,
            "name": identity.get("confirmed_name") if identity.get("state") == "CONFIRMED" else "UNKNOWN",
            "identity_state": identity.get("state", "UNRESOLVED"),
            "liveness_status": liveness.get("state", "NOT_EVALUATED"),
            "attendance_eligible": bool(entity.get("attendance_eligibility")),
            "track_generation": entity.get("track_generation", 1),
        })
        result.append(face)
    return result


def run_replay(scenario: dict[str, Any]) -> dict[str, Any]:
    """Replay a scenario and return decisions, signals, and invariant errors."""
    if not isinstance(scenario, dict):
        raise ReplayValidationError("Scenario must be an object.")
    frames = scenario.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReplayValidationError("Scenario requires a non-empty frames list.")

    policy = scenario.get("policy") or {}
    policy_config = dict(scenario.get("config") or {})
    # Replay-only thresholds are still part of the tested policy contract.
    policy_config["CORRELATION_CORE"] = {"replay": dict(policy)}
    policy_snapshot = PolicyEngine(policy_config).snapshot
    core = CorrelationCore(
        close_after_sec=float(policy.get("presence_close_after_sec", 2.0)),
        face_visible_after_sec=float(policy.get("face_visible_after_sec", 1.0)),
        quality_valid_after_sec=float(policy.get("quality_valid_after_sec", 1.5)),
        liveness_valid_after_sec=float(policy.get("liveness_valid_after_sec", 2.0)),
        identity_confirmation_observations=int(policy.get("identity_confirmation_observations", 3)),
        identity_confirmation_window=int(policy.get("identity_confirmation_window", 5)),
        track_switch_distance=float(policy.get("track_switch_distance", 250.0)),
        policy_version=policy_snapshot.policy_id,
        policy_valid=policy_snapshot.valid,
        policy_issues=policy_snapshot.issues,
    )
    security = SecuritySignalEngine(
        scenario.get("config") or {}, policy_version=policy_snapshot.policy_id)
    # A supplied roster is part of the replay contract, not a report-only
    # annotation. Configure it before any security or attendance decision so
    # the harness exercises the same authorization boundary as production.
    if "active_roster" in scenario:
        core.set_active_roster(scenario.get("active_roster") or [])
    attendance: list[dict] = []
    security_events: list[dict] = []
    transitions: list[dict] = []
    violations: list[str] = []
    identity_states: Counter[str] = Counter()
    replay_origin = time.monotonic()

    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ReplayValidationError(f"Frame {index} must be an object.")
        try:
            # Scenario times are relative seconds. Anchor them to the real
            # monotonic clock because ObservationHistory also prunes against
            # that clock during insertion.
            relative_time = float(frame.get("time", index))
            now = replay_origin + relative_time
        except (TypeError, ValueError):
            raise ReplayValidationError(f"Frame {index} has an invalid time.")
        occurred_at = str(frame.get("occurred_at") or f"replay:{relative_time:.6f}")
        tracked = _tracks(frame)
        raw_faces = _faces(frame)
        state = core.update(
            tracked=tracked,
            faces_info=raw_faces,
            source_frame_id=frame.get("frame_id", index + 1),
            camera_id=str(frame.get("camera_id") or scenario.get("camera_id") or "cam_0"),
            observed_at_monotonic=now,
            observed_at_wallclock=occurred_at,
        )
        transitions.extend(state.get("presence_transitions") or [])
        canonical = _canonical_faces(state, raw_faces)
        generations = {
            int(entity["track_id"]): int(entity.get("track_generation", 1) or 1)
            for entity in state.get("entities") or []
            if entity.get("track_id") is not None
        }
        signals = security.update(
            tracked,
            faces_info=canonical,
            now=now,
            wallclock=occurred_at,
            camera_id=str(frame.get("camera_id") or scenario.get("camera_id") or "cam_0"),
            track_generations=generations,
        )
        correlated = core.correlate_security_events(
            signals,
            source_frame_id=frame.get("frame_id", index + 1),
            camera_id=str(frame.get("camera_id") or scenario.get("camera_id") or "cam_0"),
            observed_at_monotonic=now,
            observed_at_wallclock=occurred_at,
        )
        for event in correlated.get("security_events") or []:
            security_events.append(event)
        for track_id in tracked:
            decision = core.attendance_decision(track_id)
            decision["track_id"] = track_id
            decision["frame_id"] = frame.get("frame_id", index + 1)
            attendance.append(decision)
            identity_states[str(decision.get("identity_state") or "UNRESOLVED")] += 1
            if decision["eligible"] and (
                decision.get("identity_state") != "CONFIRMED"
                or not decision.get("identity_evidence_fresh")
                or decision.get("liveness_state") != "REAL"
                or not decision.get("quality_ok")
            ):
                violations.append(
                    f"frame {decision['frame_id']} track {track_id} passed an unsafe attendance gate"
                )

    event_counts = Counter(item.get("event_type") for item in security_events)
    eligible = [item for item in attendance if item.get("eligible")]
    return {
        "schema_version": 1,
        # Replay inputs represent already-produced perception outputs. They
        # can verify policy invariants, but cannot measure the camera models
        # unless the labelled evaluation layer is run separately.
        "measurement_status": "NOT_MEASURED",
        "policy_verification": "PASS" if not violations else "FAIL",
        "policy_version": policy_snapshot.policy_id,
        "policy_valid": policy_snapshot.valid,
        "policy_issues": list(policy_snapshot.issues),
        "verdict": "PASS" if not violations else "FAIL",
        "frames": len(frames),
        "attendance": {
            "decisions": len(attendance),
            "eligible": len(eligible),
            "rejected": len(attendance) - len(eligible),
            "unknown_or_unresolved_rejected": sum(
                1 for item in attendance
                if not item.get("eligible") and item.get("identity_state") != "CONFIRMED"
            ),
        },
        "identity_states": dict(identity_states),
        "presence_transitions": transitions,
        "security_event_counts": dict(event_counts),
        "security_events": security_events,
        "attendance_decisions": attendance,
        "invariant_violations": violations,
        "notes": [
            "Replay validates policy and correlation behavior, not perception-model accuracy.",
            "Use separate holdout and adversarial data to measure recognition and liveness error rates.",
        ],
    }
