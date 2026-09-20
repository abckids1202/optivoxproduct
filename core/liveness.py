"""Entity-scoped liveness challenge and evaluation metrics.

The challenge is deliberately deterministic and model-agnostic. It is an
attendance gate, not a security-grade anti-spoof guarantee. The metrics
collector is also usable by a replay/evaluation harness without requiring a
camera or a particular liveness model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional


class LivenessMetrics:
    """Bounded counters for challenge outcomes and labelled evaluations."""

    def __init__(self):
        self.attempts = 0
        self.completions = 0
        self.timeouts = 0
        self.failures = 0
        self.confirmation_times = []
        self.scenarios = defaultdict(lambda: {
            "evaluations": 0,
            "true_accepts": 0,
            "true_rejects": 0,
            "false_accepts": 0,
            "false_rejects": 0,
        })

    def record_attempt(self) -> None:
        self.attempts += 1

    def record_completion(self, duration_sec: float) -> None:
        self.completions += 1
        self.confirmation_times.append(max(0.0, float(duration_sec)))
        self.confirmation_times = self.confirmation_times[-1000:]

    def record_timeout(self) -> None:
        self.timeouts += 1
        self.failures += 1

    def record_failure(self) -> None:
        self.failures += 1

    def record_evaluation(self, scenario: str, predicted_status: str,
                          expected_live: Optional[bool]) -> None:
        """Record a labelled result from a controlled liveness evaluation."""
        name = str(scenario or "unlabelled").strip()[:80] or "unlabelled"
        status = str(predicted_status or "NOT_EVALUATED").strip().upper()
        row = self.scenarios[name]
        row["evaluations"] += 1
        if expected_live is True:
            if status == "REAL":
                row["true_accepts"] += 1
            else:
                row["false_rejects"] += 1
        elif expected_live is False:
            if status == "REAL":
                row["false_accepts"] += 1
            else:
                row["true_rejects"] += 1

    def snapshot(self) -> dict:
        durations = self.confirmation_times
        labelled = {
            "evaluations": 0,
            "true_accepts": 0,
            "true_rejects": 0,
            "false_accepts": 0,
            "false_rejects": 0,
        }
        for row in self.scenarios.values():
            for key in labelled:
                labelled[key] += int(row[key])
        scenario_metrics = {}
        for name, row in self.scenarios.items():
            negatives = int(row["true_rejects"]) + int(row["false_accepts"])
            positives = int(row["true_accepts"]) + int(row["false_rejects"])
            scenario_metrics[name] = {
                **dict(row),
                "false_accept_rate": (
                    int(row["false_accepts"]) / negatives if negatives else None
                ),
                "false_reject_rate": (
                    int(row["false_rejects"]) / positives if positives else None
                ),
            }
        labelled["false_accept_rate"] = (
            labelled["false_accepts"] /
            (labelled["true_rejects"] + labelled["false_accepts"])
            if labelled["true_rejects"] + labelled["false_accepts"] else None
        )
        labelled["false_reject_rate"] = (
            labelled["false_rejects"] /
            (labelled["true_accepts"] + labelled["false_rejects"])
            if labelled["true_accepts"] + labelled["false_rejects"] else None
        )
        return {
            "attempts": self.attempts,
            "completions": self.completions,
            "timeouts": self.timeouts,
            "failures": self.failures,
            "completion_rate": round(self.completions / self.attempts, 4) if self.attempts else None,
            "confirmation_time_sec": {
                "count": len(durations),
                "mean": round(sum(durations) / len(durations), 4) if durations else None,
                "max": round(max(durations), 4) if durations else None,
            },
            **labelled,
            "scenarios": scenario_metrics,
        }


class LivenessChallenge:
    """Deterministic state machine: center, left, right, center.

    State is keyed by the caller-provided entity ID. A tracker ID should not
    be reused as this key after a generation change; the correlation layer
    supplies stable entity IDs for the attendance path.
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.states: Dict[str, dict] = {}
        self.metrics = LivenessMetrics()

    def _new_state(self, now: float) -> dict:
        self.metrics.record_attempt()
        return {
            "phase": "CENTER",
            "status": "UNCERTAIN",
            "started_at": float(now),
            "hold_started_at": None,
            "first_turn_sign": None,
            "passed": False,
            "attempt_number": self.metrics.attempts,
            "timeout_count": 0,
            "failure_count": 0,
            "retry_count": 0,
            "completed_at": None,
            "challenge_duration_sec": None,
        }

    def reset(self, key) -> None:
        self.states.pop(str(key), None)

    def reset_all(self) -> None:
        self.states.clear()

    def state_for(self, key) -> Optional[dict]:
        state = self.states.get(str(key))
        return dict(state) if state else None

    def metrics_snapshot(self) -> dict:
        return self.metrics.snapshot()

    def record_evaluation(self, scenario: str, predicted_status: str,
                          expected_live: Optional[bool]) -> None:
        self.metrics.record_evaluation(scenario, predicted_status, expected_live)

    def update(self, key, yaw, now: float, enabled: bool = True) -> dict:
        key = str(key)
        now = float(now)
        state = self.states.get(key)
        if state is None:
            state = self._new_state(now)
            self.states[key] = state
        if not enabled:
            if not state.get("passed"):
                state["passed"] = True
                state["phase"] = "PASSED"
                state["status"] = "REAL"
                state["completed_at"] = now
                state["challenge_duration_sec"] = max(0.0, now - state["started_at"])
                self.metrics.record_completion(state["challenge_duration_sec"])
            return {**state, "message": "Liveness challenge disabled.", "progress": 1.0}

        timeout = float(self.cfg.get("CENTER_MODE_CHALLENGE_TIMEOUT_SEC", 15.0))
        hold_seconds = float(self.cfg.get("CENTER_MODE_TURN_HOLD_SEC", 0.35))
        max_attempts = max(1, int(self.cfg.get("CENTER_MODE_MAX_LIVENESS_ATTEMPTS", 3)))
        neutral = float(self.cfg.get("CENTER_MODE_NEUTRAL_THRESHOLD", 0.07))
        turn = float(self.cfg.get("CENTER_MODE_YAW_THRESHOLD", 0.12))
        left_sign = 1 if float(self.cfg.get("CENTER_MODE_LEFT_YAW_SIGN", 1)) >= 0 else -1
        if state.get("phase") == "FAILED":
            return {
                **state,
                "message": "Liveness challenge failed. Restart the attendance flow to retry.",
                "progress": 0.0,
                "terminal": True,
            }
        if now - state["started_at"] > timeout:
            self.metrics.record_timeout()
            timeout_count = int(state.get("timeout_count") or 0) + 1
            failure_count = int(state.get("failure_count") or 0) + 1
            retry_count = int(state.get("retry_count") or 0) + 1
            if retry_count >= max_attempts:
                state.update({
                    "phase": "FAILED",
                    "status": "SPOOF_SUSPECT",
                    "timeout_count": timeout_count,
                    "failure_count": failure_count,
                    "retry_count": retry_count,
                    "completed_at": now,
                    "challenge_duration_sec": max(0.0, now - state["started_at"]),
                })
                self.states[key] = state
                return {
                    **state,
                    "message": "Liveness challenge failed after the maximum retries.",
                    "progress": 0.0,
                    "timed_out": True,
                    "terminal": True,
                }
            # Keep the replacement state in the per-entity map. Without this
            # assignment, the next frame silently starts from an untracked
            # attempt and loses timeout/failure history.
            self.reset(key)
            state = self._new_state(now)
            state["timeout_count"] = timeout_count
            state["failure_count"] = failure_count
            state["retry_count"] = retry_count
            self.states[key] = state
            return {**state, "message": "Challenge timed out. Face forward to restart.",
                    "progress": 0.0, "timed_out": True, "terminal": False}

        def held(condition: bool) -> bool:
            if not condition:
                state["hold_started_at"] = None
                return False
            if state["hold_started_at"] is None:
                state["hold_started_at"] = now
                return False
            return now - state["hold_started_at"] >= hold_seconds

        message = "Keep one clear face visible."
        if yaw is None:
            state["status"] = "UNCERTAIN"
            message = "Liveness landmarks unavailable. Keep one clear face visible."
        else:
            yaw = float(yaw)
            phase = state["phase"]
            sign = -1 if yaw < 0 else 1
            if phase == "CENTER":
                if held(abs(yaw) <= neutral):
                    state["phase"] = "TURN_LEFT"
                    state["hold_started_at"] = None
                    message = "Now turn your head to the LEFT."
                else:
                    message = "Face forward and hold still."
            elif phase == "TURN_LEFT":
                if abs(yaw) >= turn and sign == left_sign:
                    state["first_turn_sign"] = sign
                    if held(True):
                        state["phase"] = "TURN_RIGHT"
                        state["hold_started_at"] = None
                        message = "Good. Now turn your head to the RIGHT."
                    else:
                        message = "Hold the LEFT turn."
                else:
                    state["hold_started_at"] = None
                    message = "Turn to the LEFT, not the right."
            elif phase == "TURN_RIGHT":
                first = state.get("first_turn_sign") or left_sign
                if held(abs(yaw) >= turn and sign != first):
                    state["phase"] = "RETURN"
                    state["hold_started_at"] = None
                    message = "Return to the center and face forward."
                else:
                    message = "Turn to the opposite side."
            elif phase == "RETURN":
                if held(abs(yaw) <= neutral):
                    state["phase"] = "PASSED"
                    state["passed"] = True
                    state["status"] = "REAL"
                    state["hold_started_at"] = None
                    state["completed_at"] = now
                    state["challenge_duration_sec"] = max(0.0, now - state["started_at"])
                    self.metrics.record_completion(state["challenge_duration_sec"])
                    message = "Liveness passed. Finalizing attendance."
                else:
                    message = "Return to the center and face forward."
            else:
                state["passed"] = True
                state["status"] = "REAL"
                message = "Liveness passed."
        progress_map = {
            "CENTER": 0.1, "TURN_LEFT": 0.35, "TURN_RIGHT": 0.6,
            "RETURN": 0.82, "PASSED": 1.0, "FAILED": 0.0,
        }
        return {
            **state,
            "yaw": yaw,
            "message": message,
            "progress": progress_map.get(state["phase"], 0.0),
        }
