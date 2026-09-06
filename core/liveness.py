"""Entity-scoped head-turn challenge for local attendance verification."""

from __future__ import annotations

from typing import Dict, Optional


class LivenessChallenge:
    """Small deterministic state machine: center, left, right, center."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.states: Dict[str, dict] = {}

    def reset(self, key) -> None:
        self.states.pop(str(key), None)

    def reset_all(self) -> None:
        self.states.clear()

    def update(self, key, yaw, now: float, enabled: bool = True) -> dict:
        key = str(key)
        state = self.states.setdefault(key, {
            "phase": "CENTER",
            "started_at": now,
            "hold_started_at": None,
            "first_turn_sign": None,
            "passed": not enabled,
        })
        if not enabled:
            state["passed"] = True
            state["phase"] = "PASSED"
            return {**state, "message": "Liveness challenge disabled.", "progress": 1.0}

        timeout = float(self.cfg.get("CENTER_MODE_CHALLENGE_TIMEOUT_SEC", 15.0))
        hold_seconds = float(self.cfg.get("CENTER_MODE_TURN_HOLD_SEC", 0.35))
        neutral = float(self.cfg.get("CENTER_MODE_NEUTRAL_THRESHOLD", 0.07))
        turn = float(self.cfg.get("CENTER_MODE_YAW_THRESHOLD", 0.12))
        if now - state["started_at"] > timeout:
            self.reset(key)
            state = self.states[key]
            return {**state, "message": "Challenge timed out. Face forward to restart.", "progress": 0.0, "timed_out": True}

        def held(condition: bool) -> bool:
            if not condition:
                state["hold_started_at"] = None
                return False
            if state["hold_started_at"] is None:
                state["hold_started_at"] = now
                return False
            return now - state["hold_started_at"] >= hold_seconds

        if yaw is None:
            message = "Liveness landmarks unavailable. Keep one clear face visible."
        else:
            yaw = float(yaw)
            phase = state["phase"]
            if phase == "CENTER":
                if held(abs(yaw) <= neutral):
                    state["phase"] = "TURN_LEFT"
                    state["hold_started_at"] = None
                    message = "Now turn your head to the LEFT."
                else:
                    message = "Face forward and hold still."
            elif phase == "TURN_LEFT":
                if abs(yaw) >= turn:
                    state["first_turn_sign"] = -1 if yaw < 0 else 1
                    if held(True):
                        state["phase"] = "TURN_RIGHT"
                        state["hold_started_at"] = None
                        message = "Good. Now turn your head to the RIGHT."
                    else:
                        message = "Hold the LEFT turn."
                else:
                    message = "Turn your head to the LEFT."
            elif phase == "TURN_RIGHT":
                first = state.get("first_turn_sign") or -1
                if held(abs(yaw) >= turn and ((-1 if yaw < 0 else 1) != first)):
                    state["phase"] = "RETURN"
                    state["hold_started_at"] = None
                    message = "Return to the center and face forward."
                else:
                    message = "Turn to the opposite side."
            elif phase == "RETURN":
                if held(abs(yaw) <= neutral):
                    state["phase"] = "PASSED"
                    state["passed"] = True
                    state["hold_started_at"] = None
                    message = "Liveness passed. Finalizing attendance."
                else:
                    message = "Return to the center and face forward."
            else:
                state["passed"] = True
                message = "Liveness passed."
        progress_map = {"CENTER": 0.1, "TURN_LEFT": 0.35, "TURN_RIGHT": 0.6, "RETURN": 0.82, "PASSED": 1.0}
        return {**state, "yaw": yaw, "message": message, "progress": progress_map.get(state["phase"], 0.0)}

