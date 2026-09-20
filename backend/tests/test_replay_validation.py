import json

from core.replay import run_replay
from core.evaluation import (
    categorical_metrics,
    density_metrics,
    evaluate_acceptance,
    evaluate_replay,
    plate_metrics,
)
from core.evaluation_suite import run_suite


def _scenario():
    return {
        "camera_id": "cam_0",
        "active_roster": ["Ada"],
        "policy": {"identity_confirmation_observations": 3},
        "config": {"SECURITY": {"ENABLED": True, "ZONES": [{
            "id": "restricted", "bounds": [0, 0, 100, 100],
            "restricted": True, "allowed_names": ["Ada"],
        }]}},
        "frames": [
            {"frame_id": 1, "time": 0.0, "tracks": [{"id": 1, "center": [150, 150]}],
             "faces": [{"oid": 1, "name": "Ada", "identity_state": "CANDIDATE",
                        "confidence": .9, "quality_ok": True, "quality_score": 90,
                        "liveness_status": "REAL"}]},
            {"frame_id": 2, "time": .1, "tracks": [{"id": 1, "center": [150, 150]}],
             "faces": [{"oid": 1, "name": "Ada", "identity_state": "CANDIDATE",
                        "confidence": .91, "quality_ok": True, "quality_score": 91,
                        "liveness_status": "REAL"}]},
            {"frame_id": 3, "time": .2, "tracks": [{"id": 1, "center": [50, 50]}],
             "faces": [{"oid": 1, "name": "Ada", "identity_state": "CANDIDATE",
                        "confidence": .92, "quality_ok": True, "quality_score": 92,
                        "liveness_status": "REAL"}]},
        ],
    }


def test_replay_confirms_known_identity_only_after_temporal_votes():
    result = run_replay(_scenario())
    assert result["verdict"] == "PASS"
    assert result["measurement_status"] == "NOT_MEASURED"
    assert result["policy_verification"] == "PASS"
    assert result["attendance"]["eligible"] == 1
    assert result["identity_states"]["CONFIRMED"] >= 1


def test_replay_unknown_identity_cannot_pass_attendance_or_become_named_intrusion():
    scenario = _scenario()
    scenario["frames"].append({
        "frame_id": 4, "time": .3,
        "tracks": [
            {"id": 1, "center": [50, 50]},
            {"id": 2, "center": [150, 150]},
        ],
        "faces": [
            {"oid": 1, "name": "UNKNOWN", "identity_state": "UNRESOLVED",
             "confidence": .1, "quality_ok": True, "quality_score": 90,
             "liveness_status": "REAL"},
            {"oid": 2, "name": "UNKNOWN", "identity_state": "UNRESOLVED",
             "confidence": .1, "quality_ok": True, "quality_score": 90,
             "liveness_status": "REAL"},
        ],
    })
    scenario["frames"].append({
        "frame_id": 5, "time": .4,
        "tracks": [
            {"id": 1, "center": [50, 50]},
            {"id": 2, "center": [50, 50]},
        ],
        "faces": [
            {"oid": 1, "name": "UNKNOWN", "identity_state": "UNRESOLVED",
             "confidence": .1, "quality_ok": True, "quality_score": 90,
             "liveness_status": "REAL"},
            {"oid": 2, "name": "UNKNOWN", "identity_state": "UNRESOLVED",
             "confidence": .1, "quality_ok": True, "quality_score": 90,
             "liveness_status": "REAL"},
        ],
    })
    result = run_replay(scenario)
    assert result["attendance"]["unknown_or_unresolved_rejected"] >= 1
    intrusion_targets = [event["target"] for event in result["security_events"] if event["event_type"] == "ZONE_INTRUSION"]
    assert "UNKNOWN" in intrusion_targets
    assert "Ada" not in intrusion_targets
    assert result["verdict"] == "PASS"


def test_replay_active_roster_rejects_confirmed_but_inactive_identity():
    scenario = _scenario()
    scenario["active_roster"] = []
    result = run_replay(scenario)

    assert result["attendance"]["eligible"] == 0
    intrusion_targets = [
        event["target"] for event in result["security_events"]
        if event["event_type"] in {"ZONE_ENTRY", "ZONE_INTRUSION"}
    ]
    assert intrusion_targets
    assert set(intrusion_targets) == {"UNKNOWN"}
    assert result["verdict"] == "PASS"


def test_replay_records_occlusion_and_long_session_close():
    scenario = _scenario()
    scenario["frames"].extend([
        {"frame_id": 4, "time": .3, "tracks": [], "faces": []},
        {"frame_id": 5, "time": 2.5, "tracks": [], "faces": []},
    ])
    result = run_replay(scenario)
    transition_types = [item["type"] for item in result["presence_transitions"]]
    assert "occluded" in transition_types
    assert "closed" in transition_types


def test_replay_input_can_be_serialized_without_sensitive_model_data():
    encoded = json.dumps(_scenario())
    assert "embedding" not in encoded
    assert "frame_bytes" not in encoded


def test_replay_event_ids_are_stable_for_the_same_trace():
    first = run_replay(_scenario())
    second = run_replay(_scenario())
    assert [event["envelope"]["event_id"] for event in first["security_events"]] == [
        event["envelope"]["event_id"] for event in second["security_events"]
    ]


def test_evaluation_report_measures_labeled_identity_liveness_security_and_tracking():
    scenario = _scenario()
    for frame in scenario["frames"]:
        frame["faces"][0]["expected_name"] = "Ada"
        frame["faces"][0]["expected_live"] = True
        frame["tracks"][0]["ground_truth_id"] = "student-1"
    scenario["expected_security_events"] = ["ZONE_ENTRY", "ZONE_INTRUSION"]
    scenario["vehicle_speed_labels"] = [
        {"expected_kmh": 36, "predicted_kmh": 40},
        {"expected_kmh": 20, "predicted_kmh": 18},
    ]

    replay = run_replay(scenario)
    report = evaluate_replay(scenario, replay)

    assert report["status"] == "MEASURED"
    assert report["recognition"]["status"] == "MEASURED"
    assert report["recognition"]["false_accepts"] == 0
    assert report["recognition"]["true_unknown_rejects"] == 0
    assert report["liveness"]["false_positive"] == 0
    assert report["tracking"]["status"] == "MEASURED"
    assert report["tracking"]["id_switches"] == 0
    assert report["security"]["status"] == "MEASURED"
    assert report["vehicle"]["mean_absolute_error_kmh"] == 3.0


def test_evaluation_report_counts_unknown_false_accepts_and_tracker_switches():
    scenario = _scenario()
    for frame in scenario["frames"]:
        frame["faces"][0]["expected_name"] = "UNKNOWN"
        frame["faces"][0]["expected_live"] = False
        frame["tracks"][0]["ground_truth_id"] = "unknown-1"
    scenario["frames"][2]["faces"][0]["expected_name"] = "UNKNOWN"
    for index, frame in enumerate(scenario["frames"]):
        frame["tracks"].append({
            "id": (4, 9, 4)[index],
            "center": [240, 240],
            "ground_truth_id": "other-1",
        })

    report = evaluate_replay(scenario, run_replay(scenario))

    assert report["recognition"]["false_accepts"] >= 1
    assert report["liveness"]["false_positive"] >= 1
    assert report["tracking"]["id_switches"] == 2


def test_evaluation_suite_aggregates_unknown_rejection_accuracy():
    scenario = {
        "scenario_id": "unknown-rejection",
        "active_roster": ["Ada"],
        "frames": [
            {"frame_id": 1, "time": 0, "tracks": [{"id": 1, "center": [20, 20]}],
             "faces": [{"oid": 1, "name": "UNKNOWN", "identity_state": "UNRESOLVED",
                        "confidence": 0.1, "quality_ok": True, "liveness_status": "REAL",
                        "expected_name": "UNKNOWN"}]},
            {"frame_id": 2, "time": 0.2, "tracks": [{"id": 1, "center": [20, 20]}],
             "faces": [{"oid": 1, "name": "UNKNOWN", "identity_state": "UNRESOLVED",
                        "confidence": 0.1, "quality_ok": True, "liveness_status": "REAL",
                        "expected_name": "UNKNOWN"}]},
        ],
    }

    report = run_suite([scenario])

    assert report["aggregate"]["recognition"]["true_unknown_rejects"] == 2
    assert report["aggregate"]["recognition"]["unknown_rejection_accuracy"] == 1.0


def test_unlabeled_replay_does_not_claim_model_accuracy():
    scenario = _scenario()
    report = evaluate_replay(scenario, run_replay(scenario))

    assert report["status"] == "NOT_MEASURED"
    assert report["recognition"]["status"] == "NOT_MEASURED"
    assert report["liveness"]["status"] == "NOT_MEASURED"
    assert report["security"]["status"] == "NOT_MEASURED"
    assert report["vehicle"]["status"] == "NOT_MEASURED"


def test_security_evaluation_counts_repeated_events_instead_of_collapsing_to_a_set():
    scenario = _scenario()
    scenario["expected_security_events"] = ["ZONE_ENTRY", "ZONE_ENTRY"]
    report = evaluate_replay(scenario, run_replay(scenario))

    assert report["security"]["true_positive"] == 1
    assert report["security"]["false_negative"] == 1


def test_evaluation_suite_aggregates_labelled_cases_and_fails_malformed_cases():
    first = _scenario()
    first["scenario_id"] = "known-intrusion"
    for frame in first["frames"]:
        frame["faces"][0]["expected_name"] = "Ada"
        frame["faces"][0]["expected_live"] = True
        frame["tracks"][0]["ground_truth_id"] = "student-1"
    first["expected_security_events"] = ["ZONE_ENTRY"]

    second = {"scenario_id": "malformed", "frames": [{"tracks": "not-a-list"}]}
    report = run_suite([first, second])

    assert report["status"] == "FAIL"
    assert report["cases"] == 2
    assert report["aggregate"]["recognition"]["samples"] == 1
    assert report["aggregate"]["recognition"]["observed_samples"] == 3
    assert report["aggregate"]["recognition"]["deferred"] == 2
    assert report["aggregate"]["security"]["true_positive"] == 1
    assert report["cases_detail"][1]["error"]


def test_plate_metrics_preserve_unreadable_uncertainty():
    result = plate_metrics(["AB 123 CD", None, "XY-9"], ["AB123CD", None, "invented"])
    assert result["exact_matches"] == 1
    assert result["exact_match_rate"] == 0.5
    assert result["false_reads_on_unreadable"] == 0
    assert result["uncertainty_preservation_rate"] == 1.0


def test_age_band_and_density_metrics_are_explicitly_measured():
    assert categorical_metrics(["teen", "adult"], ["teen", "child"])["accuracy"] == 0.5
    result = density_metrics([10, 20], [12, 17])
    assert result["mean_absolute_error"] == 2.5
    assert result["max_absolute_error"] == 3.0


def test_replay_evaluation_exposes_optional_vehicle_and_demographic_sections():
    scenario = {
        "frames": [{"frame_id": 1, "time": 0, "tracks": [], "faces": []}],
        "plate_labels": [{"expected": "AB123", "predicted": None}],
        "age_band_labels": [{"expected_band": "adult", "predicted_band": "adult"}],
        "density_labels": [{"expected_count": 4, "predicted_count": 5}],
    }
    report = evaluate_replay(scenario, {"attendance_decisions": [], "security_events": []})
    assert report["plate"]["status"] == "MEASURED"
    assert report["demographics"]["decision_use"] == "aggregate_only"
    assert report["density"]["mean_absolute_error"] == 1.0

    suite = run_suite([{**scenario, "scenario_id": "optional-labels"}])
    assert suite["aggregate"]["plate"]["exact_match_rate"] == 0.0
    assert suite["aggregate"]["demographics"]["accuracy"] == 1.0
    assert suite["aggregate"]["density"]["mean_absolute_error"] == 1.0


def test_acceptance_gate_requires_measurement_and_applies_configured_limits():
    scenario = {
        "scenario_id": "thresholded-density",
        "frames": [{"frame_id": 1, "time": 0, "tracks": [], "faces": []}],
        "density_labels": [{"expected_count": 4, "predicted_count": 5}],
    }
    passing = run_suite(
        [scenario],
        {"density": {"mean_absolute_error_max": 1.0}},
    )
    failing = run_suite(
        [scenario],
        {"density": {"mean_absolute_error_max": 0.5}},
    )

    assert passing["acceptance"]["status"] == "PASS"
    assert failing["acceptance"]["status"] == "FAIL"
    assert evaluate_acceptance(
        {"aggregate": {"recognition": {"status": "NOT_MEASURED"}}},
        {"recognition": {"accuracy_min": 0.9}},
    )["status"] == "NOT_MEASURED"


def test_acceptance_gate_requires_declared_scenario_coverage():
    first = {"scenario_id": "known-face", "tags": ["known", "liveness"], **_scenario()}
    second = {"scenario_id": "unknown-face", "tags": ["unknown"], **_scenario()}
    passing = run_suite(
        [first, second],
        {"coverage": {"required_tags": ["known", "unknown"]}},
    )
    failing = run_suite(
        [first],
        {"coverage": {"required_tags": ["known", "unknown"]}},
    )

    assert passing["acceptance"]["status"] == "PASS"
    assert failing["acceptance"]["status"] == "FAIL"
    assert "unknown" not in failing["coverage"]["scenario_tags"]
