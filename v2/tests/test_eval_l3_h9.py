from collections import namedtuple

from v2.eval_l3.hypothesis_h9 import (
    build_actual_hazard_events, build_predicted_events,
    total_km_driven_m, compute_h9_metrics,
)

_KeyRow = namedtuple("KeyRow", ["bound", "run", "sample_id", "instance_id", "pred_ttc"])
_EgoState = namedtuple("EgoState", ["t", "x", "y", "vx", "vy"])


def _samples_index(entries):
    """entries: {sample_id: (timestamp_us, scene_name)}."""
    return {sid: {"timestamp_us": ts, "scene_name": scene} for sid, (ts, scene) in entries.items()}


# ---------------------------------------------------------------------
# build_actual_hazard_events / build_predicted_events: event collapsing
# ---------------------------------------------------------------------

def test_actual_hazard_events_collapses_consecutive_danger_into_one_event():
    samples_index = _samples_index({
        "s0": (0, "scene-a"), "s1": (1, "scene-a"), "s2": (2, "scene-a"), "s3": (3, "scene-a"),
    })
    # danger at s0,s1 (contiguous) then safe at s2, danger again at s3 -> TWO events
    gt_ttc_lookup = {
        ("s0", "A"): {"ttc": 1.0}, ("s1", "A"): {"ttc": 1.2},
        ("s2", "A"): {"ttc": 10.0}, ("s3", "A"): {"ttc": 0.5},
    }
    events = build_actual_hazard_events(gt_ttc_lookup, samples_index, threshold=2.5)
    assert len(events["A"]) == 2
    assert events["A"][0] == frozenset({"s0", "s1"})
    assert events["A"][1] == frozenset({"s3"})


def test_actual_hazard_events_orders_by_timestamp_not_sample_id_string():
    # s10 has an EARLIER timestamp than s2 -- string order would get this wrong
    samples_index = _samples_index({"s2": (100, "scene-a"), "s10": (0, "scene-a")})
    gt_ttc_lookup = {("s10", "A"): {"ttc": 1.0}, ("s2", "A"): {"ttc": 1.0}}
    events = build_actual_hazard_events(gt_ttc_lookup, samples_index, threshold=2.5)
    assert events["A"] == [frozenset({"s10", "s2"})]   # still one contiguous event


def test_build_predicted_events_tracks_covered_samples_independent_of_threshold():
    samples_index = _samples_index({"s0": (0, "scene-a"), "s1": (1, "scene-a")})
    rows = [
        _KeyRow("hungarian", "lidar", "s0", "A", 10.0),   # covered, but not below threshold
        _KeyRow("hungarian", "lidar", "s1", "A", 1.0),    # covered AND below threshold
    ]
    events, covered = build_predicted_events(rows, samples_index, threshold=2.5)
    assert events["A"] == [frozenset({"s1"})]
    assert covered["A"] == {"s0", "s1"}   # both samples covered, regardless of threshold


# ---------------------------------------------------------------------
# total_km_driven_m: scene isolation
# ---------------------------------------------------------------------

def test_total_km_driven_sums_within_scene_only_not_across_boundary():
    samples_index = _samples_index({
        "s0": (0, "scene-a"), "s1": (1, "scene-a"),   # 3-4-5 triangle -> 5m apart
        "s2": (2, "scene-b"),                          # different scene, far away
    })
    ego_by_sample = {
        "s0": _EgoState(0.0, 0.0, 0.0, 0.0, 0.0),
        "s1": _EgoState(1.0, 3.0, 4.0, 0.0, 0.0),
        "s2": _EgoState(2.0, 10000.0, 10000.0, 0.0, 0.0),   # would add a huge, bogus delta if not isolated
    }
    km = total_km_driven_m(ego_by_sample, samples_index)
    assert abs(km - 0.005) < 1e-9   # only the 5m within scene-a counts; scene-b has only 1 sample, no delta


def test_total_km_driven_skips_samples_with_no_ego_state():
    samples_index = _samples_index({"s0": (0, "scene-a"), "s1": (1, "scene-a")})
    ego_by_sample = {"s0": _EgoState(0.0, 0.0, 0.0, 0.0, 0.0)}   # s1 missing
    km = total_km_driven_m(ego_by_sample, samples_index)
    assert km == 0.0


# ---------------------------------------------------------------------
# compute_h9_metrics: TP/FP/FN classification
# ---------------------------------------------------------------------

def test_compute_h9_metrics_true_positive_when_events_overlap():
    actual = {"A": [frozenset({"s0", "s1"})]}
    predicted = {"A": [frozenset({"s1", "s2"})]}   # overlaps at s1
    covered = {"A": {"s0", "s1", "s2"}}
    m = compute_h9_metrics(actual, predicted, covered, total_km=10.0)
    assert m["tp"] == 1
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_compute_h9_metrics_false_positive_when_predicted_event_has_no_real_hazard():
    actual = {}   # instance A never actually a hazard
    predicted = {"A": [frozenset({"s0"})]}
    covered = {"A": {"s0"}}
    m = compute_h9_metrics(actual, predicted, covered, total_km=2.0)
    assert m["tp"] == 0
    assert m["fp"] == 1
    assert m["false_alarms_per_km"] == 0.5   # 1 FP / 2 km


def test_compute_h9_metrics_fn_covered_vs_fn_uncovered():
    actual = {
        "A": [frozenset({"s0"})],   # covered by SOME row, but that row never crossed threshold
        "B": [frozenset({"s0"})],   # never covered by any row at all
    }
    predicted = {}   # no predicted events crossed threshold for either
    covered = {"A": {"s0", "s1"}, "B": set()}
    m = compute_h9_metrics(actual, predicted, covered, total_km=10.0)
    assert m["fn_covered"] == 1   # A
    assert m["fn_uncovered"] == 1   # B
    assert m["fn"] == 2
    assert m["recall"] == 0.0


def test_compute_h9_metrics_none_when_denominator_empty():
    m = compute_h9_metrics({}, {}, {}, total_km=10.0)
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["false_alarms_per_km"] == 0.0   # 0 FP / 10 km is a real, defined zero rate


def test_compute_h9_metrics_false_alarms_per_km_none_when_zero_distance():
    predicted = {"A": [frozenset({"s0"})]}
    m = compute_h9_metrics({}, predicted, {"A": {"s0"}}, total_km=0.0)
    assert m["false_alarms_per_km"] is None
