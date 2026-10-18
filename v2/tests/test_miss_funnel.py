from collections import namedtuple

from v2.miss_funnel import (
    build_funnel_rows, summarize_funnel, expected_for_sensor,
    _covered_by, raw_event_candidates_by_sample, write_funnel_csv_and_summarize,
    LIDAR_MIN_PTS, RADAR_MIN_PTS, CAMERA_MIN_VISIBILITY_TOKEN,
)

_Obj = namedtuple("Obj", ["x", "y"])
_Event = namedtuple("Event", ["sample_id", "x", "y"])


def test_expected_for_sensor_thresholds():
    assert expected_for_sensor("lidar", {"num_lidar_pts": LIDAR_MIN_PTS}) is True
    assert expected_for_sensor("lidar", {"num_lidar_pts": LIDAR_MIN_PTS - 1}) is False
    assert expected_for_sensor("radar", {"num_radar_pts": RADAR_MIN_PTS}) is True
    assert expected_for_sensor("radar", {"num_radar_pts": 0}) is False
    assert expected_for_sensor("camera_mono", {"visibility_token": CAMERA_MIN_VISIBILITY_TOKEN}) is True
    assert expected_for_sensor("camera_mono", {"visibility_token": "1"}) is False
    assert expected_for_sensor("camera_mono", {"visibility_token": "4"}) is True


def test_covered_by_within_threshold():
    objs_by_sample = {"s0": [_Obj(10.1, 20.0)]}
    assert _covered_by(objs_by_sample, "s0", 10.0, 20.0, match_dist_threshold=1.0) is True
    assert _covered_by(objs_by_sample, "s0", 50.0, 50.0, match_dist_threshold=1.0) is False
    assert _covered_by(objs_by_sample, "missing", 10.0, 20.0, match_dist_threshold=1.0) is False


def test_raw_event_candidates_by_sample_groups_by_sample_id():
    events = [_Event("s0", 1.0, 2.0), _Event("s0", 3.0, 4.0), _Event("s1", 5.0, 6.0)]
    out = raw_event_candidates_by_sample(events)
    assert len(out["s0"]) == 2
    assert len(out["s1"]) == 1
    assert out["s0"][0].x == 1.0


def test_build_funnel_rows_each_stage_independent():
    EgoState = namedtuple("EgoState", ["t", "x", "y", "vx", "vy"])
    gt_by_sample = {"s0": [("A", 10.0, 20.0)]}
    ego_by_sample = {"s0": EgoState(0.0, 0.0, 0.0, 0.0, 0.0)}
    ann_lookup = {
        ("s0", "A"): {"num_lidar_pts": LIDAR_MIN_PTS, "num_radar_pts": 0, "visibility_token": "1"},
    }
    # lidar: detected and tracked, but NOT fused (fused candidates empty)
    raw_candidates = {
        "lidar": {"s0": [_Obj(10.1, 20.0)]},
        "radar": {"s0": []},
        "camera_mono": {"s0": []},
    }
    tracked_candidates = {
        "lidar": {"s0": [_Obj(10.1, 20.0)]},
        "radar": {"s0": []},
        "camera_mono": {"s0": []},
    }
    fused_candidates = {"s0": []}   # nothing made it to fused

    rows = build_funnel_rows(gt_by_sample, ann_lookup, ego_by_sample,
                              raw_candidates, tracked_candidates, fused_candidates)
    by_sensor = {r["sensor"]: r for r in rows}

    assert by_sensor["lidar"]["expected"] is True
    assert by_sensor["lidar"]["detected"] is True
    assert by_sensor["lidar"]["tracked"] is True
    assert by_sensor["lidar"]["fused"] is False   # dropped at the fusion stage

    assert by_sensor["radar"]["expected"] is False   # num_radar_pts == 0
    assert by_sensor["camera_mono"]["expected"] is False   # visibility_token == "1"


def test_summarize_funnel_pct_and_missed_range():
    rows = [
        {"sensor": "lidar", "expected": True, "detected": True, "tracked": True, "fused": True, "range_m": 10.0},
        {"sensor": "lidar", "expected": True, "detected": False, "tracked": False, "fused": False, "range_m": 55.0},
        {"sensor": "lidar", "expected": False, "detected": True, "tracked": False, "fused": False, "range_m": 5.0},
    ]
    s = summarize_funnel(rows, "lidar")
    assert s["n_total_gt_keys"] == 3
    assert s["n_expected"] == 2
    assert s["pct_detected_of_expected"] == 50.0
    assert s["pct_tracked_of_expected"] == 50.0
    assert s["pct_fused_of_expected"] == 50.0
    assert s["median_range_m_missed_at_detection"] == 55.0


def test_summarize_funnel_no_expected_returns_none_pct():
    rows = [{"sensor": "radar", "expected": False, "detected": False, "tracked": False, "fused": False, "range_m": 1.0}]
    s = summarize_funnel(rows, "radar")
    assert s["n_expected"] == 0
    assert s["pct_detected_of_expected"] is None
    assert s["median_range_m_missed_at_detection"] is None


def test_write_funnel_csv_and_summarize_matches_list_based_summarize(tmp_path):
    """The streaming single-pass summarizer (used by the real,
    memory-conscious run) must produce the exact same numbers as the
    list-based summarize_funnel() on an equivalent in-memory row set --
    it's a different code path purely for memory, not a different
    definition."""
    rows = [
        {"sample_id": "s0", "instance_id": "A", "sensor": "lidar", "expected": True, "detected": True, "tracked": True, "fused": True, "range_m": 10.0},
        {"sample_id": "s1", "instance_id": "A", "sensor": "lidar", "expected": True, "detected": False, "tracked": False, "fused": False, "range_m": 55.0},
        {"sample_id": "s0", "instance_id": "B", "sensor": "lidar", "expected": False, "detected": True, "tracked": False, "fused": False, "range_m": 5.0},
        {"sample_id": "s0", "instance_id": "A", "sensor": "radar", "expected": False, "detected": False, "tracked": False, "fused": False, "range_m": 10.0},
    ]
    list_based = {
        "lidar": summarize_funnel(rows, "lidar"),
        "radar": summarize_funnel(rows, "radar"),
        "camera_mono": summarize_funnel(rows, "camera_mono"),
    }
    streamed = write_funnel_csv_and_summarize(iter(rows), tmp_path / "out.csv")

    for sensor in ("lidar", "radar", "camera_mono"):
        assert streamed[sensor] == list_based[sensor]

    assert (tmp_path / "out.csv").exists()
    with open(tmp_path / "out.csv") as f:
        assert len(f.readlines()) == len(rows) + 1   # header + rows
