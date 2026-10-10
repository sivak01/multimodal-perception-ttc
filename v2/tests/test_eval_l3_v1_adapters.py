from v2.evaluate import GTPoint, EgoState, compute_ground_truth_ttc
from v2.eval_l3.v1_adapters import (
    build_v1_track_histories, match_v1_lock_once, build_v1_composition_lookup,
)


def _samples(n):
    return [f"sample_{i:04d}" for i in range(n)]


def _write_v1_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fused_id", "sample_id", "timestamp", "x", "y", "vx", "vy",
                          "distance", "closing_speed", "ttc", "method"])
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------
# build_v1_track_histories
# ---------------------------------------------------------------------

def test_build_v1_track_histories_groups_by_fused_id(tmp_path):
    samples = _samples(2)
    ego_by_sample = {s: EgoState(t=float(i), x=0.0, y=0.0, vx=0.0, vy=0.0) for i, s in enumerate(samples)}
    csv_path = tmp_path / "ttc_lidar.csv"
    _write_v1_csv(csv_path, [
        ["trk_a", samples[0], 100, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, "direct_2pt"],
        ["trk_a", samples[1], 200, 2.0, 0.0, -1.0, 0.0, 2.0, 1.0, 2.0, "direct_2pt"],
        ["trk_b", samples[0], 100, 5.0, 0.0, -1.0, 0.0, 5.0, 1.0, 5.0, "direct_2pt"],
    ])
    histories = build_v1_track_histories(csv_path, ego_by_sample)
    assert set(histories.keys()) == {"trk_a", "trk_b"}
    assert len(histories["trk_a"]) == 2
    assert len(histories["trk_b"]) == 1
    # sorted by t
    assert histories["trk_a"][0].t <= histories["trk_a"][1].t


def test_build_v1_track_histories_skips_rows_without_ego_state(tmp_path):
    csv_path = tmp_path / "ttc_lidar.csv"
    _write_v1_csv(csv_path, [["trk_a", "sample_9999", 100, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, "direct_2pt"]])
    histories = build_v1_track_histories(csv_path, ego_by_sample={})
    assert histories == {}


# ---------------------------------------------------------------------
# match_v1_lock_once -- reuses v2.evaluate.match_track_to_ground_truth
# unchanged, so this mainly verifies the v1 CSV -> history adaptation.
# ---------------------------------------------------------------------

def test_match_v1_lock_once_matches_a_closing_object(tmp_path):
    samples = _samples(3)
    ego_by_sample = {s: EgoState(t=float(i), x=0.0, y=0.0, vx=0.0, vy=0.0) for i, s in enumerate(samples)}
    gt_by_sample = {
        samples[0]: [("car", 50.0, 0.0)],
        samples[1]: [("car", 40.0, 0.0)],
        samples[2]: [("car", 30.0, 0.0)],
    }
    gt_trajectories = {
        "car": [GTPoint(0.0, 50.0, 0.0, samples[0]), GTPoint(1.0, 40.0, 0.0, samples[1]),
                GTPoint(2.0, 30.0, 0.0, samples[2])],
    }
    gt_ttc_lookup = compute_ground_truth_ttc(gt_trajectories, ego_by_sample)

    csv_path = tmp_path / "ttc_lidar.csv"
    _write_v1_csv(csv_path, [
        ["trk1", samples[1], 100, 40.1, 0.0, -10.0, 0.0, 40.1, 10.0, 4.01, "direct_2pt"],
        ["trk1", samples[2], 200, 30.1, 0.0, -10.0, 0.0, 30.1, 10.0, 3.01, "direct_2pt"],
    ])
    pairs = match_v1_lock_once(csv_path, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert len(pairs) == 2
    assert all(p["instance_id"] == "car" for p in pairs)
    assert all(p["track_id"] == "trk1" for p in pairs)


def test_match_v1_lock_once_no_match_beyond_threshold(tmp_path):
    samples = _samples(1)
    ego_by_sample = {samples[0]: EgoState(t=0.0, x=0.0, y=0.0, vx=0.0, vy=0.0)}
    gt_by_sample = {samples[0]: [("car", 50.0, 0.0)]}
    gt_ttc_lookup = {}   # no defined GT TTC at all -> nothing should match anyway
    csv_path = tmp_path / "ttc_lidar.csv"
    _write_v1_csv(csv_path, [["trk1", samples[0], 100, 500.0, 0.0, 0.0, 0.0, 500.0, 0.0, float("inf"), "direct_2pt"]])
    pairs = match_v1_lock_once(csv_path, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert pairs == []


# ---------------------------------------------------------------------
# build_v1_composition_lookup
# ---------------------------------------------------------------------

def test_build_v1_composition_lookup_counts_sensors(tmp_path, monkeypatch):
    import csv as csv_module
    step4_dir = tmp_path / "step_4"
    step4_dir.mkdir()
    fused_path = step4_dir / "fused_tracks_all.csv"
    with open(fused_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerow(["fused_id", "sample_id", "timestamp", "x", "y", "z", "vx", "vy", "sensors"])
        writer.writerow(["fused_a", "sample_0000", 1, 0, 0, 0, 0, 0, "lidar"])
        writer.writerow(["fused_b", "sample_0000", 1, 0, 0, 0, 0, 0, "lidar+radar"])
        writer.writerow(["fused_c", "sample_0001", 1, 0, 0, 0, 0, 0, "camera_mono+lidar+radar"])

    import v2.eval_l3.v1_adapters as v1_adapters_module
    fake_config = type("FakeConfig", (), {"STEP4_DIR": step4_dir})
    monkeypatch.setattr(v1_adapters_module, "legacy_config", fake_config)

    lookup = build_v1_composition_lookup()
    assert lookup[("sample_0000", "fused_a")] == 1
    assert lookup[("sample_0000", "fused_b")] == 2
    assert lookup[("sample_0001", "fused_c")] == 3
