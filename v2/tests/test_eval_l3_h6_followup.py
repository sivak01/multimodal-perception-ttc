from v2.evaluate import GTPoint, EgoState
from v2.eval_l3.h6_common import KeyRow
from v2.eval_l3.hypothesis_h6 import build_paired_rows
from v2.eval_l3.h6_followup import (
    count_scenes, leave_one_scene_out, all_12_comparisons,
    build_gt_closing_speed_lookup,
)


def _row(run, scene_id, sample_id, instance_id, pred_ttc, gt_ttc=4.0, n_sensors=None, bound="hungarian"):
    return KeyRow(
        bound=bound, run=run, scene_id=scene_id, sample_id=sample_id, instance_id=instance_id,
        pred_ttc=pred_ttc, gt_ttc=gt_ttc, range_bucket=(0, 20), class_group="vehicle",
        motion="parked", odd="day", in_corridor=True, n_sensors=n_sensors,
    )


# ---------------------------------------------------------------------
# count_scenes / leave_one_scene_out
# ---------------------------------------------------------------------

def _paired(rows):
    return build_paired_rows(rows, bound="hungarian", sensor="radar")


def test_count_scenes_unique_and_sorted():
    rows = [
        _row("fused", "scene-b", "s0", "A", 4.1), _row("radar", "scene-b", "s0", "A", 5.0),
        _row("fused", "scene-a", "s1", "B", 4.1), _row("radar", "scene-a", "s1", "B", 5.0),
        _row("fused", "scene-a", "s2", "C", 4.1), _row("radar", "scene-a", "s2", "C", 5.0),
    ]
    scenes = count_scenes(_paired(rows))
    assert scenes == ["scene-a", "scene-b"]


def test_leave_one_scene_out_removes_only_that_scene():
    rows = [
        _row("fused", "scene-a", "s0", "A", 4.1), _row("radar", "scene-a", "s0", "A", 5.0),
        _row("fused", "scene-b", "s1", "B", 4.2), _row("radar", "scene-b", "s1", "B", 6.0),
    ]
    paired = _paired(rows)
    result = leave_one_scene_out(paired)
    assert result["scene-a"]["n_removed"] == 1
    assert result["scene-a"]["n_remaining"] == 1
    assert result["scene-b"]["n_removed"] == 1
    assert result["scene-b"]["n_remaining"] == 1


def test_leave_one_scene_out_detects_single_scene_driving_the_effect():
    # 9 keys from scene-a with diff=0 (neutral), 1 key from scene-b with a huge diff.
    # Removing scene-b should flip the overall mean_diff toward 0.
    rows = []
    for i in range(9):
        rows.append(_row("fused", "scene-a", f"s{i}", f"obj{i}", 4.0))
        rows.append(_row("radar", "scene-a", f"s{i}", f"obj{i}", 4.0))
    rows.append(_row("fused", "scene-b", "s100", "obj100", 10.0))   # fused err = 6
    rows.append(_row("radar", "scene-b", "s100", "obj100", 4.1))    # radar err = 0.1 -> diff ~5.9
    paired = _paired(rows)
    result = leave_one_scene_out(paired)
    assert abs(result["scene-b"]["mean_diff_remaining"] - 0.0) < 1e-9   # removing the one outlier scene -> flat 0 diff
    assert result["scene-a"]["mean_diff_remaining"] > 5.0               # removing scene-a leaves only the big outlier


# ---------------------------------------------------------------------
# all_12_comparisons
# ---------------------------------------------------------------------

def test_all_12_comparisons_shape_and_bonferroni_threshold():
    rows = []
    for scene in ("scene-a", "scene-b", "scene-c"):
        for i in range(5):
            sid, iid = f"{scene}_{i}", f"obj_{scene}_{i}"
            rows.append(_row("fused", scene, sid, iid, 1.0, gt_ttc=2.0))
            rows.append(_row("lidar", scene, sid, iid, 1.1, gt_ttc=2.0))
            rows.append(_row("radar", scene, sid, iid, 2.5, gt_ttc=2.0))
            rows.append(_row("camera_mono", scene, sid, iid, 3.0, gt_ttc=2.0))
    cells = all_12_comparisons(rows, rows, alpha=0.05)
    assert len(cells) == 12
    pipelines = {c["pipeline"] for c in cells}
    bounds = {c["bound"] for c in cells}
    sensors = {c["sensor"] for c in cells}
    assert pipelines == {"v1", "v2"}
    assert bounds == {"hungarian", "lock_once"}
    assert sensors == {"lidar", "radar", "camera_mono"}
    assert all(abs(c["bonferroni_threshold"] - 0.05 / 12) < 1e-12 for c in cells)


# ---------------------------------------------------------------------
# build_gt_closing_speed_lookup
# ---------------------------------------------------------------------

def test_build_gt_closing_speed_lookup_matches_hand_computation():
    samples = ["sample_0000", "sample_0001"]
    gt_points = [
        GTPoint(t=0.0, x=50.0, y=0.0, sample_id=samples[0]),
        GTPoint(t=1.0, x=40.0, y=0.0, sample_id=samples[1]),
    ]
    ego_by_sample = {s: EgoState(t=float(i), x=0.0, y=0.0, vx=0.0, vy=0.0) for i, s in enumerate(samples)}
    lookup = build_gt_closing_speed_lookup({"car": gt_points}, ego_by_sample)
    assert (samples[0], "car") not in lookup   # first point has no preceding point
    key = (samples[1], "car")
    assert key in lookup
    assert abs(lookup[key] - 10.0) < 1e-9   # object closes 10m in 1s -> closing speed 10 m/s


def test_build_gt_closing_speed_lookup_skips_missing_ego():
    gt_points = [
        GTPoint(t=0.0, x=50.0, y=0.0, sample_id="sample_0000"),
        GTPoint(t=1.0, x=40.0, y=0.0, sample_id="sample_0001"),
    ]
    lookup = build_gt_closing_speed_lookup({"car": gt_points}, ego_by_sample={})
    assert lookup == {}
