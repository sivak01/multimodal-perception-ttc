from v2.eval_l3.h6_common import KeyRow
from v2.eval_l3.hypothesis_h6 import (
    build_paired_rows, compute_stats, run_all_slices, verdict_for,
    best_sensor_for_pass_fail, in_ttc_band, in_combined_corridor_and_band,
)


def _row(run, sample_id, instance_id, pred_ttc, gt_ttc=4.0, scene_id="s1",
         range_bucket=(0, 20), class_group="vehicle", motion="moving", odd="day",
         in_corridor=True, n_sensors=None, bound="hungarian"):
    return KeyRow(
        bound=bound, run=run, scene_id=scene_id, sample_id=sample_id, instance_id=instance_id,
        pred_ttc=pred_ttc, gt_ttc=gt_ttc, range_bucket=range_bucket, class_group=class_group,
        motion=motion, odd=odd, in_corridor=in_corridor, n_sensors=n_sensors,
    )


# ---------------------------------------------------------------------
# build_paired_rows
# ---------------------------------------------------------------------

def test_build_paired_rows_pairs_shared_keys_only():
    rows = [
        _row("fused", "s0", "A", pred_ttc=4.1, gt_ttc=4.0),
        _row("lidar", "s0", "A", pred_ttc=5.0, gt_ttc=4.0),
        _row("fused", "s0", "B", pred_ttc=3.0, gt_ttc=4.0),   # no matching lidar row for B
    ]
    paired = build_paired_rows(rows, bound="hungarian", sensor="lidar")
    assert len(paired) == 1
    p = paired[0]
    assert abs(p.fused_err - 0.1) < 1e-9
    assert abs(p.sensor_err - 1.0) < 1e-9
    assert abs(p.diff - (0.1 - 1.0)) < 1e-9


def test_build_paired_rows_excludes_infinite_errors():
    rows = [
        _row("fused", "s0", "A", pred_ttc=float("inf"), gt_ttc=4.0),
        _row("lidar", "s0", "A", pred_ttc=5.0, gt_ttc=4.0),
    ]
    paired = build_paired_rows(rows, bound="hungarian", sensor="lidar")
    assert paired == []


def test_build_paired_rows_respects_bound():
    rows = [
        _row("fused", "s0", "A", pred_ttc=4.1, gt_ttc=4.0, bound="hungarian"),
        _row("lidar", "s0", "A", pred_ttc=5.0, gt_ttc=4.0, bound="lock_once"),
    ]
    paired = build_paired_rows(rows, bound="hungarian", sensor="lidar")
    assert paired == []


# ---------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------

def test_compute_stats_fused_clearly_better():
    rows = [
        _row("fused", f"s{i}", f"obj{i}", pred_ttc=4.1, gt_ttc=4.0, scene_id=f"scene{i % 3}")
        for i in range(10)
    ] + [
        _row("lidar", f"s{i}", f"obj{i}", pred_ttc=6.0, gt_ttc=4.0, scene_id=f"scene{i % 3}")
        for i in range(10)
    ]
    paired = build_paired_rows(rows, bound="hungarian", sensor="lidar")
    stats = compute_stats(paired)
    assert stats["n"] == 10
    assert abs(stats["mean_fused_err"] - 0.1) < 1e-9
    assert abs(stats["mean_sensor_err"] - 2.0) < 1e-9
    assert stats["mean_diff"] < 0
    assert stats["fused_wins_pct"] == 100.0


def test_compute_stats_empty_returns_none_fields():
    stats = compute_stats([])
    assert stats["n"] == 0
    assert stats["mean_diff"] is None
    assert stats["ci_excludes_zero"] is False


# ---------------------------------------------------------------------
# slice filters
# ---------------------------------------------------------------------

def test_in_ttc_band():
    f = in_ttc_band(0, 3)
    make = lambda gt: _row("fused", "s", "a", pred_ttc=1.0, gt_ttc=gt)
    paired = build_paired_rows([make(2.0), _row("lidar", "s", "a", pred_ttc=1.0, gt_ttc=2.0)], "hungarian", "lidar")
    assert f(paired[0]) is True


def test_in_combined_corridor_and_band():
    f = in_combined_corridor_and_band(0, 5)
    rows = [
        _row("fused", "s0", "A", pred_ttc=1.0, gt_ttc=4.0, in_corridor=True),
        _row("lidar", "s0", "A", pred_ttc=1.0, gt_ttc=4.0, in_corridor=True),
    ]
    paired = build_paired_rows(rows, "hungarian", "lidar")
    assert f(paired[0]) is True

    rows_out = [
        _row("fused", "s0", "B", pred_ttc=1.0, gt_ttc=4.0, in_corridor=False),
        _row("lidar", "s0", "B", pred_ttc=1.0, gt_ttc=4.0, in_corridor=False),
    ]
    paired_out = build_paired_rows(rows_out, "hungarian", "lidar")
    assert f(paired_out[0]) is False


# ---------------------------------------------------------------------
# run_all_slices / verdict_for
# ---------------------------------------------------------------------

def test_run_all_slices_includes_pass_fail_slice():
    rows = [
        _row("fused", "s0", "A", pred_ttc=1.0, gt_ttc=2.0, in_corridor=True, n_sensors=2),
        _row("lidar", "s0", "A", pred_ttc=3.0, gt_ttc=2.0, in_corridor=True),
    ]
    results = run_all_slices(rows, bound="hungarian", sensor="lidar")
    assert "in_corridor_ttc_0_5s" in results
    assert results["overall"]["n"] == 1
    assert results["k2"]["n"] == 1
    assert results["k3"]["n"] == 0


def test_verdict_confirmed():
    stats = {"n": 50, "mean_diff": -1.0, "ci_excludes_zero": True, "ci95_low": -1.5, "ci95_high": -0.5,
              "mean_fused_err": 1.0, "mean_sensor_err": 2.0}
    verdict, reason = verdict_for(stats)
    assert verdict == "CONFIRMED"


def test_verdict_refuted_favours_sensor():
    stats = {"n": 50, "mean_diff": 1.0, "ci_excludes_zero": True, "ci95_low": 0.5, "ci95_high": 1.5,
              "mean_fused_err": 2.0, "mean_sensor_err": 1.0}
    verdict, reason = verdict_for(stats)
    assert verdict == "REFUTED"
    assert "WORSE" in reason


def test_verdict_refuted_ci_includes_zero():
    stats = {"n": 50, "mean_diff": -0.2, "ci_excludes_zero": False, "ci95_low": -0.5, "ci95_high": 0.3,
              "mean_fused_err": 1.0, "mean_sensor_err": 1.2}
    verdict, reason = verdict_for(stats)
    assert verdict == "REFUTED"


def test_verdict_inconclusive_small_n():
    stats = {"n": 5, "mean_diff": -1.0, "ci_excludes_zero": True, "ci95_low": -1.5, "ci95_high": -0.5,
              "mean_fused_err": 1.0, "mean_sensor_err": 2.0}
    verdict, reason = verdict_for(stats)
    assert verdict == "INCONCLUSIVE"


# ---------------------------------------------------------------------
# best_sensor_for_pass_fail
# ---------------------------------------------------------------------

def test_best_sensor_for_pass_fail_picks_lowest_mean_error():
    rows = [
        _row("fused", "s0", "A", pred_ttc=1.0, gt_ttc=2.0, in_corridor=True),
        _row("lidar", "s0", "A", pred_ttc=2.5, gt_ttc=2.0, in_corridor=True),   # lidar err=0.5
        _row("fused", "s0", "B", pred_ttc=1.0, gt_ttc=2.0, in_corridor=True),
        _row("radar", "s0", "B", pred_ttc=5.0, gt_ttc=2.0, in_corridor=True),  # radar err=3.0
    ]
    best, per_sensor = best_sensor_for_pass_fail(rows, bound="hungarian")
    assert best == "lidar"
    assert per_sensor["lidar"]["mean_sensor_err"] == 0.5
    assert per_sensor["radar"]["mean_sensor_err"] == 3.0
    assert per_sensor["camera_mono"]["n"] == 0
