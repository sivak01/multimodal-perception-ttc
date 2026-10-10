"""
v2/eval_l3/hypothesis_h6.py — H6: does fused beat each single sensor,
on the SAME GT keys, paired per key? Run ONCE PER PIPELINE (v1, v2),
never mixing the two (rule 2).

Pass/fail rule, given verbatim: CONFIRMED only if fused beats the BEST
single sensor in the in-corridor, GT-TTC-0-5s band, with a scene-
bootstrap 95% CI (of the paired diff |fused_err|-|best_sensor_err|)
that excludes 0. Otherwise REFUTED (CI includes 0) or INCONCLUSIVE
(n < config_l3.MIN_N_FOR_RELIABLE_CI).
"""
import math
from collections import namedtuple
from statistics import mean, median

from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3.h6_common import abs_error
from v2.eval_l3.bootstrap import group_rows_by_scene, scene_bootstrap_ci, ci_excludes_zero, paired_sign_test

SENSORS = ("lidar", "radar", "camera_mono")

PairedRow = namedtuple("PairedRow", [
    "scene_id", "sample_id", "instance_id", "gt_ttc",
    "fused_err", "sensor_err", "diff",
    "range_bucket", "class_group", "motion", "odd", "in_corridor", "n_sensors",
])


def build_paired_rows(key_rows, bound, sensor, fused_run="fused"):
    """Pairs fused's and sensor's own matched rows on the SAME
    (sample_id, instance_id) key, for the given bound, keeping only
    keys where BOTH give a finite abs_error. diff = |fused_err| -
    |sensor_err|; negative means fused was more accurate at that key."""
    fused_by_key = {(r.sample_id, r.instance_id): r for r in key_rows if r.bound == bound and r.run == fused_run}
    sensor_by_key = {(r.sample_id, r.instance_id): r for r in key_rows if r.bound == bound and r.run == sensor}

    paired = []
    for key, frow in fused_by_key.items():
        srow = sensor_by_key.get(key)
        if srow is None:
            continue
        fe, se = abs_error(frow), abs_error(srow)
        if fe is None or se is None:
            continue
        paired.append(PairedRow(
            scene_id=frow.scene_id, sample_id=frow.sample_id, instance_id=frow.instance_id,
            gt_ttc=frow.gt_ttc, fused_err=fe, sensor_err=se, diff=fe - se,
            range_bucket=frow.range_bucket, class_group=frow.class_group, motion=frow.motion,
            odd=frow.odd, in_corridor=frow.in_corridor, n_sensors=frow.n_sensors,
        ))
    return paired


def _diff_mean(rows):
    diffs = [r.diff for r in rows]
    return mean(diffs) if diffs else None


def compute_stats(paired_rows):
    n = len(paired_rows)
    if n == 0:
        return {
            "n": 0, "mean_diff": None, "median_diff": None, "fused_wins_pct": None,
            "ci95_low": None, "ci95_high": None, "ci_excludes_zero": False,
            "n_bootstrap_resamples_used": 0, "mean_fused_err": None, "mean_sensor_err": None,
            "paired_sign_test": paired_sign_test([]),
        }

    diffs = [r.diff for r in paired_rows]
    mean_diff, med_diff = mean(diffs), median(diffs)
    fused_wins_pct = round(sum(1 for d in diffs if d < 0) / n * 100, 2)

    rows_by_scene = group_rows_by_scene(paired_rows, scene_of_row=lambda r: r.scene_id)
    point, ci_lo, ci_hi, n_boot = scene_bootstrap_ci(rows_by_scene, _diff_mean)

    # Sign test convention: positive = fused better for that key (so
    # negate diff, where negative diff already means fused better).
    sign_test = paired_sign_test([-d for d in diffs])

    return {
        "n": n,
        "mean_diff": round(mean_diff, 4), "median_diff": round(med_diff, 4),
        "fused_wins_pct": fused_wins_pct,
        "ci95_low": round(ci_lo, 4) if ci_lo is not None else None,
        "ci95_high": round(ci_hi, 4) if ci_hi is not None else None,
        "ci_excludes_zero": ci_excludes_zero(ci_lo, ci_hi),
        "n_bootstrap_resamples_used": n_boot,
        "mean_fused_err": round(mean(r.fused_err for r in paired_rows), 4),
        "mean_sensor_err": round(mean(r.sensor_err for r in paired_rows), 4),
        "paired_sign_test": sign_test,
    }


def in_ttc_band(lo, hi):
    return lambda r: math.isfinite(r.gt_ttc) and lo <= r.gt_ttc < hi


def in_combined_corridor_and_band(lo, hi):
    return lambda r: r.in_corridor is True and math.isfinite(r.gt_ttc) and lo <= r.gt_ttc < hi


SLICES = [
    ("overall", None),
    ("k2", lambda r: r.n_sensors == 2),
    ("k3", lambda r: r.n_sensors == 3),
    ("in_corridor", lambda r: r.in_corridor is True),
    ("ttc_0_3s", in_ttc_band(0, 3)),
    ("ttc_3_5s", in_ttc_band(3, 5)),
    ("vru", lambda r: r.class_group == "vru"),
    ("vehicle", lambda r: r.class_group == "vehicle"),
    ("day", lambda r: r.odd == "day"),
    ("night", lambda r: r.odd == "night"),
]

PASS_FAIL_SLICE_NAME = "in_corridor_ttc_0_5s"
PASS_FAIL_RESTRICT_FN = in_combined_corridor_and_band(0, 5)


def run_all_slices(key_rows, bound, sensor):
    all_paired = build_paired_rows(key_rows, bound, sensor)
    results = {}
    for slice_name, restrict_fn in SLICES:
        rows = all_paired if restrict_fn is None else [r for r in all_paired if restrict_fn(r)]
        results[slice_name] = compute_stats(rows)
    pass_fail_rows = [r for r in all_paired if PASS_FAIL_RESTRICT_FN(r)]
    results[PASS_FAIL_SLICE_NAME] = compute_stats(pass_fail_rows)
    return results


def best_sensor_for_pass_fail(key_rows, bound):
    """Which of SENSORS has the lowest mean OWN error (not fused's),
    restricted to in-corridor + GT-TTC-0-5s, on THAT sensor's own
    paired-with-fused population (each sensor's shared-key set with
    fused can differ in size/composition, so "best" is judged on each
    sensor's own fair comparison population, not an unconditioned
    standalone number -- consistent with rule 2's same-GT-keys
    fairness requirement). Returns (best_sensor_name, per_sensor_stats)."""
    per_sensor = {}
    for sensor in SENSORS:
        paired = build_paired_rows(key_rows, bound, sensor)
        restricted = [r for r in paired if PASS_FAIL_RESTRICT_FN(r)]
        stats = compute_stats(restricted)
        per_sensor[sensor] = stats
    candidates = {s: st["mean_sensor_err"] for s, st in per_sensor.items() if st["mean_sensor_err"] is not None}
    best = min(candidates, key=candidates.get) if candidates else None
    return best, per_sensor


def verdict_for(stats, min_n=cfg.MIN_N_FOR_RELIABLE_CI):
    if stats["n"] < min_n:
        return "INCONCLUSIVE", f"only n={stats['n']} paired keys (< {min_n}) -- too few for a reliable CI"
    if stats["mean_diff"] is None:
        return "INCONCLUSIVE", "no valid paired keys in this slice"
    if stats["ci_excludes_zero"] and stats["mean_diff"] < 0:
        return "CONFIRMED", (
            f"fused mean err ({stats['mean_fused_err']}) beats sensor mean err ({stats['mean_sensor_err']}); "
            f"95% CI of the diff [{stats['ci95_low']}, {stats['ci95_high']}] excludes 0"
        )
    if stats["ci_excludes_zero"] and stats["mean_diff"] > 0:
        return "REFUTED", (
            f"fused mean err ({stats['mean_fused_err']}) is WORSE than sensor mean err ({stats['mean_sensor_err']}); "
            f"95% CI [{stats['ci95_low']}, {stats['ci95_high']}] excludes 0, favouring the sensor"
        )
    return "REFUTED", f"95% CI [{stats['ci95_low']}, {stats['ci95_high']}] includes 0"
