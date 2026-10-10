"""
v2/eval_l3/h6_followup.py — three follow-up checks on the H6 v1
in-corridor + GT-TTC-0-5s finding (fused loses to radar), requested
directly rather than as a new pre-registered hypothesis. No
config_l3.py threshold is read, added, or changed by anything here.

1. How many real scenes the n=36/45 paired keys actually come from,
   and leave-one-scene-out: does removing any single scene flip or
   erase the result?
2. A full, honest multiple-comparisons correction across ALL 12
   (pipeline x bound x sensor) comparisons in this same slice -- not
   just the 4 "best sensor" cells used for the original pass/fail
   call, since radar was the BEST-LOOKING sensor, picked after seeing
   the data; correcting only across the post-hoc-selected cells
   understates how many real chances there were for a false positive.
3. Mechanism check: on GT-parked objects in this slice, is fused's
   larger TTC error explained by a worse closing-speed estimate
   specifically (not just a generic position/TTC error)? Compares
   each of fused's and radar's OWN closing_speed column (v1 already
   computed it, Step 5) against GT's own closing speed (derived here
   the same finite-difference way evaluate.compute_ground_truth_ttc()
   already does internally for ttc, just also exposing the
   closing_speed intermediate that function computes and discards).
"""
import csv
from statistics import mean, median

import config as legacy_config   # repo-root config.py: paths only
from v2.evaluate_matched import candidates_from_v1_csv
from v2.ttc import compute_ttc
from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3.hypothesis_h6 import SENSORS, build_paired_rows, compute_stats, PASS_FAIL_RESTRICT_FN
from v2.eval_l3.hungarian_match import match_samplewise_hungarian
from v2.eval_l3.v1_adapters import match_v1_lock_once
from v2.eval_l3.bootstrap import group_rows_by_scene, scene_bootstrap_p_value, bonferroni_correct


# ---------------------------------------------------------------------
# Part 1: scene count + leave-one-scene-out
# ---------------------------------------------------------------------

def count_scenes(paired_rows):
    return sorted({r.scene_id for r in paired_rows})


def _diff_mean(rows):
    diffs = [r.diff for r in rows]
    return mean(diffs) if diffs else None


def leave_one_scene_out(paired_rows, stat_fn=None):
    """{scene_id: {"n_removed":, "n_remaining":, "mean_diff_remaining":}}
    -- stat_fn defaults to the plain pooled mean diff. Shows whether
    any single scene is solely responsible for the overall result."""
    stat_fn = stat_fn if stat_fn is not None else _diff_mean
    result = {}
    for scene in count_scenes(paired_rows):
        removed = [r for r in paired_rows if r.scene_id == scene]
        remaining = [r for r in paired_rows if r.scene_id != scene]
        val = stat_fn(remaining) if remaining else None
        result[scene] = {
            "n_removed": len(removed), "n_remaining": len(remaining),
            "mean_diff_remaining": round(val, 4) if val is not None else None,
        }
    return result


# ---------------------------------------------------------------------
# Part 2: all 12 comparisons, corrected together
# ---------------------------------------------------------------------

def all_12_comparisons(v1_rows, v2_rows, alpha=0.05):
    """One cell per (pipeline, bound, sensor) in the SAME in-corridor +
    GT-TTC-0-5s slice (PASS_FAIL_RESTRICT_FN), regardless of whether
    that sensor turned out to be the "best" one -- the full
    pre-selection family, so Bonferroni's n=12 reflects every real
    comparison that could have been picked, not just the one(s) that
    looked good."""
    cells = []
    for pipeline, rows in (("v1", v1_rows), ("v2", v2_rows)):
        for bound in ("hungarian", "lock_once"):
            for sensor in SENSORS:
                paired = build_paired_rows(rows, bound, sensor)
                restricted = [r for r in paired if PASS_FAIL_RESTRICT_FN(r)]
                stats = compute_stats(restricted)
                rows_by_scene = group_rows_by_scene(restricted, scene_of_row=lambda r: r.scene_id)
                p = scene_bootstrap_p_value(rows_by_scene, _diff_mean,
                                             n_resamples=cfg.BOOTSTRAP_N_RESAMPLES, seed=cfg.BOOTSTRAP_SEED)
                cells.append({
                    "pipeline": pipeline, "bound": bound, "sensor": sensor,
                    "n": stats["n"], "mean_diff": stats["mean_diff"],
                    "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"],
                    "p_value": p,
                })

    p_values = [c["p_value"] for c in cells]
    bonf = bonferroni_correct(p_values, alpha=alpha)
    for cell, (_p, survives) in zip(cells, bonf):
        cell["bonferroni_threshold"] = alpha / len(p_values)
        cell["bonferroni_survives"] = survives
    return cells


# ---------------------------------------------------------------------
# Part 3: mechanism check -- closing speed, GT-parked objects only
# ---------------------------------------------------------------------

def build_gt_closing_speed_lookup(gt_trajectories, ego_by_sample):
    """{(sample_id, instance_id): closing_speed}, the SAME finite-
    difference GT velocity evaluate.compute_ground_truth_ttc() already
    uses internally -- exposing the closing_speed intermediate that
    function computes and discards (it only keeps "ttc"), not a
    second, independently-derived formula."""
    lookup = {}
    for instance_id, points in gt_trajectories.items():
        points = sorted(points, key=lambda p: p.t)
        for i in range(1, len(points)):
            p0, p1 = points[i - 1], points[i]
            dt = p1.t - p0.t
            if dt <= 0:
                continue
            vx, vy = (p1.x - p0.x) / dt, (p1.y - p0.y) / dt
            ego = ego_by_sample.get(p1.sample_id)
            if ego is None:
                continue
            _distance, closing_speed, _ttc = compute_ttc(p1.x, p1.y, vx, vy, ego.x, ego.y, ego.vx, ego.vy)
            lookup[(p1.sample_id, instance_id)] = closing_speed
    return lookup


def build_v1_closing_speed_lookup(csv_path):
    """{(sample_id, track_id): closing_speed} read directly off one v1
    ttc_<sensor>.csv's own 'closing_speed' column -- v1 already
    computed this (Step 5); not re-derived here."""
    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lookup[(row["sample_id"], row["fused_id"])] = float(row["closing_speed"])
            except (KeyError, ValueError):
                continue
    return lookup


def _v1_key_to_track_id(run_name, bound, gt_by_sample, gt_ttc_lookup, ego_by_sample, match_dist_threshold):
    csv_path = legacy_config.STEP5_DIR / f"ttc_{run_name}.csv"
    if bound == "hungarian":
        candidates_by_sample = candidates_from_v1_csv(csv_path)
        matched = match_samplewise_hungarian(run_name, candidates_by_sample, gt_by_sample, gt_ttc_lookup, match_dist_threshold)
        return {(r.sample_id, r.instance_id): r.track_id for r in matched}
    pairs = match_v1_lock_once(csv_path, gt_by_sample, gt_ttc_lookup, ego_by_sample, match_dist_threshold)
    return {(p["sample_id"], p["instance_id"]): p["track_id"] for p in pairs}


def closing_speed_mechanism_check(parked_keys, bound, gt_closing_speed_lookup,
                                   gt_by_sample, gt_ttc_lookup, ego_by_sample,
                                   match_dist_threshold=cfg.MATCH_DIST_THRESHOLD_M):
    """parked_keys: iterable of (sample_id, instance_id) -- the
    GT-parked keys already identified in the target slice (paired
    fused-vs-radar). For each of {fused, radar}, re-matches v1's own
    CSV (same algorithm/threshold as everywhere else in H6, NOT
    retuned) to recover each key's own track_id, looks up that track's
    OWN closing_speed from its CSV, and compares it against GT's.
    Returns {"fused": stats, "radar": stats} where stats has n, mae,
    median_abs_error, mean_estimate, mean_gt."""
    results = {}
    for run_name in ("fused", "radar"):
        key_to_track = _v1_key_to_track_id(run_name, bound, gt_by_sample, gt_ttc_lookup, ego_by_sample, match_dist_threshold)
        cs_lookup = build_v1_closing_speed_lookup(legacy_config.STEP5_DIR / f"ttc_{run_name}.csv")

        abs_errors, estimates, gts = [], [], []
        for key in parked_keys:
            track_id = key_to_track.get(key)
            if track_id is None:
                continue
            sample_id, _instance_id = key
            estimate = cs_lookup.get((sample_id, track_id))
            gt_cs = gt_closing_speed_lookup.get(key)
            if estimate is None or gt_cs is None:
                continue
            abs_errors.append(abs(estimate - gt_cs))
            estimates.append(estimate)
            gts.append(gt_cs)

        results[run_name] = {
            "n": len(abs_errors),
            "mae_closing_speed": round(mean(abs_errors), 4) if abs_errors else None,
            "median_abs_error": round(median(abs_errors), 4) if abs_errors else None,
            "mean_estimate": round(mean(estimates), 4) if estimates else None,
            "mean_gt": round(mean(gts), 4) if gts else None,
        }
    return results
