"""
v2/eval_l3/hypothesis_h1.py — H1 (refined): within the fused run, does a
GT key whose fused output was corroborated by 2+ sensors ("merged") beat
a key covered by only one ("single") -- compared FAIRLY, i.e. only
across groups of keys that share the same GT instance AND the same
range bucket (so a merged key is never compared against an unrelated,
possibly easier/harder single-sensor key)?

Pass/fail rule, given verbatim by this session's own instructions:
  CONFIRMED (fusion helps): merged beats single within the same
    instance+range-bucket grouping, AND the 95% CI of the (merged -
    single) MAE difference excludes 0 (and is negative, i.e. merged IS
    better -- see _verdict() below for why both conditions are checked
    explicitly rather than only "CI excludes 0").
  REFUTED: the CI includes 0, or favours single (diff > 0, CI excludes
    0 on the positive side).
  INCONCLUSIVE: fewer than config_l3.MIN_N_FOR_RELIABLE_CI qualifying
    (instance, range_bucket) groups exist -- stated as its own
    category per this session's rule 4 ("say clearly when n < 30"),
    rather than silently forcing every result into CONFIRMED/REFUTED.
"""
import math
from collections import defaultdict
from statistics import mean, median

from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3.bootstrap import group_rows_by_scene, scene_bootstrap_ci, ci_excludes_zero, paired_sign_test


def _percentile(values, pct):
    """Simple nearest-rank percentile on a pre-filtered finite list --
    documented as approximate-but-deterministic, not interpolated; fine
    for the P90 "how bad does it get" reporting this is used for."""
    if not values:
        return None
    s = sorted(values)
    idx = round(pct / 100 * (len(s) - 1))
    return s[idx]


def _finite_abs_errors(rows):
    return [r.abs_error for r in rows if r.abs_error is not None and math.isfinite(r.abs_error)]


def _finite_rel_errors(rows):
    return [r.rel_error for r in rows if r.rel_error is not None and math.isfinite(r.rel_error)]


def mae_stat(rows):
    errs = _finite_abs_errors(rows)
    return mean(errs) if errs else None


def median_stat(rows):
    errs = _finite_abs_errors(rows)
    return median(errs) if errs else None


def p90_stat(rows):
    errs = _finite_abs_errors(rows)
    return _percentile(errs, 90) if errs else None


def rel_mae_stat(rows):
    errs = _finite_rel_errors(rows)
    return mean(errs) if errs else None


METRIC_FUNCS = {"mae": mae_stat, "median": median_stat, "p90": p90_stat, "rel_mae": rel_mae_stat}


def build_qualifying_groups(rows, bound, restrict_fn=None):
    """rows: full EnrichedKeyRow list (both bounds, all rows). Filters
    to the given bound (+ restrict_fn if given), keeps only rows with a
    defined abs_error, groups by (instance_id, range_bucket), and keeps
    only groups containing BOTH a merged and a single row. Returns
    (merged_rows, single_rows, groups) where groups is
    {(instance_id, range_bucket): {"merged": [...], "single": [...]}}
    restricted to qualifying groups only."""
    candidates = [r for r in rows if r.bound == bound and r.abs_error is not None]
    if restrict_fn is not None:
        candidates = [r for r in candidates if restrict_fn(r)]

    by_group = defaultdict(lambda: {"merged": [], "single": []})
    for r in candidates:
        by_group[(r.instance_id, r.range_bucket)][r.composition].append(r)

    groups = {k: v for k, v in by_group.items() if v["merged"] and v["single"]}
    merged_rows = [r for g in groups.values() for r in g["merged"]]
    single_rows = [r for g in groups.values() for r in g["single"]]
    return merged_rows, single_rows, groups


def _diff_stat_fn(metric_fn):
    """Builds a stat_fn(pooled_rows) -> merged_metric - single_metric,
    for scene_bootstrap_ci -- a resampled pool still contains both
    compositions tagged per-row, so this recomputes the split fresh on
    every resample rather than resampling two separate pools
    independently (which would lose the paired, same-scene structure)."""
    def _stat(rows):
        merged = [r for r in rows if r.composition == "merged"]
        single = [r for r in rows if r.composition == "single"]
        m, s = metric_fn(merged), metric_fn(single)
        if m is None or s is None:
            return None
        return m - s
    return _stat


def compare_merged_vs_single(rows, bound, restrict_fn=None):
    """The full H1 comparison for one bound and one optional
    restriction (None = part a; in-corridor-only / ttc-band-only =
    part b). Returns a dict with per-metric point estimates + CIs,
    n's, and the paired sign test."""
    merged_rows, single_rows, groups = build_qualifying_groups(rows, bound, restrict_fn)
    n_groups = len(groups)

    pooled = merged_rows + single_rows
    rows_by_scene = group_rows_by_scene(pooled, scene_of_row=lambda r: r.scene_id)

    metrics = {}
    for name, fn in METRIC_FUNCS.items():
        point_merged = fn(merged_rows)
        point_single = fn(single_rows)
        diff_point, ci_lo, ci_hi, n_boot = scene_bootstrap_ci(rows_by_scene, _diff_stat_fn(fn))
        metrics[name] = {
            "merged": round(point_merged, 4) if point_merged is not None else None,
            "single": round(point_single, 4) if point_single is not None else None,
            "diff_merged_minus_single": round(diff_point, 4) if diff_point is not None else None,
            "ci95_low": round(ci_lo, 4) if ci_lo is not None else None,
            "ci95_high": round(ci_hi, 4) if ci_hi is not None else None,
            "ci_excludes_zero": ci_excludes_zero(ci_lo, ci_hi),
            "n_bootstrap_resamples_used": n_boot,
        }

    # Paired sign test: one paired diff per qualifying group, using
    # each group's own mean abs error (single - merged; positive =
    # merged better for that group).
    paired_diffs = []
    for g in groups.values():
        m_errs, s_errs = _finite_abs_errors(g["merged"]), _finite_abs_errors(g["single"])
        if m_errs and s_errs:
            paired_diffs.append(mean(s_errs) - mean(m_errs))
    sign_test = paired_sign_test(paired_diffs)

    return {
        "bound": bound,
        "n_qualifying_groups": n_groups,
        "n_merged_rows": len(merged_rows),
        "n_single_rows": len(single_rows),
        "metrics": metrics,
        "paired_sign_test": sign_test,
    }


def verdict_for(comparison, min_n=cfg.MIN_N_FOR_RELIABLE_CI):
    """Applies this session's own pass/fail rule to the MAE metric
    (the rule's own stated metric) of one compare_merged_vs_single()
    result. Returns "CONFIRMED" / "REFUTED" / "INCONCLUSIVE" plus the
    one-line reason."""
    if comparison["n_qualifying_groups"] < min_n:
        return "INCONCLUSIVE", (
            f"only {comparison['n_qualifying_groups']} qualifying (instance, range_bucket) groups "
            f"(< {min_n}) -- too few independent units for a reliable CI"
        )

    mae = comparison["metrics"]["mae"]
    if mae["diff_merged_minus_single"] is None:
        return "INCONCLUSIVE", "MAE undefined for merged or single in this slice"

    if mae["ci_excludes_zero"] and mae["diff_merged_minus_single"] < 0:
        return "CONFIRMED", (
            f"merged MAE ({mae['merged']}) beats single MAE ({mae['single']}); "
            f"95% CI of the difference [{mae['ci95_low']}, {mae['ci95_high']}] excludes 0"
        )

    if mae["ci_excludes_zero"] and mae["diff_merged_minus_single"] > 0:
        return "REFUTED", (
            f"merged MAE ({mae['merged']}) is WORSE than single MAE ({mae['single']}); "
            f"95% CI of the difference [{mae['ci95_low']}, {mae['ci95_high']}] excludes 0, favouring single"
        )

    return "REFUTED", (
        f"95% CI of the MAE difference [{mae['ci95_low']}, {mae['ci95_high']}] includes 0 "
        "(per this session's own rule: CI includes 0 => REFUTED, not INCONCLUSIVE)"
    )
