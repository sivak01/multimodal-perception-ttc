"""
v2/eval_l3/bootstrap.py — scene-level bootstrap CIs and a paired sign
test, per this session's rule 4: frames inside one scene are not
independent (consecutive samples of the same real object), so
resampling individual rows would understate real uncertainty.
Resampling whole SCENES with replacement treats each scene as one
independent unit instead.
"""
import math
import random
from statistics import NormalDist

from v2.eval_l3 import config_l3 as cfg


def group_rows_by_scene(rows, scene_of_row):
    """rows: any list. scene_of_row: callable(row) -> scene_id.
    Returns {scene_id: [row, ...]}."""
    by_scene = {}
    for r in rows:
        by_scene.setdefault(scene_of_row(r), []).append(r)
    return by_scene


def scene_bootstrap_ci(rows_by_scene, stat_fn,
                        n_resamples=cfg.BOOTSTRAP_N_RESAMPLES, seed=cfg.BOOTSTRAP_SEED,
                        ci_pct=cfg.BOOTSTRAP_CI_PCT):
    """rows_by_scene: {scene_id: [row, ...]}. stat_fn: callable(list of
    rows) -> float or None (None means "undefined for this resample",
    e.g. an empty group after resampling -- skipped, not treated as 0).

    Resamples the scene_ids themselves (not the rows) with replacement,
    n_resamples times, with a fixed seed for reproducibility; each
    resample's pooled row list is EVERY row belonging to each of the
    resampled scene_ids (a scene drawn twice contributes its rows
    twice). Returns (point_estimate, ci_low, ci_high, n_resamples_used)
    -- point_estimate is stat_fn() on the REAL (unresampled) full pool,
    not the mean of the bootstrap distribution (standard practice: the
    bootstrap estimates the CI around the observed point estimate, it
    does not replace it).

    If rows_by_scene has fewer than 2 scenes, or stat_fn is undefined
    on the real full pool, returns (point_estimate_or_None, None, None, 0)
    -- a CI cannot be meaningfully built from fewer than 2 independent
    units.
    """
    scene_ids = list(rows_by_scene.keys())
    all_rows = [r for rows in rows_by_scene.values() for r in rows]
    point_estimate = stat_fn(all_rows) if all_rows else None

    if len(scene_ids) < 2 or point_estimate is None:
        return point_estimate, None, None, 0

    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        resampled_scenes = [rng.choice(scene_ids) for _ in scene_ids]
        pooled = []
        for sid in resampled_scenes:
            pooled.extend(rows_by_scene[sid])
        val = stat_fn(pooled)
        if val is not None and math.isfinite(val):
            samples.append(val)

    if len(samples) < 2:
        return point_estimate, None, None, len(samples)

    samples.sort()
    alpha = (100 - ci_pct) / 2 / 100
    lo_idx = max(0, int(math.floor(alpha * len(samples))))
    hi_idx = min(len(samples) - 1, int(math.ceil((1 - alpha) * len(samples))) - 1)
    return point_estimate, samples[lo_idx], samples[hi_idx], len(samples)


def ci_excludes_zero(ci_low, ci_high):
    """True only if BOTH bounds are on the same side of 0 -- i.e. the
    whole 95% interval excludes 0. None/None (CI not computable) is
    reported as False (cannot claim exclusion without a CI)."""
    if ci_low is None or ci_high is None:
        return False
    return (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)


def paired_sign_test(paired_diffs):
    """paired_diffs: list of (a - b) for each paired unit (e.g. one
    (instance, range_bucket) group's mean single-error minus mean
    merged-error -- positive means merged was better for that group).
    Exact two-sided binomial sign test against the null that + and -
    are equally likely (ties, exact 0.0, are dropped from n per the
    standard sign-test convention -- they carry no directional
    information). Returns dict with n, n_positive, n_negative, n_ties,
    p_value. No scipy dependency (binomial CDF computed directly via
    math.comb, exact, not a normal approximation) -- small, well-
    defined numbers of groups are expected here, not large-n regimes
    where that would matter anyway."""
    positive = sum(1 for d in paired_diffs if d > 0)
    negative = sum(1 for d in paired_diffs if d < 0)
    ties = len(paired_diffs) - positive - negative
    n = positive + negative

    if n == 0:
        return {"n": 0, "n_positive": positive, "n_negative": negative, "n_ties": ties, "p_value": None}

    k = min(positive, negative)   # two-sided: double the smaller tail
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p_value = min(1.0, 2 * tail)
    return {"n": n, "n_positive": positive, "n_negative": negative, "n_ties": ties, "p_value": round(p_value, 6)}


# Exposed for modules that want a quick z-based sanity check without a
# full bootstrap (not used for the headline CIs themselves, which are
# always the scene bootstrap above per rule 4) -- kept for test
# coverage of NormalDist usage only; not part of the H1 report.
def _normal_ci_half_width(std, n, ci_pct=cfg.BOOTSTRAP_CI_PCT):
    if n <= 1:
        return None
    z = NormalDist().inv_cdf(0.5 + ci_pct / 200)
    return z * std / math.sqrt(n)
