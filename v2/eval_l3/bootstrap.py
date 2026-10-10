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


def _bootstrap_resamples(rows_by_scene, stat_fn, n_resamples, seed):
    """Shared resampling core for scene_bootstrap_ci() and
    scene_bootstrap_p_value() -- same scene_ids, same seed, same
    n_resamples, so the CI and the p-value for one comparison are
    always drawn from the IDENTICAL bootstrap distribution rather than
    two independently-reseeded runs. Returns the raw list of finite
    stat_fn() values across resamples (non-finite/None dropped, same
    convention as scene_bootstrap_ci)."""
    scene_ids = list(rows_by_scene.keys())
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
    return samples


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

    samples = _bootstrap_resamples(rows_by_scene, stat_fn, n_resamples, seed)

    if len(samples) < 2:
        return point_estimate, None, None, len(samples)

    samples = sorted(samples)
    alpha = (100 - ci_pct) / 2 / 100
    lo_idx = max(0, int(math.floor(alpha * len(samples))))
    hi_idx = min(len(samples) - 1, int(math.ceil((1 - alpha) * len(samples))) - 1)
    return point_estimate, samples[lo_idx], samples[hi_idx], len(samples)


def scene_bootstrap_p_value(rows_by_scene, stat_fn,
                             n_resamples=cfg.BOOTSTRAP_N_RESAMPLES, seed=cfg.BOOTSTRAP_SEED):
    """Two-sided bootstrap p-value for testing stat_fn's pooled value
    against 0 (e.g. stat_fn = merged_MAE - single_MAE), drawn from the
    SAME resample distribution scene_bootstrap_ci() would use (same
    seed) -- so this p-value and that CI are two views of one
    bootstrap run, never independently reseeded, and "CI excludes 0"
    and "p < alpha" agree by construction (both ask what fraction of
    the bootstrap distribution crosses 0).

    p = 2 * min(fraction of resamples <= 0, fraction of resamples >= 0),
    capped at 1.0 -- the standard percentile-bootstrap two-sided
    p-value. Returns None if there are fewer than 2 scenes or fewer
    than 2 usable resamples (same guard as scene_bootstrap_ci)."""
    if len(rows_by_scene) < 2:
        return None
    samples = _bootstrap_resamples(rows_by_scene, stat_fn, n_resamples, seed)
    if len(samples) < 2:
        return None
    n = len(samples)
    frac_le = sum(1 for s in samples if s <= 0) / n
    frac_ge = sum(1 for s in samples if s >= 0) / n
    return min(1.0, 2 * min(frac_le, frac_ge))


def ci_excludes_zero(ci_low, ci_high):
    """True only if BOTH bounds are on the same side of 0 -- i.e. the
    whole 95% interval excludes 0. None/None (CI not computable) is
    reported as False (cannot claim exclusion without a CI)."""
    if ci_low is None or ci_high is None:
        return False
    return (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)


def _log_binom_pmf_half(n, k):
    """log(C(n, k) * 0.5**n), via lgamma -- stays in float range for
    any n (unlike forming the exact integer math.comb(n, k) first,
    which for n in the thousands can have hundreds of digits and
    overflow float conversion before the 0.5**n factor ever shrinks
    it back down -- found via a real H6 "overall" slice with several
    thousand paired keys, not a hypothetical)."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + n * math.log(0.5)


def paired_sign_test(paired_diffs):
    """paired_diffs: list of (a - b) for each paired unit (e.g. one
    (instance, range_bucket) group's mean single-error minus mean
    merged-error -- positive means merged was better for that group).
    Exact two-sided binomial sign test against the null that + and -
    are equally likely (ties, exact 0.0, are dropped from n per the
    standard sign-test convention -- they carry no directional
    information). Returns dict with n, n_positive, n_negative, n_ties,
    p_value.

    No scipy dependency -- computed directly via the exact binomial
    PMF, but in LOG space (see _log_binom_pmf_half) with a log-sum-exp
    reduction, not by forming math.comb(n, k) as a giant exact integer
    first (that only works for small n; overflows float conversion for
    n in the thousands, which real H6 slices reach)."""
    positive = sum(1 for d in paired_diffs if d > 0)
    negative = sum(1 for d in paired_diffs if d < 0)
    ties = len(paired_diffs) - positive - negative
    n = positive + negative

    if n == 0:
        return {"n": 0, "n_positive": positive, "n_negative": negative, "n_ties": ties, "p_value": None}

    k = min(positive, negative)   # two-sided: double the smaller tail
    log_terms = [_log_binom_pmf_half(n, i) for i in range(0, k + 1)]
    max_log = max(log_terms)
    tail = math.exp(max_log) * sum(math.exp(t - max_log) for t in log_terms)
    p_value = min(1.0, 2 * tail)
    return {"n": n, "n_positive": positive, "n_negative": negative, "n_ties": ties, "p_value": round(p_value, 6)}


def bonferroni_correct(p_values, alpha=0.05):
    """Simplest, most conservative family-wise correction: the per-test
    significance threshold is alpha / len(p_values); a p-value
    "survives" if it is below that threshold (not if p*len(p_values) is
    compared to alpha -- same conclusion, computed the more standard
    way around). Returns a list of (p_value, survives_bool) in the
    SAME order as the input."""
    n = len(p_values)
    if n == 0:
        return []
    threshold = alpha / n
    return [(p, (p is not None and p < threshold)) for p in p_values]


def holm_bonferroni_correct(p_values, alpha=0.05):
    """Less conservative, still family-wise-error-rate-controlling,
    step-down correction: sort ascending, compare the i-th smallest
    p-value (1-indexed) against alpha/(n-i+1), stop at the first
    failure (every later, larger p-value is then also treated as not
    surviving, per the Holm procedure). Returns survives_bool per
    ORIGINAL input index (order-preserving), not the sorted order."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(
        [(i, p) for i, p in enumerate(p_values) if p is not None],
        key=lambda ip: ip[1],
    )
    survives = [False] * n
    for rank, (orig_i, p) in enumerate(indexed):   # rank is 0-indexed
        threshold = alpha / (n - rank)
        if p < threshold:
            survives[orig_i] = True
        else:
            break   # Holm's procedure: stop at the first non-rejection
    return survives


def benjamini_hochberg_correct(p_values, alpha=0.05):
    """False-discovery-rate control (less conservative than either
    Bonferroni variant above): sort ascending, find the LARGEST i such
    that the i-th smallest p-value <= (i/n)*alpha, and everything at or
    below that rank survives. Returns survives_bool per ORIGINAL input
    index."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(
        [(i, p) for i, p in enumerate(p_values) if p is not None],
        key=lambda ip: ip[1],
    )
    survives = [False] * n
    largest_surviving_rank = -1
    for rank, (orig_i, p) in enumerate(indexed):   # rank is 0-indexed
        if p <= ((rank + 1) / n) * alpha:
            largest_surviving_rank = rank
    for rank, (orig_i, p) in enumerate(indexed):
        if rank <= largest_surviving_rank:
            survives[orig_i] = True
    return survives


# Exposed for modules that want a quick z-based sanity check without a
# full bootstrap (not used for the headline CIs themselves, which are
# always the scene bootstrap above per rule 4) -- kept for test
# coverage of NormalDist usage only; not part of the H1 report.
def _normal_ci_half_width(std, n, ci_pct=cfg.BOOTSTRAP_CI_PCT):
    if n <= 1:
        return None
    z = NormalDist().inv_cdf(0.5 + ci_pct / 200)
    return z * std / math.sqrt(n)
