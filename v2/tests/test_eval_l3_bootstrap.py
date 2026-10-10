from statistics import mean

from v2.eval_l3.bootstrap import (
    group_rows_by_scene, scene_bootstrap_ci, scene_bootstrap_p_value,
    ci_excludes_zero, paired_sign_test,
    bonferroni_correct, holm_bonferroni_correct, benjamini_hochberg_correct,
)


def test_group_rows_by_scene():
    rows = [("s1", 1), ("s1", 2), ("s2", 3)]
    grouped = group_rows_by_scene(rows, scene_of_row=lambda r: r[0])
    assert set(grouped.keys()) == {"s1", "s2"}
    assert len(grouped["s1"]) == 2
    assert len(grouped["s2"]) == 1


def test_scene_bootstrap_ci_point_estimate_matches_real_pool():
    # 3 scenes, values chosen so the TRUE mean is exactly 2.0
    rows_by_scene = {"A": [1.0, 2.0], "B": [3.0], "C": [2.0, 2.0]}
    point, lo, hi, n_used = scene_bootstrap_ci(rows_by_scene, mean, n_resamples=500, seed=1)
    assert abs(point - 2.0) < 1e-9
    assert lo is not None and hi is not None
    assert lo <= point <= hi
    assert n_used > 0


def test_scene_bootstrap_ci_reproducible_with_fixed_seed():
    rows_by_scene = {"A": [1.0, 5.0, 2.0], "B": [3.0, 3.0], "C": [10.0], "D": [0.5, 0.5]}
    r1 = scene_bootstrap_ci(rows_by_scene, mean, n_resamples=300, seed=42)
    r2 = scene_bootstrap_ci(rows_by_scene, mean, n_resamples=300, seed=42)
    assert r1 == r2


def test_scene_bootstrap_ci_too_few_scenes_returns_no_ci():
    point, lo, hi, n_used = scene_bootstrap_ci({"A": [1.0, 2.0]}, mean, n_resamples=100, seed=1)
    assert point == 1.5
    assert lo is None and hi is None and n_used == 0


def test_scene_bootstrap_ci_handles_none_stat_for_some_resamples():
    # stat_fn returns None whenever a resampled pool has no "B"-tagged
    # values -- simulates a composition split that can vanish on some
    # resamples; should not crash, should just drop those resamples.
    rows_by_scene = {
        "A": [("x", 1.0)],
        "B": [("y", 2.0)],
        "C": [("x", 3.0)],
    }

    def stat_fn(rows):
        ys = [v for tag, v in rows if tag == "y"]
        return mean(ys) if ys else None

    point, lo, hi, n_used = scene_bootstrap_ci(rows_by_scene, stat_fn, n_resamples=200, seed=7)
    assert point == 2.0
    assert n_used <= 200


def test_ci_excludes_zero():
    assert ci_excludes_zero(0.5, 1.5) is True
    assert ci_excludes_zero(-1.5, -0.5) is True
    assert ci_excludes_zero(-0.5, 0.5) is False
    assert ci_excludes_zero(None, None) is False
    assert ci_excludes_zero(0.0, 1.0) is False   # boundary touches zero -> does not exclude


def test_paired_sign_test_all_positive():
    result = paired_sign_test([1.0, 2.0, 0.5, 3.0])
    assert result["n"] == 4
    assert result["n_positive"] == 4
    assert result["n_negative"] == 0
    assert result["p_value"] < 0.2   # 4/4 one-sided is already fairly extreme (2*(1/16)=0.125)


def test_paired_sign_test_mixed_and_ties():
    result = paired_sign_test([1.0, -1.0, 0.0, 2.0, -0.5])
    assert result["n_ties"] == 1
    assert result["n"] == 4   # ties excluded from n
    assert result["n_positive"] == 2 and result["n_negative"] == 2
    assert result["p_value"] == 1.0   # perfectly balanced -> p=1


def test_paired_sign_test_empty():
    result = paired_sign_test([])
    assert result["n"] == 0
    assert result["p_value"] is None


# ---------------------------------------------------------------------
# scene_bootstrap_p_value
# ---------------------------------------------------------------------

def test_scene_bootstrap_p_value_clearly_nonzero_effect_is_significant():
    # every scene's values are far from 0 and all positive -> virtually
    # every resample's mean will be positive -> tiny two-sided p-value
    rows_by_scene = {"A": [10.0, 11.0], "B": [9.0], "C": [10.5, 10.0], "D": [12.0]}
    p = scene_bootstrap_p_value(rows_by_scene, mean, n_resamples=500, seed=1)
    assert p is not None
    assert p < 0.01


def test_scene_bootstrap_p_value_symmetric_around_zero_is_not_significant():
    # values straddle 0 roughly symmetrically -> resampled means land on
    # both sides often -> p should be large (not significant)
    rows_by_scene = {"A": [-5.0, 5.0], "B": [-4.0], "C": [4.0], "D": [0.1, -0.1]}
    p = scene_bootstrap_p_value(rows_by_scene, mean, n_resamples=1000, seed=2)
    assert p is not None
    assert p > 0.2


def test_scene_bootstrap_p_value_consistent_with_ci_exclusion():
    """Same resamples (same seed) drive both scene_bootstrap_ci() and
    scene_bootstrap_p_value() -- CI-excludes-zero at 95% and p<0.05
    should agree for the same input."""
    rows_by_scene = {"A": [3.0, 4.0], "B": [2.5], "C": [3.5, 3.0], "D": [2.0, 4.5], "E": [3.2]}
    _point, lo, hi, _n = scene_bootstrap_ci(rows_by_scene, mean, n_resamples=1000, seed=42)
    p = scene_bootstrap_p_value(rows_by_scene, mean, n_resamples=1000, seed=42)
    assert ci_excludes_zero(lo, hi) is True
    assert p < 0.05


def test_scene_bootstrap_p_value_too_few_scenes_returns_none():
    assert scene_bootstrap_p_value({"A": [1.0, 2.0]}, mean) is None


def test_scene_bootstrap_p_value_reproducible_with_fixed_seed():
    rows_by_scene = {"A": [1.0, 5.0, 2.0], "B": [3.0, 3.0], "C": [10.0], "D": [0.5, 0.5]}
    p1 = scene_bootstrap_p_value(rows_by_scene, mean, n_resamples=300, seed=42)
    p2 = scene_bootstrap_p_value(rows_by_scene, mean, n_resamples=300, seed=42)
    assert p1 == p2


# ---------------------------------------------------------------------
# multiple-comparison corrections
# ---------------------------------------------------------------------

def test_bonferroni_correct_basic():
    # alpha/6 = 0.008333...; only the first p-value survives
    results = bonferroni_correct([0.001, 0.01, 0.2, 0.5, 0.03, 0.04], alpha=0.05)
    assert [survives for _p, survives in results] == [True, False, False, False, False, False]


def test_bonferroni_correct_empty():
    assert bonferroni_correct([]) == []


def test_bonferroni_correct_none_values_never_survive():
    results = bonferroni_correct([None, 0.001], alpha=0.05)
    assert results[0] == (None, False)
    assert results[1][1] is True


def test_holm_bonferroni_less_conservative_than_bonferroni():
    """A classic case where Holm rejects more than plain Bonferroni:
    p-values [0.001, 0.009, 0.02, 0.03, 0.04, 0.045], alpha=0.05.
    Plain Bonferroni threshold = 0.05/6 = 0.00833 -> only p=0.001 survives.
    Holm: sorted ranks 1..6, thresholds 0.00833, 0.01, 0.0125, 0.01667,
    0.025, 0.05 -> 0.001<0.00833 ok, 0.009<0.01 ok, 0.02<0.0125 FAILS,
    stop -> first two survive."""
    p_values = [0.001, 0.009, 0.02, 0.03, 0.04, 0.045]
    holm = holm_bonferroni_correct(p_values, alpha=0.05)
    bonf = [survives for _p, survives in bonferroni_correct(p_values, alpha=0.05)]
    assert holm == [True, True, False, False, False, False]
    assert sum(holm) >= sum(bonf)


def test_holm_bonferroni_empty():
    assert holm_bonferroni_correct([]) == []


def test_benjamini_hochberg_at_least_as_permissive_as_holm():
    p_values = [0.001, 0.009, 0.02, 0.03, 0.04, 0.045]
    bh = benjamini_hochberg_correct(p_values, alpha=0.05)
    holm = holm_bonferroni_correct(p_values, alpha=0.05)
    assert sum(bh) >= sum(holm)


def test_benjamini_hochberg_known_case():
    # BH thresholds for n=4, alpha=0.05: rank*0.05/4 = 0.0125, 0.025, 0.0375, 0.05
    # sorted p: 0.01, 0.02, 0.03, 0.2 -> compare: 0.01<=0.0125 ok(rank1),
    # 0.02<=0.025 ok(rank2), 0.03<=0.0375 ok(rank3), 0.2<=0.05 fail(rank4)
    # largest surviving rank = 3 (0-indexed rank 2) -> first 3 survive
    p_values = [0.03, 0.01, 0.2, 0.02]
    bh = benjamini_hochberg_correct(p_values, alpha=0.05)
    assert bh == [True, True, False, True]


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg_correct([]) == []
