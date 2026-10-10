from statistics import mean

from v2.eval_l3.bootstrap import (
    group_rows_by_scene, scene_bootstrap_ci, ci_excludes_zero, paired_sign_test,
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
