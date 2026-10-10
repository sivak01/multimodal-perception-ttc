from v2.eval_l3.build_fused_dataset import EnrichedKeyRow
from v2.eval_l3.hypothesis_h1 import (
    _percentile, mae_stat, median_stat, p90_stat, rel_mae_stat,
    build_qualifying_groups, compare_merged_vs_single, verdict_for,
)


def _row(scene_id, instance_id, range_bucket, composition, abs_error, gt_ttc=4.0,
         bound="hungarian", rel_error=None):
    return EnrichedKeyRow(
        bound=bound, scene_id=scene_id, sample_id=f"{scene_id}_{instance_id}_{abs_error}",
        instance_id=instance_id, pred_ttc=gt_ttc + abs_error, gt_ttc=gt_ttc,
        abs_error=abs_error, rel_error=(abs_error / gt_ttc if rel_error is None else rel_error),
        range_m=10.0, range_bucket=range_bucket, class_group="vehicle", motion="moving",
        odd="day", in_corridor=True, n_sensors_corroborating=2 if composition == "merged" else 1,
        composition=composition,
    )


# ---------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------

def test_percentile_basic():
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([], 90) is None
    assert _percentile([10], 90) == 10


# ---------------------------------------------------------------------
# metric stat functions
# ---------------------------------------------------------------------

def test_mae_median_p90_stats():
    rows = [
        _row("s1", "A", (0, 20), "merged", 1.0),
        _row("s1", "A", (0, 20), "merged", 2.0),
        _row("s1", "A", (0, 20), "merged", 100.0),
    ]
    assert abs(mae_stat(rows) - (1 + 2 + 100) / 3) < 1e-9
    assert median_stat(rows) == 2.0
    assert p90_stat(rows) == 100.0   # dominated by the outlier, same shape as the earlier H7 finding


def test_rel_mae_stat():
    rows = [_row("s1", "A", (0, 20), "merged", 2.0, gt_ttc=4.0)]   # rel_error = 0.5
    assert abs(rel_mae_stat(rows) - 0.5) < 1e-9


def test_stats_return_none_on_empty():
    assert mae_stat([]) is None
    assert median_stat([]) is None
    assert p90_stat([]) is None
    assert rel_mae_stat([]) is None


# ---------------------------------------------------------------------
# build_qualifying_groups
# ---------------------------------------------------------------------

def test_build_qualifying_groups_excludes_single_composition_groups():
    rows = [
        _row("s1", "A", (0, 20), "merged", 1.0),
        _row("s1", "A", (0, 20), "single", 2.0),   # A/(0,20) has BOTH -> qualifies
        _row("s1", "B", (0, 20), "merged", 1.0),   # B/(0,20) has ONLY merged -> excluded
        _row("s2", "C", (20, 40), "single", 3.0),  # C/(20,40) has ONLY single -> excluded
    ]
    merged_rows, single_rows, groups = build_qualifying_groups(rows, bound="hungarian")
    assert len(groups) == 1
    assert ("A", (0, 20)) in groups
    assert len(merged_rows) == 1 and len(single_rows) == 1


def test_build_qualifying_groups_filters_by_bound():
    rows = [
        _row("s1", "A", (0, 20), "merged", 1.0, bound="hungarian"),
        _row("s1", "A", (0, 20), "single", 2.0, bound="lock_once"),   # different bound -> never pairs with the row above
    ]
    merged_rows, single_rows, groups = build_qualifying_groups(rows, bound="hungarian")
    assert groups == {}


def test_build_qualifying_groups_restrict_fn():
    rows = [
        _row("s1", "A", (0, 20), "merged", 1.0),
        _row("s1", "A", (0, 20), "single", 2.0),
    ]
    merged_rows, single_rows, groups = build_qualifying_groups(
        rows, bound="hungarian", restrict_fn=lambda r: r.in_corridor is False,
    )
    assert groups == {}   # all synthetic rows have in_corridor=True


# ---------------------------------------------------------------------
# compare_merged_vs_single (structure + a hand-checkable favourable case)
# ---------------------------------------------------------------------

def test_compare_merged_vs_single_merged_clearly_better():
    rows = []
    # 3 scenes, each with 2 qualifying (instance, range_bucket) groups,
    # merged consistently ~1s better than single in every group
    for scene_i, scene in enumerate(["s1", "s2", "s3"]):
        for g in range(2):
            instance = f"obj_{scene_i}_{g}"
            rows.append(_row(scene, instance, (0, 20), "merged", 1.0))
            rows.append(_row(scene, instance, (0, 20), "single", 2.0))

    result = compare_merged_vs_single(rows, bound="hungarian")
    assert result["n_qualifying_groups"] == 6
    assert result["metrics"]["mae"]["merged"] == 1.0
    assert result["metrics"]["mae"]["single"] == 2.0
    assert result["metrics"]["mae"]["diff_merged_minus_single"] == -1.0
    assert result["paired_sign_test"]["n_positive"] == 6   # merged better in every single group
    assert result["paired_sign_test"]["n_negative"] == 0


def test_compare_merged_vs_single_no_qualifying_groups_returns_none_metrics():
    rows = [_row("s1", "A", (0, 20), "merged", 1.0)]   # no matching single row -> no qualifying group
    result = compare_merged_vs_single(rows, bound="hungarian")
    assert result["n_qualifying_groups"] == 0
    assert result["metrics"]["mae"]["diff_merged_minus_single"] is None


# ---------------------------------------------------------------------
# verdict_for
# ---------------------------------------------------------------------

def _comparison(n_groups, diff, ci_lo, ci_hi):
    return {
        "n_qualifying_groups": n_groups,
        "metrics": {"mae": {
            "merged": 1.0, "single": 1.0 - diff, "diff_merged_minus_single": diff,
            "ci95_low": ci_lo, "ci95_high": ci_hi,
            "ci_excludes_zero": (ci_lo is not None and ci_hi is not None and (ci_lo > 0) == (ci_hi > 0) and ci_lo != 0),
        }},
    }


def test_verdict_confirmed():
    comp = _comparison(n_groups=50, diff=-1.0, ci_lo=-1.5, ci_hi=-0.5)
    verdict, reason = verdict_for(comp)
    assert verdict == "CONFIRMED"


def test_verdict_refuted_ci_includes_zero():
    comp = _comparison(n_groups=50, diff=-0.2, ci_lo=-0.5, ci_hi=0.3)
    verdict, reason = verdict_for(comp)
    assert verdict == "REFUTED"
    assert "includes 0" in reason


def test_verdict_refuted_favours_single():
    comp = _comparison(n_groups=50, diff=1.0, ci_lo=0.5, ci_hi=1.5)
    verdict, reason = verdict_for(comp)
    assert verdict == "REFUTED"
    assert "WORSE" in reason


def test_verdict_inconclusive_small_n():
    comp = _comparison(n_groups=5, diff=-1.0, ci_lo=-1.5, ci_hi=-0.5)
    verdict, reason = verdict_for(comp)
    assert verdict == "INCONCLUSIVE"
    assert "5" in reason
