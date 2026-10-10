from v2.eval_l3.build_fused_dataset import EnrichedKeyRow
from v2.eval_l3.multiple_comparisons_h1 import (
    _parse_optional_float, _parse_range_bucket, load_enriched_rows,
)


def test_parse_optional_float():
    assert _parse_optional_float("1.5") == 1.5
    assert _parse_optional_float("") is None


def test_parse_range_bucket_plain():
    assert _parse_range_bucket("(0, 20)") == (0.0, 20.0)
    assert _parse_range_bucket("(40, 60)") == (40.0, 60.0)


def test_parse_range_bucket_inf():
    lo, hi = _parse_range_bucket("(60, inf)")
    assert lo == 60.0
    import math
    assert math.isinf(hi) and hi > 0


def test_load_enriched_rows_round_trip(tmp_path):
    rows = [
        EnrichedKeyRow(
            bound="hungarian", scene_id="scene-0061", sample_id="sample_0001", instance_id="inst_a",
            pred_ttc=3.5, gt_ttc=4.0, abs_error=0.5, rel_error=0.125,
            range_m=58.92, range_bucket=(40, 60), class_group="vru", motion="parked",
            odd="day", in_corridor=False, n_sensors_corroborating=1, composition="single",
        ),
        EnrichedKeyRow(
            bound="lock_once", scene_id="scene-1077", sample_id="sample_0300", instance_id="inst_b",
            pred_ttc=float("inf"), gt_ttc=float("inf"), abs_error=None, rel_error=None,
            range_m=61.0, range_bucket=(60, float("inf")), class_group=None, motion=None,
            odd="night", in_corridor=None, n_sensors_corroborating=2, composition="merged",
        ),
    ]
    from v2.eval_l3.build_fused_dataset import write_csv
    csv_path = tmp_path / "enriched.csv"
    write_csv(rows, csv_path)

    loaded = load_enriched_rows(csv_path)
    assert len(loaded) == 2

    r0 = loaded[0]
    assert r0.bound == "hungarian" and r0.instance_id == "inst_a"
    assert r0.pred_ttc == 3.5 and r0.abs_error == 0.5
    assert r0.range_bucket == (40.0, 60.0)
    assert r0.in_corridor is False
    assert r0.composition == "single"

    r1 = loaded[1]
    import math
    assert math.isinf(r1.pred_ttc) and math.isinf(r1.gt_ttc)
    assert r1.abs_error is None and r1.rel_error is None
    assert math.isinf(r1.range_bucket[1])
    assert r1.class_group is None and r1.motion is None
    assert r1.composition == "merged"
