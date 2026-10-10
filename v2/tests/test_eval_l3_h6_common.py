import math

from v2.eval_l3.h6_common import KeyRow, write_csv, read_csv, abs_error


def _row(**overrides):
    base = dict(
        bound="hungarian", run="fused", scene_id="scene-0061", sample_id="sample_0001",
        instance_id="inst_a", pred_ttc=3.5, gt_ttc=4.0, range_bucket=(0, 20),
        class_group="vru", motion="moving", odd="day", in_corridor=True, n_sensors=2,
    )
    base.update(overrides)
    return KeyRow(**base)


def test_abs_error_finite():
    assert abs(abs_error(_row(pred_ttc=3.5, gt_ttc=4.0)) - 0.5) < 1e-9


def test_abs_error_none_when_either_side_infinite():
    assert abs_error(_row(pred_ttc=float("inf"), gt_ttc=4.0)) is None
    assert abs_error(_row(pred_ttc=3.0, gt_ttc=float("inf"))) is None


def test_csv_round_trip_basic_fields(tmp_path):
    rows = [_row()]
    path = tmp_path / "keys.csv"
    write_csv(rows, path)
    loaded = read_csv(path)
    assert len(loaded) == 1
    r = loaded[0]
    assert r.bound == "hungarian" and r.run == "fused"
    assert r.pred_ttc == 3.5 and r.gt_ttc == 4.0
    assert r.range_bucket == (0.0, 20.0)
    assert r.class_group == "vru" and r.motion == "moving" and r.odd == "day"
    assert r.in_corridor is True
    assert r.n_sensors == 2


def test_csv_round_trip_inf_and_none_fields(tmp_path):
    rows = [_row(
        pred_ttc=float("inf"), gt_ttc=float("inf"), range_bucket=(60, float("inf")),
        class_group=None, motion=None, in_corridor=None, n_sensors=None,
    )]
    path = tmp_path / "keys.csv"
    write_csv(rows, path)
    loaded = read_csv(path)
    r = loaded[0]
    assert math.isinf(r.pred_ttc) and math.isinf(r.gt_ttc)
    assert math.isinf(r.range_bucket[1])
    assert r.class_group is None and r.motion is None
    assert r.in_corridor is None
    assert r.n_sensors is None


def test_csv_round_trip_in_corridor_false(tmp_path):
    rows = [_row(in_corridor=False)]
    path = tmp_path / "keys.csv"
    write_csv(rows, path)
    loaded = read_csv(path)
    assert loaded[0].in_corridor is False
