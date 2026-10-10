from collections import namedtuple

from v2.eval_l3.gt_context import GTContext, enrich_key

EgoState = namedtuple("EgoState", ["t", "x", "y", "vx", "vy"])


def _ctx(**overrides):
    base = dict(
        samples_index={"sample_0001": {"scene_name": "scene-0061", "scene_token": "tok1"}},
        ego_by_sample={"sample_0001": EgoState(t=0.0, x=0.0, y=0.0, vx=1.0, vy=0.0)},
        gt_by_sample={"sample_0001": [("inst_a", 10.0, 0.0)]},
        gt_ttc_lookup={},
        gt_speed_lookup={("sample_0001", "inst_a"): 5.0},
        category_lookup={("sample_0001", "inst_a"): "vehicle.car"},
        scene_odd_lookup={"tok1": "Night, big street"},
        ego_heading_lookup={"sample_0001": 0.0},   # facing +x
        gt_by_sample_pos={("sample_0001", "inst_a"): (10.0, 0.0)},
    )
    base.update(overrides)
    return GTContext(**base)


def test_enrich_key_basic_fields():
    ctx = _ctx()
    result = enrich_key(ctx, "sample_0001", "inst_a")
    assert result is not None
    assert result["scene_id"] == "scene-0061"
    assert result["range_m"] == 10.0
    assert result["range_bucket"] == (0, 20)
    assert result["class_group"] == "vehicle"
    assert result["motion"] == "moving"
    assert result["odd"] == "night"
    assert result["in_corridor"] is True   # 10m straight ahead, within half-width


def test_enrich_key_returns_none_when_key_unresolvable():
    ctx = _ctx(gt_by_sample_pos={})
    assert enrich_key(ctx, "sample_0001", "inst_a") is None


def test_enrich_key_returns_none_when_no_ego_state():
    ctx = _ctx(ego_by_sample={})
    assert enrich_key(ctx, "sample_0001", "inst_a") is None


def test_enrich_key_in_corridor_false_when_far_off_axis():
    ctx = _ctx(gt_by_sample_pos={("sample_0001", "inst_a"): (10.0, 10.0)})
    result = enrich_key(ctx, "sample_0001", "inst_a")
    assert result["in_corridor"] is False   # 10m lateral >> CORRIDOR_HALF_WIDTH_M


def test_enrich_key_handles_unresolvable_category_and_speed():
    ctx = _ctx(category_lookup={}, gt_speed_lookup={})
    result = enrich_key(ctx, "sample_0001", "inst_a")
    assert result["class_group"] is None
    assert result["motion"] is None


def test_enrich_key_in_corridor_none_when_no_heading():
    ctx = _ctx(ego_heading_lookup={})
    result = enrich_key(ctx, "sample_0001", "inst_a")
    assert result["in_corridor"] is None
