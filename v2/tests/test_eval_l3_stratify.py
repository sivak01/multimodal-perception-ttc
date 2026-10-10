import math

from v2.eval_l3.stratify import (
    class_group, moving_or_parked, range_bucket, range_bucket_label,
    ttc_band, in_h1_ttc_band, ego_frame_offset, in_corridor, classify_odd,
)


def test_class_group_vru():
    assert class_group("human.pedestrian.adult") == "vru"
    assert class_group("human.pedestrian.child") == "vru"
    assert class_group("vehicle.bicycle") == "vru"
    assert class_group("vehicle.motorcycle") == "vru"


def test_class_group_vehicle():
    assert class_group("vehicle.car") == "vehicle"
    assert class_group("vehicle.truck") == "vehicle"
    assert class_group("vehicle.bus.rigid") == "vehicle"


def test_class_group_other():
    assert class_group("movable_object.barrier") == "other"
    assert class_group("static_object.bicycle_rack") == "other"


def test_moving_or_parked():
    assert moving_or_parked(0.1) == "parked"
    assert moving_or_parked(0.5) == "parked"   # exactly at threshold -> not > threshold -> parked
    assert moving_or_parked(0.51) == "moving"
    assert moving_or_parked(None) is None


def test_range_bucket_boundaries():
    assert range_bucket(0.0) == (0, 20)
    assert range_bucket(19.999) == (0, 20)
    assert range_bucket(20.0) == (20, 40)
    assert range_bucket(59.999) == (40, 60)
    assert range_bucket(60.0) == (60, float("inf"))
    assert range_bucket(1000.0) == (60, float("inf"))


def test_range_bucket_label():
    assert range_bucket_label((0, 20)) == "[0,20)"
    assert range_bucket_label((60, float("inf"))) == "[60,inf)"


def test_ttc_band():
    assert ttc_band(1.0) == (0, 3)
    assert ttc_band(3.0) == (3, 5)
    assert ttc_band(4.999) == (3, 5)
    assert ttc_band(5.0) == (5, 10)
    assert ttc_band(9.999) == (5, 10)
    assert ttc_band(10.0) is None            # above the last band
    assert ttc_band(float("inf")) is None    # non-closing, excluded


def test_in_h1_ttc_band():
    assert in_h1_ttc_band(0.0) is True
    assert in_h1_ttc_band(4.999) is True
    assert in_h1_ttc_band(5.0) is False
    assert in_h1_ttc_band(float("inf")) is False


def test_ego_frame_offset_object_directly_ahead():
    # ego at origin facing +x (yaw=0); object 10m ahead on the x axis
    fwd, lat = ego_frame_offset(0.0, 0.0, 0.0, 10.0, 0.0)
    assert abs(fwd - 10.0) < 1e-9
    assert abs(lat - 0.0) < 1e-9


def test_ego_frame_offset_object_to_the_left():
    # ego facing +x; object directly to the ego's left (+y) should be +lateral
    fwd, lat = ego_frame_offset(0.0, 0.0, 0.0, 0.0, 5.0)
    assert abs(fwd - 0.0) < 1e-9
    assert abs(lat - 5.0) < 1e-9


def test_ego_frame_offset_with_yaw_rotation():
    # ego facing +y (yaw=90deg); object 10m further in +y is "ahead" in ego frame
    fwd, lat = ego_frame_offset(0.0, 0.0, math.pi / 2, 0.0, 10.0)
    assert abs(fwd - 10.0) < 1e-9
    assert abs(lat - 0.0) < 1e-9


def test_in_corridor_true_and_false():
    assert in_corridor(forward_m=30.0, lateral_m=1.0) is True
    assert in_corridor(forward_m=30.0, lateral_m=3.0) is False   # outside half-width
    assert in_corridor(forward_m=61.0, lateral_m=0.0) is False   # beyond max range
    assert in_corridor(forward_m=-5.0, lateral_m=0.0) is False   # behind the ego


def test_classify_odd():
    assert classify_odd("Parked truck, construction, intersection") == "day"
    assert classify_odd("Night, big street, bus stop") == "night"
    assert classify_odd("Night, after rain, many peds") == "rain"   # rain takes priority
    assert classify_odd("NIGHT and RAIN") == "rain"                  # case-insensitive
