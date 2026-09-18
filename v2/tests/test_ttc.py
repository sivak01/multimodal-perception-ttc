import math

from v2 import config
from v2.ttc import compute_ttc


def test_closing_object_gives_positive_speed_and_finite_ttc():
    # object at x=50 moving toward a stationary ego at the origin at -10 m/s
    distance, closing_speed, ttc = compute_ttc(
        track_x=50.0, track_y=0.0, track_vx=-10.0, track_vy=0.0,
        ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0,
    )
    assert distance == 50.0
    assert abs(closing_speed - 10.0) < 1e-9
    assert abs(ttc - 5.0) < 1e-9


def test_ttc_decreases_linearly_for_constant_closing_speed():
    """Sanity check the formula's time-consistency: for a constant
    closing speed, ttc computed at t should equal (original ttc - t)."""
    for t, expected_ttc in [(0.0, 5.0), (1.0, 4.0), (2.0, 3.0)]:
        x = 50.0 - 10.0 * t
        _, closing_speed, ttc = compute_ttc(
            track_x=x, track_y=0.0, track_vx=-10.0, track_vy=0.0,
            ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0,
        )
        assert abs(closing_speed - 10.0) < 1e-9
        assert abs(ttc - expected_ttc) < 1e-9


def test_receding_object_gives_infinite_ttc_not_negative():
    # object moving AWAY from ego -- closing_speed must be negative, ttc inf, never a negative number
    distance, closing_speed, ttc = compute_ttc(
        track_x=50.0, track_y=0.0, track_vx=10.0, track_vy=0.0,
        ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0,
    )
    assert closing_speed < 0
    assert ttc == float("inf")


def test_zero_closing_speed_gives_infinite_ttc_not_a_division_error():
    distance, closing_speed, ttc = compute_ttc(
        track_x=50.0, track_y=0.0, track_vx=0.0, track_vy=10.0,   # moving perpendicular, no radial component
        ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0,
    )
    assert abs(closing_speed) < 1e-9
    assert ttc == float("inf")


def test_coincident_position_gives_infinite_ttc_not_a_crash():
    distance, closing_speed, ttc = compute_ttc(
        track_x=0.0, track_y=0.0, track_vx=5.0, track_vy=0.0,
        ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0,
    )
    assert distance == 0.0
    assert ttc == float("inf")


def test_ego_velocity_is_actually_subtracted_relative_closing_speed():
    """The confirmed design decision: an ego vehicle moving in the SAME
    direction and at the SAME speed as the track must see zero relative
    closing speed (infinite TTC), even though the track's own raw
    velocity looks like it's approaching a stationary point. This is
    exactly the case the old pipeline's ego-relative formula handled
    and a track-velocity-only formula would get wrong."""
    # track moving at -10 m/s toward the origin; ego ALSO moving at
    # -10 m/s in the same direction (i.e. driving away from the track
    # at the same rate the track approaches the origin) -- ego and
    # track maintain constant separation, so relative closing speed
    # must be exactly zero.
    distance, closing_speed, ttc = compute_ttc(
        track_x=50.0, track_y=0.0, track_vx=-10.0, track_vy=0.0,
        ego_x=0.0, ego_y=0.0, ego_vx=-10.0, ego_vy=0.0,
    )
    assert abs(closing_speed) < 1e-9
    assert ttc == float("inf")


def test_closing_speed_just_above_min_closing_speed_gives_a_legitimate_finite_ttc():
    """The deadband targets the INSTABILITY of dividing by a
    near-zero closing speed, not merely large ttc output -- a closing
    speed just above config.MIN_CLOSING_SPEED (however slow) must still
    produce a legitimate, finite ttc, not be swept into inf just for
    being a large number."""
    distance = 100.0
    closing_speed_target = config.MIN_CLOSING_SPEED + 0.01
    _, closing_speed, ttc = compute_ttc(
        track_x=distance, track_y=0.0, track_vx=-closing_speed_target, track_vy=0.0,
        ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0,
    )
    assert abs(closing_speed - closing_speed_target) < 1e-9
    assert math.isfinite(ttc)
    assert abs(ttc - distance / closing_speed_target) < 1e-6


def test_closing_speed_just_below_min_closing_speed_gives_infinite_ttc():
    """The specific failure mode this deadband exists to fix: a closing
    speed just below config.MIN_CLOSING_SPEED (technically positive,
    i.e. NOT caught by the old closing_speed <= 0 check alone) must
    still produce inf, not an astronomically large finite ttc from
    dividing by a near-zero number."""
    distance = 100.0
    closing_speed_target = config.MIN_CLOSING_SPEED - 0.01
    assert closing_speed_target > 0   # confirms this genuinely tests the NEW boundary, not the old closing_speed<=0 one
    _, closing_speed, ttc = compute_ttc(
        track_x=distance, track_y=0.0, track_vx=-closing_speed_target, track_vy=0.0,
        ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0,
    )
    assert abs(closing_speed - closing_speed_target) < 1e-9
    assert ttc == float("inf")


def test_ego_moving_toward_track_increases_closing_speed():
    # ego driving TOWARD the approaching track should show a HIGHER
    # closing speed than a stationary ego would (10 + 5 = 15 m/s)
    distance, closing_speed, ttc = compute_ttc(
        track_x=50.0, track_y=0.0, track_vx=-10.0, track_vy=0.0,
        ego_x=0.0, ego_y=0.0, ego_vx=5.0, ego_vy=0.0,
    )
    assert abs(closing_speed - 15.0) < 1e-9
    assert abs(ttc - 50.0 / 15.0) < 1e-9
