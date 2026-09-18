"""Sanity checks on v2/config.py -- mostly guarding against the exact
mistakes the module docstring warns about (no "camera" key, values are
variance not weight, etc.)."""
from v2 import config


def test_only_three_sensors_no_camera_key():
    expected = {"lidar", "radar", "camera_mono"}
    assert set(config.SENSOR_NOISE_VAR.keys()) == expected
    assert "camera" not in config.SENSOR_NOISE_VAR


def test_process_noise_is_a_single_sensor_independent_constant():
    # PROCESS_NOISE_SIGMA_A models the tracked object's own motion
    # uncertainty, not sensor measurement noise -- it must NOT be
    # indexed by sensor (an earlier draft got this wrong).
    assert isinstance(config.PROCESS_NOISE_SIGMA_A, (int, float))
    assert config.PROCESS_NOISE_SIGMA_A > 0


def test_noise_values_are_positive_variance_not_weight():
    for sensor, var in config.SENSOR_NOISE_VAR.items():
        assert var > 0
    # camera_mono is the noisiest measured sensor -- variance should be
    # the LARGEST value (bigger = noisier), the opposite ordering you'd
    # get if these were accidentally stored as trust weights.
    assert config.SENSOR_NOISE_VAR["camera_mono"] > config.SENSOR_NOISE_VAR["radar"]
    assert config.SENSOR_NOISE_VAR["radar"] > config.SENSOR_NOISE_VAR["lidar"]


def test_specific_values_match_the_brief():
    assert config.SENSOR_NOISE_VAR["lidar"] == 0.0225
    assert config.SENSOR_NOISE_VAR["radar"] == 0.25
    assert config.SENSOR_NOISE_VAR["camera_mono"] == 1.70
    assert config.PROCESS_NOISE_SIGMA_A == 2.0
    assert config.GATE_CONFIDENCE_LEVEL == 0.99
    assert config.MAX_MISSED_SECONDS == 1.5
    assert config.MIN_TRUSTED_SPEED == 1.0
    assert config.TTC_DANGER_ZONE_SECONDS == 2.0
    assert config.TRACK_LENGTH_BUCKETS == [(0, 5), (5, 20), (20, None)]


def test_gate_confidence_is_a_valid_probability():
    assert 0.0 < config.GATE_CONFIDENCE_LEVEL < 1.0


def test_min_updates_for_trusted_velocity_matches_brief_prose():
    # R8 says "fewer than 3 updates" -- not listed in the brief's shown
    # config block, added here as an explicit tunable per R7 rather than
    # a hardcoded literal in kalman_track.py.
    assert config.MIN_UPDATES_FOR_TRUSTED_VELOCITY == 3
