import numpy as np
import pytest

from v2.kalman_track import ConstantVelocityKF, ChronologyError

SIGMA_A = 2.0   # fixed per-filter process noise used throughout these tests


def test_initial_state():
    kf = ConstantVelocityKF(x=10.0, y=-5.0, t=0.0, sigma_a=SIGMA_A)
    assert np.allclose(kf.position, [10.0, -5.0])
    assert np.allclose(kf.velocity, [0.0, 0.0])
    assert kf.sigma_a == SIGMA_A
    assert kf.n_updates == 1
    assert kf.last_predict_time == 0.0
    assert kf.last_update_time == 0.0
    # this class knows nothing about sensors at all -- that bookkeeping
    # lives on Track in central_tracker.py, not here
    assert not hasattr(kf, "sensors_seen")
    assert not hasattr(kf, "is_multi_sensor")


def test_predict_advances_time_and_grows_uncertainty():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    trace_before = np.trace(kf.position_cov)
    kf.predict(t=1.0)
    assert kf.last_predict_time == 1.0
    assert np.trace(kf.position_cov) > trace_before  # uncertainty grows with no measurement


def test_predict_zero_dt_is_a_no_op():
    kf = ConstantVelocityKF(x=1.0, y=2.0, t=5.0, sigma_a=SIGMA_A)
    x_before, p_before = kf.x.copy(), kf.P.copy()
    kf.predict(t=5.0)
    assert np.allclose(kf.x, x_before)
    assert np.allclose(kf.P, p_before)


def test_predict_rejects_out_of_order_calls():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=5.0, sigma_a=SIGMA_A)
    with pytest.raises(ChronologyError):
        kf.predict(t=4.0)


def test_predict_uses_incremental_dt_not_cumulative_across_non_uniform_gaps():
    """Directly probes predict()'s own timestamp bookkeeping: two
    predict() calls separated by a NON-uniform gap must each apply
    their own true elapsed dt (t - last_predict_time), never a
    cumulative dt measured from the filter's construction time.

    Method: give the filter a known velocity via one predict+update
    (sigma_a=0.0 so there is no process noise blurring the arithmetic),
    then read back whatever velocity it actually settled on and use
    THAT as the reference for checking each subsequent predict() step
    -- so this test doesn't depend on the exact Kalman gain math
    reaching some assumed value, only on predict() applying the
    correct dt given whatever velocity is already in the state.
    """
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=0.0)
    kf.predict(t=1.0)
    kf.update(x_meas=10.0, y_meas=0.0, r_var=1e-9, t=1.0)   # near-zero r_var -> state snaps close to this measurement
    vx = kf.velocity[0]
    assert vx > 5.0   # sanity: it should have picked up a substantial positive velocity, not exactly-checked

    pos_before_1 = kf.position.copy()
    kf.predict(t=1.3)   # gap #1: dt = 0.3
    dx_1 = kf.position[0] - pos_before_1[0]
    assert abs(dx_1 - vx * 0.3) < 1e-9

    pos_before_2 = kf.position.copy()
    kf.predict(t=1.35)   # gap #2: dt = 0.05, deliberately small and different from gap #1
    dx_2 = kf.position[0] - pos_before_2[0]
    assert abs(dx_2 - vx * 0.05) < 1e-9

    # explicitly rule out a cumulative-dt bug: if predict() had used
    # (t - construction_time) instead of (t - last_predict_time), the
    # second call would have moved by vx * 1.35 from its pre-call
    # position instead of vx * 0.05 -- these must clearly differ.
    assert abs(dx_2 - vx * 1.35) > 0.1


def test_update_requires_matching_predict_time():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    kf.predict(t=1.0)
    with pytest.raises(ChronologyError):
        # forgot to predict to t=2.0 first
        kf.update(x_meas=1.0, y_meas=1.0, r_var=0.02, t=2.0)


def test_update_reduces_uncertainty_relative_to_pure_prediction():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    kf.predict(t=1.0)
    trace_after_predict = np.trace(kf.P)
    kf.update(x_meas=2.0, y_meas=0.0, r_var=0.0225, t=1.0)
    assert np.trace(kf.P) < trace_after_predict  # a real measurement should shrink uncertainty


def test_update_moves_state_toward_measurement():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    kf.predict(t=1.0)
    kf.update(x_meas=10.0, y_meas=0.0, r_var=0.0225, t=1.0)
    # position should have moved toward (not necessarily all the way to) the measurement
    assert kf.position[0] > 0.0


def test_update_does_not_track_sensor_identity():
    """update() takes r_var directly, with no sensor_name parameter at
    all -- sensor provenance is entirely Track's responsibility in
    central_tracker.py, not this class's."""
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    kf.predict(t=1.0)
    kf.update(x_meas=1.0, y_meas=0.0, r_var=0.25, t=1.0)   # no sensor_name kwarg accepted
    assert kf.n_updates == 2


def test_sigma_a_fixed_at_construction_is_unaffected_by_update_r_var():
    """sigma_a models the OBJECT's own motion uncertainty, not sensor
    noise -- it must stay the value set at construction regardless of
    what r_var later update() calls use."""
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    kf.predict(t=0.5)
    kf.update(x_meas=0.5, y_meas=0.0, r_var=0.0225, t=0.5)
    assert kf.sigma_a == SIGMA_A

    kf.predict(t=1.0)
    kf.update(x_meas=1.0, y_meas=0.0, r_var=0.25, t=1.0)   # a very different r_var
    assert kf.sigma_a == SIGMA_A


def test_trusted_velocity_deadband_by_update_count():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    kf.predict(t=1.0)
    kf.update(x_meas=5.0, y_meas=0.0, r_var=0.0225, t=1.0)
    # n_updates == 2 here, below the default min_updates=3 -- must be suppressed
    assert kf.n_updates == 2
    assert kf.trusted_velocity(min_speed=1.0) == (0.0, 0.0)


def test_trusted_velocity_deadband_by_speed():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    t = 0.0
    for _ in range(3):
        t += 0.5
        kf.predict(t=t)
        # essentially stationary, tiny jitter only -- true speed ~0
        kf.update(x_meas=0.01, y_meas=-0.01, r_var=0.0225, t=t)
    assert kf.n_updates >= 3
    vx, vy = kf.trusted_velocity(min_speed=1.0)
    assert (vx, vy) == (0.0, 0.0)


def test_trusted_velocity_reports_real_motion_once_confident():
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=SIGMA_A)
    t = 0.0
    true_vx = 10.0
    for _ in range(6):
        t += 0.5
        kf.predict(t=t)
        x_true = true_vx * t
        kf.update(x_meas=x_true, y_meas=0.0, r_var=0.0225, t=t)
    vx, vy = kf.trusted_velocity(min_speed=1.0)
    assert vx > 5.0   # should have converged reasonably close to the true 10 m/s
    assert abs(vy) < 1.0


def test_recovers_true_constant_velocity_from_noiseless_measurements():
    """End-to-end check that predict()/update() math is actually correct,
    not just internally consistent -- feed a perfectly constant-velocity
    trajectory and confirm the filter's velocity estimate converges to
    the true value."""
    true_vx, true_vy = 8.0, -3.0
    kf = ConstantVelocityKF(x=0.0, y=0.0, t=0.0, sigma_a=0.1)
    t = 0.0
    for _ in range(20):
        t += 0.1
        kf.predict(t=t)
        kf.update(x_meas=true_vx * t, y_meas=true_vy * t, r_var=0.0001, t=t)
    assert abs(kf.velocity[0] - true_vx) < 0.5
    assert abs(kf.velocity[1] - true_vy) < 0.5
