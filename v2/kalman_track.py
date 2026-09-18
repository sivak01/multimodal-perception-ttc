"""
v2/kalman_track.py — a single, sensor-agnostic constant-velocity Kalman
filter.

State:       [px, py, vx, vy] — global frame, metres / metres-per-second.
Measurement: [px, py] — position only. No sensor's raw velocity (e.g.
             radar Doppler) is ever fed into this filter — see R10 in
             v2_architecture_brief.md: TTC always uses the KF's own
             velocity state, never a raw sensor-reported speed field.

This class knows NOTHING about sensor names, config.py, OR track-level
provenance (which sensors have touched this track, whether it counts as
"multi-sensor") — that bookkeeping lives on Track in central_tracker.py,
which wraps one of these. Keeping it out of here isn't a style
preference: it's the same separation-of-concerns principle that put
sigma_a's fix in the previous deliverable — a sensor-agnostic numerical
filter should not also be the place tracking sensor identity.

The two noise numbers involved are handled differently, on purpose:

  - Measurement noise (r_var, ~R) genuinely varies by sensor, and a
    FUSED track (central_tracker's whole point) gets update() calls
    from different sensors over its lifetime -- so r_var is a per-call
    argument to update(), supplied by whichever caller just decided
    this event matched this track.

  - Process noise (sigma_a, ~Q) represents uncertainty in the TRACKED
    OBJECT's own motion, not sensor measurement noise -- it does not
    vary by sensor, and predict() runs against every active track
    BEFORE gating even decides which detection (if any) matches, so
    there is no sensor to attribute it to at that point anyway. It is
    fixed once per track at construction and reused by every predict()
    call for that track's lifetime.

central_tracker.py is the one place that imports config.SENSOR_NOISE_VAR
/ config.PROCESS_NOISE_SIGMA_A; this file stays independently testable
with no config.py dependency at all.
"""
import numpy as np


class ChronologyError(ValueError):
    """Raised when predict()/update() is called out of timestamp order,
    or update() is called without a matching predict() to the same t."""


class ConstantVelocityKF:
    def __init__(self, x, y, t, sigma_a, initial_pos_var=1.0, initial_vel_var=100.0):
        """
        x, y: initial position (global frame, metres).
        t: initial timestamp (seconds, any consistent monotonic epoch).
        sigma_a: process noise (m/s^2, config.PROCESS_NOISE_SIGMA_A) --
            fixed for this filter's entire lifetime, since it models
            the OBJECT's own motion uncertainty, not sensor noise. See
            module docstring.
        initial_pos_var / initial_vel_var: initial covariance diagonal.
            Position uncertainty starts tight (we just measured it);
            velocity uncertainty starts deliberately large since a
            single point carries no velocity information at all.
        """
        self.x = np.array([float(x), float(y), 0.0, 0.0])
        self.P = np.diag([float(initial_pos_var), float(initial_pos_var),
                           float(initial_vel_var), float(initial_vel_var)])
        self.sigma_a = float(sigma_a)
        self.last_predict_time = float(t)
        self.last_update_time = float(t)
        self.n_updates = 1

    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def velocity(self):
        return self.x[2:].copy()

    @property
    def position_cov(self):
        return self.P[:2, :2].copy()

    @staticmethod
    def _F(dt):
        """Constant-velocity state transition matrix for interval dt."""
        return np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _Q(dt, sigma_a):
        """Discretized white-noise-acceleration process noise. x and y
        are treated as decoupled (same sigma_a for both axes)."""
        q = sigma_a ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        block = np.array([[dt4 / 4.0, dt3 / 2.0],
                           [dt3 / 2.0, dt2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block   # x, vx
        Q[np.ix_([1, 3], [1, 3])] = block   # y, vy
        return Q

    def predict(self, t):
        """Advance this filter's own state/covariance from its own
        last_predict_time to t (seconds), using this filter's fixed
        self.sigma_a (set once at construction — see module docstring
        for why process noise does not vary per call the way
        measurement noise does). dt is always computed from THIS
        filter's own last_predict_time (an absolute timestamp updated
        after every call), never a shared/nominal interval and never a
        cumulative offset from construction time (v2 design rule R2) —
        mutates state in place; the resulting .position / .position_cov
        are what gating.py should use for this event.
        """
        t = float(t)
        dt = t - self.last_predict_time
        if dt < 0:
            raise ChronologyError(
                f"predict(t={t}) is before last_predict_time={self.last_predict_time} "
                "-- events must be processed in chronological order"
            )
        if dt == 0:
            return
        F = self._F(dt)
        Q = self._Q(dt, self.sigma_a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_predict_time = t

    def update(self, x_meas, y_meas, r_var, t):
        """Correct the already-predicted state with a new position
        measurement. Must be called with the SAME t just passed to
        predict() -- this class does not implicitly predict for you,
        since central_tracker.py needs the pre-update predicted state
        for gating before it knows which detection (if any) matches.

        r_var: this specific measurement's sensor noise variance (m^2)
            -- looked up by the caller from
            config.SENSOR_NOISE_VAR[sensor_name], since a fused track
            can be updated by different sensors over its lifetime. This
            class does not record which sensor it came from -- that is
            Track's job in central_tracker.py.
        """
        t = float(t)
        if abs(t - self.last_predict_time) > 1e-9:
            raise ChronologyError(
                f"update(t={t}) does not match last_predict_time={self.last_predict_time} "
                "-- call predict(t) with this same t first"
            )

        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        R = np.diag([float(r_var), float(r_var)])
        z = np.array([float(x_meas), float(y_meas)])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.n_updates += 1
        self.last_update_time = t

    def trusted_velocity(self, min_speed, min_updates=3):
        """Returns (0.0, 0.0) if this filter hasn't accumulated at
        least min_updates real measurement updates yet, or if its
        estimated speed is below min_speed -- detector/clustering
        jitter on a stationary object otherwise reads as false motion
        (v2 design rule R8). Otherwise returns the KF's own (vx, vy)
        state -- never a raw sensor-reported velocity field (R10)."""
        if self.n_updates < min_updates:
            return 0.0, 0.0
        vx, vy = float(self.x[2]), float(self.x[3])
        if np.hypot(vx, vy) < min_speed:
            return 0.0, 0.0
        return vx, vy
