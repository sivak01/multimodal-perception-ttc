"""
v2/central_tracker.py — THE shared track set.

Replaces the old late-fusion pipeline's per-sensor trackers plus a
separate cross-sensor fusion pass entirely (R1): CentralTracker is the
only tracker class, and it is used identically whether it is fed one
sensor's event stream (a single-sensor baseline) or the merged stream
of all three (the fused run). There is no separate "fusion step" --
that is the entire point of this design.

This module is the one place in v2 that imports config.py's per-sensor
values (SENSOR_NOISE_VAR, PROCESS_NOISE_SIGMA_A, GATE_CONFIDENCE_LEVEL,
MAX_MISSED_SECONDS) and decides how they're used; kalman_track.py and
gating.py stay agnostic to sensors and config on purpose (see their own
module docstrings).
"""
from collections import namedtuple

from v2 import config
from v2.kalman_track import ConstantVelocityKF
from v2.gating import mahalanobis_gate_and_assign

# One snapshot of a track's own state at one of ITS OWN update times.
# evaluate.py (deliverable 3) needs a track's trajectory over time, not
# just its final state, to match against ground truth at each real
# sample it touched -- Track.history is a list of these, oldest first.
TrackSnapshot = namedtuple("TrackSnapshot", ["t", "x", "y", "vx", "vy", "sample_id"])


class Track:
    """Wraps one ConstantVelocityKF with track-level metadata that does
    NOT belong in a sensor-agnostic Kalman filter: which sensors have
    touched this track (R6), its assigned id, and its own trajectory
    history. This separation is deliberate, not incidental -- the KF
    does only Kalman math; sensor provenance and per-update history
    bookkeeping live here instead. `is_multi_sensor` needs no separate
    pass or flag anywhere else (R6): it is always exactly
    `len(sensors_seen) > 1`.
    """
    def __init__(self, track_id, kf, first_sensor, sample_id):
        self.track_id = track_id
        self.kf = kf
        self.sensors_seen = {first_sensor}
        self.history = [TrackSnapshot(
            t=kf.last_update_time, x=kf.position[0], y=kf.position[1],
            vx=kf.velocity[0], vy=kf.velocity[1], sample_id=sample_id,
        )]

    @property
    def is_multi_sensor(self):
        return len(self.sensors_seen) > 1

    @property
    def position(self):
        return self.kf.position

    @property
    def velocity(self):
        return self.kf.velocity

    @property
    def position_cov(self):
        return self.kf.position_cov

    @property
    def n_updates(self):
        return self.kf.n_updates

    @property
    def last_update_time(self):
        return self.kf.last_update_time

    def predict(self, t):
        self.kf.predict(t)

    def update(self, x, y, r_var, sensor_name, sample_id, t):
        self.kf.update(x, y, r_var, t)
        self.sensors_seen.add(sensor_name)
        self.history.append(TrackSnapshot(
            t=t, x=self.kf.position[0], y=self.kf.position[1],
            vx=self.kf.velocity[0], vy=self.kf.velocity[1], sample_id=sample_id,
        ))

    def trusted_velocity(self, min_speed, min_updates=3):
        return self.kf.trusted_velocity(min_speed, min_updates)


class CentralTracker:
    def __init__(self, sensor_noise_var=None, sigma_a=None,
                 gate_confidence=None, max_missed_seconds=None):
        """All four parameters default to config.py's values; overriding
        them is for tests only (e.g. a tiny max_missed_seconds to make
        an eviction test fast and deterministic without a config.py
        dependency in the test itself)."""
        self.sensor_noise_var = dict(sensor_noise_var) if sensor_noise_var is not None else dict(config.SENSOR_NOISE_VAR)
        self.sigma_a = sigma_a if sigma_a is not None else config.PROCESS_NOISE_SIGMA_A
        self.gate_confidence = gate_confidence if gate_confidence is not None else config.GATE_CONFIDENCE_LEVEL
        self.max_missed_seconds = max_missed_seconds if max_missed_seconds is not None else config.MAX_MISSED_SECONDS

        self.active_tracks = {}     # tid -> Track
        self.finished_tracks = {}   # tid -> Track (evicted or scene-finalized)
        self.current_scene_token = None
        self._next_id = 0

        # (scene_token, sample_id, track_id) per accepted association --
        # run_pipeline.py (not yet built) is expected to assert every
        # track's own association history never spans two scene
        # tokens, same as the old pipeline verified for R9.
        self.association_log = []

    def _new_track_id(self):
        tid = f"trk_{self._next_id:06d}"
        self._next_id += 1
        return tid

    def _reset_for_new_scene(self, scene_token):
        """R9: on scene_token change, finalize every active track before
        continuing. No ID, motion state, or last-known-position carries
        across -- each survivor is finalized exactly as it stood."""
        self.finished_tracks.update(self.active_tracks)
        self.active_tracks = {}
        self.current_scene_token = scene_token

    def _evict_stale_tracks(self, now):
        """R5: time-based eviction. A track is evicted once `now -
        last_update_time` exceeds max_missed_seconds -- frame counting
        is meaningless in an asynchronous design, so this is always
        real elapsed seconds, never a missed-frame counter."""
        for tid in list(self.active_tracks.keys()):
            track = self.active_tracks[tid]
            if (now - track.last_update_time) > self.max_missed_seconds:
                self.finished_tracks[tid] = self.active_tracks.pop(tid)

    def process_event(self, event):
        """Handle exactly one SensorEvent (R2). Order of operations:

        1. Scene-boundary reset, if this event starts a new scene.
        2. Predict every remaining active track forward to event.t,
           using THAT TRACK's own last_predict_time -- never a shared
           dt (R2).
        3. Evict any track that's been stale too long relative to
           event.t, based on real elapsed seconds (R5) -- before
           gating, so an already-expired track can never steal this
           event instead of properly starting a new one.
        4. Mahalanobis-gate + Hungarian-assign this one detection
           against whatever tracks remain (R3/R4) -- trivially a 1xN
           problem for a single incoming detection, but gating.py is
           written generically and reused as-is. Gates on the
           INNOVATION covariance (predicted covariance + this event's
           own sensor variance), not predicted covariance alone -- see
           gating.py's module docstring for why omitting the sensor's R
           here was a real, measured bug (it fragmented low-noise
           sensors' tracks worst, exactly as their tight post-update
           covariance would predict).
        5. On a match: update() the winning track, using THIS event's
           sensor for r_var (a fused track's r_var varies update to
           update; sigma_a does not -- see kalman_track.py) and to
           extend Track.sensors_seen. On no match: spawn a new Track
           (R6: sensors_seen starts as {event.sensor}, is_multi_sensor
           becomes true automatically the first time a second sensor's
           event matches it -- no separate "is_merged" pass anywhere).
        """
        if self.current_scene_token is not None and event.scene_token != self.current_scene_token:
            self._reset_for_new_scene(event.scene_token)
        elif self.current_scene_token is None:
            self.current_scene_token = event.scene_token

        for track in self.active_tracks.values():
            track.predict(event.t)

        self._evict_stale_tracks(event.t)

        track_ids = list(self.active_tracks.keys())
        track_predictions = [
            (self.active_tracks[tid].position, self.active_tracks[tid].position_cov)
            for tid in track_ids
        ]
        detections = [(event.x, event.y)]

        pairs, _unmatched_tracks, unmatched_dets = mahalanobis_gate_and_assign(
            track_predictions, detections, self.gate_confidence,
            detection_vars=self.sensor_noise_var[event.sensor],
        )

        if pairs:
            r, _c = pairs[0]   # exactly one detection this call, so at most one pair
            tid = track_ids[r]
            self.active_tracks[tid].update(
                event.x, event.y,
                r_var=self.sensor_noise_var[event.sensor],
                sensor_name=event.sensor,
                sample_id=event.sample_id,
                t=event.t,
            )
            self.association_log.append((event.scene_token, event.sample_id, tid))
        else:
            assert unmatched_dets == [0]
            tid = self._new_track_id()
            kf = ConstantVelocityKF(event.x, event.y, event.t, sigma_a=self.sigma_a)
            self.active_tracks[tid] = Track(tid, kf, event.sensor, event.sample_id)

    def run(self, events):
        """Convenience: process_event() every event in a (possibly
        lazy) iterable, in order -- typically event_stream.merge_streams()'s
        output. Does not finalize remaining active tracks; call
        finalize() explicitly once the stream is exhausted if you want
        every track (including still-active ones) in all_tracks()."""
        for event in events:
            self.process_event(event)

    def finalize(self):
        """Move every still-active track to finished_tracks, as-is --
        call this once an event stream is exhausted so all_tracks()
        reflects every track, not just evicted/scene-finalized ones."""
        self.finished_tracks.update(self.active_tracks)
        self.active_tracks = {}

    def all_tracks(self):
        return {**self.finished_tracks, **self.active_tracks}
