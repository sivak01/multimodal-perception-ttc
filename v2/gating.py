"""
v2/gating.py — Mahalanobis-distance gating + Hungarian (optimal)
assignment between predicted track positions and new detections.

Using Mahalanobis distance (scaled by each track's own predicted
position covariance) instead of a flat Euclidean threshold means the
effective gate radius automatically widens for a fast-moving track, or
one that has coasted through missed updates and accumulated process
noise, and stays tight for a stationary, confident one. This is the v2
replacement for the old late-fusion pipeline's fixed 3.0m
ASSOC_DIST_THRESHOLD, which never adapted to how uncertain a prediction
actually was and fragmented fast-moving objects as a result (see
v2_architecture_brief.md, "root causes already diagnosed", #2).

Innovation covariance MUST include the detection's own measurement
noise R, not just the track's predicted covariance P (found from a
real, measured pattern on the full dataset: every single-sensor
baseline's track count was up sharply from the old late-fusion
pipeline -- lidar +99%, radar +58%, camera_mono +1% -- and that
ordering tracked EXACTLY with each sensor's R, smallest-R sensors
fragmenting worst). The textbook-correct gate uses the INNOVATION
covariance S = H@P@H.T + R (here, since H is just the position rows,
S = P_pred + diag([r_var, r_var])), not P_pred alone. Omitting R meant
a track just updated by a low-noise sensor collapsed P so tight that
the very next real detection of the same object -- ordinary motion,
nothing anomalous -- could fall outside the gate, which is exactly the
fragmentation pattern observed. central_tracker.py passes the current
event's own sensor variance in as detection_vars, the SAME value it
already looks up for that event's subsequent update() call if a match
succeeds.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# Substituted for any track/detection pair whose squared Mahalanobis
# distance exceeds the gate -- large enough that Hungarian will always
# prefer any feasible pairing over one at this cost, but finite so
# linear_sum_assignment (which requires a finite cost matrix) still runs.
INFEASIBLE_COST = 1e6


def gate_threshold(confidence_level, df=2):
    """Chi-square critical value for the given confidence level and
    degrees of freedom (2, for a 2D x/y position gate)."""
    return chi2.ppf(confidence_level, df=df)


def mahalanobis_sq(z, x_pred, p_pred):
    """Squared Mahalanobis distance of a 2D measurement z from a
    predicted 2D position x_pred with covariance p_pred. Despite the
    parameter name (kept for backward compatibility with existing
    callers/tests that pass a pure predicted covariance), this is
    really "whatever covariance the caller wants distance scaled by" --
    mahalanobis_gate_and_assign() below passes the full INNOVATION
    covariance (P_pred + R), not P_pred alone; see its own docstring."""
    z = np.asarray(z, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    diff = z - x_pred
    return float(diff @ np.linalg.inv(p_pred) @ diff)


def mahalanobis_gate_and_assign(track_predictions, detections, confidence_level, detection_vars):
    """
    track_predictions: list of (x_pred, P_pred) -- x_pred a length-2
        position, P_pred a 2x2 covariance. One entry per active track,
        already predicted forward to the detections' timestamp.
    detections: list of length-2 positions.
    confidence_level: chi-square gate confidence, e.g.
        config.GATE_CONFIDENCE_LEVEL.
    detection_vars: measurement noise VARIANCE (R's diagonal value,
        assumed isotropic in x/y) for each detection being gated --
        REQUIRED, not optional: see module docstring for why gating on
        P_pred alone (omitting R) was a real, measured bug, not a
        simplification. Either a single scalar (broadcast to every
        detection -- the common case, since CentralTracker only ever
        gates one detection per call) or a sequence of length
        len(detections) (one variance per detection, for a future
        caller gating several detections with different R's at once).
        There is deliberately no default value: every real caller has a
        real sensor variance to pass (central_tracker.py passes the
        same config.SENSOR_NOISE_VAR value already used for that
        event's update() call), so an implicit default here would risk
        silently reintroducing the exact bug this parameter exists to
        fix. Tests that deliberately want to isolate pure-covariance
        gating behaviour pass detection_vars=0.0 explicitly.

    Returns (assigned_pairs, unmatched_track_idx, unmatched_det_idx):
        assigned_pairs: list of (track_idx, det_idx) -- Hungarian-optimal
            AND gate-feasible pairs only (R4: pairs outside the gate are
            never assignable, even if Hungarian's raw optimum would
            otherwise pick one because nothing better existed).
        unmatched_track_idx: track indices with no feasible assignment
            this round (candidates for eviction bookkeeping upstream).
        unmatched_det_idx: detection indices with no feasible
            assignment (each should start a new track upstream).
    """
    n_tracks, n_dets = len(track_predictions), len(detections)
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    if np.isscalar(detection_vars):
        detection_vars = [float(detection_vars)] * n_dets
    else:
        detection_vars = [float(v) for v in detection_vars]
    assert len(detection_vars) == n_dets, "detection_vars must be a scalar or have one entry per detection"

    threshold = gate_threshold(confidence_level)

    cost = np.full((n_tracks, n_dets), INFEASIBLE_COST)
    for i, (x_pred, p_pred) in enumerate(track_predictions):
        for j, z in enumerate(detections):
            # innovation covariance S = P_pred + R -- see module
            # docstring for why R must be included here
            innovation_cov = np.asarray(p_pred, dtype=float) + np.diag([detection_vars[j], detection_vars[j]])
            d2 = mahalanobis_sq(z, x_pred, innovation_cov)
            if d2 <= threshold:
                cost[i, j] = d2

    row_idx, col_idx = linear_sum_assignment(cost)

    assigned_pairs = []
    matched_tracks, matched_dets = set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] < INFEASIBLE_COST:
            assigned_pairs.append((int(r), int(c)))
            matched_tracks.add(r)
            matched_dets.add(c)

    unmatched_track_idx = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_det_idx = [j for j in range(n_dets) if j not in matched_dets]

    return assigned_pairs, unmatched_track_idx, unmatched_det_idx
