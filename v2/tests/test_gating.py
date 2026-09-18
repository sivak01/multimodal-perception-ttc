import numpy as np

from v2.gating import mahalanobis_gate_and_assign, mahalanobis_sq, gate_threshold

CONFIDENCE = 0.99

# Existing tests below are deliberately testing pure Hungarian/gating
# MECHANICS (assignment, infeasibility, threshold scaling) in isolation
# from measurement noise, so they pass detection_vars=0.0 explicitly --
# see test_gating_includes_measurement_noise_in_the_innovation_covariance
# further down for the behaviour this parameter actually exists to fix.
NO_R = 0.0


def test_no_tracks_or_no_detections_returns_empty():
    assert mahalanobis_gate_and_assign([], [[0.0, 0.0]], CONFIDENCE, NO_R) == ([], [], [0])
    assert mahalanobis_gate_and_assign([((0.0, 0.0), np.eye(2))], [], CONFIDENCE, NO_R) == ([], [0], [])
    assert mahalanobis_gate_and_assign([], [], CONFIDENCE, NO_R) == ([], [], [])


def test_close_detection_matches_within_gate():
    track_predictions = [((0.0, 0.0), np.eye(2))]
    detections = [(0.1, 0.1)]
    pairs, unmatched_t, unmatched_d = mahalanobis_gate_and_assign(track_predictions, detections, CONFIDENCE, NO_R)
    assert pairs == [(0, 0)]
    assert unmatched_t == []
    assert unmatched_d == []


def test_far_detection_rejected_by_gate():
    # covariance is tight (identity), so a detection 50m away is nowhere
    # near the gate regardless of confidence level
    track_predictions = [((0.0, 0.0), np.eye(2))]
    detections = [(50.0, 50.0)]
    pairs, unmatched_t, unmatched_d = mahalanobis_gate_and_assign(track_predictions, detections, CONFIDENCE, NO_R)
    assert pairs == []
    assert unmatched_t == [0]
    assert unmatched_d == [0]


def test_hungarian_picks_the_closer_of_two_competing_tracks():
    track_predictions = [
        ((0.0, 0.0), np.eye(2)),    # track 0: detection is 0.1 away
        ((5.0, 5.0), np.eye(2)),    # track 1: detection is far away
    ]
    detections = [(0.1, 0.0)]
    pairs, unmatched_t, unmatched_d = mahalanobis_gate_and_assign(track_predictions, detections, CONFIDENCE, NO_R)
    assert pairs == [(0, 0)]
    assert unmatched_t == [1]
    assert unmatched_d == []


def test_wide_covariance_widens_the_effective_gate():
    """The whole point of Mahalanobis over a flat Euclidean gate: the
    SAME detection can be accepted or rejected depending on how
    uncertain the track's own prediction is."""
    detections = [(3.0, 0.0)]

    tight_pred = [((0.0, 0.0), np.eye(2) * 0.01)]   # very confident -- 3m is enormous relative to this
    pairs_tight, _, unmatched_d_tight = mahalanobis_gate_and_assign(tight_pred, detections, CONFIDENCE, NO_R)
    assert pairs_tight == []
    assert unmatched_d_tight == [0]

    wide_pred = [((0.0, 0.0), np.eye(2) * 100.0)]   # a fast/coasting track -- 3m is trivial here
    pairs_wide, unmatched_t_wide, _ = mahalanobis_gate_and_assign(wide_pred, detections, CONFIDENCE, NO_R)
    assert pairs_wide == [(0, 0)]
    assert unmatched_t_wide == []


def test_infeasible_pairs_are_never_assigned_even_as_last_resort():
    """R4: pairs outside the gate must be unassignable, even when
    Hungarian's raw bipartite optimum would otherwise pick one because
    nothing better existed (single track, single detection, both far
    outside any reasonable gate)."""
    track_predictions = [((0.0, 0.0), np.eye(2))]
    detections = [(1000.0, 1000.0)]
    pairs, unmatched_t, unmatched_d = mahalanobis_gate_and_assign(track_predictions, detections, CONFIDENCE, NO_R)
    assert pairs == []
    assert unmatched_t == [0]
    assert unmatched_d == [0]


def test_gating_includes_measurement_noise_in_the_innovation_covariance():
    """The specific bug found by tracing the real dataset's fragmentation
    pattern (lidar +99%/radar +58%/camera_mono +1% track count vs. the
    old pipeline, tracking EXACTLY with each sensor's R): gating on the
    track's predicted covariance P_pred ALONE, instead of the true
    innovation covariance S = P_pred + R, made a just-updated low-noise
    track's gate so tight that the next real detection of the SAME
    object -- ordinary motion, nothing anomalous -- could fall outside
    it. Same track_predictions/detection in both calls; only
    detection_vars changes."""
    # a tight predicted covariance, as a just-updated low-R track's
    # would be, and a detection offset by a plausible amount of real
    # motion (e.g. ~4 m/s over half a second)
    track_predictions = [((0.0, 0.0), np.eye(2) * 0.3)]
    detections = [(2.0, 0.0)]

    # OLD (buggy) behaviour: gate on predicted covariance alone --
    # d^2 = 2.0^2 / 0.3 = 13.3, which exceeds the df=2/confidence=0.99
    # threshold (~9.21) -- the real, same-object detection is wrongly
    # rejected, forcing a new (fragmented) track to spawn instead
    pairs_without_r, _, unmatched_d_without_r = mahalanobis_gate_and_assign(
        track_predictions, detections, CONFIDENCE, detection_vars=0.0
    )
    assert pairs_without_r == [], "sanity check: this scenario must reproduce the old rejection first"
    assert unmatched_d_without_r == [0]

    # FIXED behaviour: gate on the innovation covariance P_pred + R --
    # d^2 = 2.0^2 / (0.3 + 0.5) = 5.0, comfortably inside the same
    # threshold -- the SAME real detection is now correctly accepted
    pairs_with_r, unmatched_t_with_r, _ = mahalanobis_gate_and_assign(
        track_predictions, detections, CONFIDENCE, detection_vars=0.5
    )
    assert pairs_with_r == [(0, 0)], "including R should let this plausible same-object detection gate in"
    assert unmatched_t_with_r == []


def test_gating_accepts_a_scalar_or_a_per_detection_list_for_detection_vars():
    """detection_vars may be a single scalar (broadcast to every
    detection) or a list with one entry per detection -- both must
    produce identical results when the scalar equals every list entry."""
    track_predictions = [((0.0, 0.0), np.eye(2)), ((10.0, 10.0), np.eye(2))]
    detections = [(0.1, 0.0), (10.1, 10.0)]

    pairs_scalar, ut_scalar, ud_scalar = mahalanobis_gate_and_assign(
        track_predictions, detections, CONFIDENCE, detection_vars=0.1
    )
    pairs_list, ut_list, ud_list = mahalanobis_gate_and_assign(
        track_predictions, detections, CONFIDENCE, detection_vars=[0.1, 0.1]
    )
    assert pairs_scalar == pairs_list
    assert ut_scalar == ut_list
    assert ud_scalar == ud_list


def test_mahalanobis_sq_matches_hand_computation():
    z = np.array([2.0, 0.0])
    x_pred = np.array([0.0, 0.0])
    p_pred = np.diag([4.0, 1.0])   # sigma_x=2, sigma_y=1
    # (2-0)^2 / 4 + 0 = 1.0
    assert abs(mahalanobis_sq(z, x_pred, p_pred) - 1.0) < 1e-9


def test_gate_threshold_increases_with_confidence():
    assert gate_threshold(0.90) < gate_threshold(0.99) < gate_threshold(0.999)


def test_multiple_tracks_and_detections_full_assignment():
    # two well-separated tracks, two well-separated detections, each
    # detection close to exactly one track -- both should match cleanly
    track_predictions = [
        ((0.0, 0.0), np.eye(2)),
        ((100.0, 100.0), np.eye(2)),
    ]
    detections = [(100.1, 100.1), (0.1, -0.1)]
    pairs, unmatched_t, unmatched_d = mahalanobis_gate_and_assign(track_predictions, detections, CONFIDENCE, NO_R)
    assert set(pairs) == {(0, 1), (1, 0)}
    assert unmatched_t == []
    assert unmatched_d == []
