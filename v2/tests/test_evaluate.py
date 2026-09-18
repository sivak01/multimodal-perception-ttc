import math

from v2 import config
from v2.central_tracker import CentralTracker, TrackSnapshot
from v2.event_stream import SensorEvent
from v2.evaluate import (
    GTPoint, EgoState, MATCH_DIST_THRESHOLD,
    compute_ground_truth_ttc, match_track_to_ground_truth,
    compute_sensor_metrics, compute_length_bucket_metrics,
    compute_multi_sensor_composition,
)


# ---------------------------------------------------------------------
# Pure, hand-checkable tests: no CentralTracker involved, just plain
# TrackSnapshot / GTPoint data, so the matching/TTC-comparison logic is
# verified independent of any Kalman filter convergence behaviour.
# ---------------------------------------------------------------------

def _samples(n):
    return [f"sample_{i:04d}" for i in range(n)]


def test_ground_truth_ttc_matches_hand_computation():
    # a GT object at x=50 moving toward a stationary ego at the origin
    # at exactly -10 m/s -- same scenario as test_ttc.py's closing case
    samples = _samples(2)
    gt_points = [GTPoint(t=0.0, x=50.0, y=0.0, sample_id=samples[0]),
                 GTPoint(t=1.0, x=40.0, y=0.0, sample_id=samples[1])]
    ego_by_sample = {s: EgoState(t=float(i), x=0.0, y=0.0, vx=0.0, vy=0.0) for i, s in enumerate(samples)}

    gt_ttc_lookup = compute_ground_truth_ttc({"gt_car": gt_points}, ego_by_sample)

    # only the SECOND point produces a TTC entry -- the first point of
    # any trajectory has no preceding point to derive a velocity from
    assert (samples[0], "gt_car") not in gt_ttc_lookup
    key = (samples[1], "gt_car")
    assert key in gt_ttc_lookup
    assert abs(gt_ttc_lookup[key]["ttc"] - 4.0) < 1e-9   # distance 40 / closing speed 10


def test_match_locks_identity_and_does_not_re_search_every_point():
    samples = _samples(3)
    ego_by_sample = {s: EgoState(t=float(i), x=0.0, y=0.0, vx=0.0, vy=0.0) for i, s in enumerate(samples)}

    # two real objects: "near" starts close to the track's first point,
    # "far" is nowhere close -- track should lock onto "near" and keep
    # using it even at a later sample where gt_by_sample also contains
    # a much closer-looking (but wrong) instance
    gt_by_sample = {
        samples[0]: [("near", 50.1, 0.0), ("far", 500.0, 500.0)],
        samples[1]: [("near", 40.0, 0.0), ("decoy", 30.05, 0.0)],   # decoy is even closer to the track's point 2 than "near" is
        samples[2]: [("near", 30.0, 0.0)],
    }
    gt_trajectories = {
        "near": [GTPoint(0.0, 50.0, 0.0, samples[0]), GTPoint(1.0, 40.0, 0.0, samples[1]), GTPoint(2.0, 30.0, 0.0, samples[2])],
        "decoy": [GTPoint(1.0, 30.05, 0.0, samples[1])],
    }
    gt_ttc_lookup = compute_ground_truth_ttc(gt_trajectories, ego_by_sample)

    # track's own history: same trajectory as "near", noiseless
    history = [
        TrackSnapshot(t=0.0, x=50.0, y=0.0, vx=-10.0, vy=0.0, sample_id=samples[0]),
        TrackSnapshot(t=1.0, x=40.0, y=0.0, vx=-10.0, vy=0.0, sample_id=samples[1]),
        TrackSnapshot(t=2.0, x=30.0, y=0.0, vx=-10.0, vy=0.0, sample_id=samples[2]),
    ]

    pairs = match_track_to_ground_truth(history, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert all(p["instance_id"] == "near" for p in pairs)   # never switches to "decoy" despite it being closer at sample 1
    assert len(pairs) == 2   # sample 0 has no gt_ttc entry yet (no preceding "near" point); samples 1 and 2 do


def test_match_extrapolates_to_annotation_exact_timestamp_not_nearest_snapshot():
    """The specific case v2_architecture_brief.md's ground-truth-matching
    note is about: a fast-moving object whose track updates (its own
    sensor's native rate) do NOT land on the GT annotation's timestamp.
    Comparing against the nearest raw snapshot AS-IS would be measurably
    wrong here; extrapolating that snapshot's own velocity forward to
    the exact annotation time recovers the true position exactly (since
    this synthetic object is exactly constant-velocity)."""
    true_vx = -30.0   # fast: at this speed, 0.15s of unextrapolated error is 4.5m

    # track's own update times: 0.0, 0.4, 0.8 -- NOT aligned with the
    # GT annotation at t=0.55 below
    track_samples = ["s_a", "s_b", "s_c"]
    track_times = [0.0, 0.4, 0.8]
    events = [
        SensorEvent(t=t, sensor="lidar", x=100.0 + true_vx * t, y=0.0, sample_id=sid, scene_token="scene0")
        for t, sid in zip(track_times, track_samples)
    ]

    tracker = CentralTracker()
    tracker.run(events)
    tracker.finalize()
    tracks = tracker.all_tracks()
    assert len(tracks) == 1
    track = next(iter(tracks.values()))

    # GT annotation exists only at t=0.55, deliberately between two
    # track updates (0.4 and 0.8) -- true position there is exact
    gt_sample = "gt_sample_0001"
    true_x_at_annotation = 100.0 + true_vx * 0.55   # = 83.5

    gt_trajectories = {
        "gt_car": [
            GTPoint(t=0.0, x=100.0, y=0.0, sample_id="gt_sample_0000"),
            GTPoint(t=0.55, x=true_x_at_annotation, y=0.0, sample_id=gt_sample),
        ]
    }
    ego_by_sample = {
        "gt_sample_0000": EgoState(t=0.0, x=0.0, y=0.0, vx=0.0, vy=0.0),
        gt_sample: EgoState(t=0.55, x=0.0, y=0.0, vx=0.0, vy=0.0),
    }
    gt_by_sample = {gt_sample: [("gt_car", true_x_at_annotation, 0.0)]}
    gt_ttc_lookup = compute_ground_truth_ttc(gt_trajectories, ego_by_sample)

    # the nearest raw snapshot to t=0.55 is the t=0.4 one; using its raw
    # x AS-IS (100 + (-30)*0.4 = 88) instead of extrapolating to t=0.55
    # would be off from the true/GT position (83.5) by 4.5m -- well
    # outside MATCH_DIST_THRESHOLD (3.0m), so a broken implementation
    # would fail to lock at all and return no pairs here.
    raw_snapshot_x_at_0_4 = 100.0 + true_vx * 0.4
    assert abs(raw_snapshot_x_at_0_4 - true_x_at_annotation) > MATCH_DIST_THRESHOLD

    pairs = match_track_to_ground_truth(track.history, gt_by_sample, gt_ttc_lookup, ego_by_sample)

    assert len(pairs) == 1, "extrapolation should have let this lock and produce a pair"
    pair = pairs[0]
    assert pair["sample_id"] == gt_sample

    # gt_ttc is exact (GT trajectory is exactly constant-velocity):
    # distance 83.5, closing_speed 30 -> ttc = 83.5/30
    expected_ttc = true_x_at_annotation / 30.0
    assert abs(pair["gt_ttc"] - expected_ttc) < 1e-6

    # pred_ttc must be close to gt_ttc -- only possible if the matcher
    # used the EXTRAPOLATED position (~83.5), not the raw t=0.4
    # snapshot's own position (88, which would have failed to lock at
    # all, so this assertion also implicitly proves locking succeeded)
    assert abs(pair["pred_ttc"] - expected_ttc) < 0.3


def test_match_refuses_to_extrapolate_past_max_missed_seconds():
    """Edge case 1: a GT annotation timestamped well after a track's
    last real snapshot must NOT be extrapolated to arbitrarily far in
    the future. The cap reuses config.MAX_MISSED_SECONDS -- the same
    cutoff CentralTracker itself uses to decide a track is dead -- so a
    track that would already have been evicted by the time this
    annotation exists must produce no match here either."""
    samples = _samples(1)
    # track's last (only) snapshot is at t=0.0, moving fast
    history = [TrackSnapshot(t=0.0, x=0.0, y=0.0, vx=10.0, vy=0.0, sample_id="s0")]

    # annotation timestamped far beyond MAX_MISSED_SECONDS after the
    # track's last snapshot -- a real track this quiet would already be
    # evicted, so this must be a clean no-match, not an extrapolation
    # of vx*dt across a huge, meaningless gap
    late_t = config.MAX_MISSED_SECONDS + 5.0
    would_be_extrapolated_x = 10.0 * late_t   # what a buggy unbounded extrapolation would produce
    gt_by_sample = {samples[0]: [("gt_car", would_be_extrapolated_x, 0.0)]}
    ego_by_sample = {samples[0]: EgoState(t=late_t, x=0.0, y=0.0, vx=0.0, vy=0.0)}
    gt_ttc_lookup = {(samples[0], "gt_car"): {"ttc": 1.0, "pos": (would_be_extrapolated_x, 0.0)}}

    pairs = match_track_to_ground_truth(history, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert pairs == [], "an annotation this far past the track's last snapshot must not match, even at the exact extrapolated position"


def test_match_refuses_to_backward_extrapolate_before_first_snapshot():
    """Edge case 2: a GT annotation timestamped BEFORE the track's very
    first snapshot has no prior snapshot to extrapolate from -- this
    must be a clean no-match, never a backward extrapolation from the
    first snapshot (negative dt) and never an index wraparound onto
    the track's LAST snapshot."""
    history = [
        TrackSnapshot(t=5.0, x=0.0, y=0.0, vx=10.0, vy=0.0, sample_id="s_first"),
        TrackSnapshot(t=6.0, x=10.0, y=0.0, vx=10.0, vy=0.0, sample_id="s_last"),
    ]
    # annotation at t=1.0, well before the track's first snapshot (t=5.0)
    early_sample = "gt_sample_early"
    ego_by_sample = {early_sample: EgoState(t=1.0, x=0.0, y=0.0, vx=0.0, vy=0.0)}
    # placed exactly where a backward extrapolation from the first
    # snapshot (0.0 + 10.0*(1.0-5.0) = -40.0) or a wraparound onto the
    # last snapshot's position (10.0) might otherwise accidentally lock
    gt_by_sample = {early_sample: [("gt_car", -40.0, 0.0), ("decoy_at_last_snapshot", 10.0, 0.0)]}
    gt_ttc_lookup = {
        (early_sample, "gt_car"): {"ttc": 1.0, "pos": (-40.0, 0.0)},
        (early_sample, "decoy_at_last_snapshot"): {"ttc": 1.0, "pos": (10.0, 0.0)},
    }

    pairs = match_track_to_ground_truth(history, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert pairs == [], "an annotation before the track's first snapshot must not match at all"


def test_match_returns_nothing_when_no_gt_within_threshold():
    samples = _samples(1)
    ego_by_sample = {samples[0]: EgoState(0.0, 0.0, 0.0, 0.0, 0.0)}
    gt_by_sample = {samples[0]: [("far_away", 1000.0, 1000.0)]}
    gt_ttc_lookup = {}
    history = [TrackSnapshot(t=0.0, x=0.0, y=0.0, vx=0.0, vy=0.0, sample_id=samples[0])]

    pairs = match_track_to_ground_truth(history, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert pairs == []


# ---------------------------------------------------------------------
# Integration-style tests: run a REAL CentralTracker on synthetic
# events and evaluate its actual output -- exercises everything built
# so far together, not just evaluate.py in isolation.
# ---------------------------------------------------------------------

def _build_closing_object_scenario(n_samples=6, dt=1.0, sensors=("lidar",)):
    """A single real object at x0=50 moving at -10 m/s toward a
    stationary ego at the origin, reported noiselessly by the given
    sensor(s) in round-robin, plus matching ground truth."""
    samples = _samples(n_samples)
    true_x = [50.0 - 10.0 * i * dt for i in range(n_samples)]

    events = []
    for i, (s, x) in enumerate(zip(samples, true_x)):
        sensor = sensors[i % len(sensors)]
        events.append(SensorEvent(t=i * dt, sensor=sensor, x=x, y=0.0, sample_id=s, scene_token="scene0"))

    gt_points = [GTPoint(t=i * dt, x=x, y=0.0, sample_id=s) for i, (s, x) in enumerate(zip(samples, true_x))]
    gt_by_sample = {s: [("gt_car", x, 0.0)] for s, x in zip(samples, true_x)}
    ego_by_sample = {s: EgoState(t=i * dt, x=0.0, y=0.0, vx=0.0, vy=0.0) for i, s in enumerate(samples)}

    return events, gt_points, gt_by_sample, ego_by_sample


def test_single_sensor_run_produces_low_mae_against_matching_ground_truth():
    events, gt_points, gt_by_sample, ego_by_sample = _build_closing_object_scenario(sensors=("lidar",))
    gt_ttc_lookup = compute_ground_truth_ttc({"gt_car": gt_points}, ego_by_sample)

    tracker = CentralTracker()
    tracker.run(events)
    tracker.finalize()
    tracks = tracker.all_tracks()

    assert len(tracks) == 1

    metrics = compute_sensor_metrics("lidar", tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert metrics["n_tracks_total"] == 1
    assert metrics["n_tracks_gt_matched"] == 1
    assert metrics["n_matched_pairs"] > 0
    assert metrics["mae"] is not None
    assert metrics["mae"] < 1.0   # noiseless input against matching ground truth -- should be small


def test_multi_sensor_composition_reflects_a_genuinely_fused_track():
    events, gt_points, gt_by_sample, ego_by_sample = _build_closing_object_scenario(sensors=("lidar", "radar"))
    gt_ttc_lookup = compute_ground_truth_ttc({"gt_car": gt_points}, ego_by_sample)

    tracker = CentralTracker()
    tracker.run(events)
    tracker.finalize()
    tracks = tracker.all_tracks()
    assert len(tracks) == 1
    assert next(iter(tracks.values())).is_multi_sensor is True

    composition = compute_multi_sensor_composition(tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    assert composition["n_multi_sensor_tracks"] == 1
    assert composition["n_single_sensor_tracks"] == 0
    assert composition["multi_sensor_fraction_pct"] == 100.0
    assert composition["mae_multi_sensor"] is not None
    assert composition["mae_single_sensor"] is None   # no single-sensor tracks at all in this run


def test_length_bucket_metrics_places_track_in_the_correct_bucket():
    events, gt_points, gt_by_sample, ego_by_sample = _build_closing_object_scenario(n_samples=6, sensors=("lidar",))
    gt_ttc_lookup = compute_ground_truth_ttc({"gt_car": gt_points}, ego_by_sample)

    tracker = CentralTracker()
    tracker.run(events)
    tracker.finalize()
    tracks = tracker.all_tracks()

    buckets = compute_length_bucket_metrics(tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample)
    # 6 history points falls in the "short" bucket (0, 5)? no -- 6 is in [5, 20)
    assert buckets[(5, 20)]["n_tracks"] == 1
    assert buckets[(0, 5)]["n_tracks"] == 0
    assert buckets[(20, None)]["n_tracks"] == 0
    total_tracks_across_buckets = sum(b["n_tracks"] for b in buckets.values())
    assert total_tracks_across_buckets == len(tracks)
