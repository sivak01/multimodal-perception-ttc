"""
v2/evaluate.py — metrics for a completed CentralTracker run: MAE/RMSE/
match-rate/track-counts per sensor and for fused, the track-length-
bucket breakdown, and (for the fused run) the multi-sensor composition
breakdown -- all listed in v2_architecture_brief.md as metrics that
"must be reported on every run", not optional diagnostics.

Ground-truth ("GT") matching follows the same locked-identity scheme
the old late-fusion pipeline used: a track locks onto the nearest real
object within MATCH_DIST_THRESHOLD at the first sample where one is
available, then keeps using that SAME identity for the rest of its
life -- it does not re-search for a closer match every point.

This module is deliberately data-source-agnostic: it takes plain
Python structures (dicts, namedtuples) as input, not nuScenes objects
or file paths -- adapters.py / run_pipeline.py (not yet built) are
responsible for turning real dataset annotations into the GTPoint /
gt_by_sample / ego_trajectory shapes used here, which keeps this file
independently testable with synthetic ground truth.
"""
import math
from collections import namedtuple

import numpy as np

from v2 import config
from v2.ttc import compute_ttc

# Same value the old pipeline used for its locked-identity GT matching.
MATCH_DIST_THRESHOLD = 3.0

# One real annotated position for one ground-truth instance at one
# sample. Ground-truth trajectories are built the same way a track's
# own history is: one point per real sample, in time order.
GTPoint = namedtuple("GTPoint", ["t", "x", "y", "sample_id"])

# Ego vehicle's own state at one sample -- needed by compute_ttc() for
# both predicted and ground-truth TTC, since true closing speed is
# relative to the ego's own motion too (see ttc.py).
EgoState = namedtuple("EgoState", ["t", "x", "y", "vx", "vy"])


def compute_ground_truth_ttc(gt_trajectories, ego_by_sample):
    """
    gt_trajectories: {instance_id: [GTPoint, ...]}, any order (sorted
        internally by t).
    ego_by_sample: {sample_id: EgoState}.

    Returns {(sample_id, instance_id): {"ttc": float, "pos": (x, y)}}
    -- ground-truth TTC computed the SAME way as predicted TTC: a
    finite-difference velocity from the instance's own two most recent
    real positions, fed into the same ttc.compute_ttc(), so the later
    comparison against predicted TTC is genuinely apples-to-apples
    rather than two different formulas.
    """
    gt_ttc_lookup = {}
    for instance_id, points in gt_trajectories.items():
        points = sorted(points, key=lambda p: p.t)
        for i in range(1, len(points)):
            p0, p1 = points[i - 1], points[i]
            dt = p1.t - p0.t
            if dt <= 0:
                continue
            vx = (p1.x - p0.x) / dt
            vy = (p1.y - p0.y) / dt

            ego = ego_by_sample.get(p1.sample_id)
            if ego is None:
                continue

            _, _, ttc = compute_ttc(p1.x, p1.y, vx, vy, ego.x, ego.y, ego.vx, ego.vy)
            gt_ttc_lookup[(p1.sample_id, instance_id)] = {"ttc": ttc, "pos": (p1.x, p1.y)}
    return gt_ttc_lookup


def position_at(snapshot, target_t):
    """Extrapolate a TrackSnapshot's position to target_t using its own
    (vx, vy). This is exactly what re-predicting a fresh KF to target_t
    would produce for the mean state under a constant-velocity model
    (velocity does not change under predict() -- only the covariance
    grows), so no KF cloning is needed, just this arithmetic. Returns
    (x, y)."""
    dt = target_t - snapshot.t
    return snapshot.x + snapshot.vx * dt, snapshot.y + snapshot.vy * dt


def match_track_to_ground_truth(history, gt_by_sample, gt_ttc_lookup, ego_by_sample,
                                 match_dist_threshold=MATCH_DIST_THRESHOLD):
    """
    history: a Track's own .history (list of TrackSnapshot, oldest
        first) -- see central_tracker.py. Recorded at the track's own
        update times, which do NOT generally coincide with GT
        annotation timestamps in the async design.
    gt_by_sample: {sample_id: [(instance_id, x, y), ...]} -- every real
        annotated object present at that sample.
    gt_ttc_lookup: output of compute_ground_truth_ttc().
    ego_by_sample: {sample_id: EgoState} -- also the source of truth
        for every real annotation sample's own timestamp.

    Returns a list of dicts, one per GT-annotated sample (within the
    track's alive window) with both a locked GT identity and an
    available GT TTC:
        {"t":, "sample_id":, "instance_id":, "pred_ttc":, "gt_ttc":}.

    sample_annotation boxes exist only at their own real keyframe
    timestamps, which a track updated at its own sensor's native rate
    will generally NOT land on exactly. This function therefore does
    NOT compare a GT annotation against whichever raw snapshot happens
    to be nearest -- it extrapolates the track's most recent real
    update forward (via position_at(), using that update's own
    velocity) to each annotation's EXACT timestamp, and compares
    against that.

    Two windowing guards keep this extrapolation bounded rather than
    arbitrary, reusing config.MAX_MISSED_SECONDS (the same cutoff
    CentralTracker itself uses to decide a track is dead) instead of a
    second, possibly-inconsistent constant:
      - An annotation timestamped before the track's very first
        snapshot has no prior snapshot to extrapolate from at all --
        this is a clean no-match, never a backward extrapolation or an
        index wraparound onto the last snapshot.
      - An annotation timestamped more than MAX_MISSED_SECONDS after
        the most recent usable snapshot is treated as no-match too --
        CentralTracker would already have evicted a track that quiet
        by then, so extrapolating that far forward would be
        extrapolating past the point the real track's life ended.

    Locks onto the nearest GT instance within match_dist_threshold at
    the first annotated sample (using the extrapolated position) where
    one exists, then reuses that SAME instance id for the rest of the
    track's life -- matching the old pipeline's locked-identity design.
    """
    if not history:
        return []

    # Every annotated sample this track could conceivably be compared
    # against, in time order -- both lists are already time-ordered, so
    # the scan below advances history_idx monotonically (O(n+m) total).
    target_samples = sorted(
        (sid for sid in gt_by_sample if sid in ego_by_sample),
        key=lambda sid: ego_by_sample[sid].t,
    )

    pairs = []
    locked_instance = None
    history_idx = 0

    for sample_id in target_samples:
        target_t = ego_by_sample[sample_id].t
        if target_t < history[0].t:
            continue   # this annotation predates the track's first detection

        while history_idx + 1 < len(history) and history[history_idx + 1].t <= target_t:
            history_idx += 1
        ref = history[history_idx]
        if ref.t > target_t:
            continue   # defensive; the guard above should make this unreachable

        if target_t - ref.t > config.MAX_MISSED_SECONDS:
            continue   # too far past the track's last usable snapshot -- it would already be dead by now

        x, y = position_at(ref, target_t)
        vx, vy = ref.vx, ref.vy

        if locked_instance is None:
            candidates = gt_by_sample.get(sample_id, [])
            if not candidates:
                continue
            best_instance, best_dist = None, float("inf")
            for instance_id, gx, gy in candidates:
                d = math.hypot(x - gx, y - gy)
                if d < best_dist:
                    best_instance, best_dist = instance_id, d
            if best_dist < match_dist_threshold:
                locked_instance = best_instance
            else:
                continue

        key = (sample_id, locked_instance)
        if key not in gt_ttc_lookup:
            continue

        ego = ego_by_sample[sample_id]
        _, _, pred_ttc = compute_ttc(x, y, vx, vy, ego.x, ego.y, ego.vx, ego.vy)
        pairs.append({
            "t": target_t,
            "sample_id": sample_id,
            "instance_id": locked_instance,
            "pred_ttc": pred_ttc,
            "gt_ttc": gt_ttc_lookup[key]["ttc"],
        })

    return pairs


def _valid_pairs(pairs):
    return [p for p in pairs if math.isfinite(p["pred_ttc"]) and math.isfinite(p["gt_ttc"])]


def _mae_rmse(pairs):
    valid = _valid_pairs(pairs)
    if not valid:
        return None, None, 0
    errors = np.array([abs(p["pred_ttc"] - p["gt_ttc"]) for p in valid])
    return round(float(errors.mean()), 3), round(float(np.sqrt((errors ** 2).mean())), 3), len(valid)


def compute_sensor_metrics(sensor_name, tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample):
    """
    tracks: {track_id: Track} -- one sensor's own tracks (or the fused
        run's tracks, with sensor_name="fused").

    Returns the per-run metrics dict the brief requires: MAE, RMSE,
    match rate, total tracks, GT-matched tracks.
    """
    n_tracks_total = len(tracks)
    n_tracks_matched = 0
    all_pairs = []

    for track in tracks.values():
        pairs = match_track_to_ground_truth(track.history, gt_by_sample, gt_ttc_lookup, ego_by_sample)
        if pairs:
            n_tracks_matched += 1
        all_pairs.extend(pairs)

    mae, rmse, n_matched_pairs = _mae_rmse(all_pairs)

    return {
        "sensor": sensor_name,
        "n_tracks_total": n_tracks_total,
        "n_tracks_gt_matched": n_tracks_matched,
        "match_rate_pct": round(n_tracks_matched / n_tracks_total * 100, 2) if n_tracks_total else 0.0,
        "n_matched_pairs": n_matched_pairs,
        "mae": mae,
        "rmse": rmse,
    }


def _length_bucket_for(n, buckets):
    for lo, hi in buckets:
        if n >= lo and (hi is None or n < hi):
            return (lo, hi)
    return buckets[-1]


def compute_length_bucket_metrics(tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample, buckets=None):
    """The dilution-anomaly breakdown from the old pipeline's audit
    (medium tracks best, long tracks worst, driven by a handful of
    sparsely-matched long tracks) -- checks whether it reappears here.
    Returns {(lo, hi): {"n_tracks":, "n_points":, "mae":, "rmse":}}.
    """
    buckets = buckets if buckets is not None else config.TRACK_LENGTH_BUCKETS
    bucketed = {b: [] for b in buckets}
    for track in tracks.values():
        bucketed[_length_bucket_for(len(track.history), buckets)].append(track)

    results = {}
    for b, bucket_tracks in bucketed.items():
        pairs = []
        for track in bucket_tracks:
            pairs.extend(match_track_to_ground_truth(track.history, gt_by_sample, gt_ttc_lookup, ego_by_sample))
        mae, rmse, n_matched_pairs = _mae_rmse(pairs)
        results[b] = {
            "n_tracks": len(bucket_tracks),
            "n_points": n_matched_pairs,
            "mae": mae,
            "rmse": rmse,
        }
    return results


def compute_multi_sensor_composition(tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample):
    """Fused-run-only diagnostic: what fraction of tracks ever saw more
    than one sensor, and how does MAE compare between multi-sensor and
    single-sensor tracks. Since camera_mono is a full fusion
    participant in this design (unlike the old pipeline's excluded
    LiDAR-assisted camera), this number is genuinely new information.
    """
    n_multi, n_single = 0, 0
    multi_pairs, single_pairs = [], []

    for track in tracks.values():
        pairs = match_track_to_ground_truth(track.history, gt_by_sample, gt_ttc_lookup, ego_by_sample)
        if track.is_multi_sensor:
            n_multi += 1
            multi_pairs.extend(pairs)
        else:
            n_single += 1
            single_pairs.extend(pairs)

    total = n_multi + n_single
    mae_multi, rmse_multi, n_multi_pairs = _mae_rmse(multi_pairs)
    mae_single, rmse_single, n_single_pairs = _mae_rmse(single_pairs)

    return {
        "n_multi_sensor_tracks": n_multi,
        "n_single_sensor_tracks": n_single,
        "multi_sensor_fraction_pct": round(n_multi / total * 100, 2) if total else 0.0,
        "mae_multi_sensor": mae_multi,
        "n_multi_sensor_matched_pairs": n_multi_pairs,
        "mae_single_sensor": mae_single,
        "n_single_sensor_matched_pairs": n_single_pairs,
    }
