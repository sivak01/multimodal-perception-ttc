"""
v2/run_pipeline.py — orchestrator: runs the three single-sensor
baselines (lidar, radar, camera_mono) and the one fused run
(lidar+radar+camera_mono) through the SAME CentralTracker class (R1),
evaluates each against real nuScenes ground truth, and prints the full
metrics table with all three breakdowns v2_architecture_brief.md
requires on every run: per-sensor MAE/RMSE/match-rate/track-counts,
the track-length-bucket breakdown, and the fused run's multi-sensor
composition breakdown.

Also implements R9's final check: after every run, independently
verify -- via samples_index.json's own scene_token per sample_id, NOT
just CentralTracker's internal _reset_for_new_scene() bookkeeping --
that no single track's history ever touches more than one scene. This
mirrors the old pipeline's own two-pronged verification (an in-run
assertion plus an external scan of every saved track's sample_ids
against samples_index.json's scene_name), run here against the real,
full 10-scene dataset.

Per v2_architecture_brief.md: do not tune anything to make fusion look
better. If fused MAE ends up above lidar's reference 3.54, that is
reported plainly, same as every other number here.
"""
import time

from v2 import adapters
from v2.central_tracker import CentralTracker
from v2.event_stream import merge_streams
from v2.evaluate import (
    compute_ground_truth_ttc,
    compute_length_bucket_metrics,
    compute_multi_sensor_composition,
    compute_sensor_metrics,
)

SENSOR_ORDER = ["lidar", "radar", "camera_mono", "fused"]

# Reference only (v2_architecture_brief.md) -- NOT a target to tune
# toward, just what this run's numbers should land near as a sanity
# check, and the number fused is being honestly compared against.
LIDAR_MAE_REFERENCE = 3.54


def _assert_no_track_spans_two_scenes(tracks, samples_index, run_name):
    """R9, checked independently of CentralTracker's own bookkeeping:
    for every track, look up each of its history sample_ids' REAL
    scene_token from samples_index.json and assert they are all the
    same. CentralTracker._reset_for_new_scene() should make a violation
    impossible by construction -- this is defense-in-depth verification
    against the real dataset, not a substitute for that design."""
    violations = []
    for track_id, track in tracks.items():
        scene_tokens = {samples_index[snap.sample_id]["scene_token"] for snap in track.history}
        if len(scene_tokens) > 1:
            violations.append((track_id, scene_tokens))
    assert not violations, (
        f"[{run_name}] {len(violations)} track(s) span more than one scene, e.g. {violations[0]} -- "
        "a track's history must never cross a scene boundary (R9)"
    )


def _run_tracker(label, events):
    print(f"\n--- Running {label} ---")
    t0 = time.time()
    tracker = CentralTracker()
    tracker.run(events)
    tracker.finalize()
    print(f"  {len(tracker.all_tracks())} tracks in {time.time() - t0:.1f}s")
    return tracker


def run():
    samples_index = adapters.load_samples_index()

    print("Loading ground truth and ego trajectory...")
    ego_by_sample = adapters.ego_pose_lookup()
    gt_trajectories, gt_by_sample = adapters.gt_annotations_lookup()
    gt_ttc_lookup = compute_ground_truth_ttc(gt_trajectories, ego_by_sample)

    # Materialize each sensor's events to a list EXACTLY ONCE. Each of
    # lidar_events()/radar_events()/camera_mono_events() is a generator
    # -- passing the SAME generator object to merge_streams() a second
    # time (after already consuming it for a solo run) would silently
    # yield nothing, since a generator can't be re-iterated. Lists can
    # be iterated any number of times, so materializing once here and
    # reusing the same list for both a sensor's solo run AND its place
    # in the fused merge is both correct and avoids doing every disk
    # read / NuScenes timestamp lookup twice.
    print("Materializing event streams (each generator consumed exactly once)...")
    t0 = time.time()
    lidar = list(adapters.lidar_events())
    radar = list(adapters.radar_events())
    camera_mono = list(adapters.camera_mono_events())
    print(f"  lidar={len(lidar)} radar={len(radar)} camera_mono={len(camera_mono)} "
          f"events materialized in {time.time() - t0:.1f}s")

    trackers = {
        "lidar": _run_tracker("lidar", lidar),
        "radar": _run_tracker("radar", radar),
        "camera_mono": _run_tracker("camera_mono", camera_mono),
    }
    trackers["fused"] = _run_tracker(
        "fused (lidar+radar+camera_mono)",
        merge_streams(lidar, radar, camera_mono),
    )

    print("\n--- R9 check: no track spans two scenes ---")
    for run_name in SENSOR_ORDER:
        _assert_no_track_spans_two_scenes(trackers[run_name].all_tracks(), samples_index, run_name)
    print(f"  OK -- no track in any of {len(SENSOR_ORDER)} runs spans more than one scene_token "
          f"(checked against the real {len(samples_index)}-sample dataset)")

    print("\n=== Per-sensor / fused metrics ===")
    header = f"  {'sensor':12s}  {'MAE':>6s}  {'RMSE':>6s}  {'match%':>7s}  {'n_tracks':>8s}  {'n_gt_matched':>12s}  {'n_pairs':>7s}"
    print(header)
    metrics_by_sensor = {}
    for sensor_name in SENSOR_ORDER:
        tracks = trackers[sensor_name].all_tracks()
        metrics = compute_sensor_metrics(sensor_name, tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample)
        metrics_by_sensor[sensor_name] = metrics
        print(f"  {sensor_name:12s}  {str(metrics['mae']):>6s}  {str(metrics['rmse']):>6s}  "
              f"{metrics['match_rate_pct']:>6.2f}%  {metrics['n_tracks_total']:>8d}  "
              f"{metrics['n_tracks_gt_matched']:>12d}  {metrics['n_matched_pairs']:>7d}")

    fused_mae = metrics_by_sensor["fused"]["mae"]
    if fused_mae is not None and fused_mae > LIDAR_MAE_REFERENCE:
        print(f"\n  NOTE: fused MAE ({fused_mae}) is ABOVE lidar's reference ({LIDAR_MAE_REFERENCE}) -- "
              "reported plainly, not tuned around. See the multi-sensor composition breakdown below.")
    elif fused_mae is not None:
        print(f"\n  NOTE: fused MAE ({fused_mae}) beats lidar's reference ({LIDAR_MAE_REFERENCE}).")

    print("\n=== Track-length bucket breakdown ===")
    for sensor_name in SENSOR_ORDER:
        tracks = trackers[sensor_name].all_tracks()
        buckets = compute_length_bucket_metrics(tracks, gt_by_sample, gt_ttc_lookup, ego_by_sample)
        print(f"  {sensor_name}:")
        for (lo, hi), b in buckets.items():
            label = f"[{lo},{hi if hi is not None else 'inf'})"
            print(f"    {label:10s}  n_tracks={b['n_tracks']:4d}  n_points={str(b['n_points']):>5s}  "
                  f"MAE={b['mae']}  RMSE={b['rmse']}")

    print("\n=== Multi-sensor composition (fused run only) ===")
    composition = compute_multi_sensor_composition(
        trackers["fused"].all_tracks(), gt_by_sample, gt_ttc_lookup, ego_by_sample
    )
    print(f"  multi-sensor tracks:  {composition['n_multi_sensor_tracks']} "
          f"({composition['multi_sensor_fraction_pct']}%)")
    print(f"  single-sensor tracks: {composition['n_single_sensor_tracks']}")
    print(f"  MAE multi-sensor:  {composition['mae_multi_sensor']}  "
          f"(n={composition['n_multi_sensor_matched_pairs']})")
    print(f"  MAE single-sensor: {composition['mae_single_sensor']}  "
          f"(n={composition['n_single_sensor_matched_pairs']})")

    return {
        "metrics": metrics_by_sensor,
        "composition": composition,
    }


if __name__ == "__main__":
    run()
