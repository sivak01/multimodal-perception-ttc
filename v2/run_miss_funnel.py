"""
v2/run_miss_funnel.py — H3 (revised) driver, docs/hypotheses-2026-10-10.md.
Builds the expected -> detected -> tracked -> fused miss funnel per
(GT key, sensor in {lidar, radar, camera_mono}), against the CURRENTLY
CONFIGURED dataset (config.py's DATA_ROOT/NUSCENES_VERSION — trainval
as of this run, per this session's explicit choice to fix the "mini
result inflated" concern rather than reproduce it).

Run with: python -m v2.run_miss_funnel > v2_miss_funnel_report.txt
Writes v2_miss_funnel.csv alongside it.

Memory-conscious by design, after four real OOM kills on this 16GB
machine running earlier, progressively-less-naive versions of this
script. Three fixes compound:
  1. The funnel rows are streamed straight to CSV
     (miss_funnel.write_funnel_csv_and_summarize()) rather than
     materialized as one big list first — on the real dataset that's
     450K+ rows.
  2. Each solo tracker's heavy internal state (its full Track/history
     objects, up to ~168K tracks for the fused run) is freed
     immediately after extracting the lightweight per-sample
     candidate list this script needs from it, instead of holding all
     4 trackers resident simultaneously.
  3. adapters.lidar_events()/radar_events()/camera_mono_events() are
     true generators (confirmed by reading adapters.py directly —
     they `yield` per sample, re-reading already-extracted JSON files
     off disk, not pre-loading anything) — this script never
     materializes them into a persistent list at all. Each is called
     FRESH wherever it's needed (once for its raw-candidate grouping,
     once for its solo tracker run, and lidar/radar/camera_mono each
     once more inside the fused merge_streams() call) — trading some
     extra disk I/O (small local JSON files, re-read 2-3x per sensor)
     for a much smaller peak memory footprint, since no sensor's full
     event list is ever resident as a persistent Python object.
"""
import gc
import time

from v2 import adapters, miss_funnel
from v2.central_tracker import CentralTracker
from v2.event_stream import merge_streams
from v2.evaluate_matched import candidates_from_tracker
import config as legacy_config


def _tracked_tracks(tracker):
    """Tracks that survived with 2+ real updates -- CentralTracker's
    own all_tracks() includes every track ever created, even 1-hit
    ones immediately evicted (no length filter exists anywhere in
    central_tracker.py). This filter is what makes "tracked" a
    genuinely distinct funnel stage from "detected"."""
    return {tid: t for tid, t in tracker.all_tracks().items() if len(t.history) >= 2}


def _solo_candidates(label, events, all_sample_ids, ego_by_sample):
    """Runs one solo CentralTracker on `events`, extracts its
    tracked-stage candidates, then frees the tracker itself (its Track
    objects, each with a full KF + history) before returning -- so at
    most one solo tracker's full state is resident at a time, instead
    of all three (lidar/radar/camera_mono) simultaneously."""
    print(f"--- Running {label} ---")
    t0 = time.time()
    tracker = CentralTracker()
    tracker.run(events)
    tracker.finalize()
    n_tracks = len(tracker.all_tracks())
    print(f"  {n_tracks} tracks ({time.time() - t0:.1f}s)")
    candidates = candidates_from_tracker(_tracked_tracks(tracker), all_sample_ids, ego_by_sample)
    del tracker
    gc.collect()
    return candidates


def run():
    print(f"Dataset: {legacy_config.NUSCENES_VERSION} at {legacy_config.DATA_ROOT}")
    print("Loading ground truth, ego trajectory, and per-annotation lookups...")
    ego_by_sample = adapters.ego_pose_lookup()
    gt_trajectories, gt_by_sample = adapters.gt_annotations_lookup()
    ann_lookup = miss_funnel.build_annotation_lookup()

    all_sample_ids = sorted((sid for sid in gt_by_sample if sid in ego_by_sample),
                             key=lambda sid: ego_by_sample[sid].t)

    EVENT_FNS = {"lidar": adapters.lidar_events, "radar": adapters.radar_events,
                 "camera_mono": adapters.camera_mono_events}

    raw_candidates_by_sample = {}
    tracked_candidates_by_sample = {}
    for sensor, event_fn in EVENT_FNS.items():
        print(f"Building {sensor} detected-stage (raw event) candidates...")
        t0 = time.time()
        raw_candidates_by_sample[sensor] = miss_funnel.raw_event_candidates_by_sample(event_fn())
        n_events = sum(len(v) for v in raw_candidates_by_sample[sensor].values())
        print(f"  {n_events} events grouped ({time.time() - t0:.1f}s)")

        tracked_candidates_by_sample[sensor] = _solo_candidates(sensor, event_fn(), all_sample_ids, ego_by_sample)
        gc.collect()

    print("Running fused tracker (lidar+radar+camera_mono merged stream, fresh generators)...")
    fused_candidates_by_sample = _solo_candidates(
        "fused", merge_streams(adapters.lidar_events(), adapters.radar_events(), adapters.camera_mono_events()),
        all_sample_ids, ego_by_sample,
    )
    gc.collect()

    print("Streaming funnel rows to CSV + accumulating summary (single pass, no full row list)...")
    rows_gen = miss_funnel.build_funnel_rows(
        gt_by_sample, ann_lookup, ego_by_sample,
        raw_candidates_by_sample, tracked_candidates_by_sample, fused_candidates_by_sample,
    )
    summaries = miss_funnel.write_funnel_csv_and_summarize(rows_gen, "v2_miss_funnel.csv")
    print(f"  wrote v2_miss_funnel.csv")

    print("\n=== H3 (revised): miss funnel, expected -> detected -> tracked -> fused ===")
    for sensor in miss_funnel.SENSORS:
        s = summaries[sensor]
        print(f"  {sensor}: n_total_gt_keys={s['n_total_gt_keys']}  n_expected={s['n_expected']}")
        print(f"    detected: {s['pct_detected_of_expected']}% of expected")
        print(f"    tracked:  {s['pct_tracked_of_expected']}% of expected")
        print(f"    fused:    {s['pct_fused_of_expected']}% of expected")
        print(f"    median range_m of detection-stage misses: {s['median_range_m_missed_at_detection']}")

    return summaries


if __name__ == "__main__":
    run()
