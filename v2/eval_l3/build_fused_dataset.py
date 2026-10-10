"""
v2/eval_l3/build_fused_dataset.py — builds the enriched per-GT-key
dataset H1 (and later eval_l3 hypotheses) run against: one row per
(bound, sample_id, instance_id) for the FUSED run, carrying its
predicted TTC under BOTH matching bounds (rule 5) plus every
stratification field (rule 6) and the merged/single composition label.

Composition label, decided independently of the fused tracker's own
bookkeeping (rule 1 forbids touching central_tracker.py, and
Track.history/TrackSnapshot do not record which sensor produced each
individual past snapshot anyway -- only the track-level, whole-life
`sensors_seen` set, which would leak FUTURE corroboration onto a PAST
instant). Instead: independently run the three single-sensor solo
trackers (lidar, radar, camera_mono -- the exact same CentralTracker
class, same as v2/run_pipeline.py already does), and for each GT key,
count how many of those three solo runs have ANY candidate within
MATCH_DIST_THRESHOLD_M of that exact real object at that exact sample.
"merged" = 2 or 3 solo sensors independently had something there;
"single" = 0 or 1. This directly measures "was this real instant
actually corroborated by more than one sensor", with no dependency on
the fused tracker's own internal history bookkeeping.

Does not modify central_tracker.py, adapters.py, evaluate.py, or
evaluate_matched.py -- only imports and calls them.
"""
import math
import time
from collections import namedtuple

from v2 import adapters, config
from v2.central_tracker import CentralTracker
from v2.event_stream import merge_streams
from v2.evaluate import compute_ground_truth_ttc, match_track_to_ground_truth
from v2.evaluate_matched import candidates_from_tracker, build_gt_speed_lookup
from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3 import stratify
from v2.eval_l3.ego_heading import build_ego_heading_lookup
from v2.eval_l3.ground_truth_extra import build_category_lookup, build_scene_odd_lookup
from v2.eval_l3.hungarian_match import match_samplewise_hungarian

EnrichedKeyRow = namedtuple("EnrichedKeyRow", [
    "bound", "scene_id", "sample_id", "instance_id",
    "pred_ttc", "gt_ttc", "abs_error", "rel_error",
    "range_m", "range_bucket", "class_group", "motion", "odd",
    "in_corridor", "n_sensors_corroborating", "composition",
])


def _solo_candidates_cover(solo_candidates_by_sample, sample_id, gx, gy, match_dist_threshold):
    for cand in solo_candidates_by_sample.get(sample_id, []):
        if math.hypot(cand.x - gx, cand.y - gy) <= match_dist_threshold:
            return True
    return False


def _enrich_pairs(bound_name, pairs, gt_by_sample_pos, samples_index, category_lookup,
                   scene_odd_lookup, gt_speed_lookup, ego_by_sample, ego_heading_lookup,
                   solo_candidates):
    """pairs: iterable of dicts with at least sample_id, instance_id,
    pred_ttc, gt_ttc (the shape both match_samplewise_hungarian's
    MatchedKeyRow._asdict()-like access and evaluate.py's
    match_track_to_ground_truth() pair dicts already share)."""
    rows = []
    for p in pairs:
        sample_id, instance_id = p["sample_id"], p["instance_id"]
        gx, gy = gt_by_sample_pos.get((sample_id, instance_id), (None, None))
        if gx is None:
            continue

        ego = ego_by_sample.get(sample_id)
        if ego is None:
            continue
        range_m = math.hypot(gx - ego.x, gy - ego.y)

        yaw = ego_heading_lookup.get(sample_id)
        corridor = None
        if yaw is not None:
            fwd, lat = stratify.ego_frame_offset(ego.x, ego.y, yaw, gx, gy)
            corridor = stratify.in_corridor(fwd, lat)

        category = category_lookup.get((sample_id, instance_id))
        cgroup = stratify.class_group(category) if category is not None else None

        scene_id = samples_index[sample_id]["scene_name"]
        scene_token = samples_index[sample_id]["scene_token"]
        odd = stratify.classify_odd(scene_odd_lookup.get(scene_token, ""))

        gt_speed = gt_speed_lookup.get((sample_id, instance_id))
        motion = stratify.moving_or_parked(gt_speed)

        n_corroborating = sum(
            1 for sensor_cands in solo_candidates.values()
            if _solo_candidates_cover(sensor_cands, sample_id, gx, gy, cfg.MATCH_DIST_THRESHOLD_M)
        )
        composition = "merged" if n_corroborating >= 2 else "single"

        pred_ttc, gt_ttc = p["pred_ttc"], p["gt_ttc"]
        if math.isfinite(pred_ttc) and math.isfinite(gt_ttc):
            abs_error = abs(pred_ttc - gt_ttc)
            rel_error = abs_error / gt_ttc if gt_ttc != 0 else None
        else:
            abs_error, rel_error = None, None

        rows.append(EnrichedKeyRow(
            bound=bound_name, scene_id=scene_id, sample_id=sample_id, instance_id=instance_id,
            pred_ttc=pred_ttc, gt_ttc=gt_ttc, abs_error=abs_error, rel_error=rel_error,
            range_m=round(range_m, 2), range_bucket=stratify.range_bucket(range_m),
            class_group=cgroup, motion=motion, odd=odd, in_corridor=corridor,
            n_sensors_corroborating=n_corroborating, composition=composition,
        ))
    return rows


def build(print_progress=True):
    def log(msg):
        if print_progress:
            print(msg)

    log("Loading ground truth, ego trajectory, and supplementary lookups...")
    samples_index = adapters.load_samples_index()
    ego_by_sample = adapters.ego_pose_lookup()
    gt_trajectories, gt_by_sample = adapters.gt_annotations_lookup()
    gt_ttc_lookup = compute_ground_truth_ttc(gt_trajectories, ego_by_sample)
    gt_speed_lookup = build_gt_speed_lookup(gt_trajectories)
    category_lookup = build_category_lookup()
    scene_odd_lookup = build_scene_odd_lookup()
    ego_heading_lookup = build_ego_heading_lookup()

    gt_by_sample_pos = {}
    for sample_id, gts in gt_by_sample.items():
        for instance_id, gx, gy in gts:
            gt_by_sample_pos[(sample_id, instance_id)] = (gx, gy)

    log("Materializing event streams...")
    t0 = time.time()
    lidar = list(adapters.lidar_events())
    radar = list(adapters.radar_events())
    camera_mono = list(adapters.camera_mono_events())
    log(f"  lidar={len(lidar)} radar={len(radar)} camera_mono={len(camera_mono)} ({time.time() - t0:.1f}s)")

    def run_tracker(label, events):
        log(f"--- Running {label} ---")
        t0 = time.time()
        tracker = CentralTracker()
        tracker.run(events)
        tracker.finalize()
        log(f"  {len(tracker.all_tracks())} tracks ({time.time() - t0:.1f}s)")
        return tracker

    lidar_tracker = run_tracker("lidar", lidar)
    radar_tracker = run_tracker("radar", radar)
    camera_mono_tracker = run_tracker("camera_mono", camera_mono)
    fused_tracker = run_tracker("fused", merge_streams(lidar, radar, camera_mono))

    target_sample_ids = sorted(
        (sid for sid in gt_by_sample if sid in ego_by_sample),
        key=lambda sid: ego_by_sample[sid].t,
    )

    log("Building solo-run candidates (for independent merged/single corroboration counting)...")
    solo_candidates = {
        "lidar": candidates_from_tracker(lidar_tracker.all_tracks(), target_sample_ids, ego_by_sample),
        "radar": candidates_from_tracker(radar_tracker.all_tracks(), target_sample_ids, ego_by_sample),
        "camera_mono": candidates_from_tracker(camera_mono_tracker.all_tracks(), target_sample_ids, ego_by_sample),
    }

    log("Matching bound 1/2: per-frame Hungarian (optimistic)...")
    fused_candidates = candidates_from_tracker(fused_tracker.all_tracks(), target_sample_ids, ego_by_sample)
    hungarian_rows_raw = match_samplewise_hungarian(
        "fused", fused_candidates, gt_by_sample, gt_ttc_lookup, cfg.MATCH_DIST_THRESHOLD_M,
    )
    hungarian_pairs = [r._asdict() for r in hungarian_rows_raw]

    log("Matching bound 2/2: lock-once (pessimistic, v2/evaluate.py's own scheme)...")
    lock_once_pairs = []
    for track in fused_tracker.all_tracks().values():
        lock_once_pairs.extend(
            match_track_to_ground_truth(track.history, gt_by_sample, gt_ttc_lookup, ego_by_sample)
        )

    log("Enriching both bounds with stratification + composition labels...")
    enriched = []
    enriched.extend(_enrich_pairs(
        "hungarian", hungarian_pairs, gt_by_sample_pos, samples_index, category_lookup,
        scene_odd_lookup, gt_speed_lookup, ego_by_sample, ego_heading_lookup, solo_candidates,
    ))
    enriched.extend(_enrich_pairs(
        "lock_once", lock_once_pairs, gt_by_sample_pos, samples_index, category_lookup,
        scene_odd_lookup, gt_speed_lookup, ego_by_sample, ego_heading_lookup, solo_candidates,
    ))
    log(f"  {len(enriched)} enriched rows total ({len(hungarian_pairs)} hungarian + {len(lock_once_pairs)} lock_once)")
    return enriched


def write_csv(rows, path):
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(EnrichedKeyRow._fields)
        for r in rows:
            writer.writerow(list(r))
