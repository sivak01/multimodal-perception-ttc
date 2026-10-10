"""
v2/eval_l3/build_h6_dataset_v2.py — builds the v2 per-key dataset H6
needs: one row per (bound, run, sample_id, instance_id) for run in
{lidar, radar, camera_mono, fused}, under BOTH matching bounds (rule
5), enriched with gt_context's pipeline-agnostic stratification
fields. Runs the four CentralTracker instances fresh (same class,
same adapters/event streams as v2/run_pipeline.py and
build_fused_dataset.py already use) -- does not modify
central_tracker.py, adapters.py, evaluate.py, or evaluate_matched.py.

n_sensors (corroboration count) is only meaningful for the fused run
here, and is derived the SAME independent way build_fused_dataset.py
already does: counting how many of the three solo single-sensor runs
have a candidate within MATCH_DIST_THRESHOLD_M of the real object at
that exact sample (see that module's own docstring for why this is
used instead of the fused tracker's whole-life is_multi_sensor flag).
"""
import time
from collections import defaultdict

from v2 import adapters
from v2.central_tracker import CentralTracker
from v2.event_stream import merge_streams
from v2.evaluate import match_track_to_ground_truth
from v2.evaluate_matched import candidates_from_tracker
from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3.gt_context import build_gt_context, enrich_key
from v2.eval_l3.h6_common import KeyRow
from v2.eval_l3.hungarian_match import match_samplewise_hungarian

RUNS = ("lidar", "radar", "camera_mono", "fused")


def _solo_candidates_cover(solo_candidates_by_sample, sample_id, gx, gy, match_dist_threshold):
    import math
    for cand in solo_candidates_by_sample.get(sample_id, []):
        if math.hypot(cand.x - gx, cand.y - gy) <= match_dist_threshold:
            return True
    return False


def build(print_progress=True):
    def log(msg):
        if print_progress:
            print(msg)

    log("Building GT context (shared with the v1 builder)...")
    ctx = build_gt_context()

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

    trackers = {
        "lidar": run_tracker("lidar", lidar),
        "radar": run_tracker("radar", radar),
        "camera_mono": run_tracker("camera_mono", camera_mono),
    }
    trackers["fused"] = run_tracker("fused", merge_streams(lidar, radar, camera_mono))

    target_sample_ids = sorted(
        (sid for sid in ctx.gt_by_sample if sid in ctx.ego_by_sample),
        key=lambda sid: ctx.ego_by_sample[sid].t,
    )

    log("Building per-run candidates (for Hungarian matching + fused corroboration counting)...")
    candidates_by_run = {
        run_name: candidates_from_tracker(tracker.all_tracks(), target_sample_ids, ctx.ego_by_sample)
        for run_name, tracker in trackers.items()
    }
    solo_candidates = {k: v for k, v in candidates_by_run.items() if k != "fused"}

    rows = []

    log("Matching bound 1/2: per-frame Hungarian...")
    for run_name in RUNS:
        matched = match_samplewise_hungarian(
            run_name, candidates_by_run[run_name], ctx.gt_by_sample, ctx.gt_ttc_lookup,
            cfg.MATCH_DIST_THRESHOLD_M,
        )
        for r in matched:
            enriched = enrich_key(ctx, r.sample_id, r.instance_id)
            if enriched is None:
                continue
            n_sensors = None
            if run_name == "fused":
                gx, gy = ctx.gt_by_sample_pos[(r.sample_id, r.instance_id)]
                n_sensors = sum(
                    1 for sc in solo_candidates.values()
                    if _solo_candidates_cover(sc, r.sample_id, gx, gy, cfg.MATCH_DIST_THRESHOLD_M)
                )
            rows.append(KeyRow(
                bound="hungarian", run=run_name, scene_id=enriched["scene_id"],
                sample_id=r.sample_id, instance_id=r.instance_id,
                pred_ttc=r.pred_ttc, gt_ttc=r.gt_ttc,
                range_bucket=enriched["range_bucket"], class_group=enriched["class_group"],
                motion=enriched["motion"], odd=enriched["odd"], in_corridor=enriched["in_corridor"],
                n_sensors=n_sensors,
            ))

    log("Matching bound 2/2: lock-once...")
    for run_name in RUNS:
        for track in trackers[run_name].all_tracks().values():
            pairs = match_track_to_ground_truth(track.history, ctx.gt_by_sample, ctx.gt_ttc_lookup, ctx.ego_by_sample)
            for p in pairs:
                enriched = enrich_key(ctx, p["sample_id"], p["instance_id"])
                if enriched is None:
                    continue
                n_sensors = None
                if run_name == "fused":
                    gx, gy = ctx.gt_by_sample_pos[(p["sample_id"], p["instance_id"])]
                    n_sensors = sum(
                        1 for sc in solo_candidates.values()
                        if _solo_candidates_cover(sc, p["sample_id"], gx, gy, cfg.MATCH_DIST_THRESHOLD_M)
                    )
                rows.append(KeyRow(
                    bound="lock_once", run=run_name, scene_id=enriched["scene_id"],
                    sample_id=p["sample_id"], instance_id=p["instance_id"],
                    pred_ttc=p["pred_ttc"], gt_ttc=p["gt_ttc"],
                    range_bucket=enriched["range_bucket"], class_group=enriched["class_group"],
                    motion=enriched["motion"], odd=enriched["odd"], in_corridor=enriched["in_corridor"],
                    n_sensors=n_sensors,
                ))

    log(f"  {len(rows)} total v2 key rows")
    return rows
