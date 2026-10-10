"""
v2/eval_l3/build_h6_dataset_v1.py — builds the v1 per-key dataset H6
needs, from v1's EXISTING output/step_5/ttc_<sensor>.csv files only
(no v1 notebook is read, modified, or re-run). Same KeyRow shape and
same gt_context enrichment as build_h6_dataset_v2.py, so both
pipelines' slices mean exactly the same thing -- but written to a
SEPARATE file and never joined with v2's rows (rule 2: never compare
v1 numbers with v2 numbers).

Hungarian bound: v1's own 'ttc' column is used as pred_ttc directly (v1
already computed it with its own UKF/CTRV filter) -- this module
doesn't re-derive TTC, only re-matches v1's existing (x, y, ttc) rows
against GT per-sample, same as evaluate_matched.score_v1_outputs()
already does with the greedy matcher (this uses the Hungarian-optimal
one instead, for the "optimistic" bound).

Lock-once bound: v1_adapters.match_v1_lock_once(), which reuses
v2.evaluate.match_track_to_ground_truth() unchanged against v1's own
track-id ('fused_id' column) groupings.

n_sensors: v1's OWN real composition from
v1_adapters.build_v1_composition_lookup() (output/step_4/
fused_tracks_all.csv's 'sensors' column) -- only meaningful for the
fused run.
"""
import config as legacy_config   # repo-root config.py: paths only
from v2.evaluate_matched import candidates_from_v1_csv
from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3.gt_context import build_gt_context, enrich_key
from v2.eval_l3.h6_common import KeyRow
from v2.eval_l3.hungarian_match import match_samplewise_hungarian
from v2.eval_l3.v1_adapters import match_v1_lock_once, build_v1_composition_lookup

RUNS = ("lidar", "radar", "camera_mono", "fused")


def build(print_progress=True):
    def log(msg):
        if print_progress:
            print(msg)

    log("Building GT context (shared with the v2 builder)...")
    ctx = build_gt_context()

    log("Loading v1's real fusion composition (output/step_4/fused_tracks_all.csv)...")
    composition_lookup = build_v1_composition_lookup()

    rows = []

    for run_name in RUNS:
        csv_path = legacy_config.STEP5_DIR / f"ttc_{run_name}.csv"
        if not csv_path.exists():
            log(f"  (skipping {run_name}: {csv_path} not found)")
            continue

        log(f"--- {run_name}: Hungarian bound ---")
        candidates_by_sample = candidates_from_v1_csv(csv_path)
        matched = match_samplewise_hungarian(
            run_name, candidates_by_sample, ctx.gt_by_sample, ctx.gt_ttc_lookup,
            cfg.MATCH_DIST_THRESHOLD_M,
        )
        for r in matched:
            enriched = enrich_key(ctx, r.sample_id, r.instance_id)
            if enriched is None:
                continue
            n_sensors = composition_lookup.get((r.sample_id, r.track_id)) if run_name == "fused" else None
            rows.append(KeyRow(
                bound="hungarian", run=run_name, scene_id=enriched["scene_id"],
                sample_id=r.sample_id, instance_id=r.instance_id,
                pred_ttc=r.pred_ttc, gt_ttc=r.gt_ttc,
                range_bucket=enriched["range_bucket"], class_group=enriched["class_group"],
                motion=enriched["motion"], odd=enriched["odd"], in_corridor=enriched["in_corridor"],
                n_sensors=n_sensors,
            ))

        log(f"--- {run_name}: lock-once bound ---")
        pairs = match_v1_lock_once(csv_path, ctx.gt_by_sample, ctx.gt_ttc_lookup, ctx.ego_by_sample,
                                    match_dist_threshold=cfg.MATCH_DIST_THRESHOLD_M)
        for p in pairs:
            enriched = enrich_key(ctx, p["sample_id"], p["instance_id"])
            if enriched is None:
                continue
            n_sensors = composition_lookup.get((p["sample_id"], p["track_id"])) if run_name == "fused" else None
            rows.append(KeyRow(
                bound="lock_once", run=run_name, scene_id=enriched["scene_id"],
                sample_id=p["sample_id"], instance_id=p["instance_id"],
                pred_ttc=p["pred_ttc"], gt_ttc=p["gt_ttc"],
                range_bucket=enriched["range_bucket"], class_group=enriched["class_group"],
                motion=enriched["motion"], odd=enriched["odd"], in_corridor=enriched["in_corridor"],
                n_sensors=n_sensors,
            ))

    log(f"  {len(rows)} total v1 key rows")
    return rows
