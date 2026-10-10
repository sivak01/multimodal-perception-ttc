"""
v2/eval_l3/gt_context.py — the one shared bundle of GT/dataset-only
lookups (NOT pipeline output) that both the v1 and v2 H6 dataset
builders need: ego pose/heading, GT trajectories/TTC, GT category,
scene ODD. Building this once and sharing it between the v1 and v2
builders keeps their enrichment (range/corridor/class/motion/odd)
IDENTICAL by construction, which is what rule 2's "same GT keys"
fairness requirement needs -- a key enriched two different ways in the
two pipelines would make a within-pipeline slice silently inconsistent
with what the other pipeline means by the same slice name.
"""
import math
from collections import namedtuple

from v2 import adapters
from v2.evaluate import compute_ground_truth_ttc
from v2.evaluate_matched import build_gt_speed_lookup
from v2.eval_l3 import stratify
from v2.eval_l3.ego_heading import build_ego_heading_lookup
from v2.eval_l3.ground_truth_extra import build_category_lookup, build_scene_odd_lookup

GTContext = namedtuple("GTContext", [
    "samples_index", "ego_by_sample", "gt_by_sample", "gt_ttc_lookup",
    "gt_speed_lookup", "category_lookup", "scene_odd_lookup", "ego_heading_lookup",
    "gt_by_sample_pos",
])


def build_gt_context():
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

    return GTContext(
        samples_index=samples_index, ego_by_sample=ego_by_sample,
        gt_by_sample=gt_by_sample, gt_ttc_lookup=gt_ttc_lookup,
        gt_speed_lookup=gt_speed_lookup, category_lookup=category_lookup,
        scene_odd_lookup=scene_odd_lookup, ego_heading_lookup=ego_heading_lookup,
        gt_by_sample_pos=gt_by_sample_pos,
    )


def enrich_key(ctx, sample_id, instance_id):
    """Returns a dict of GT-side-only fields for one (sample_id,
    instance_id) key, or None if the key/sample can't be resolved
    (e.g. no ego state at that sample). Identical for v1 and v2 -- no
    pipeline-specific input here at all."""
    pos = ctx.gt_by_sample_pos.get((sample_id, instance_id))
    ego = ctx.ego_by_sample.get(sample_id)
    if pos is None or ego is None:
        return None
    gx, gy = pos
    range_m = math.hypot(gx - ego.x, gy - ego.y)

    yaw = ctx.ego_heading_lookup.get(sample_id)
    in_corridor = None
    if yaw is not None:
        fwd, lat = stratify.ego_frame_offset(ego.x, ego.y, yaw, gx, gy)
        in_corridor = stratify.in_corridor(fwd, lat)

    category = ctx.category_lookup.get((sample_id, instance_id))
    class_group = stratify.class_group(category) if category is not None else None

    scene_id = ctx.samples_index[sample_id]["scene_name"]
    scene_token = ctx.samples_index[sample_id]["scene_token"]
    odd = stratify.classify_odd(ctx.scene_odd_lookup.get(scene_token, ""))

    gt_speed = ctx.gt_speed_lookup.get((sample_id, instance_id))
    motion = stratify.moving_or_parked(gt_speed)

    return {
        "scene_id": scene_id, "range_m": round(range_m, 2),
        "range_bucket": stratify.range_bucket(range_m),
        "class_group": class_group, "motion": motion, "odd": odd,
        "in_corridor": in_corridor,
    }
