"""
v2/eval_l3/ground_truth_extra.py — supplementary real-dataset lookups
v2/adapters.py does not build (category_name per GT key, each scene's
own free-text description for ODD classification). Kept as new,
additive lookups here rather than changing adapters.py's tested return
shapes (same convention v2/coverage_model.py already established for
num_lidar_pts/num_radar_pts).
"""
from v2 import adapters


def build_category_lookup():
    """{(sample_id, instance_id): category_name} for every real
    sample_annotation -- one small pass over the same records
    adapters.gt_annotations_lookup() and coverage_model.py's
    build_annotation_point_count_lookup() already walk."""
    samples_index = adapters.load_samples_index()
    nusc = adapters._get_nuscenes()
    lookup = {}
    for sample_id in adapters._sample_ids_sorted():
        info = samples_index[sample_id]
        sample = nusc.get("sample", info["sample_token"])
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            lookup[(sample_id, ann["instance_token"])] = ann["category_name"]
    return lookup


def build_scene_odd_lookup():
    """{scene_token: scene_description} -- raw descriptions, one per
    real scene present in samples_index.json. stratify.classify_odd()
    turns this into day/night/rain; kept separate so the raw text is
    available for the verdict doc's own limitations section."""
    samples_index = adapters.load_samples_index()
    nusc = adapters._get_nuscenes()
    scene_tokens = {info["scene_token"] for info in samples_index.values()}
    return {st: nusc.get("scene", st)["description"] for st in scene_tokens}
