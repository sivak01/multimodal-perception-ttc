"""
v2/eval_l3/ego_heading.py — per-sample ego vehicle YAW (body heading),
needed for H1's in-corridor check (stratify.ego_frame_offset()) but not
built by v2/adapters.ego_pose_lookup(), which only keeps (x, y, vx, vy)
-- no rotation -- since v2's own tracker/TTC math never needs ego
orientation, only ego position/velocity.

Reads the SAME lidar_meta.json file adapters.ego_pose_lookup() already
reads (one real ego_pose per sample), just additionally pulling its
'rotation' quaternion -- a new, additive lookup, not a change to that
function's tested return shape (same convention as coverage_model.py /
ground_truth_extra.py).
"""
import json

import config as legacy_config
from pyquaternion import Quaternion

from v2 import adapters


def build_ego_heading_lookup():
    """{sample_id: yaw_radians} -- ego's own body heading in the global
    frame (same frame src/geometry.py's transforms and every tracked
    position in this project already use)."""
    samples_index = adapters.load_samples_index()
    lookup = {}
    for sample_id in adapters._sample_ids_sorted():
        meta_path = legacy_config.STEP1_DIR / "lidar" / sample_id / "lidar_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        yaw, _pitch, _roll = Quaternion(meta["ego_pose"]["rotation"]).yaw_pitch_roll
        lookup[sample_id] = yaw
    return lookup
