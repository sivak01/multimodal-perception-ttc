"""
v2/eval_l3/stratify.py — pure classification functions used to stratify
every eval_l3 metric (rule 6). No I/O here by design, so every function
is hand-checkably unit-testable; the modules that fetch real data
(ground_truth_extra.py, ego_heading.py) are kept separate.
"""
import math

from v2.eval_l3 import config_l3 as cfg

VRU_CATEGORIES_EXACT = {"vehicle.bicycle", "vehicle.motorcycle"}
VRU_CATEGORY_PREFIX = "human.pedestrian."
VEHICLE_CATEGORY_PREFIX = "vehicle."


def class_group(category_name):
    """'vru' (pedestrian/bicycle/motorcycle), 'vehicle' (car/truck/bus/
    construction/trailer), or 'other' (movable_object.*,
    static_object.*, animal -- not part of this project's tracked
    classes at all, see repo-root CLAUDE.md's YOLO class filter)."""
    if category_name.startswith(VRU_CATEGORY_PREFIX) or category_name in VRU_CATEGORIES_EXACT:
        return "vru"
    if category_name.startswith(VEHICLE_CATEGORY_PREFIX):
        return "vehicle"
    return "other"


def moving_or_parked(gt_speed_mps, moving_threshold=cfg.MOVING_SPEED_MPS):
    if gt_speed_mps is None:
        return None
    return "moving" if gt_speed_mps > moving_threshold else "parked"


def range_bucket(range_m, buckets=None):
    """(lo, hi) tuple the range falls into, per RANGE_BUCKETS_M (half-
    open, lo inclusive, hi exclusive; the last bucket's hi is inf)."""
    buckets = buckets if buckets is not None else cfg.RANGE_BUCKETS_M
    for lo, hi in buckets:
        if range_m >= lo and range_m < hi:
            return (lo, hi)
    return buckets[-1]


def range_bucket_label(bucket):
    lo, hi = bucket
    return f"[{lo},{'inf' if math.isinf(hi) else hi})"


def ttc_band(gt_ttc_s, bands=None):
    """Which of CRITICAL_TTC_BANDS_S a FINITE gt_ttc falls into, or
    None if gt_ttc is inf (not closing) or falls above the last band."""
    if not math.isfinite(gt_ttc_s):
        return None
    bands = bands if bands is not None else cfg.CRITICAL_TTC_BANDS_S
    for lo, hi in bands:
        if gt_ttc_s >= lo and gt_ttc_s < hi:
            return (lo, hi)
    return None


def in_h1_ttc_band(gt_ttc_s, lo=cfg.H1_TTC_BAND_LOW_S, hi=cfg.H1_TTC_BAND_HIGH_S):
    """H1 part (b)'s specific 0-5s filter -- a finite GT TTC in [lo, hi)."""
    return math.isfinite(gt_ttc_s) and lo <= gt_ttc_s < hi


def ego_frame_offset(ego_x, ego_y, ego_yaw_rad, obj_x, obj_y):
    """Rotates (obj - ego) into the ego's own forward/lateral frame.
    forward: + ahead of the ego's own heading (its direction of
    travel). lateral: + to the ego's left, by the standard
    right-handed convention (matches src/geometry.py's own world
    frame -- see ego_heading.py for how ego_yaw_rad is derived).
    Returns (forward_m, lateral_m)."""
    dx, dy = obj_x - ego_x, obj_y - ego_y
    cos_yaw, sin_yaw = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
    forward = dx * cos_yaw + dy * sin_yaw
    lateral = -dx * sin_yaw + dy * cos_yaw
    return forward, lateral


def in_corridor(forward_m, lateral_m,
                 half_width=cfg.CORRIDOR_HALF_WIDTH_M, max_range=cfg.CORRIDOR_MAX_RANGE_M):
    """'in ego path': ahead of the car (forward >= 0), within
    max_range, and within half_width of the centerline laterally."""
    return 0.0 <= forward_m <= max_range and abs(lateral_m) <= half_width


def classify_odd(scene_description):
    """'rain', 'night', or 'day' from a scene's free-text description
    keywords -- rain takes priority over night when a description
    mentions both (e.g. nuScenes-mini's own scene-1094, "Night, after
    rain, ..."), since rain is the rarer and arguably more operationally
    distinct condition; stated here rather than left as an implicit
    tie-break. Real nuScenes-mini scene descriptions verified against
    this function: 7 day, 2 night (scene-1077, scene-1100), 1 rain
    (scene-1094, which also mentions night)."""
    text = scene_description.lower()
    if "rain" in text:
        return "rain"
    if "night" in text:
        return "night"
    return "day"
