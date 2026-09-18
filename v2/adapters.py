"""
v2/adapters.py — bridges the EXISTING pipeline's on-disk Step 0-2
outputs into SensorEvent streams. Per v2_architecture_brief.md, this is
the only v2/ file that touches existing notebook outputs (or the
NuScenes devkit) at all; every other v2/ module is data-source-agnostic.

Radar timestamp gap (found while writing this file, reported before
fixing it -- see the conversation this was built in): Step 1.2 (radar
parsing) never captured each radar channel's own native sample_data
timestamp anywhere in its on-disk output (not in the per-channel JSON,
not in radar_parsed_data.csv) -- only camera_meta.json (Step 1.1) did
that for cameras. Approximating radar's timestamp with its parent
sample's shared timestamp_us was rejected: that is exactly the
per-sensor-vs-shared-timestamp gap R2 exists to eliminate, and doing it
silently for radar alone would undermine the design for one of only
three sensors here. Modifying Step 1.2 to add the missing field was
also rejected (out of scope without asking, for no benefit the option
below doesn't already provide).

Fix: every timestamp used in this file -- lidar, radar, AND
camera_mono -- is looked up directly via the NuScenes devkit:
sample_token -> nusc.sample['data'][channel] -> nusc.sample_data
['timestamp']. This is the exact chain Step 0 already uses once per
sample (picking an arbitrary channel) to build samples_index.json;
here it is applied per-channel, uniformly across all three sensors,
rather than trusting camera_meta.json's already-correct field for
camera and inventing a different method for lidar. Verified against
sample_0000: RADAR_FRONT's real sample_data timestamp differs from the
sample's shared timestamp_us by +16227us (~16ms) -- a small, physically
plausible offset, not zero (proving this is not silently falling back
to the shared timestamp) and not implausibly large.

Per-scan radar clustering (added after the first cut of this file
yielded one event per raw point): lidar arrives already one-centroid-
per-object (Step 2.1's DBSCAN) and camera_mono arrives already one-box-
per-object (YOLO), but raw radar points do NOT -- one real object
triggers 3-8 separate echoes within a single channel's single scan.
Left unclustered, this design's single-event-at-a-time
CentralTracker.process_event() would feed near-simultaneous same-scan
blips in as if they were repeated noisy measurements of ONE position,
when they are actually different points on one object's surface -- a
plausible, previously-unexplained contributor to radar's 2.4x
fragmentation ratio in the old pipeline's audit. Fixed with the same
DBSCAN-centroid approach the old pipeline's Step 3.2 used
(config.RADAR_CLUSTER_EPS_M, min_samples=1), but applied per-channel-
per-scan here rather than pooled across all 5 channels -- pooling
across channels would mix points from slightly different real
timestamps, which conflicts with this design's whole point (R2: each
channel is its own asynchronously-timestamped stream).
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

import config as legacy_config  # the EXISTING repo-root config.py (paths only) -- NOT v2/config.py
from src.geometry import point_to_global
from v2 import config as v2_config
from v2.event_stream import SensorEvent

_nusc = None


def _get_nuscenes():
    """Lazy singleton -- NuScenes() takes ~1s to load; only pay that
    cost once even if all three generators run in the same process."""
    global _nusc
    if _nusc is None:
        from nuscenes.nuscenes import NuScenes
        _nusc = NuScenes(version=legacy_config.NUSCENES_VERSION,
                          dataroot=str(legacy_config.DATA_ROOT), verbose=False)
    return _nusc


def load_samples_index():
    path = legacy_config.STEP0_DIR / "samples_index.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _channel_timestamp_s(sample_token, channel):
    """This ONE channel's own real native capture time, in seconds --
    see the module docstring for why this doesn't just read
    timestamp_us off samples_index.json."""
    nusc = _get_nuscenes()
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"][channel])
    return sd["timestamp"] / 1e6


def _cluster_scan_points(points):
    """Groups ONE channel's ONE scan's raw radar points within
    config.RADAR_CLUSTER_EPS_M of each other into one representative
    point per real object (the cluster's own centroid, including its
    z) -- min_samples=1 so an isolated blip still becomes its own
    single-point detection, same convention as the old pipeline. See
    module docstring for why this happens per-channel-per-scan rather
    than pooled across channels."""
    if len(points) <= 1:
        return points
    xyz = np.array([[p["x"], p["y"], p.get("z", 0.0)] for p in points])
    labels = DBSCAN(eps=v2_config.RADAR_CLUSTER_EPS_M, min_samples=1).fit_predict(xyz)
    clustered = []
    for label in sorted(set(labels)):
        centroid = xyz[labels == label].mean(axis=0)
        clustered.append({"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])})
    return clustered


def _sample_ids_sorted():
    """Sample folder names in the same order Step 0 assigned them --
    already chronological per scene (see samples_index.json's own
    prev_token/next_token chain), so iterating in this order and
    sorting only WITHIN one sample's own handful of channels/clusters
    is enough to keep each returned generator's overall .t sequence
    non-decreasing, without ever materializing the whole dataset at
    once (heapq.merge, event_stream.py's docstring, needs exactly this:
    each stream already sorted, not necessarily eager)."""
    return sorted(load_samples_index().keys())


def lidar_events():
    """One SensorEvent per kept LiDAR cluster (output/step_2/lidar/
    <sample>/lidar_clusters.json -- already ground-removed, DBSCAN-
    clustered, and shape-filtered by Step 2.1), lifted from LIDAR_TOP's
    own local frame into the global frame via the real per-sample
    sensor_to_ego/ego_pose calibration Step 1.3 already recorded."""
    samples_index = load_samples_index()

    for sample_id in _sample_ids_sorted():
        info = samples_index[sample_id]
        sample_token, scene_token = info["sample_token"], info["scene_token"]

        clusters_path = legacy_config.STEP2_DIR / "lidar" / sample_id / "lidar_clusters.json"
        meta_path = legacy_config.STEP1_DIR / "lidar" / sample_id / "lidar_meta.json"
        if not clusters_path.exists() or not meta_path.exists():
            continue

        with open(clusters_path, encoding="utf-8") as f:
            clusters = json.load(f)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        if not clusters:
            continue

        t = _channel_timestamp_s(sample_token, "LIDAR_TOP")
        calibrated_sensor = {"translation": meta["sensor_to_ego_translation"],
                              "rotation": meta["sensor_to_ego_rotation"]}
        ego_pose = meta["ego_pose"]

        # all clusters in one LiDAR sweep share the same real capture
        # time, so no within-sample sort is needed here (unlike radar/
        # camera_mono, which have multiple channels with slightly
        # different real timestamps each)
        for cluster in clusters.values():
            gx, gy, _gz = point_to_global(cluster["centroid"], ego_pose, calibrated_sensor)
            yield SensorEvent(t=t, sensor="lidar", x=float(gx), y=float(gy),
                               sample_id=sample_id, scene_token=scene_token)


def radar_events():
    """One SensorEvent per already-decluttered radar point
    (output/step_2/radar/<sample>/<channel>.json -- Step 2.2's dyn_prop
    filter already keeps only {0, 2, 6}, i.e. moving/oncoming/crossing-
    moving), lifted from that channel's own local frame into the global
    frame via that channel's own per-sample calibration."""
    samples_index = load_samples_index()
    radar_dir = legacy_config.STEP2_DIR / "radar"

    for sample_id in _sample_ids_sorted():
        info = samples_index[sample_id]
        sample_token, scene_token = info["sample_token"], info["scene_token"]

        sample_dir = radar_dir / sample_id
        if not sample_dir.exists():
            continue

        channel_files = sorted(sample_dir.glob("*.json"))
        # this sample's channels, each timestamped -- sorted so this
        # sample's own contribution to the stream is non-decreasing
        # even though the 5 radar channels don't fire at exactly the
        # same instant
        per_channel = []
        for channel_path in channel_files:
            with open(channel_path, encoding="utf-8") as f:
                data = json.load(f)
            channel = data["radar_channel"]
            t = _channel_timestamp_s(sample_token, channel)
            per_channel.append((t, data))
        per_channel.sort(key=lambda pair: pair[0])

        for t, data in per_channel:
            calib = data["calibration"]
            calibrated_sensor = {"translation": calib["sensor_to_ego_translation"],
                                  "rotation": calib["sensor_to_ego_rotation"]}
            ego_pose = calib["ego_pose"]
            for point in _cluster_scan_points(data["points"]):
                gx, gy, _gz = point_to_global((point["x"], point["y"], point["z"]), ego_pose, calibrated_sensor)
                yield SensorEvent(t=t, sensor="radar", x=float(gx), y=float(gy),
                                   sample_id=sample_id, scene_token=scene_token)


def camera_mono_events():
    """One SensorEvent per YOLO detection with a valid monocular 3D
    lift (output/step_2/yolo_mono/<sample>/<camera>.json, Step 2.3.2 --
    similar-triangles depth-from-height, no LiDAR). These are already
    in the global frame (global_x/global_y), so unlike lidar/radar no
    point_to_global() call is needed here -- only the real per-channel
    timestamp lookup, since yolo_mono's own JSON doesn't carry one."""
    samples_index = load_samples_index()
    yolo_mono_dir = legacy_config.STEP2_DIR / "yolo_mono"

    for sample_id in _sample_ids_sorted():
        info = samples_index[sample_id]
        sample_token, scene_token = info["sample_token"], info["scene_token"]

        sample_dir = yolo_mono_dir / sample_id
        if not sample_dir.exists():
            continue

        channel_files = sorted(sample_dir.glob("*.json"))
        per_channel = []
        for channel_path in channel_files:
            channel = channel_path.stem   # e.g. "CAM_FRONT"
            t = _channel_timestamp_s(sample_token, channel)
            with open(channel_path, encoding="utf-8") as f:
                detections = json.load(f)
            per_channel.append((t, detections))
        per_channel.sort(key=lambda pair: pair[0])

        for t, detections in per_channel:
            for det in detections:
                if not det.get("has_3d_position", False):
                    continue
                yield SensorEvent(t=t, sensor="camera_mono",
                                   x=float(det["global_x"]), y=float(det["global_y"]),
                                   sample_id=sample_id, scene_token=scene_token)


def ego_pose_lookup():
    """Returns {sample_id: EgoState(t, x, y, vx, vy)} -- the ego
    vehicle's own global position at each sample's real timestamp.

    Unlike a track's .history (which needs evaluate.position_at() to
    extrapolate to a GT annotation's exact timestamp, since a track can
    update many times between two annotated samples at its own
    sensor's native rate), sample_id-keying here is sufficient with no
    extrapolation: this dict is only ever looked up at the GT
    annotation sample_ids themselves (compute_ground_truth_ttc() and
    match_track_to_ground_truth()'s TTC calls both key ego state by the
    SAME sample_id as the annotation being compared), never at an
    arbitrary in-between timestamp -- so there is no timestamp mismatch
    for this dict to paper over in the first place.
    (from Step 1.3's already-recorded LIDAR_TOP ego_pose, one per
    sample), with velocity derived properly rather than the old
    pipeline's bug: nuScenes' ego_pose has no velocity field, and
    Step_6_Visualization.ipynb derived one with a raw two-point finite
    difference between consecutive ego_pose entries -- exactly the bug
    class this whole rebuild exists to eliminate, made worse here since
    ego velocity feeds into EVERY TTC computation (predicted and
    ground-truth alike), so noise there propagates everywhere.

    Instead, ego's own consecutive global positions (sorted by real
    timestamp) are fed through a ConstantVelocityKF -- the same
    primitive already built and tested in deliverable 1, just applied
    to the ego vehicle's own trajectory instead of an external tracked
    object -- and the smoothed velocity is read back out. r_var uses
    config.EGO_POSITION_NOISE_VAR (nuScenes' fused localization stack
    is far more accurate than any of the three tracked sensors);
    sigma_a reuses config.PROCESS_NOISE_SIGMA_A, since it is already
    defined as a sensor/object-independent constant, not something
    that needs a second copy for ego.

    Resets to a fresh KF on every scene_token change, mirroring R9
    (central_tracker.py's own scene-boundary reset) -- different scenes
    are different driving-log segments, not one continuous trajectory,
    so a KF blindly carried across a scene cut would treat a real
    position jump as valid motion and emit a garbage velocity spike
    right at the boundary.
    """
    from v2.evaluate import EgoState
    from v2.kalman_track import ConstantVelocityKF

    samples_index = load_samples_index()

    ego_by_sample = {}
    kf = None
    current_scene_token = None

    for sample_id in _sample_ids_sorted():
        info = samples_index[sample_id]
        meta_path = legacy_config.STEP1_DIR / "lidar" / sample_id / "lidar_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        t = info["timestamp_us"] / 1e6
        x, y = float(meta["ego_pose"]["translation"][0]), float(meta["ego_pose"]["translation"][1])
        scene_token = info["scene_token"]

        if kf is None or scene_token != current_scene_token:
            kf = ConstantVelocityKF(x, y, t, sigma_a=v2_config.PROCESS_NOISE_SIGMA_A)
            current_scene_token = scene_token
        else:
            kf.predict(t)
            kf.update(x, y, v2_config.EGO_POSITION_NOISE_VAR, t)

        vx, vy = kf.velocity
        ego_by_sample[sample_id] = EgoState(t=t, x=x, y=y, vx=float(vx), vy=float(vy))

    return ego_by_sample


def gt_annotations_lookup():
    """Returns (gt_trajectories, gt_by_sample), both built from ONE
    pass over every real sample_annotation in the dataset -- no
    category filter, matching the old pipeline's own GT-loading
    convention (Step_6_Visualization.ipynb) for a like-for-like
    comparison rather than silently narrowing the evaluated object set:

      gt_trajectories: {instance_token: [GTPoint, ...]} -- the shape
          evaluate.compute_ground_truth_ttc() expects.
      gt_by_sample: {sample_id: [(instance_token, x, y), ...]} -- the
          shape evaluate.match_track_to_ground_truth() expects.

    Every annotation's own t is its parent sample's real timestamp
    (samples_index.json's timestamp_us) -- nuScenes defines a sample's
    timestamp to always equal its LIDAR_TOP keyframe's own capture
    time, so this is already that channel's real timestamp, not an
    approximation.
    """
    from v2.evaluate import GTPoint

    samples_index = load_samples_index()
    nusc = _get_nuscenes()

    gt_trajectories = defaultdict(list)
    gt_by_sample = defaultdict(list)

    for sample_id in _sample_ids_sorted():
        info = samples_index[sample_id]
        t = info["timestamp_us"] / 1e6
        sample = nusc.get("sample", info["sample_token"])
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            x, y = float(ann["translation"][0]), float(ann["translation"][1])
            instance_token = ann["instance_token"]
            gt_trajectories[instance_token].append(GTPoint(t=t, x=x, y=y, sample_id=sample_id))
            gt_by_sample[sample_id].append((instance_token, x, y))

    return dict(gt_trajectories), dict(gt_by_sample)
