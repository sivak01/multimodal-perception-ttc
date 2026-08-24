# src/geometry.py — shared coordinate-transform and camera-projection utilities.
#
# transform_matrix() and the point-to-global / global-to-camera functions below
# were previously copy-pasted into Step 1.1, Step 2.3.1, Step 3.1, and Step 3.2
# (four independent copies of the same nuScenes transform-chain math). They now
# live here once; each notebook imports what it needs.

import warnings

import numpy as np
from pyquaternion import Quaternion


def transform_matrix(translation, rotation_quat, inverse=False):
    """
    Builds a 4x4 homogeneous transform matrix from a translation vector
    and a quaternion [w, x, y, z] (nuScenes quaternion convention).

    If inverse=True, returns the inverse transform (used when going
    FROM global back INTO a sensor frame — e.g. global -> camera's ego).
    """
    rot = Quaternion(rotation_quat)
    tm = np.eye(4)

    if inverse:
        rot_inv = rot.inverse
        tm[:3, :3] = rot_inv.rotation_matrix
        tm[:3, 3] = -(rot_inv.rotation_matrix @ np.array(translation))
    else:
        tm[:3, :3] = rot.rotation_matrix
        tm[:3, 3] = translation

    return tm


def point_to_global(point_xyz, ego_pose, calibrated_sensor):
    """Transforms a single point from ITS OWN sensor's frame into the global frame.
    Critically: pass the calibration for the SPECIFIC channel this point came from —
    each sensor has a different sensor_to_ego transform."""
    point = np.array(point_xyz, dtype=float).reshape(3, 1)
    point_h = np.vstack([point, [[1.0]]])

    T1 = transform_matrix(calibrated_sensor["translation"], calibrated_sensor["rotation"])
    T2 = transform_matrix(ego_pose["translation"], ego_pose["rotation"])

    point_h = T2 @ (T1 @ point_h)
    return point_h[:3].flatten()


def points_to_global(points_xyz, src_ego_pose, src_calibrated_sensor):
    """Batch: sensor frame (N,3) -> global frame (N,3)."""
    N = points_xyz.shape[0]
    points_h = np.hstack([points_xyz, np.ones((N, 1))]).T   # (4, N)

    T1 = transform_matrix(src_calibrated_sensor["translation"], src_calibrated_sensor["rotation"])
    T2 = transform_matrix(src_ego_pose["translation"], src_ego_pose["rotation"])

    points_h = T2 @ (T1 @ points_h)
    return points_h[:3].T   # (N, 3)


def project_point_to_camera(point_xyz,
                             src_ego_pose, src_calibrated_sensor,
                             cam_ego_pose, cam_calibrated_sensor,
                             camera_intrinsic,
                             image_width=1600, image_height=900):
    """
    Projects a single 3D point from a SOURCE sensor's frame (e.g. LiDAR,
    radar, or a fused UKF state in global frame) into a TARGET camera's
    image plane.

    Five-step chain:
      1. source sensor frame -> ego frame        (at SOURCE timestamp)
      2. ego frame -> global frame                (at SOURCE timestamp)
      3. global frame -> ego frame                (at CAMERA timestamp — bridges the time gap)
      4. ego frame -> camera sensor frame          (at CAMERA timestamp)
      5. camera 3D point -> image pixel            (using camera_intrinsic)

    Returns:
        (u, v, depth) if the point is in front of the camera and depth > 0
        None if the point is behind the camera (invalid projection)
    """
    point = np.array(point_xyz, dtype=float).reshape(3, 1)
    point_h = np.vstack([point, [[1.0]]])

    T1 = transform_matrix(src_calibrated_sensor["translation"], src_calibrated_sensor["rotation"])
    point_h = T1 @ point_h

    T2 = transform_matrix(src_ego_pose["translation"], src_ego_pose["rotation"])
    point_h = T2 @ point_h

    T3 = transform_matrix(cam_ego_pose["translation"], cam_ego_pose["rotation"], inverse=True)
    point_h = T3 @ point_h

    T4 = transform_matrix(cam_calibrated_sensor["translation"], cam_calibrated_sensor["rotation"], inverse=True)
    point_h = T4 @ point_h

    point_cam = point_h[:3]
    depth = float(point_cam[2, 0])

    if depth <= 0:
        return None  # point is behind the camera — do not project

    K = np.array(camera_intrinsic)
    pixel = K @ point_cam
    u = float(pixel[0, 0] / pixel[2, 0])
    v = float(pixel[1, 0] / pixel[2, 0])

    if not (0 <= u < image_width and 0 <= v < image_height):
        return None  # projects outside the visible image — discard

    return (u, v, depth)


# Below this mean per-point magnitude (metres), points look like a raw sensor-local
# cloud rather than nuScenes global-frame coordinates. Calibrated against this repo's
# actual data: a typical LIDAR_TOP sweep has mean |xyz| ~10m (median ~6m), while this
# dataset's global frame sits around 1100-1250m from origin — a 100m cutoff sits well
# clear of both, so it won't false-negative on real local point clouds.
_LOCAL_FRAME_MAGNITUDE_WARN_THRESHOLD_M = 100.0


def global_points_to_camera(points_global, cam_ego_pose, cam_calibrated_sensor,
                             camera_intrinsic, image_width=1600, image_height=900):
    """
    Batch: global frame (N,3) -> camera pixel (u, v, depth).
    Returns arrays of shape (N,) for u, v, depth, and a boolean valid mask
    (in front of camera AND inside image bounds).

    points_global MUST already be in the global frame (e.g. via point_to_global /
    points_to_global) — this function has no way to detect a sensor-local point cloud
    passed in by mistake and will silently produce plausible-looking (u, v) pixels
    for it. As a cheap sanity net, a suspiciously small mean magnitude (typical of a
    raw local sweep, atypical for this dataset's global frame) triggers a warning —
    it does not raise, since a legitimately close-to-global-origin point set is
    possible in principle.
    """
    if len(points_global) > 0:
        mean_mag = float(np.mean(np.linalg.norm(points_global, axis=1)))
        if mean_mag < _LOCAL_FRAME_MAGNITUDE_WARN_THRESHOLD_M:
            warnings.warn(
                f"global_points_to_camera: input points have mean |xyz|={mean_mag:.1f}m, "
                f"well below this dataset's typical global-frame magnitude — did you forget "
                f"to call points_to_global() first? (points_global must already be global-frame)",
                stacklevel=2,
            )

    N = points_global.shape[0]
    points_h = np.hstack([points_global, np.ones((N, 1))]).T   # (4, N)

    T3 = transform_matrix(cam_ego_pose["translation"], cam_ego_pose["rotation"], inverse=True)
    T4 = transform_matrix(cam_calibrated_sensor["translation"], cam_calibrated_sensor["rotation"], inverse=True)

    points_cam = (T4 @ (T3 @ points_h))[:3]   # (3, N)
    depth = points_cam[2, :]

    K = np.array(camera_intrinsic)
    pixel = K @ points_cam
    with np.errstate(divide='ignore', invalid='ignore'):
        u = pixel[0, :] / pixel[2, :]
        v = pixel[1, :] / pixel[2, :]

    valid = (depth > 0) & (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    return u, v, depth, valid


def project_points_to_camera_batch(points_local, src_ego_pose, src_calibrated_sensor,
                                    cam_ego_pose, cam_calibrated_sensor,
                                    camera_intrinsic, image_width=1600, image_height=900):
    """
    Batch-safe equivalent of project_point_to_camera() for many points sharing the
    SAME source/camera pose (e.g. one LiDAR sweep projected onto one camera image).

    project_point_to_camera() rebuilds all four transform matrices (T1-T4) from
    scratch on every call, which is fine for a single interactive point but wasteful
    in a loop — the same quaternion-to-rotation-matrix work gets redone per point.
    This function computes T1-T4 once, combines them into a single 4x4 matrix, and
    applies it to all N points in one matrix multiply.

    points_local: (N,3) array, in the SOURCE sensor's own local frame (same
    convention as project_point_to_camera's point_xyz — NOT global-frame; for
    already-global points use global_points_to_camera instead).

    Returns arrays of shape (N,) for u, v, depth, and a boolean valid mask (in front
    of camera AND inside image bounds) — same contract as global_points_to_camera().
    """
    points_local = np.asarray(points_local, dtype=float)
    N = points_local.shape[0]
    if N == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty.astype(bool)

    points_h = np.hstack([points_local, np.ones((N, 1))]).T   # (4, N)

    T1 = transform_matrix(src_calibrated_sensor["translation"], src_calibrated_sensor["rotation"])
    T2 = transform_matrix(src_ego_pose["translation"], src_ego_pose["rotation"])
    T3 = transform_matrix(cam_ego_pose["translation"], cam_ego_pose["rotation"], inverse=True)
    T4 = transform_matrix(cam_calibrated_sensor["translation"], cam_calibrated_sensor["rotation"], inverse=True)

    # Same five-step chain as project_point_to_camera, precomputed into one matrix.
    M = T4 @ T3 @ T2 @ T1
    points_cam = (M @ points_h)[:3]   # (3, N)
    depth = points_cam[2, :]

    K = np.array(camera_intrinsic)
    pixel = K @ points_cam
    with np.errstate(divide='ignore', invalid='ignore'):
        u = pixel[0, :] / pixel[2, :]
        v = pixel[1, :] / pixel[2, :]

    valid = (depth > 0) & (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    return u, v, depth, valid
