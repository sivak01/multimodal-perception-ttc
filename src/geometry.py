# src/geometry.py — shared coordinate-transform and camera-projection utilities.
#
# transform_matrix() and the point-to-global / global-to-camera functions below
# were previously copy-pasted into Step 1.1, Step 2.3.1, Step 3.1, and Step 3.2
# (four independent copies of the same nuScenes transform-chain math). They now
# live here once; each notebook imports what it needs.

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


def global_points_to_camera(points_global, cam_ego_pose, cam_calibrated_sensor,
                             camera_intrinsic, image_width=1600, image_height=900):
    """
    Batch: global frame (N,3) -> camera pixel (u, v, depth).
    Returns arrays of shape (N,) for u, v, depth, and a boolean valid mask
    (in front of camera AND inside image bounds).
    """
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
