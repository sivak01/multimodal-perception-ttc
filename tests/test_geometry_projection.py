"""
Cross-validates src.geometry.project_point_to_camera() against nuScenes devkit's
OWN, independently-implemented projection chain (Box.translate/rotate + view_points()).

Hand-checked algebra catches derivation errors; it doesn't catch a K-matrix layout
mismatch or an off-by-one against how a particular nuscenes-devkit version loads
camera_intrinsic. This test settles that by comparing pixel coordinates directly
against the dataset authors' own implementation, on real ground-truth 3D boxes,
across 5 samples spanning >=2 scenes (including one high-yaw-rate turn — the case
most likely to expose a hidden rotation-order bug, since a straight-line drive can't
distinguish a wrong quaternion convention from a correct one).

Run with: pytest tests/test_geometry_projection.py -v -s
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT, NUSCENES_VERSION, STEP0_DIR
from src.geometry import project_point_to_camera

PIXEL_TOL = 1.0
CAMERAS = ["CAM_FRONT", "CAM_BACK"]
N_RANDOM_SAMPLES = 4  # + 1 dedicated turn sample = 5 total
RANDOM_SEED = 42

# project_point_to_camera expects point_xyz in a SOURCE sensor's local frame and
# chains sensor->ego->global via src_calibrated_sensor/src_ego_pose. GT annotation
# translations are already global, so an identity source transform makes the
# sensor->ego->global leg a no-op and effectively feeds the global point straight
# through — a supported usage the function's own docstring anticipates ("or a fused
# UKF state in global frame").
IDENTITY_POSE = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}


@pytest.fixture(scope="module")
def nusc():
    return NuScenes(version=NUSCENES_VERSION, dataroot=str(DATA_ROOT), verbose=False)


@pytest.fixture(scope="module")
def samples_index():
    with open(STEP0_DIR / "samples_index.json") as f:
        return json.load(f)


def _find_turn_sample(nusc, samples_index):
    """Sample with the largest ego yaw change from the previous sample, using
    LIDAR_TOP's ego_pose (2Hz keyframe rate, same as every other sensor's keyframe)."""
    ordered = sorted(samples_index.keys())
    best_sid, best_dyaw, prev_yaw = None, -1.0, None
    for sid in ordered:
        sample = nusc.get("sample", samples_index[sid]["sample_token"])
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        yaw = Quaternion(ep["rotation"]).yaw_pitch_roll[0]
        if prev_yaw is not None:
            dyaw = abs(((yaw - prev_yaw) + np.pi) % (2 * np.pi) - np.pi)
            if dyaw > best_dyaw:
                best_dyaw, best_sid = dyaw, sid
        prev_yaw = yaw
    return best_sid, best_dyaw


def _pick_samples(nusc, samples_index):
    """4 random samples spanning >=2 distinct scenes, plus 1 dedicated turn sample."""
    turn_sid, turn_dyaw = _find_turn_sample(nusc, samples_index)

    by_scene = {}
    for sid, info in samples_index.items():
        by_scene.setdefault(info["scene_name"], []).append(sid)

    rng = random.Random(RANDOM_SEED)
    scene_names = sorted(by_scene.keys())
    rng.shuffle(scene_names)

    chosen, seen_scenes = [], set()
    for scene_name in scene_names:
        candidates = [s for s in by_scene[scene_name] if s != turn_sid]
        if not candidates:
            continue
        chosen.append(rng.choice(candidates))
        seen_scenes.add(scene_name)
        if len(chosen) >= N_RANDOM_SAMPLES and len(seen_scenes) >= 2:
            break

    assert len(seen_scenes) >= 2, "could not find samples across >=2 distinct scenes"
    return chosen[:N_RANDOM_SAMPLES] + [turn_sid], turn_sid, turn_dyaw


@pytest.fixture(scope="module")
def selected_samples(nusc, samples_index):
    chosen, turn_sid, turn_dyaw = _pick_samples(nusc, samples_index)
    print(f"\nSelected samples: {chosen}")
    print(f"Turn sample: {turn_sid} (ego yaw change {np.degrees(turn_dyaw):.1f} deg from previous sample)")
    return chosen, turn_sid


def _first_visible_annotation(nusc, cam_sd_token, ann_tokens):
    """First GT box actually visible in this camera, via the devkit's OWN
    get_sample_data() -> Box(camera frame) chain — independent of src/geometry.py."""
    for ann_token in ann_tokens:
        _, box_list, cam_intrinsic = nusc.get_sample_data(
            cam_sd_token, box_vis_level=BoxVisibility.ANY, selected_anntokens=[ann_token]
        )
        if not box_list or box_list[0].center[2] <= 0:
            continue
        return nusc.get("sample_annotation", ann_token), box_list[0], cam_intrinsic
    return None, None, None


def _project_ours(nusc, ann, cam_sd_token):
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    cam_ego_pose = {"translation": ep["translation"], "rotation": ep["rotation"]}
    cam_calib = {"translation": cs["translation"], "rotation": cs["rotation"]}
    return project_point_to_camera(
        ann["translation"],            # already global — identity source transform below
        IDENTITY_POSE, IDENTITY_POSE,  # src_ego_pose, src_calibrated_sensor: no-ops
        cam_ego_pose, cam_calib,
        cs["camera_intrinsic"],
        image_width=sd["width"], image_height=sd["height"],
    )


def _project_devkit(box, cam_intrinsic):
    pixel = view_points(box.center.reshape(3, 1), np.array(cam_intrinsic), normalize=True)
    return float(pixel[0, 0]), float(pixel[1, 0])


def _run_check(nusc, samples_index, sample_id, cam_name):
    sample = nusc.get("sample", samples_index[sample_id]["sample_token"])
    cam_sd_token = sample["data"][cam_name]

    ann, box, cam_intrinsic = _first_visible_annotation(nusc, cam_sd_token, sample["anns"])
    assert ann is not None, (
        f"{sample_id}/{cam_name}: no GT annotation visible in this camera — "
        f"can't cross-validate on this (sample, camera) pair"
    )

    ours = _project_ours(nusc, ann, cam_sd_token)
    assert ours is not None, (
        f"{sample_id}/{cam_name}: our function returned None (behind camera / out of "
        f"frame) but the devkit found this box visible — likely disagreement, not just "
        f"a missing point"
    )
    u_ours, v_ours, _ = ours
    u_dev, v_dev = _project_devkit(box, cam_intrinsic)
    return abs(u_ours - u_dev), abs(v_ours - v_dev), u_ours, v_ours, u_dev, v_dev


@pytest.mark.parametrize("cam_name", CAMERAS)
@pytest.mark.parametrize("sample_idx", range(N_RANDOM_SAMPLES + 1))
def test_projection_matches_devkit(nusc, samples_index, selected_samples, sample_idx, cam_name):
    chosen, turn_sid = selected_samples
    sample_id = chosen[sample_idx]
    du, dv, u_ours, v_ours, u_dev, v_dev = _run_check(nusc, samples_index, sample_id, cam_name)

    tag = " [TURN SAMPLE]" if sample_id == turn_sid else ""
    print(f"{sample_id}/{cam_name}{tag}: ours=({u_ours:.2f},{v_ours:.2f}) "
          f"devkit=({u_dev:.2f},{v_dev:.2f}) du={du:.4f}px dv={dv:.4f}px")

    assert du < PIXEL_TOL, f"{sample_id}/{cam_name}{tag}: u mismatch {du:.4f}px >= {PIXEL_TOL}px"
    assert dv < PIXEL_TOL, f"{sample_id}/{cam_name}{tag}: v mismatch {dv:.4f}px >= {PIXEL_TOL}px"


@pytest.mark.parametrize("cam_name", CAMERAS)
def test_turn_sample_no_rotation_order_bug(nusc, samples_index, selected_samples, cam_name):
    """Dedicated, separately-reported check on the highest-yaw-rate sample — a
    hidden quaternion/rotation-order bug is far more likely to surface during a turn
    than on a straight-line drive, where wrong and right conventions can coincide."""
    _, turn_sid = selected_samples
    du, dv, u_ours, v_ours, u_dev, v_dev = _run_check(nusc, samples_index, turn_sid, cam_name)

    print(f"[TURN SAMPLE RESULT] {turn_sid}/{cam_name}: ours=({u_ours:.2f},{v_ours:.2f}) "
          f"devkit=({u_dev:.2f},{v_dev:.2f}) du={du:.4f}px dv={dv:.4f}px")

    assert du < PIXEL_TOL, f"TURN SAMPLE {turn_sid}/{cam_name}: u mismatch {du:.4f}px >= {PIXEL_TOL}px"
    assert dv < PIXEL_TOL, f"TURN SAMPLE {turn_sid}/{cam_name}: v mismatch {dv:.4f}px >= {PIXEL_TOL}px"
