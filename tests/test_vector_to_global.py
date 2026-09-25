"""
tests/test_vector_to_global.py — synthetic, no dataset dependency.

Confirms the one property that distinguishes vector_to_global() from
point_to_global(): translation must NEVER affect a direction/velocity
vector, only rotation should.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.geometry import vector_to_global, point_to_global

IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0]   # nuScenes [w, x, y, z] identity quaternion


def test_identity_transform_leaves_vector_unchanged():
    calib = {"translation": [0.0, 0.0, 0.0], "rotation": IDENTITY_ROTATION}
    ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": IDENTITY_ROTATION}
    out = vector_to_global([3.0, -1.5, 0.0], ego_pose, calib)
    assert np.allclose(out, [3.0, -1.5, 0.0])


def test_translation_alone_does_not_affect_a_vector():
    """The key property: point_to_global() of the same input WOULD shift by
    the translation; vector_to_global() must not, since a velocity has no
    position to translate."""
    calib = {"translation": [5.0, -2.0, 1.0], "rotation": IDENTITY_ROTATION}
    ego_pose = {"translation": [100.0, 200.0, 0.0], "rotation": IDENTITY_ROTATION}
    vec = [4.0, 0.0, 0.0]

    vec_out = vector_to_global(vec, ego_pose, calib)
    assert np.allclose(vec_out, vec), "translation must not affect a direction vector"

    point_out = point_to_global(vec, ego_pose, calib)
    assert not np.allclose(point_out, vec), "sanity check: point_to_global SHOULD shift by translation"


def test_90_degree_rotation_rotates_the_vector_correctly():
    # 90 deg rotation about +z: [w,x,y,z] = [cos(45deg), 0, 0, sin(45deg)]
    half = np.pi / 4
    rot_90_z = [np.cos(half), 0.0, 0.0, np.sin(half)]
    calib = {"translation": [0.0, 0.0, 0.0], "rotation": rot_90_z}
    ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": IDENTITY_ROTATION}

    out = vector_to_global([1.0, 0.0, 0.0], ego_pose, calib)
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_composes_sensor_to_ego_and_ego_to_global_rotations():
    half = np.pi / 4
    rot_90_z = [np.cos(half), 0.0, 0.0, np.sin(half)]
    # two successive 90 degree rotations about z = 180 degrees
    calib = {"translation": [1.0, 2.0, 3.0], "rotation": rot_90_z}
    ego_pose = {"translation": [10.0, 20.0, 30.0], "rotation": rot_90_z}

    out = vector_to_global([1.0, 0.0, 0.0], ego_pose, calib)
    assert np.allclose(out, [-1.0, 0.0, 0.0], atol=1e-9)
