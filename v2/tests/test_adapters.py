"""
v2/tests/test_adapters.py — integration tests against the REAL on-disk
Step 0-2 outputs and the real NuScenes devkit (not synthetic data, by
necessity -- adapters.py's whole job is bridging real pipeline output).

Skipped automatically if this machine doesn't have the dataset/output
directories (e.g. a CI box without the ~GB nuScenes mini dataset
checked out), same defensive spirit as the existing notebooks'
skipped_samples tracking for missing per-sample files.
"""
import json

import pytest

import config as legacy_config
from v2 import adapters

# Applied individually to the tests below that need the real, checked-out
# nuScenes mini dataset + a real Step 0-2 pipeline run -- NOT applied at
# module level, since the synthetic clustering test further down is fully
# self-contained (monkeypatched paths, no real dataset needed) and must
# still run on a machine without the ~GB dataset present.
_needs_real_dataset = pytest.mark.skipif(
    not (legacy_config.DATA_ROOT.exists() and (legacy_config.STEP2_DIR / "radar").exists()),
    reason="real nuScenes dataset / pipeline output not present on this machine",
)


@_needs_real_dataset
def test_radar_channel_timestamp_differs_from_parent_sample_by_a_plausible_small_amount():
    """The specific fix this file exists to prove: radar_events() must
    NOT silently fall back to the parent sample's shared timestamp_us
    (that would make the offset exactly 0), and must not produce
    something wildly implausible either (samples are ~0.5s apart at
    2Hz, so a real per-channel jitter must be well under that)."""
    samples_index = adapters.load_samples_index()
    sample_id = "sample_0000"
    info = samples_index[sample_id]

    radar_t_s = adapters._channel_timestamp_s(info["sample_token"], "RADAR_FRONT")
    radar_t_us = radar_t_s * 1e6

    diff_us = radar_t_us - info["timestamp_us"]
    assert diff_us != 0, "radar timestamp must not equal the parent sample's shared timestamp exactly"
    assert 0 < abs(diff_us) < 100_000, (
        f"expected a small (sub-100ms) plausible offset between a radar channel's own "
        f"capture time and its parent sample's shared timestamp, got {diff_us}us"
    )


@_needs_real_dataset
def test_lidar_events_are_globally_sorted_by_timestamp_and_in_global_frame():
    events = []
    for i, ev in enumerate(adapters.lidar_events()):
        events.append(ev)
        if i >= 200:   # enough to span several samples without reading the whole (small) dataset
            break

    assert len(events) > 0
    ts = [ev.t for ev in events]
    assert ts == sorted(ts), "lidar_events() must yield in non-decreasing timestamp order"
    assert all(ev.sensor == "lidar" for ev in events)
    assert all(ev.sample_id and ev.scene_token for ev in events)
    # this dataset's global frame sits ~1100-1250m from origin (src/geometry.py's
    # own calibrated threshold) -- a local/unlifted centroid would be within a few
    # metres of 0, which this checks was NOT what got yielded
    assert (events[0].x ** 2 + events[0].y ** 2) ** 0.5 > 100


@_needs_real_dataset
def test_radar_events_are_globally_sorted_by_timestamp_and_in_global_frame():
    events = []
    for i, ev in enumerate(adapters.radar_events()):
        events.append(ev)
        if i >= 200:
            break

    assert len(events) > 0
    ts = [ev.t for ev in events]
    assert ts == sorted(ts), "radar_events() must yield in non-decreasing timestamp order"
    assert all(ev.sensor == "radar" for ev in events)
    assert (events[0].x ** 2 + events[0].y ** 2) ** 0.5 > 100


@_needs_real_dataset
def test_camera_mono_events_are_globally_sorted_by_timestamp_and_in_global_frame():
    events = []
    for i, ev in enumerate(adapters.camera_mono_events()):
        events.append(ev)
        if i >= 200:
            break

    assert len(events) > 0
    ts = [ev.t for ev in events]
    assert ts == sorted(ts), "camera_mono_events() must yield in non-decreasing timestamp order"
    assert all(ev.sensor == "camera_mono" for ev in events)
    assert (events[0].x ** 2 + events[0].y ** 2) ** 0.5 > 100


@_needs_real_dataset
def test_lidar_and_radar_and_camera_mono_events_merge_into_one_chronological_stream():
    """Sanity check on the whole point of R2: merging three independently-
    real-timestamped streams should still produce one non-decreasing
    sequence, not require any of them to share a nominal frame clock."""
    from v2.event_stream import merge_streams

    def _capped(gen, n):
        for i, ev in enumerate(gen):
            if i >= n:
                return
            yield ev

    merged = list(merge_streams(
        _capped(adapters.lidar_events(), 50),
        _capped(adapters.radar_events(), 50),
        _capped(adapters.camera_mono_events(), 50),
    ))
    ts = [ev.t for ev in merged]
    assert ts == sorted(ts)
    assert {ev.sensor for ev in merged} == {"lidar", "radar", "camera_mono"}


# ---------------------------------------------------------------------
# Fully synthetic (no real dataset needed): proves radar_events() itself
# clusters near-simultaneous same-scan blips rather than passing every
# raw point straight through. Uses monkeypatched paths + a stubbed
# timestamp lookup instead of the real NuScenes devkit/dataset, so this
# runs on any machine, unlike the tests above.
# ---------------------------------------------------------------------

def test_radar_events_clusters_near_simultaneous_same_scan_blips(tmp_path, monkeypatch):
    """3 raw points within 1m of each other, in ONE channel's ONE scan
    (simulating multiple echoes off one real object) -- radar_events()
    must yield ONE clustered SensorEvent here, not three raw ones."""
    sample_id = "sample_0000"
    scene_token = "scene_test"
    sample_token = "tok_test"

    step0_dir = tmp_path / "step_0"
    step2_dir = tmp_path / "step_2"
    step0_dir.mkdir(parents=True)
    (step2_dir / "radar" / sample_id).mkdir(parents=True)

    samples_index = {
        sample_id: {"sample_token": sample_token, "scene_token": scene_token, "timestamp_us": 1_000_000}
    }
    with open(step0_dir / "samples_index.json", "w", encoding="utf-8") as f:
        json.dump(samples_index, f)

    # identity calibration -- point_to_global() should pass local
    # coordinates through unchanged, so the expected clustered global
    # position is just the mean of the three raw local points
    radar_channel_data = {
        "sample_id": sample_id,
        "radar_channel": "RADAR_FRONT",
        "calibration": {
            "sensor_to_ego_translation": [0.0, 0.0, 0.0],
            "sensor_to_ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego_pose": {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        },
        "points": [
            {"x": 20.0, "y": 5.0, "z": 0.0},
            {"x": 20.3, "y": 5.2, "z": 0.0},   # within 1m of the first -- same object's echoes
            {"x": 19.7, "y": 4.8, "z": 0.0},   # within 1m of the first
        ],
    }
    with open(step2_dir / "radar" / sample_id / "RADAR_FRONT.json", "w", encoding="utf-8") as f:
        json.dump(radar_channel_data, f)

    monkeypatch.setattr(legacy_config, "STEP0_DIR", step0_dir)
    monkeypatch.setattr(legacy_config, "STEP2_DIR", step2_dir)
    monkeypatch.setattr(adapters, "_channel_timestamp_s", lambda sample_token, channel: 1.0)

    events = list(adapters.radar_events())

    assert len(events) == 1, f"expected 3 same-scan blips within RADAR_CLUSTER_EPS_M to collapse into 1 event, got {len(events)}"
    ev = events[0]
    assert ev.sensor == "radar"
    assert ev.sample_id == sample_id
    assert ev.scene_token == scene_token
    assert abs(ev.x - 20.0) < 1e-6   # mean of the 3 local x's, identity calibration
    assert abs(ev.y - 5.0) < 1e-6    # mean of the 3 local y's


def test_radar_events_does_not_over_cluster_two_genuinely_separate_objects(tmp_path, monkeypatch):
    """Two points far apart (>> RADAR_CLUSTER_EPS_M) in the same scan
    are two different real objects -- must remain two separate events,
    not be merged just because they're in the same channel/scan."""
    sample_id = "sample_0000"
    scene_token = "scene_test"
    sample_token = "tok_test"

    step0_dir = tmp_path / "step_0"
    step2_dir = tmp_path / "step_2"
    step0_dir.mkdir(parents=True)
    (step2_dir / "radar" / sample_id).mkdir(parents=True)

    samples_index = {
        sample_id: {"sample_token": sample_token, "scene_token": scene_token, "timestamp_us": 1_000_000}
    }
    with open(step0_dir / "samples_index.json", "w", encoding="utf-8") as f:
        json.dump(samples_index, f)

    radar_channel_data = {
        "sample_id": sample_id,
        "radar_channel": "RADAR_FRONT",
        "calibration": {
            "sensor_to_ego_translation": [0.0, 0.0, 0.0],
            "sensor_to_ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego_pose": {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        },
        "points": [
            {"x": 20.0, "y": 5.0, "z": 0.0},
            {"x": 60.0, "y": -10.0, "z": 0.0},   # far away -- a different real object
        ],
    }
    with open(step2_dir / "radar" / sample_id / "RADAR_FRONT.json", "w", encoding="utf-8") as f:
        json.dump(radar_channel_data, f)

    monkeypatch.setattr(legacy_config, "STEP0_DIR", step0_dir)
    monkeypatch.setattr(legacy_config, "STEP2_DIR", step2_dir)
    monkeypatch.setattr(adapters, "_channel_timestamp_s", lambda sample_token, channel: 1.0)

    events = list(adapters.radar_events())
    assert len(events) == 2


def test_radar_events_uses_the_radar_channels_own_ego_pose_not_lidar_tops(tmp_path, monkeypatch):
    """Same bug class as the radar timestamp fix: the vehicle moves
    between when each sensor actually fires within one nuScenes sample,
    so each radar channel's sample_data has its OWN ego_pose_token,
    distinct from LIDAR_TOP's. Confirmed against the real devkit
    (RADAR_FRONT/sample_0000's on-disk ego_pose exactly matches that
    channel's own devkit ego_pose, and differs from LIDAR_TOP's) that
    radar_events() already reads ego_pose from the radar channel's OWN
    on-disk JSON, not from lidar_meta.json -- this test locks that in
    with a deliberately wrong-looking LIDAR_TOP ego_pose present for
    the same sample, to prove it's never even consulted."""
    sample_id = "sample_0000"
    scene_token = "scene_test"
    sample_token = "tok_test"

    step0_dir = tmp_path / "step_0"
    step1_dir = tmp_path / "step_1"
    step2_dir = tmp_path / "step_2"
    step0_dir.mkdir(parents=True)
    (step2_dir / "radar" / sample_id).mkdir(parents=True)
    (step1_dir / "lidar" / sample_id).mkdir(parents=True)

    samples_index = {
        sample_id: {"sample_token": sample_token, "scene_token": scene_token, "timestamp_us": 1_000_000}
    }
    with open(step0_dir / "samples_index.json", "w", encoding="utf-8") as f:
        json.dump(samples_index, f)

    # a deliberately very different ego_pose under LIDAR_TOP's own
    # per-sample file -- radar_events() must never touch this
    wrong_lidar_meta = {"ego_pose": {"translation": [999.0, 999.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}}
    with open(step1_dir / "lidar" / sample_id / "lidar_meta.json", "w", encoding="utf-8") as f:
        json.dump(wrong_lidar_meta, f)

    # RADAR_FRONT's OWN ego_pose (identity sensor_to_ego, so the lifted
    # global point should land exactly at this ego_pose's translation)
    radar_channel_data = {
        "sample_id": sample_id,
        "radar_channel": "RADAR_FRONT",
        "calibration": {
            "sensor_to_ego_translation": [0.0, 0.0, 0.0],
            "sensor_to_ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego_pose": {"translation": [100.0, 200.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        },
        "points": [{"x": 0.0, "y": 0.0, "z": 0.0}],
    }
    with open(step2_dir / "radar" / sample_id / "RADAR_FRONT.json", "w", encoding="utf-8") as f:
        json.dump(radar_channel_data, f)

    monkeypatch.setattr(legacy_config, "STEP0_DIR", step0_dir)
    monkeypatch.setattr(legacy_config, "STEP1_DIR", step1_dir)
    monkeypatch.setattr(legacy_config, "STEP2_DIR", step2_dir)
    monkeypatch.setattr(adapters, "_channel_timestamp_s", lambda sample_token, channel: 1.0)

    events = list(adapters.radar_events())

    assert len(events) == 1
    ev = events[0]
    assert abs(ev.x - 100.0) < 1e-6 and abs(ev.y - 200.0) < 1e-6, (
        f"expected the radar channel's OWN ego_pose (100, 200), got ({ev.x}, {ev.y}) -- "
        "looks like LIDAR_TOP's ego_pose (999, 999) leaked in instead"
    )


# ---------------------------------------------------------------------
# ego_pose_lookup(): fully synthetic too -- the KF-vs-raw-finite-
# difference requirement is a precise, easily-violated correctness
# property (ego velocity feeds every TTC computation), so it gets the
# same rigor as everything else in this suite rather than being taken
# on faith.
# ---------------------------------------------------------------------

def _write_fake_step0_and_lidar_meta(tmp_path, samples):
    """samples: list of (sample_id, scene_token, timestamp_us, x, y).
    Writes a minimal samples_index.json + one lidar_meta.json per
    sample -- exactly the two on-disk inputs ego_pose_lookup() reads."""
    step0_dir = tmp_path / "step_0"
    step1_dir = tmp_path / "step_1"
    step0_dir.mkdir(parents=True)

    samples_index = {}
    for sample_id, scene_token, timestamp_us, x, y in samples:
        samples_index[sample_id] = {
            "sample_token": f"tok_{sample_id}", "scene_token": scene_token, "timestamp_us": timestamp_us,
        }
        meta_dir = step1_dir / "lidar" / sample_id
        meta_dir.mkdir(parents=True)
        meta = {"ego_pose": {"translation": [x, y, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}}
        with open(meta_dir / "lidar_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    with open(step0_dir / "samples_index.json", "w", encoding="utf-8") as f:
        json.dump(samples_index, f)

    return step0_dir, step1_dir


def test_ego_pose_lookup_recovers_true_constant_velocity_from_noiseless_positions(tmp_path, monkeypatch):
    true_vx = 8.0   # m/s
    samples = [(f"sample_{i:04d}", "scene0", i * 500_000, true_vx * (i * 0.5), 0.0) for i in range(6)]
    step0_dir, step1_dir = _write_fake_step0_and_lidar_meta(tmp_path, samples)
    monkeypatch.setattr(legacy_config, "STEP0_DIR", step0_dir)
    monkeypatch.setattr(legacy_config, "STEP1_DIR", step1_dir)

    ego_by_sample = adapters.ego_pose_lookup()

    last = ego_by_sample["sample_0005"]
    assert abs(last.vx - true_vx) < 0.5
    assert abs(last.vy) < 0.5


def test_ego_pose_lookup_actually_routes_through_constantvelocitykf_predict_and_update(tmp_path, monkeypatch):
    """Structural proof, not just a numeric coincidence check: patches
    ConstantVelocityKF itself with a spy subclass and asserts
    ego_pose_lookup() actually calls predict()/update() on it once per
    subsequent sample -- i.e. this really goes through the KF machinery
    (state + covariance carried across every sample), not a hand-rolled
    two-point (x2-x1)/(t2-t1) formula that happens to produce a
    similar-looking number."""
    import v2.kalman_track as kalman_track_module

    calls = {"predict": 0, "update": 0}
    RealKF = kalman_track_module.ConstantVelocityKF

    class SpyKF(RealKF):
        def predict(self, t):
            calls["predict"] += 1
            return super().predict(t)

        def update(self, x_meas, y_meas, r_var, t):
            calls["update"] += 1
            return super().update(x_meas, y_meas, r_var, t)

    monkeypatch.setattr(kalman_track_module, "ConstantVelocityKF", SpyKF)

    samples = [(f"sample_{i:04d}", "scene0", i * 500_000, 5.0 * (i * 0.5), 0.0) for i in range(5)]
    step0_dir, step1_dir = _write_fake_step0_and_lidar_meta(tmp_path, samples)
    monkeypatch.setattr(legacy_config, "STEP0_DIR", step0_dir)
    monkeypatch.setattr(legacy_config, "STEP1_DIR", step1_dir)

    adapters.ego_pose_lookup()

    # 5 samples, one scene -> 1 construction (not spied) + 4 predict/update pairs
    assert calls["predict"] == 4
    assert calls["update"] == 4


def test_ego_pose_lookup_resets_velocity_at_scene_boundary(tmp_path, monkeypatch):
    """A big spatial jump between two different scenes' recordings must
    NOT be read as real ego motion -- the first sample of a new scene
    must start a fresh KF (velocity (0,0), matching a freshly-
    constructed filter's own convention -- one point carries no
    velocity information), not extend the previous scene's trajectory."""
    samples = [
        ("sample_0000", "sceneA", 0, 0.0, 0.0),
        ("sample_0001", "sceneA", 500_000, 4.0, 0.0),
        ("sample_0002", "sceneB", 1_000_000, 9000.0, 9000.0),   # huge jump: different recording entirely
    ]
    step0_dir, step1_dir = _write_fake_step0_and_lidar_meta(tmp_path, samples)
    monkeypatch.setattr(legacy_config, "STEP0_DIR", step0_dir)
    monkeypatch.setattr(legacy_config, "STEP1_DIR", step1_dir)

    ego_by_sample = adapters.ego_pose_lookup()

    first_of_new_scene = ego_by_sample["sample_0002"]
    assert first_of_new_scene.vx == 0.0 and first_of_new_scene.vy == 0.0, (
        "the first sample of a new scene must reset to zero velocity, not carry forward "
        "a huge spurious velocity computed across the scene boundary"
    )


# ---------------------------------------------------------------------
# gt_annotations_lookup(): stubs the NuScenes devkit object itself
# (rather than needing the real ~GB dataset) to verify the two returned
# shapes are correctly paired from one pass over the same annotations.
# ---------------------------------------------------------------------

class _FakeNuScenes:
    def __init__(self, samples_by_token, anns_by_token):
        self._samples = samples_by_token
        self._anns = anns_by_token

    def get(self, table, token):
        if table == "sample":
            return self._samples[token]
        if table == "sample_annotation":
            return self._anns[token]
        raise KeyError(table)


def test_gt_annotations_lookup_pairs_trajectories_and_by_sample_from_the_same_data(tmp_path, monkeypatch):
    samples = [("sample_0000", "scene0", 0), ("sample_0001", "scene0", 500_000)]
    step0_dir, _ = _write_fake_step0_and_lidar_meta(
        tmp_path, [(sid, scene, ts, 0.0, 0.0) for sid, scene, ts in samples]
    )
    monkeypatch.setattr(legacy_config, "STEP0_DIR", step0_dir)

    fake_samples = {
        "tok_sample_0000": {"anns": ["ann_a_0"]},
        "tok_sample_0001": {"anns": ["ann_a_1", "ann_b_1"]},
    }
    fake_anns = {
        "ann_a_0": {"instance_token": "inst_a", "translation": [10.0, 20.0, 0.0]},
        "ann_a_1": {"instance_token": "inst_a", "translation": [11.0, 20.0, 0.0]},
        "ann_b_1": {"instance_token": "inst_b", "translation": [50.0, 60.0, 0.0]},
    }
    monkeypatch.setattr(adapters, "_get_nuscenes", lambda: _FakeNuScenes(fake_samples, fake_anns))

    gt_trajectories, gt_by_sample = adapters.gt_annotations_lookup()

    assert set(gt_trajectories.keys()) == {"inst_a", "inst_b"}
    assert len(gt_trajectories["inst_a"]) == 2   # seen at both samples
    assert len(gt_trajectories["inst_b"]) == 1   # seen only at sample_0001

    assert len(gt_by_sample["sample_0000"]) == 1
    assert gt_by_sample["sample_0000"][0][0] == "inst_a"
    assert len(gt_by_sample["sample_0001"]) == 2
    assert {entry[0] for entry in gt_by_sample["sample_0001"]} == {"inst_a", "inst_b"}

    # same underlying position data in both shapes -- not two independently-built views
    inst_a_point_at_sample1 = next(p for p in gt_trajectories["inst_a"] if p.sample_id == "sample_0001")
    by_sample_entry = next(e for e in gt_by_sample["sample_0001"] if e[0] == "inst_a")
    assert inst_a_point_at_sample1.x == by_sample_entry[1]
    assert inst_a_point_at_sample1.y == by_sample_entry[2]
