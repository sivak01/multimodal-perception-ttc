"""
v2/tests/test_diagnostics.py — fully synthetic (monkeypatched on-disk
yolo_mono fixtures, no real dataset needed), same pattern as
test_adapters.py.
"""
import json

import config as legacy_config
from v2.central_tracker import Track, TrackSnapshot
from v2.diagnostics import flag_class_inconsistent_tracks
from v2.kalman_track import ConstantVelocityKF


def _write_yolo_mono_detection(tmp_path, sample_id, channel, global_x, global_y, class_name):
    yolo_dir = tmp_path / "step_2" / "yolo_mono" / sample_id
    yolo_dir.mkdir(parents=True, exist_ok=True)
    detections = [{
        "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "confidence": 0.9,
        "class_id": 0, "class_name": class_name, "has_3d_position": True,
        "global_x": global_x, "global_y": global_y, "global_z": 0.0,
        "estimated_depth_m": 10.0, "assumed_height_m": 1.5,
    }]
    with open(yolo_dir / f"{channel}.json", "w", encoding="utf-8") as f:
        json.dump(detections, f)


def _make_track(track_id, positions_and_samples):
    """positions_and_samples: [(x, y, sample_id), ...]. Builds a real
    Track/KF so .history has real TrackSnapshots at those positions."""
    x0, y0, sid0 = positions_and_samples[0]
    kf = ConstantVelocityKF(x0, y0, 0.0, sigma_a=2.0)
    track = Track(track_id, kf, "camera_mono", sid0)
    t = 1.0
    for x, y, sid in positions_and_samples[1:]:
        kf.predict(t)
        kf.update(x, y, 1.7, t)
        track.history.append(TrackSnapshot(t=t, x=x, y=y, vx=0.0, vy=0.0, sample_id=sid))
        t += 1.0
    return track


def test_flags_a_track_whose_nearest_class_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy_config, "STEP2_DIR", tmp_path / "step_2")
    _write_yolo_mono_detection(tmp_path, "sample_0000", "CAM_FRONT", 0.0, 0.0, "car")
    _write_yolo_mono_detection(tmp_path, "sample_0001", "CAM_FRONT", 5.0, 0.0, "car")
    _write_yolo_mono_detection(tmp_path, "sample_0002", "CAM_FRONT", 10.0, 0.0, "person")

    track = _make_track("trk_A", [(0.0, 0.0, "sample_0000"), (5.0, 0.0, "sample_0001"), (10.0, 0.0, "sample_0002")])

    flagged = flag_class_inconsistent_tracks({"trk_A": track})

    assert len(flagged) == 1
    entry = flagged[0]
    assert entry["track_id"] == "trk_A"
    assert entry["n_class_changes"] == 1
    transition = entry["transitions"][0]
    assert transition["from_class"] == "car"
    assert transition["to_class"] == "person"
    assert transition["sample_id"] == "sample_0002"
    assert transition["n_prior_updates"] == 2   # 2 real updates (indices 0,1) happened before this transition


def test_does_not_flag_a_consistently_classed_track(tmp_path, monkeypatch):
    """Same scenario shape as the flagged test, but every raw detection
    is 'car' throughout -- must NOT be flagged, proving this isn't
    triggered by mere position movement."""
    monkeypatch.setattr(legacy_config, "STEP2_DIR", tmp_path / "step_2")
    _write_yolo_mono_detection(tmp_path, "sample_0000", "CAM_FRONT", 0.0, 0.0, "car")
    _write_yolo_mono_detection(tmp_path, "sample_0001", "CAM_FRONT", 5.0, 0.0, "car")
    _write_yolo_mono_detection(tmp_path, "sample_0002", "CAM_FRONT", 10.0, 0.0, "car")

    track = _make_track("trk_B", [(0.0, 0.0, "sample_0000"), (5.0, 0.0, "sample_0001"), (10.0, 0.0, "sample_0002")])

    flagged = flag_class_inconsistent_tracks({"trk_B": track})
    assert flagged == []


def test_known_blind_spot_same_class_different_object_is_not_flagged(tmp_path, monkeypatch):
    """Documents the function's own stated limitation: a track that
    conflates two DIFFERENT real objects of the SAME class produces no
    class-label signal at all, so it is NOT flagged -- this is
    confirmed-expected behaviour (see the module docstring's
    known-blind-spot note, and the real 118-snapshot pedestrian-cluster
    track this was found against), not a bug to fix here."""
    monkeypatch.setattr(legacy_config, "STEP2_DIR", tmp_path / "step_2")
    _write_yolo_mono_detection(tmp_path, "sample_0000", "CAM_FRONT", 0.0, 0.0, "person")
    _write_yolo_mono_detection(tmp_path, "sample_0001", "CAM_FRONT", 1.0, 0.0, "person")   # a DIFFERENT real person, same class

    track = _make_track("trk_C", [(0.0, 0.0, "sample_0000"), (1.0, 0.0, "sample_0001")])

    flagged = flag_class_inconsistent_tracks({"trk_C": track})
    assert flagged == []


def test_skips_tracks_shorter_than_min_history(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy_config, "STEP2_DIR", tmp_path / "step_2")
    _write_yolo_mono_detection(tmp_path, "sample_0000", "CAM_FRONT", 0.0, 0.0, "car")

    kf = ConstantVelocityKF(0.0, 0.0, 0.0, sigma_a=2.0)
    track = Track("trk_D", kf, "camera_mono", "sample_0000")   # single-snapshot track

    flagged = flag_class_inconsistent_tracks({"trk_D": track}, min_history=2)
    assert flagged == []
