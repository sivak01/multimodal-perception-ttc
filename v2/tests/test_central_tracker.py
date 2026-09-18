from v2.central_tracker import CentralTracker
from v2.event_stream import SensorEvent, merge_streams


def ev(t, sensor, x, y, sample_id="s0", scene_token="scene0"):
    return SensorEvent(t=t, sensor=sensor, x=x, y=y, sample_id=sample_id, scene_token=scene_token)


def test_two_sensors_observing_one_moving_object_yield_one_track():
    """The core R1/R2/R6 claim of this whole design: feeding the SAME
    real object's detections from two different sensors, interleaved
    by real timestamp, must produce ONE track that ends up
    multi-sensor -- not two separate single-sensor tracks."""
    true_vx = 5.0   # m/s, moving along +x from the origin

    lidar_events = [ev(t, "lidar", true_vx * t, 0.0) for t in [0.0, 0.10, 0.20, 0.30, 0.40]]
    radar_events = [ev(t, "radar", true_vx * t, 0.0) for t in [0.05, 0.15, 0.25, 0.35, 0.45]]

    tracker = CentralTracker()
    tracker.run(merge_streams(lidar_events, radar_events))
    tracker.finalize()

    tracks = tracker.all_tracks()
    assert len(tracks) == 1, f"expected exactly one track, got {len(tracks)}: {list(tracks.keys())}"

    track = next(iter(tracks.values()))
    assert track.sensors_seen == {"lidar", "radar"}
    assert track.is_multi_sensor is True
    assert track.n_updates == 10   # 1 initial detection + 9 subsequent updates

    # and the filter should have actually converged to something close
    # to the true velocity, not just "matched enough to not split"
    vx, vy = track.trusted_velocity(min_speed=1.0)
    assert abs(vx - true_vx) < 1.5
    assert abs(vy) < 1.0


def test_two_genuinely_different_simultaneous_objects_stay_two_tracks():
    """The other side of the same claim: gating must still correctly
    keep unrelated objects apart -- this isn't a design that merges
    everything indiscriminately."""
    near_object = [ev(t, "lidar", 5.0 * t, 0.0) for t in [0.0, 0.1, 0.2, 0.3]]
    far_object = [ev(t, "radar", 100.0, 100.0 + 0.01 * t) for t in [0.02, 0.12, 0.22, 0.32]]

    tracker = CentralTracker()
    tracker.run(merge_streams(near_object, far_object))
    tracker.finalize()

    tracks = tracker.all_tracks()
    assert len(tracks) == 2
    sensor_sets = sorted(tuple(sorted(t.sensors_seen)) for t in tracks.values())
    assert sensor_sets == [("lidar",), ("radar",)]


def test_time_based_eviction_forces_a_new_track_not_a_stale_match():
    """R5: without eviction, a track left unpredicted for a long time
    would accumulate huge process-noise covariance and happily
    Mahalanobis-match almost anything -- eviction must remove it BEFORE
    gating gets a chance to do that, based on real elapsed seconds
    since its last real update."""
    tracker = CentralTracker(max_missed_seconds=0.5)

    tracker.process_event(ev(0.0, "lidar", 0.0, 0.0))
    assert len(tracker.active_tracks) == 1
    first_tid = next(iter(tracker.active_tracks))

    # 10 seconds later, same position -- far beyond max_missed_seconds,
    # so the original track must already be gone by the time gating runs
    tracker.process_event(ev(10.0, "lidar", 0.0, 0.0))

    assert first_tid in tracker.finished_tracks
    assert first_tid not in tracker.active_tracks
    tracker.finalize()
    assert len(tracker.all_tracks()) == 2   # the stale original + the new one, never merged


def test_scene_boundary_reset_never_carries_a_track_across_scenes():
    """R9: on scene_token change, every active track is finalized
    before continuing -- an identical position in a new scene must
    start a brand-new track, never extend the old one."""
    tracker = CentralTracker()

    tracker.process_event(ev(0.0, "lidar", 0.0, 0.0, scene_token="scene0"))
    first_tid = next(iter(tracker.active_tracks))

    tracker.process_event(ev(0.1, "lidar", 0.0, 0.0, scene_token="scene1"))

    assert first_tid in tracker.finished_tracks
    assert first_tid not in tracker.active_tracks
    tracker.finalize()
    assert len(tracker.all_tracks()) == 2


def test_unmatched_detection_spawns_a_new_track_with_correct_initial_sensor():
    tracker = CentralTracker()
    tracker.process_event(ev(0.0, "camera_mono", 3.0, 4.0))
    assert len(tracker.active_tracks) == 1
    track = next(iter(tracker.active_tracks.values()))
    assert track.sensors_seen == {"camera_mono"}
    assert track.is_multi_sensor is False
    assert tuple(track.position) == (3.0, 4.0)


def test_track_history_records_one_snapshot_per_real_update():
    """evaluate.py (deliverable 3) needs a track's trajectory over
    time, not just its final state, to match against ground truth at
    each real sample it touched."""
    events = [ev(t, "lidar", 5.0 * t, 0.0, sample_id=f"sample_{i:04d}")
              for i, t in enumerate([0.0, 0.1, 0.2, 0.3])]

    tracker = CentralTracker()
    tracker.run(events)
    tracker.finalize()

    track = next(iter(tracker.all_tracks().values()))
    assert len(track.history) == 4
    assert [snap.sample_id for snap in track.history] == ["sample_0000", "sample_0001", "sample_0002", "sample_0003"]
    assert [snap.t for snap in track.history] == [0.0, 0.1, 0.2, 0.3]
