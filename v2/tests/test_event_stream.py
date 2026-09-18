from v2.event_stream import SensorEvent, merge_streams


def ev(t, sensor="lidar"):
    return SensorEvent(t=t, sensor=sensor, x=0.0, y=0.0, sample_id="s", scene_token="scene0")


def test_merge_two_sorted_streams_into_one_chronological_stream():
    lidar = [ev(0.0), ev(0.2), ev(0.4)]
    radar = [ev(0.1, "radar"), ev(0.3, "radar")]
    merged = list(merge_streams(lidar, radar))
    ts = [e.t for e in merged]
    assert ts == sorted(ts)
    assert [e.sensor for e in merged] == ["lidar", "radar", "lidar", "radar", "lidar"]


def test_merge_three_streams():
    a = [ev(0.0, "lidar"), ev(3.0, "lidar")]
    b = [ev(1.0, "radar")]
    c = [ev(2.0, "camera_mono")]
    merged = list(merge_streams(a, b, c))
    assert [e.t for e in merged] == [0.0, 1.0, 2.0, 3.0]


def test_merge_is_lazy_and_works_on_generators():
    def gen():
        yield ev(0.0)
        yield ev(1.0)

    merged = merge_streams(gen(), [ev(0.5, "radar")])
    assert [e.t for e in merged] == [0.0, 0.5, 1.0]


def test_merge_empty_streams():
    assert list(merge_streams([], [])) == []
    assert list(merge_streams([ev(0.0)], [])) == [ev(0.0)]
