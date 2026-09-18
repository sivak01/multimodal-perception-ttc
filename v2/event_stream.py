"""
v2/event_stream.py — the SensorEvent type, and merging multiple
per-sensor streams into one globally chronological timeline.

R2 (v2_architecture_brief.md): each detection is a SensorEvent carrying
its own real sensor timestamp; merge_streams() uses heapq.merge to
produce one time-ordered stream across sensors without ever
materializing and re-sorting a combined list.
"""
import heapq
from dataclasses import dataclass


@dataclass(frozen=True)
class SensorEvent:
    """One detection from one sensor, already in the form CentralTracker
    needs. adapters.py (not yet built) is responsible for producing
    these from the existing pipeline's on-disk outputs.

    t: this detection's own real sensor timestamp, in seconds, on a
        single consistent monotonic epoch shared across all sensors --
        NOT a shared/nominal sample interval. Two events from different
        sensors legitimately have different t even for "the same
        instant" in the old frame-based pipeline, and that difference
        is the whole point of this design (R2).
    sensor: one of "lidar", "radar", "camera_mono". There is no
        "camera" sensor in v2 -- see v2/config.py.
    x, y: position, already in the GLOBAL frame (metres). No sensor-
        local coordinate ever reaches this type.
    sample_id: the nuScenes sample this detection's parent frame
        belongs to, carried through so ground-truth matching can look
        up sample_annotation later (evaluate.py, not yet built).
    scene_token: the nuScenes scene this detection belongs to --
        CentralTracker resets on every change (R9).
    """
    t: float
    sensor: str
    x: float
    y: float
    sample_id: str
    scene_token: str


def merge_streams(*event_streams):
    """Merge any number of per-sensor SensorEvent iterables -- each
    ALREADY sorted by its own .t -- into one globally chronological
    iterator, keyed on .t. Uses heapq.merge (R2): O(N log k) for k
    streams, never re-sorts a combined list, and works on lazy
    generators just as well as lists, so adapters.py can stream
    directly from disk without holding everything in memory at once.
    """
    return heapq.merge(*event_streams, key=lambda e: e.t)
