"""
v2/diagnostics.py — post-hoc track-quality checks that need to reach
back into the ORIGINAL raw sensor detections (not just a track's own
filtered state), to answer a question gating/covariance alone can
never structurally answer: did this track's identity silently drift
from one real object onto a different one?

Deliberately kept separate from evaluate.py, which is explicitly
data-source-agnostic (per its own module docstring, taking only plain
Python structures -- no file paths, no nuScenes objects). This module
needs real on-disk pipeline output, same as adapters.py, so it lives
here instead rather than compromising evaluate.py's own contract.

Found useful, and worth keeping permanently rather than as a one-off
script: while investigating whether raising PROCESS_NOISE_SIGMA_A (to
fix lidar/radar fragmentation) introduces cross-object over-merging in
camera_mono, position/GT-instance proximity alone proved untrustworthy
in dense scenes (the same lesson already learned once during the
original LiDAR fragmentation investigation) -- a manual nearest-raw-
detection-class trace caught a track that had silently absorbed a car,
then several different pedestrians, under one track_id. This is cheap
to run on every future pipeline/tuning change, not just this one
decision, so it belongs in the codebase rather than a throwaway script.
"""
import json

import config as legacy_config


def _nearest_camera_mono_detection(sample_id, x, y, cache):
    """Nearest raw YOLO detection (by global position) to (x, y) among
    every camera_mono detection recorded for this sample, across all 6
    camera channels.

    cache: {sample_id: [(global_x, global_y, detection_dict), ...]} --
    the caller should reuse ONE cache dict across many calls (e.g. one
    per flag_class_inconsistent_tracks() run), since many different
    tracks will revisit the same sample_id and each sample's on-disk
    JSON should only be read once, not once per track per snapshot.

    Returns (detection_dict, distance) or (None, inf) if this sample
    has no camera_mono output on disk at all.
    """
    if sample_id not in cache:
        entries = []
        yolo_dir = legacy_config.STEP2_DIR / "yolo_mono" / sample_id
        if yolo_dir.exists():
            for cam_file in sorted(yolo_dir.glob("*.json")):
                with open(cam_file, encoding="utf-8") as f:
                    dets = json.load(f)
                for det in dets:
                    if det.get("has_3d_position"):
                        entries.append((det["global_x"], det["global_y"], det))
        cache[sample_id] = entries

    best_det, best_dist = None, float("inf")
    for dx, dy, det in cache[sample_id]:
        d = ((dx - x) ** 2 + (dy - y) ** 2) ** 0.5
        if d < best_dist:
            best_dist, best_det = d, det
    return best_det, best_dist


def flag_class_inconsistent_tracks(tracks, min_history=2):
    """
    For every track with >= min_history snapshots, finds the nearest
    raw detection to EACH snapshot (by position) and checks whether its
    reported class label changes anywhere across the track's life. A
    track whose own state silently drifted onto a DIFFERENT semantic
    class of real object (e.g. a car, then a pedestrian) is an
    unambiguous over-merge -- position/covariance-only gating has no
    notion of class at all, so it cannot structurally distinguish this
    from ordinary continued tracking of one object.

    KNOWN BLIND SPOT, not fixed by this function: a track that
    conflates several DISTINCT real objects of the SAME class (e.g.
    several different people standing close together in a group) will
    NOT be flagged here, since the nearest class label never changes.
    Confirmed to actually occur on the real dataset (a 118-snapshot
    camera_mono track absorbing detections from a group of ~5 nearby
    pedestrians, all "person", zero class changes) -- this function is
    a fast, automatable complement for the CROSS-class case, not a
    replacement for a position/GT-instance-switch-based check for the
    same-class case.

    tracks: {track_id: Track}, from CentralTracker.all_tracks().
    Currently only supports camera_mono detections (the only ones in
    this pipeline carrying a class label -- lidar clusters and radar
    points carry none); there is no `sensor` parameter since there is
    nothing else to plug in yet.

    Returns a list of dicts, one per FLAGGED track, sorted by
    n_class_changes descending then history length descending:
        {"track_id":, "history_len":, "n_class_changes":,
         "transitions": [{"index_in_history":, "sample_id":,
                           "from_class":, "to_class":,
                           "n_prior_updates":}, ...]}
    A transition's n_prior_updates is how many real updates the track
    had BEFORE it -- a transition with a low count happened while the
    track's posterior covariance was still loose (freshly spawned), so
    it is the least "protected" by an already-tightened P and the
    highest-priority one to manually trace first.
    """
    flagged = []
    cache = {}
    for track_id, track in tracks.items():
        if len(track.history) < min_history:
            continue

        classes = []
        for snap in track.history:
            det, _dist = _nearest_camera_mono_detection(snap.sample_id, snap.x, snap.y, cache)
            classes.append(det["class_name"] if det else None)

        transitions = []
        for i in range(1, len(classes)):
            if classes[i] is not None and classes[i - 1] is not None and classes[i] != classes[i - 1]:
                transitions.append({
                    "index_in_history": i,
                    "sample_id": track.history[i].sample_id,
                    "from_class": classes[i - 1],
                    "to_class": classes[i],
                    "n_prior_updates": i,
                })

        if transitions:
            flagged.append({
                "track_id": track_id,
                "history_len": len(track.history),
                "n_class_changes": len(transitions),
                "transitions": transitions,
            })

    flagged.sort(key=lambda f: (-f["n_class_changes"], -f["history_len"]))
    return flagged
