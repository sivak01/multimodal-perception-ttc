"""
v2/miss_funnel.py — H3 (revised), docs/hypotheses-2026-10-10.md:
"Do we miss objects the sensors could see?" Builds a 4-stage miss
funnel per (GT key, sensor): expected -> detected -> tracked -> fused,
so a big drop between any two adjacent stages identifies exactly WHERE
real, physically-visible objects get lost (detection vs. tracking vs.
fusion), instead of one aggregate "missed" number that can't say which
stage caused it.

Supersedes v2/coverage_model.py's 2-stage (expected/actual) check for
H3's purposes: same lidar/radar expected-coverage idea, but with
camera_mono added, a tighter lidar threshold (>=10 real hits, not just
>0 -- a single stray point is not a reliable detection opportunity),
and detected/tracked/fused split apart so a miss can be attributed to
a specific stage. Additive only: does not modify coverage_model.py,
central_tracker.py, adapters.py, evaluate.py, evaluate_matched.py, or
any other existing v2 file.

Run against the CURRENTLY CONFIGURED dataset (config.py's
DATA_ROOT/NUSCENES_VERSION) via v2/run_miss_funnel.py -- this session
ran it against trainval specifically, since the hypothesis's own note
("mini result inflated") flags mini as the wrong dataset to trust here.

Expected thresholds (this session's own fixed choice, ASSUMED rather
than measured, same honesty convention as config.py's SENSOR_NOISE_VAR):
  - lidar:       num_lidar_pts >= 10 (nuScenes' own real-return count)
  - radar:       num_radar_pts >= 1
  - camera_mono: visibility_token != "1" (>=40% of the object's own
    silhouette visible in at least one camera -- nuScenes' own
    annotation-level visibility category: "1"=0-40%, "2"=40-60%,
    "3"=60-80%, "4"=80-100%. Not independently modeled per-camera,
    same scope-limit reasoning as coverage_model.py's own FOV note:
    nuScenes' own field already reflects real occlusion at that
    instant, which a geometric model built from calibration alone
    could not improve on.)

Funnel stages, each an independent boolean check at the SAME exact
(sample_id, instance_id) key:
  - detected_<sensor>: that sensor's own RAW per-frame SensorEvent
    (BEFORE any tracking at all) lands within MATCH_DIST_THRESHOLD of
    the real GT position at that exact sample.
  - tracked_<sensor>: that same key is covered by one of that sensor's
    SOLO CentralTracker tracks that survived with length>=2 -- a real,
    re-confirmed track, not a single transient hit immediately
    evicted. CentralTracker.all_tracks() includes 1-hit tracks too (no
    length filter exists anywhere in central_tracker.py, confirmed by
    reading it directly), so this stage needs an explicit filter --
    see run_miss_funnel.py's _tracked_tracks() -- that's what makes
    "tracked" genuinely distinct from "detected" rather than nearly
    coinciding with it.
  - fused_<sensor>: that same key is covered by the FUSED tracker's
    own output (merge_streams(lidar, radar, camera_mono) fed through
    one CentralTracker, same length>=2 filter) -- i.e. it survived all
    the way to the end-to-end pipeline result. One shared fused
    candidate set is used for all three sensor rows (the fused tracker
    covers the whole scene, not per-sensor).

Stages are checked independently, not forced monotonic by
construction -- in practice detected/tracked/fused are each expected
to be subsets of the stage before, since a track cannot exist without
a detection feeding it, but this module does not assume or enforce
that; it reports what the real data shows.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from v2 import adapters
from v2.evaluate_matched import MATCH_DIST_THRESHOLD

LIDAR_MIN_PTS = 10
RADAR_MIN_PTS = 1
CAMERA_MIN_VISIBILITY_TOKEN = "2"   # nuScenes: "1"=v0-40, "2"=v40-60, "3"=v60-80, "4"=v80-100

SENSORS = ("lidar", "radar", "camera_mono")


def build_annotation_lookup():
    """{(sample_id, instance_id): {"num_lidar_pts":, "num_radar_pts":,
    "visibility_token":}} -- a separate pass over the same real
    sample_annotation records adapters.gt_annotations_lookup() already
    walks, capturing the three fields this module's expected-coverage
    check needs (coverage_model.py's own build_annotation_point_count_lookup()
    captures only two of these; left untouched rather than modified)."""
    samples_index = adapters.load_samples_index()
    nusc = adapters._get_nuscenes()
    lookup = {}
    for sample_id in adapters._sample_ids_sorted():
        info = samples_index[sample_id]
        sample = nusc.get("sample", info["sample_token"])
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            lookup[(sample_id, ann["instance_token"])] = {
                "num_lidar_pts": ann["num_lidar_pts"],
                "num_radar_pts": ann["num_radar_pts"],
                "visibility_token": ann["visibility_token"],
            }
    return lookup


def expected_for_sensor(sensor, ann):
    if sensor == "lidar":
        return ann["num_lidar_pts"] >= LIDAR_MIN_PTS
    if sensor == "radar":
        return ann["num_radar_pts"] >= RADAR_MIN_PTS
    if sensor == "camera_mono":
        return ann["visibility_token"] >= CAMERA_MIN_VISIBILITY_TOKEN  # tokens are "1".."4"; lexicographic == numeric here
    raise ValueError(f"unknown sensor: {sensor!r}")


def _covered_by(objs_by_sample, sample_id, gx, gy, match_dist_threshold=MATCH_DIST_THRESHOLD):
    """objs_by_sample[sample_id]: any iterable of objects with .x/.y
    attributes -- raw SensorEvents (detected stage) and
    evaluate_matched.candidates_from_tracker()'s Candidate namedtuples
    (tracked/fused stages) both already satisfy this, so one function
    serves all three stages."""
    for obj in objs_by_sample.get(sample_id, []):
        if math.hypot(obj.x - gx, obj.y - gy) <= match_dist_threshold:
            return True
    return False


def raw_event_candidates_by_sample(events):
    """{sample_id: [SensorEvent, ...]} directly from a raw event
    stream -- no tracker involved at all. This IS the funnel's
    "detected" stage by definition."""
    out = defaultdict(list)
    for ev in events:
        out[ev.sample_id].append(ev)
    return out


def build_funnel_rows(gt_by_sample, ann_lookup, ego_by_sample,
                       raw_candidates_by_sample, tracked_candidates_by_sample,
                       fused_candidates_by_sample, match_dist_threshold=MATCH_DIST_THRESHOLD):
    """Generator: one row per (GT key, sensor in SENSORS): expected,
    detected, tracked, fused -- each an independent boolean check --
    plus range_m (ego-to-object distance, diagnostic only).

    A generator rather than a list by design: on the real trainval
    dataset this is 450K+ rows, and materializing them all at once
    would stack on top of the already-large event/track/candidate
    data this module's caller holds concurrently -- exactly the kind
    of extra peak-memory allocation that caused this script to be
    killed for low memory on a 16GB machine (see run_miss_funnel.py's
    write_funnel_csv_and_summarize(), which consumes this generator in
    one streaming pass instead of materializing it as a list).

    raw_candidates_by_sample / tracked_candidates_by_sample: {sensor:
        {sample_id: [...]}} -- one set per sensor.
    fused_candidates_by_sample: {sample_id: [...]} -- ONE shared set,
        not per-sensor (the fused tracker covers the whole scene).
    """
    for sample_id, gts in gt_by_sample.items():
        ego = ego_by_sample.get(sample_id)
        for instance_id, gx, gy in gts:
            ann = ann_lookup.get((sample_id, instance_id))
            if ann is None:
                continue
            range_m = round(math.hypot(gx - ego.x, gy - ego.y), 2) if ego is not None else None
            for sensor in SENSORS:
                expected = expected_for_sensor(sensor, ann)
                detected = _covered_by(raw_candidates_by_sample[sensor], sample_id, gx, gy, match_dist_threshold)
                tracked = _covered_by(tracked_candidates_by_sample[sensor], sample_id, gx, gy, match_dist_threshold)
                fused = _covered_by(fused_candidates_by_sample, sample_id, gx, gy, match_dist_threshold)
                yield {
                    "sample_id": sample_id, "instance_id": instance_id, "sensor": sensor,
                    "range_m": range_m, "expected": expected, "detected": detected,
                    "tracked": tracked, "fused": fused,
                }


def summarize_funnel(funnel_rows, sensor):
    rows = [r for r in funnel_rows if r["sensor"] == sensor]
    expected = [r for r in rows if r["expected"]]
    n_expected = len(expected)

    def _pct(flag):
        if not n_expected:
            return None
        return round(sum(1 for r in expected if r[flag]) / n_expected * 100, 2)

    missed_at_detection = [r["range_m"] for r in expected if not r["detected"] and r["range_m"] is not None]

    return {
        "sensor": sensor,
        "n_total_gt_keys": len(rows),
        "n_expected": n_expected,
        "pct_detected_of_expected": _pct("detected"),
        "pct_tracked_of_expected": _pct("tracked"),
        "pct_fused_of_expected": _pct("fused"),
        "median_range_m_missed_at_detection": round(median(missed_at_detection), 2) if missed_at_detection else None,
    }


def write_funnel_csv(funnel_rows, path):
    path = Path(path)
    fieldnames = ["sample_id", "instance_id", "sensor", "range_m", "expected", "detected", "tracked", "fused"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(funnel_rows)


def write_funnel_csv_and_summarize(funnel_rows, path):
    """Single streaming pass over build_funnel_rows()'s generator:
    writes every row to `path` AND accumulates the same per-sensor
    counts summarize_funnel() computes from a materialized list --
    without ever holding the full row set in memory. Returns
    {sensor: summary_dict}, same shape summarize_funnel() returns per
    sensor. This is the memory-conscious entry point real runs should
    use; summarize_funnel() itself is kept (and still tested) for
    small/in-memory use (e.g. unit tests, exploratory analysis on an
    already-loaded CSV)."""
    path = Path(path)
    fieldnames = ["sample_id", "instance_id", "sensor", "range_m", "expected", "detected", "tracked", "fused"]

    counts = {
        sensor: {"n_total": 0, "n_expected": 0, "n_detected": 0, "n_tracked": 0, "n_fused": 0, "missed_ranges": []}
        for sensor in SENSORS
    }

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in funnel_rows:
            writer.writerow(row)
            c = counts[row["sensor"]]
            c["n_total"] += 1
            if row["expected"]:
                c["n_expected"] += 1
                if row["detected"]:
                    c["n_detected"] += 1
                else:
                    if row["range_m"] is not None:
                        c["missed_ranges"].append(row["range_m"])
                if row["tracked"]:
                    c["n_tracked"] += 1
                if row["fused"]:
                    c["n_fused"] += 1

    summaries = {}
    for sensor, c in counts.items():
        n_expected = c["n_expected"]

        def _pct(n):
            return round(n / n_expected * 100, 2) if n_expected else None

        summaries[sensor] = {
            "sensor": sensor,
            "n_total_gt_keys": c["n_total"],
            "n_expected": n_expected,
            "pct_detected_of_expected": _pct(c["n_detected"]),
            "pct_tracked_of_expected": _pct(c["n_tracked"]),
            "pct_fused_of_expected": _pct(c["n_fused"]),
            "median_range_m_missed_at_detection": round(median(c["missed_ranges"]), 2) if c["missed_ranges"] else None,
        }
    return summaries
