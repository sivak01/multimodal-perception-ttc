"""
v2/eval_l3/v1_adapters.py — the only file that reads v1's own output
(output/step_4/fused_tracks_all.csv, output/step_5/ttc_<sensor>.csv)
for the H6 analysis. Reads existing v1 output only -- never modifies,
re-runs, or re-derives anything a v1 notebook already computed.

Lock-once matching for v1 deliberately REUSES v2.evaluate.
match_track_to_ground_truth() unchanged, rather than re-implementing
lock-once matching a second time: a v1 track's CSV rows are grouped by
its own track-id column (confusingly named "fused_id" even for
single-sensor CSVs -- see repo-root CLAUDE.md) into a v2-shaped
TrackSnapshot-compatible "history" list, which is exactly what that
function already expects. v1 rows sit exactly on the real GT sample
grid already (no extrapolation needed the way an async v2 track
would), so each snapshot's own t is simply that sample's real ego
timestamp.
"""
import csv
from collections import defaultdict, namedtuple

import config as legacy_config   # repo-root config.py: paths only
from v2.evaluate import match_track_to_ground_truth

FakeSnapshot = namedtuple("FakeSnapshot", ["t", "x", "y", "vx", "vy", "sample_id"])


def build_v1_track_histories(csv_path, ego_by_sample):
    """{track_id: [FakeSnapshot, ...]} (time-sorted) from one
    output/step_5/ttc_<sensor>.csv, grouped by that CSV's own
    'fused_id' column. Skips rows whose sample_id has no ego state."""
    by_track = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample_id = row["sample_id"]
            ego = ego_by_sample.get(sample_id)
            if ego is None:
                continue
            try:
                x, y, vx, vy = float(row["x"]), float(row["y"]), float(row["vx"]), float(row["vy"])
            except (KeyError, ValueError):
                continue
            by_track[row["fused_id"]].append(
                FakeSnapshot(t=ego.t, x=x, y=y, vx=vx, vy=vy, sample_id=sample_id)
            )
    for tid in by_track:
        by_track[tid].sort(key=lambda s: s.t)
    return by_track


def match_v1_lock_once(csv_path, gt_by_sample, gt_ttc_lookup, ego_by_sample, match_dist_threshold=3.0):
    """Returns a list of pair dicts, same shape
    match_track_to_ground_truth() itself returns
    ({"t","sample_id","instance_id","pred_ttc","gt_ttc"}), with
    "track_id" added so callers can tell which v1 track produced each
    pair (not used by the matching logic itself)."""
    histories = build_v1_track_histories(csv_path, ego_by_sample)
    all_pairs = []
    for track_id, history in histories.items():
        pairs = match_track_to_ground_truth(
            history, gt_by_sample, gt_ttc_lookup, ego_by_sample, match_dist_threshold,
        )
        for p in pairs:
            p["track_id"] = track_id
        all_pairs.extend(pairs)
    return all_pairs


def build_v1_composition_lookup():
    """{(sample_id, fused_id): n_sensors} read directly from v1's OWN
    output/step_4/fused_tracks_all.csv 'sensors' column (e.g.
    "lidar+radar" -> 2) -- the real, already-recorded composition of
    v1's own fusion merge step. Unlike v2 (whose CentralTracker.Track
    has no per-snapshot sensor field -- see build_fused_dataset.py's
    own docstring for why that has to be independently re-derived via
    3 solo-sensor reruns), v1's Step 4 already wrote this down."""
    path = legacy_config.STEP4_DIR / "fused_tracks_all.csv"
    lookup = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = len(row["sensors"].split("+"))
            lookup[(row["sample_id"], row["fused_id"])] = n
    return lookup
