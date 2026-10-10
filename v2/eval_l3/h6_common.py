"""
v2/eval_l3/h6_common.py — the shared row shape and CSV I/O for H6's
per-key dataset, used identically by both the v1 and v2 builders (and
never mixed between them -- rule 2 forbids comparing v1 numbers with
v2 numbers, so build_h6_dataset_v1.py and build_h6_dataset_v2.py each
write their OWN separate CSV; hypothesis_h6.py is run once per file).
"""
import csv
import math
from collections import namedtuple
from pathlib import Path

KeyRow = namedtuple("KeyRow", [
    "bound", "run", "scene_id", "sample_id", "instance_id",
    "pred_ttc", "gt_ttc", "range_bucket", "class_group", "motion", "odd",
    "in_corridor", "n_sensors",
])


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(KeyRow._fields)
        for r in rows:
            writer.writerow(list(r))


def _parse_range_bucket(s):
    """Mirrors multiple_comparisons_h1.py's own parser: literal_eval
    can't handle the bare word "inf" in "(60, inf)"."""
    lo_str, hi_str = s.strip("()").split(",")
    hi_str = hi_str.strip()
    hi = float("inf") if hi_str == "inf" else float(hi_str)
    return (float(lo_str.strip()), hi)


def _parse_optional_str(s):
    return s if s != "" else None


def _parse_optional_bool(s):
    if s == "":
        return None
    return s == "True"


def _parse_optional_int(s):
    return int(s) if s != "" else None


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            rows.append(KeyRow(
                bound=d["bound"], run=d["run"], scene_id=d["scene_id"],
                sample_id=d["sample_id"], instance_id=d["instance_id"],
                pred_ttc=float(d["pred_ttc"]), gt_ttc=float(d["gt_ttc"]),
                range_bucket=_parse_range_bucket(d["range_bucket"]),
                class_group=_parse_optional_str(d["class_group"]),
                motion=_parse_optional_str(d["motion"]),
                odd=d["odd"],
                in_corridor=_parse_optional_bool(d["in_corridor"]),
                n_sensors=_parse_optional_int(d["n_sensors"]),
            ))
    return rows


def abs_error(row):
    if math.isfinite(row.pred_ttc) and math.isfinite(row.gt_ttc):
        return abs(row.pred_ttc - row.gt_ttc)
    return None
