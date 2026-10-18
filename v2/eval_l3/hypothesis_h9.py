"""
v2/eval_l3/hypothesis_h9.py — H9 (revised), docs/hypotheses-2026-10-10.md:
"How many false alarms? Precision, recall and false alarms per km for
warn at 2.5s and brake at 1.5s." Reuses config_l3.py's FCW_TTC_S/
AEB_TTC_S (reserved there explicitly "for a future hypothesis, not
used by H1 itself" -- this is that hypothesis) as the warn/brake
thresholds, and the ALREADY-BUILT H6 per-key datasets (h6_v1_keys.csv
/ h6_v2_keys.csv) for predicted TTC -- no tracker is rebuilt here.

Why NOT just threshold the H6 CSVs directly: those rows only exist
where the per-sample matcher (Hungarian or lock-once) found some real
GT instance within MATCH_DIST_THRESHOLD_M of a predicted candidate --
a real hazard the system had ZERO candidate anywhere near is silently
ABSENT from that CSV, not represented with pred_ttc=inf. Counting
false negatives only from H6's own rows would systematically overstate
recall by hiding every completely-undetected hazard. Fixed here by
building the REAL hazard-event population independently from
v2.evaluate.compute_ground_truth_ttc() (every real annotated instance,
matched or not -- the same GT-only computation gt_context.py already
shares between the v1/v2 H6 builders), and only using the H6 CSVs for
the PREDICTED side.

Event definition (not per-frame): per real instance_id, GT/predicted
"danger" is a rising-edge run of consecutive (by real timestamp, not
sample_id string order) samples where ttc <= threshold. Counting every
FRAME below threshold as a separate alarm/hazard would massively
inflate both the false-alarm count and the hazard count for a single
slow approach spanning many frames -- not how FCW/AEB false-alarm
rates are reported in the literature (alarms/km, not frames/km).

TP/FP/FN, per (pipeline, bound, run, threshold):
  - TP: a predicted event overlaps (shares >=1 real sample_id) an
    actual hazard event for the SAME instance_id.
  - FN: an actual hazard event with no overlapping predicted event for
    that instance -- split into:
      - FN_covered: the H6 CSV has >=1 row for that instance during
        the hazard's own sample range (so SOMETHING was tracked
        nearby, it just never predicted ttc<=threshold) -- an
        algorithm/filtering miss.
      - FN_uncovered: the H6 CSV has ZERO rows for that instance
        during the hazard's own sample range at all -- the system
        never had any candidate within match distance there; a
        detection/coverage miss, not a filtering one. Same
        distinction H3's miss funnel already draws (expected-but-
        undetected), reapplied here at the hazard-event level.
  - FP: a predicted event for a real instance_id whose sample range
    does not overlap any real hazard event for that SAME instance --
    a false alarm on a real but genuinely non-hazardous object.

Scope limit, stated rather than hidden: FP here is always tied to a
REAL GT instance, because every H6 CSV row is itself already matched
to one. A "phantom" false alarm -- a predicted track with NO real
object anywhere nearby at all -- cannot be measured from this CSV and
is NOT counted here. Measuring that needs the pipeline's own full,
unmatched per-frame track output (available on disk already for v1,
output/step_4/fused_tracks_all.csv; not yet saved for v2), which is
real further work, not assumed away.

False alarms per km: FP count / total km driven over the WHOLE
dataset (every sample, not just ones with a GT hazard or a match) --
see total_km_driven_m(). This needs ego_by_sample only, no tracker.
"""
import math
from collections import defaultdict

from v2.eval_l3 import config_l3 as cfg

THRESHOLDS = {"warn": cfg.FCW_TTC_S, "brake": cfg.AEB_TTC_S}


def _sorted_sample_ids(sample_ids, samples_index):
    return sorted(sample_ids, key=lambda sid: samples_index[sid]["timestamp_us"])


def _events_from_ttc_by_instance(ttc_by_instance_sample, samples_index, threshold):
    """ttc_by_instance_sample: {instance_id: {sample_id: ttc}}.
    Returns {instance_id: [frozenset(sample_ids), ...]} -- one entry
    per rising-edge, time-contiguous run where ttc <= threshold."""
    events_by_instance = {}
    for instance_id, by_sample in ttc_by_instance_sample.items():
        ordered = _sorted_sample_ids(by_sample.keys(), samples_index)
        events = []
        current = []
        for sid in ordered:
            ttc = by_sample[sid]
            if ttc <= threshold:
                current.append(sid)
            elif current:
                events.append(frozenset(current))
                current = []
        if current:
            events.append(frozenset(current))
        if events:
            events_by_instance[instance_id] = events
    return events_by_instance


def build_actual_hazard_events(gt_ttc_lookup, samples_index, threshold):
    """gt_ttc_lookup: {(sample_id, instance_id): {"ttc":, "pos":}} --
    v2.evaluate.compute_ground_truth_ttc()'s own output, covering
    EVERY real annotated instance regardless of whether the pipeline
    ever detected it."""
    by_instance = defaultdict(dict)
    for (sample_id, instance_id), entry in gt_ttc_lookup.items():
        by_instance[instance_id][sample_id] = entry["ttc"]
    return _events_from_ttc_by_instance(by_instance, samples_index, threshold)


def build_predicted_events(key_rows, samples_index, threshold):
    """key_rows: h6_common.KeyRow list, ALREADY FILTERED to one
    (bound, run). Also returns covered_samples_by_instance (every
    sample_id this (bound, run) ever produced a row for, per
    instance, regardless of threshold) -- needed to tell FN_covered
    apart from FN_uncovered."""
    by_instance = defaultdict(dict)
    covered_samples_by_instance = defaultdict(set)
    for row in key_rows:
        by_instance[row.instance_id][row.sample_id] = row.pred_ttc
        covered_samples_by_instance[row.instance_id].add(row.sample_id)
    events = _events_from_ttc_by_instance(by_instance, samples_index, threshold)
    return events, covered_samples_by_instance


def total_km_driven_m(ego_by_sample, samples_index):
    """Total distance in km, summing consecutive (by real timestamp)
    ego position deltas WITHIN each scene only -- never across a
    scene boundary, same R9 scene-isolation discipline the rest of
    this codebase already follows."""
    by_scene = defaultdict(list)
    for sample_id, info in samples_index.items():
        if sample_id not in ego_by_sample:
            continue
        by_scene[info["scene_name"]].append(sample_id)

    total_m = 0.0
    for scene_name, sample_ids in by_scene.items():
        ordered = _sorted_sample_ids(sample_ids, samples_index)
        for i in range(1, len(ordered)):
            e0, e1 = ego_by_sample[ordered[i - 1]], ego_by_sample[ordered[i]]
            total_m += math.hypot(e1.x - e0.x, e1.y - e0.y)
    return total_m / 1000.0


def compute_h9_metrics(actual_events, predicted_events, covered_samples_by_instance, total_km):
    """actual_events / predicted_events: {instance_id: [frozenset(sample_ids), ...]}.
    Returns the full TP/FP/FN + precision/recall/false-alarms-per-km
    breakdown."""
    tp = 0
    fp = 0
    fn_covered = 0
    fn_uncovered = 0

    all_instances = set(actual_events) | set(predicted_events)
    for instance_id in all_instances:
        a_events = actual_events.get(instance_id, [])
        p_events = predicted_events.get(instance_id, [])
        covered_samples = covered_samples_by_instance.get(instance_id, set())

        for a in a_events:
            if any(a & p for p in p_events):
                tp += 1
            elif a & covered_samples:
                fn_covered += 1
            else:
                fn_uncovered += 1

        for p in p_events:
            if not any(a & p for a in a_events):
                fp += 1

    fn = fn_covered + fn_uncovered
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    fa_per_km = fp / total_km if total_km else None

    return {
        "tp": tp, "fp": fp, "fn": fn, "fn_covered": fn_covered, "fn_uncovered": fn_uncovered,
        "n_actual_hazard_events": sum(len(v) for v in actual_events.values()),
        "n_predicted_events": sum(len(v) for v in predicted_events.values()),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "false_alarms_per_km": round(fa_per_km, 4) if fa_per_km is not None else None,
        "total_km": round(total_km, 2),
    }
