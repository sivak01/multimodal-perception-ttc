"""
v2/eval_l3/run_h9.py — H9 (revised) driver, docs/hypotheses-2026-10-10.md:
"Do we miss objects the sensors could see? How many false alarms?
Precision, recall and false alarms per km for warn at 2.5s and brake
at 1.5s. 57% of keys have infinite GT TTC, so this metric is needed."

Reuses the ALREADY-BUILT H6 per-key datasets
(output/eval_l3/h6_v1_keys.csv, h6_v2_keys.csv) for predicted TTC --
no tracker is rebuilt. Only real cost here: one GT-context build
(v2.eval_l3.gt_context.build_gt_context(), a single NuScenes metadata
load) -- far cheaper than H1/H3/H6's ~70-minute multi-tracker
rebuilds, since no event stream is materialized and no CentralTracker
runs.

Run with: python -m v2.eval_l3.run_h9 > v2_h9_report.txt
Writes output/eval_l3/h9_metrics.csv and output/eval_l3/h9_verdict.md.
Requires h6_v1_keys.csv / h6_v2_keys.csv to already exist (run H6
first if not).
"""
import csv as csv_mod
import time
from pathlib import Path

from v2.eval_l3.gt_context import build_gt_context
from v2.eval_l3.h6_common import read_csv
from v2.eval_l3.hypothesis_h9 import (
    THRESHOLDS, build_actual_hazard_events, build_predicted_events,
    total_km_driven_m, compute_h9_metrics,
)

OUT_DIR = Path("output/eval_l3")
BOUNDS = ("hungarian", "lock_once")
RUNS = ("lidar", "radar", "camera_mono", "fused")
PIPELINE_CSVS = {"v1": OUT_DIR / "h6_v1_keys.csv", "v2": OUT_DIR / "h6_v2_keys.csv"}


def run():
    print("Building GT context (one NuScenes load, no tracker rebuild)...")
    t0 = time.time()
    ctx = build_gt_context()
    print(f"  done in {time.time() - t0:.1f}s")

    print("Computing total km driven (whole dataset, within-scene consecutive deltas)...")
    total_km = total_km_driven_m(ctx.ego_by_sample, ctx.samples_index)
    print(f"  total: {total_km:.2f} km")

    # Actual hazard events don't depend on which (bound, run) we're
    # scoring, so build each threshold's population once and reuse it
    # across every (pipeline, bound, run) cell below.
    print("Building the REAL (ground-truth-only) hazard-event population, per threshold...")
    actual_events_by_threshold = {}
    for label, threshold in THRESHOLDS.items():
        actual_events_by_threshold[label] = build_actual_hazard_events(ctx.gt_ttc_lookup, ctx.samples_index, threshold)
        n = sum(len(v) for v in actual_events_by_threshold[label].values())
        print(f"  {label} (<= {threshold}s): {n} real hazard events across {len(actual_events_by_threshold[label])} instances")

    results = []
    for pipeline, csv_path in PIPELINE_CSVS.items():
        if not csv_path.exists():
            print(f"  (skipping {pipeline}: {csv_path} not found -- run H6 first)")
            continue
        print(f"Loading {pipeline} key rows from {csv_path}...")
        key_rows = read_csv(csv_path)
        print(f"  {len(key_rows)} rows")

        for bound in BOUNDS:
            for run_name in RUNS:
                filtered = [r for r in key_rows if r.bound == bound and r.run == run_name]
                if not filtered:
                    continue
                for label, threshold in THRESHOLDS.items():
                    predicted_events, covered = build_predicted_events(filtered, ctx.samples_index, threshold)
                    metrics = compute_h9_metrics(actual_events_by_threshold[label], predicted_events, covered, total_km)
                    results.append({
                        "pipeline": pipeline, "bound": bound, "run": run_name,
                        "event_label": label, "threshold_s": threshold, **metrics,
                    })
                    print(f"  [{pipeline}/{bound}/{run_name}/{label}] "
                          f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
                          f"precision={metrics['precision']} recall={metrics['recall']} "
                          f"FA/km={metrics['false_alarms_per_km']}")

    write_report(results, total_km)
    return results


def write_report(results, total_km):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "h9_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pipeline", "bound", "run", "event_label", "threshold_s",
                      "tp", "fp", "fn", "fn_covered", "fn_uncovered",
                      "n_actual_hazard_events", "n_predicted_events",
                      "precision", "recall", "false_alarms_per_km", "total_km"]
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {csv_path}")

    lines = []
    lines.append("# H9 (revised): precision / recall / false alarms per km\n")
    lines.append("warn <= 2.5s (config_l3.FCW_TTC_S), brake <= 1.5s (config_l3.AEB_TTC_S).\n")
    lines.append(f"Total distance driven (whole trainval dataset, within-scene consecutive deltas): **{total_km:.2f} km**.\n")
    lines.append("Methodology: event-based, not per-frame -- a real hazard or a predicted "
                  "alarm is one contiguous (by real timestamp) run of consecutive samples "
                  "below threshold for the same real instance, not one event per frame "
                  "(counting every frame separately would massively inflate both counts for "
                  "one slow approach spanning many samples). Recall's denominator "
                  "(n_actual_hazard_events) comes from "
                  "v2.evaluate.compute_ground_truth_ttc()'s FULL real-hazard population -- "
                  "every real annotated instance, matched or not -- NOT just the H6 CSV's "
                  "matched subset, which silently omits any hazard the system had zero "
                  "candidate anywhere near (see hypothesis_h9.py's module docstring).\n")
    lines.append("**Scope limit**: false alarms here are always tied to a real GT instance "
                  "(every H6 row already is). A \"phantom\" false alarm -- a predicted track "
                  "with no real object anywhere nearby -- is NOT counted; that needs the "
                  "pipeline's own full, unmatched per-frame track output, available on disk "
                  "for v1 (output/step_4/fused_tracks_all.csv) but not yet saved for v2.\n")

    lines.append("| pipeline | bound | run | event | threshold_s | TP | FP | FN (covered/uncovered) | precision | recall | FA/km |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['pipeline']} | {r['bound']} | {r['run']} | {r['event_label']} | {r['threshold_s']} | "
                      f"{r['tp']} | {r['fp']} | {r['fn_covered']}/{r['fn_uncovered']} | "
                      f"{r['precision']} | {r['recall']} | {r['false_alarms_per_km']} |")

    verdict_path = OUT_DIR / "h9_verdict.md"
    verdict_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {verdict_path}")


if __name__ == "__main__":
    run()
