"""
v2/eval_l3/run_h6.py — H6 entry point: does fused beat each single
sensor on the same GT keys, for v1 AND v2 SEPARATELY (rule 2: never
compare v1 numbers with v2 numbers -- this script never does; it runs
the identical analysis twice, once per pipeline, and reports both
side by side without ever subtracting or ranking one against the
other).

Run with: python -m v2.eval_l3.run_h6
Writes:
  output/eval_l3/h6_v2_keys.csv, output/eval_l3/h6_v1_keys.csv (the
  per-key datasets)
  output/eval_l3/h6_verdict.md (results table + verdict + limitations,
  per rule 8)
"""
import time
from pathlib import Path

from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3.h6_common import write_csv
from v2.eval_l3.build_h6_dataset_v2 import build as build_v2
from v2.eval_l3.build_h6_dataset_v1 import build as build_v1
from v2.eval_l3.hypothesis_h6 import (
    SENSORS, SLICES, PASS_FAIL_SLICE_NAME,
    run_all_slices, verdict_for, best_sensor_for_pass_fail,
)

OUT_DIR = Path("output/eval_l3")
BOUNDS = ("hungarian", "lock_once")
BOUND_LABEL = {"hungarian": "per-frame Hungarian (optimistic)", "lock_once": "lock-once (pessimistic)"}
PIPELINES = ("v2", "v1")


def _fmt_row(pipeline, bound, sensor, slice_name, stats):
    return (f"| {pipeline} | {BOUND_LABEL[bound]} | {sensor} | {slice_name} | {stats['n']} | "
            f"{stats['mean_fused_err']} | {stats['mean_sensor_err']} | {stats['mean_diff']} | "
            f"{stats['median_diff']} | {stats['fused_wins_pct']} | "
            f"[{stats['ci95_low']}, {stats['ci95_high']}] | "
            f"{stats['paired_sign_test']['p_value']} |")


def analyze_pipeline(key_rows, pipeline_label):
    """Returns (table_lines, pass_fail_results) for one pipeline's
    key_rows. pass_fail_results: {bound: {"best_sensor":, "stats":,
    "verdict":, "reason":, "per_sensor":}}."""
    table_lines = []
    pass_fail_results = {}

    for bound in BOUNDS:
        for sensor in SENSORS:
            results = run_all_slices(key_rows, bound, sensor)
            for slice_name, _restrict_fn in SLICES:
                table_lines.append(_fmt_row(pipeline_label, bound, sensor, slice_name, results[slice_name]))
            table_lines.append(_fmt_row(pipeline_label, bound, sensor, PASS_FAIL_SLICE_NAME,
                                         results[PASS_FAIL_SLICE_NAME]))

        best_sensor, per_sensor_stats = best_sensor_for_pass_fail(key_rows, bound)
        if best_sensor is None:
            pass_fail_results[bound] = {
                "best_sensor": None, "stats": None, "verdict": "INCONCLUSIVE",
                "reason": "no sensor has any paired keys in the in-corridor, TTC-0-5s slice",
                "per_sensor": per_sensor_stats,
            }
            continue
        best_stats = per_sensor_stats[best_sensor]
        verdict, reason = verdict_for(best_stats)
        pass_fail_results[bound] = {
            "best_sensor": best_sensor, "stats": best_stats, "verdict": verdict,
            "reason": reason, "per_sensor": per_sensor_stats,
        }

    return table_lines, pass_fail_results


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building v2 dataset (fresh tracker runs) ===")
    t0 = time.time()
    v2_rows = build_v2()
    print(f"  done in {time.time() - t0:.1f}s")
    write_csv(v2_rows, OUT_DIR / "h6_v2_keys.csv")
    print(f"  wrote {OUT_DIR / 'h6_v2_keys.csv'} ({len(v2_rows)} rows)")

    print("\n=== Building v1 dataset (existing CSVs only) ===")
    t0 = time.time()
    v1_rows = build_v1()
    print(f"  done in {time.time() - t0:.1f}s")
    write_csv(v1_rows, OUT_DIR / "h6_v1_keys.csv")
    print(f"  wrote {OUT_DIR / 'h6_v1_keys.csv'} ({len(v1_rows)} rows)")

    print("\n=== Analyzing v2 ===")
    v2_table, v2_pass_fail = analyze_pipeline(v2_rows, "v2")
    print("\n=== Analyzing v1 ===")
    v1_table, v1_pass_fail = analyze_pipeline(v1_rows, "v1")

    for pipeline, pf in (("v2", v2_pass_fail), ("v1", v1_pass_fail)):
        for bound, r in pf.items():
            print(f"[{pipeline}/{bound}] best_sensor={r['best_sensor']} -> {r['verdict']}")

    lines = []
    lines.append("# H6: does fused beat each single sensor, on the same GT keys?\n")
    lines.append("Run separately for v1 and v2 (rule 2: never compared against each other). "
                  "Paired diff = |fused_err| - |sensor_err| per shared key; negative = fused wins.\n")
    lines.append("Pass/fail rule (given verbatim): **CONFIRMED** only if fused beats the BEST single "
                  "sensor in the in-corridor, GT-TTC-0-5s band, with a scene-bootstrap 95% CI (of the "
                  "paired diff) that excludes 0. Otherwise **REFUTED** (CI includes 0, or favours the "
                  f"sensor) or **INCONCLUSIVE** (n < {cfg.MIN_N_FOR_RELIABLE_CI}).\n")

    lines.append("## Pass/fail determination\n")
    for pipeline, pf in (("v2", v2_pass_fail), ("v1", v1_pass_fail)):
        lines.append(f"\n### {pipeline}\n")
        for bound in BOUNDS:
            r = pf[bound]
            lines.append(f"**{BOUND_LABEL[bound]}**\n")
            if r["best_sensor"] is None:
                lines.append(f"- No sensor has paired keys in the in-corridor+0-5s band -- {r['verdict']}.\n")
                continue
            s = r["stats"]
            lines.append(f"- Best single sensor in this slice (lowest own mean error): **{r['best_sensor']}** "
                          f"(mean_sensor_err={s['mean_sensor_err']}, n={s['n']})")
            lines.append(f"- Fused mean error: {s['mean_fused_err']}; diff (fused-best): {s['mean_diff']}; "
                          f"95% CI [{s['ci95_low']}, {s['ci95_high']}]; fused wins {s['fused_wins_pct']}% of keys")
            lines.append(f"- paired sign test: {s['paired_sign_test']}")
            lines.append(f"- **VERDICT: {r['verdict']}** -- {r['reason']}\n")
            lines.append("  Other sensors in this same slice, for context:")
            for sensor, st in r["per_sensor"].items():
                if sensor == r["best_sensor"]:
                    continue
                lines.append(f"  - {sensor}: mean_sensor_err={st['mean_sensor_err']}, n={st['n']}")
            lines.append("")

    lines.append("\n## Full results table (all slices, both bounds, all 3 sensors, both pipelines)\n")
    lines.append("| pipeline | bound | sensor | slice | n | fused mean err | sensor mean err | "
                  "mean diff | median diff | % fused wins | 95% CI | sign-test p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines.extend(v2_table)
    lines.extend(v1_table)

    lines.append("\n## Limitations\n")
    lines.append("- v2's fused composition (n_sensors, for the k2/k3 slices) is independently derived "
                  "from the three solo-sensor trackers' own candidates (see build_h6_dataset_v2.py / "
                  "build_fused_dataset.py's docstring) -- v2's CentralTracker.Track carries no "
                  "per-snapshot sensor field. v1's composition is read directly from its own "
                  "output/step_4/fused_tracks_all.csv 'sensors' column -- v1's real, already-recorded "
                  "composition, not re-derived. This is a genuine methodological difference between "
                  "the two pipelines' k2/k3 slices, stated here rather than hidden by matching labels.")
    lines.append("- v1's lock-once bound reuses v2.evaluate.match_track_to_ground_truth() unchanged, "
                  "fed v1's own track-id ('fused_id' column) groupings -- v1 rows sit exactly on the "
                  "real GT sample grid already, so no extrapolation is needed (unlike a genuinely async "
                  "v2 track).")
    lines.append("- \"Best single sensor\" (for the pass/fail determination) is judged on EACH sensor's "
                  "own paired-with-fused population in the in-corridor+0-5s slice, not an unconditioned "
                  "standalone accuracy number -- consistent with rule 2's same-GT-keys fairness "
                  "requirement, but it does mean each sensor's own n can differ.")
    lines.append("- The GT-side enrichment (range/corridor/class/motion/odd) is identical code shared by "
                  "both pipelines (gt_context.py) -- a slice named the same way means the same thing in "
                  "both pipelines' tables, even though the two tables are never directly compared.")
    lines.append(f"- Scene-level bootstrap: {cfg.BOOTSTRAP_N_RESAMPLES} resamples, seed {cfg.BOOTSTRAP_SEED}, "
                  f"{cfg.BOOTSTRAP_CI_PCT}% CI, resampling the real scenes with replacement, per rule 4.")
    lines.append("- All thresholds (config_l3.py) were fixed before any H6 result was computed and were "
                  "not tuned afterward.")

    verdict_path = OUT_DIR / "h6_verdict.md"
    verdict_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {verdict_path}")
    return {"v2": v2_pass_fail, "v1": v1_pass_fail}


if __name__ == "__main__":
    run()
