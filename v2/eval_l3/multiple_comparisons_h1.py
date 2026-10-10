"""
v2/eval_l3/multiple_comparisons_h1.py — applies a multiple-comparisons
correction across H1's 6 pre-specified (bound, slice) comparisons.

run_h1.py's own per-slice verdict used "does the 95% scene-bootstrap CI
of the MAE difference exclude 0" as its significance criterion. This
script computes the EXACT bootstrap-consistent two-sided p-value for
that same criterion (bootstrap.scene_bootstrap_p_value -- same
resamples, same seed, as the CI already reported, so "CI excludes 0"
and "p < alpha" agree by construction), then asks: across all 6
comparisons tested, does the one slice that looked CONFIRMED on its
own (lock-once, TTC 0-5s) survive a correction for having tested 6
hypotheses at once?

Reuses output/eval_l3/fused_enriched_keys.csv (already written by
run_h1.py) rather than re-running the ~8-minute tracker pipeline --
reconstructs the exact same EnrichedKeyRow objects from that CSV.

Run with: python -m v2.eval_l3.multiple_comparisons_h1
Writes output/eval_l3/h1_multiple_comparisons.md.
"""
import csv
from pathlib import Path

from v2.eval_l3 import config_l3 as cfg
from v2.eval_l3.build_fused_dataset import EnrichedKeyRow
from v2.eval_l3.bootstrap import (
    group_rows_by_scene, scene_bootstrap_p_value,
    bonferroni_correct, holm_bonferroni_correct, benjamini_hochberg_correct,
)
from v2.eval_l3.hypothesis_h1 import build_qualifying_groups, mae_stat, _diff_stat_fn
from v2.eval_l3.run_h1 import SLICES, BOUNDS, BOUND_LABEL

OUT_DIR = Path("output/eval_l3")
ENRICHED_CSV = OUT_DIR / "fused_enriched_keys.csv"


def _parse_optional_float(s):
    return float(s) if s != "" else None


def _parse_range_bucket(s):
    """literal_eval can't parse "(60, inf)" directly -- inf is a bare
    Name, not a numeric literal. Parse the two numbers out directly
    instead (lo is always a plain int/float; hi is either a number or
    the literal word "inf")."""
    lo_str, hi_str = s.strip("()").split(",")
    hi_str = hi_str.strip()
    hi = float("inf") if hi_str == "inf" else float(hi_str)
    return (float(lo_str.strip()), hi)


def load_enriched_rows(path=ENRICHED_CSV):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            rows.append(EnrichedKeyRow(
                bound=d["bound"], scene_id=d["scene_id"], sample_id=d["sample_id"],
                instance_id=d["instance_id"],
                pred_ttc=float(d["pred_ttc"]), gt_ttc=float(d["gt_ttc"]),
                abs_error=_parse_optional_float(d["abs_error"]),
                rel_error=_parse_optional_float(d["rel_error"]),
                range_m=float(d["range_m"]), range_bucket=_parse_range_bucket(d["range_bucket"]),
                class_group=d["class_group"] or None, motion=d["motion"] or None,
                odd=d["odd"],
                in_corridor=(d["in_corridor"] == "True") if d["in_corridor"] != "" else None,
                n_sensors_corroborating=int(d["n_sensors_corroborating"]),
                composition=d["composition"],
            ))
    return rows


def run():
    print(f"Loading enriched rows from {ENRICHED_CSV} (no pipeline re-run needed)...")
    rows = load_enriched_rows()
    print(f"  {len(rows)} rows loaded")

    comparisons = []   # (bound, slice_key, slice_label, p_value, mae_diff, n_groups)
    for bound in BOUNDS:
        for slice_key, slice_label, restrict_fn in SLICES:
            merged_rows, single_rows, groups = build_qualifying_groups(rows, bound, restrict_fn)
            pooled = merged_rows + single_rows
            rows_by_scene = group_rows_by_scene(pooled, scene_of_row=lambda r: r.scene_id)
            diff_point = _diff_stat_fn(mae_stat)(pooled)
            p_value = scene_bootstrap_p_value(
                rows_by_scene, _diff_stat_fn(mae_stat),
                n_resamples=cfg.BOOTSTRAP_N_RESAMPLES, seed=cfg.BOOTSTRAP_SEED,
            )
            comparisons.append({
                "bound": bound, "slice_key": slice_key, "slice_label": slice_label,
                "n_groups": len(groups), "mae_diff": diff_point, "p_value": p_value,
            })

    p_values = [c["p_value"] for c in comparisons]
    bonf = bonferroni_correct(p_values, alpha=0.05)
    holm = holm_bonferroni_correct(p_values, alpha=0.05)
    bh = benjamini_hochberg_correct(p_values, alpha=0.05)

    lines = []
    lines.append("# H1 multiple-comparisons correction (6 pre-specified slices)\n")
    lines.append("Uses the exact bootstrap-consistent two-sided p-value for each slice's MAE-diff "
                  "CI (`bootstrap.scene_bootstrap_p_value`, same resamples/seed as run_h1.py's own "
                  "CIs, so 'CI excludes 0' and 'p<0.05' agree by construction for each slice on its own).\n")
    lines.append("| # | bound | slice | n groups | MAE diff | raw p | Bonferroni (a=.05/6) | Holm-Bonferroni | Benjamini-Hochberg (FDR) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for i, c in enumerate(comparisons):
        _p_b, surv_b = bonf[i]
        surv_h = holm[i]
        surv_bh = bh[i]
        p_str = f"{c['p_value']:.4f}" if c["p_value"] is not None else "n/a"
        lines.append(
            f"| {i+1} | {BOUND_LABEL[c['bound']]} | {c['slice_label'].split(':')[0]} | "
            f"{c['n_groups']} | {c['mae_diff']:.4f} | {p_str} | "
            f"{'SURVIVES' if surv_b else 'does not survive'} | "
            f"{'SURVIVES' if surv_h else 'does not survive'} | "
            f"{'SURVIVES' if surv_bh else 'does not survive'} |"
        )
        print(f"[{i+1}] {c['bound']:10s} {c['slice_key']:20s} n={c['n_groups']:4d} "
              f"diff={c['mae_diff']:.4f} p={p_str} "
              f"bonf={'SURVIVES' if surv_b else 'no'} holm={'SURVIVES' if surv_h else 'no'} "
              f"bh={'SURVIVES' if surv_bh else 'no'}")

    lines.append(f"\nBonferroni threshold for 6 comparisons at alpha=0.05: p < {0.05/6:.5f}.\n")

    target_idx = next(i for i, c in enumerate(comparisons)
                       if c["bound"] == "lock_once" and c["slice_key"] == "b_ttc_band_0_5s")
    target = comparisons[target_idx]
    lines.append("## The headline result specifically\n")
    lines.append(f"Lock-once, TTC 0-5s band: MAE diff={target['mae_diff']:.4f}, "
                  f"raw p={target['p_value']:.4f}.\n")
    lines.append(f"- Bonferroni: {'SURVIVES' if bonf[target_idx][1] else 'does NOT survive'} "
                  f"(needs p < {0.05/6:.5f})")
    lines.append(f"- Holm-Bonferroni: {'SURVIVES' if holm[target_idx] else 'does NOT survive'}")
    lines.append(f"- Benjamini-Hochberg (FDR): {'SURVIVES' if bh[target_idx] else 'does NOT survive'}")

    out_path = OUT_DIR / "h1_multiple_comparisons.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return comparisons, {"bonferroni": bonf, "holm": holm, "benjamini_hochberg": bh}


if __name__ == "__main__":
    run()
