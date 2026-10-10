"""
v2/eval_l3/hungarian_match.py — per-frame OPTIMAL (Hungarian) bipartite
matching bound, per this session's rule 5. v2/evaluate_matched.py's own
_match_samplewise() is a greedy nearest-pairs-first approximation of
this (documented in that module's own docstring as a deliberate,
simpler choice); this module instead solves each sample's GT-to-
candidate assignment exactly via scipy.optimize.linear_sum_assignment,
giving the OPTIMISTIC bound (the best any assignment-only scheme could
do at that instant, since Hungarian is provably optimal for a fixed
cost matrix -- no identity persisted between samples here either, same
as the greedy version, so this is still a "no-lock" scheme).

A large FINITE infeasible-cost sentinel (not inf) keeps the solver
well-defined, same convention v2/gating.py's own Hungarian call uses
for out-of-gate pairs -- re-read here, not re-exported, so this module
has no import-time dependency on gating.py's own gate math.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from v2.evaluate_matched import MatchedKeyRow

INFEASIBLE_COST = 1e6


def match_samplewise_hungarian(run_name, candidates_by_sample, gt_by_sample, gt_ttc_lookup,
                                match_dist_threshold):
    """Same signature/contract as evaluate_matched._match_samplewise,
    swapping greedy nearest-pairs for an exact per-sample Hungarian
    solve. Returns a list of MatchedKeyRow."""
    rows = []
    for sample_id, gts in gt_by_sample.items():
        cands = candidates_by_sample.get(sample_id)
        if not cands or not gts:
            continue

        n_gt, n_cand = len(gts), len(cands)
        cost = np.full((n_gt, n_cand), INFEASIBLE_COST)
        for gi, (instance_id, gx, gy) in enumerate(gts):
            for ci, cand in enumerate(cands):
                d = math.hypot(cand.x - gx, cand.y - gy)
                if d <= match_dist_threshold:
                    cost[gi, ci] = d

        row_ind, col_ind = linear_sum_assignment(cost)
        for gi, ci in zip(row_ind, col_ind):
            if cost[gi, ci] >= INFEASIBLE_COST:
                continue   # Hungarian still has to return a full assignment; this pairing was never actually feasible

            instance_id = gts[gi][0]
            cand = cands[ci]
            gt_entry = gt_ttc_lookup.get((sample_id, instance_id))
            if gt_entry is None:
                continue

            rows.append(MatchedKeyRow(
                run=run_name, sample_id=sample_id, instance_id=instance_id,
                track_id=cand.track_id, dist=round(float(cost[gi, ci]), 4),
                pred_ttc=cand.pred_ttc, gt_ttc=gt_entry["ttc"],
                is_multi_sensor=cand.is_multi_sensor,
            ))
    return rows
