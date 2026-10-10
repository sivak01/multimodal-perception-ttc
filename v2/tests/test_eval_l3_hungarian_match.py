import itertools
import math

from v2.evaluate_matched import Candidate
from v2.eval_l3.hungarian_match import match_samplewise_hungarian


def _gt_ttc(entries):
    return {k: {"ttc": v, "pos": (0, 0)} for k, v in entries.items()}


def test_hungarian_matches_brute_force_optimum_on_a_3v2_instance():
    """3 GT points competing for only 2 candidates -- more GT than
    candidates means SOME GT must go unmatched, and which one is left
    out is a genuine optimization choice (not resolvable as trivially
    as a 2-vs-2 case, where metric/triangle-inequality constraints make
    greedy nearest-pairs and the true optimum coincide automatically).
    Verifies Hungarian's result against a brute-force enumeration of
    every possible assignment, rather than a hand-guessed answer."""
    gts = [("A", 0.0, 0.0), ("B", 3.0, 0.0), ("C", 3.4, 0.0)]
    gt_by_sample = {"s0": gts}
    gt_ttc_lookup = _gt_ttc({("s0", "A"): 4.0, ("s0", "B"): 5.0, ("s0", "C"): 6.0})
    cand_defs = [("cand_1", 1.0, 0.0), ("cand_2", 3.5, 0.0)]
    candidates_by_sample = {
        "s0": [Candidate(track_id=tid, x=x, y=y, pred_ttc=4.0, is_multi_sensor=False) for tid, x, y in cand_defs],
    }
    threshold = 5.0

    # Brute force: every way to assign each candidate to a distinct GT
    # (or to "unassigned"), feasible pairs only, minimizing total cost.
    gt_names = [g[0] for g in gts]
    gt_pos = {g[0]: (g[1], g[2]) for g in gts}
    cand_pos = {tid: (x, y) for tid, x, y in cand_defs}
    best_cost, best_assignment = None, None
    for perm in itertools.permutations(gt_names, len(cand_defs)):
        cost = 0.0
        feasible = True
        for (tid, _x, _y), gt_name in zip(cand_defs, perm):
            gx, gy = gt_pos[gt_name]
            cx, cy = cand_pos[tid]
            d = math.hypot(cx - gx, cy - gy)
            if d > threshold:
                feasible = False
                break
            cost += d
        if feasible and (best_cost is None or cost < best_cost):
            best_cost, best_assignment = cost, dict(zip([c[0] for c in cand_defs], perm))

    rows = match_samplewise_hungarian("fused", candidates_by_sample, gt_by_sample, gt_ttc_lookup, match_dist_threshold=threshold)
    hungarian_assignment = {r.track_id: r.instance_id for r in rows}
    hungarian_cost = sum(r.dist for r in rows)

    assert hungarian_assignment == best_assignment
    assert abs(hungarian_cost - best_cost) < 1e-6


def test_hungarian_respects_distance_threshold():
    gt_by_sample = {"s0": [("A", 0.0, 0.0)]}
    gt_ttc_lookup = _gt_ttc({("s0", "A"): 4.0})
    candidates_by_sample = {"s0": [Candidate(track_id="far", x=100.0, y=0.0, pred_ttc=4.0, is_multi_sensor=False)]}
    rows = match_samplewise_hungarian("fused", candidates_by_sample, gt_by_sample, gt_ttc_lookup, match_dist_threshold=3.0)
    assert rows == []


def test_hungarian_skips_keys_without_gt_ttc():
    gt_by_sample = {"s0": [("A", 0.0, 0.0)]}
    gt_ttc_lookup = {}   # no entry for ("s0", "A")
    candidates_by_sample = {"s0": [Candidate(track_id="t1", x=0.1, y=0.0, pred_ttc=4.0, is_multi_sensor=False)]}
    rows = match_samplewise_hungarian("fused", candidates_by_sample, gt_by_sample, gt_ttc_lookup, match_dist_threshold=3.0)
    assert rows == []


def test_hungarian_empty_sample_is_skipped_cleanly():
    rows = match_samplewise_hungarian("fused", {}, {"s0": [("A", 0.0, 0.0)]}, {}, match_dist_threshold=3.0)
    assert rows == []
