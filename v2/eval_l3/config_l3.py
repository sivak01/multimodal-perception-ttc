"""
v2/eval_l3/config_l3.py — single config block for every v2/eval_l3/
threshold, per this session's own ground rules:

  1. Never modify v1 notebooks, the v2 tracker (central_tracker.py), or
     any existing v2/*.py file -- all new code lives under v2/eval_l3/,
     tests under v2/tests/, outputs under output/eval_l3/ (gitignored
     via the repo's existing blanket "output/*" rule), never at repo
     root.
  2. Every threshold lives HERE, labelled ASSUMED or MEASURED. None of
     these were tuned after looking at any eval_l3 result -- they are
     the exact values given at the start of this task, used as-is.

ASSUMED means a reasonable, stated engineering choice, not independently
measured from this dataset (same honesty convention v2/config.py and
repo-root config.py already use for SENSOR_NOISE_VAR/SENSOR_TRUST_WEIGHT).
Every value below is ASSUMED -- none were derived from this project's own
data, by design (that is what "do not tune after seeing results" means in
practice: these were fixed BEFORE the first eval_l3 number was computed).
"""

# ── Corridor model (H1's "in ego path" check) ───────────────────────
# ASSUMED. An object is "in ego path" if, in the EGO-FRAME (longitudinal
# forward axis = ego's own direction of travel, see ego_heading.py for
# how that axis is derived), its lateral offset is within
# CORRIDOR_HALF_WIDTH_M and its forward (longitudinal) distance is
# within [0, CORRIDOR_MAX_RANGE_M] -- ahead of the car, not behind it.
CORRIDOR_HALF_WIDTH_M = 2.5
CORRIDOR_MAX_RANGE_M = 60

# ── Closest-point-of-approach model (crossing-object relevance) ─────
# ASSUMED. Reserved for a future hypothesis in this session (not used by
# H1 itself) -- kept here per rule 3 ("put EVERY threshold in one config
# block"), not introduced ad hoc later.
CPA_MAX_M = 2.0
CPA_HORIZON_S = 5

# ── Forward-collision-warning / autonomous-emergency-braking bands ──
# ASSUMED. Also reserved for a future hypothesis; not used by H1 itself.
FCW_TTC_S = 2.5
AEB_TTC_S = 1.5

# ── TTC severity bands ───────────────────────────────────────────────
# ASSUMED. Half-open intervals [lo, hi), same convention as
# repo-root config.py's TRACK_LENGTH_BUCKETS. H1 part (b) uses the
# union of the first two bands (0-5s) as its "TTC band 0-5 s" filter.
CRITICAL_TTC_BANDS_S = [(0, 3), (3, 5), (5, 10)]
H1_TTC_BAND_LOW_S, H1_TTC_BAND_HIGH_S = 0, 5   # = bands[0] + bands[1], stated explicitly for H1's own use

# ── Range buckets (ego-to-object distance, metres) ──────────────────
# ASSUMED. Half-open, last bucket unbounded above.
RANGE_BUCKETS_M = [(0, 20), (20, 40), (40, 60), (60, float("inf"))]

# ── Moving vs. parked ────────────────────────────────────────────────
# ASSUMED. World-frame GT speed threshold (same semantics as
# v2/evaluate_matched.py's by_motion_split, re-stated here as this
# session's own fixed value rather than imported, so eval_l3 does not
# silently drift if that module's own default ever changes).
MOVING_SPEED_MPS = 0.5

# ── GT-key matching distance (shared by both matching bounds) ───────
# ASSUMED. Same value v2/evaluate.py's MATCH_DIST_THRESHOLD and
# v2/evaluate_matched.py's MATCH_DIST_THRESHOLD already use -- restated
# here, not imported, for the same reason as MOVING_SPEED_MPS above.
MATCH_DIST_THRESHOLD_M = 3.0

# ── Scene-level bootstrap ───────────────────────────────────────────
# Not a modeling assumption -- a fixed statistical procedure, specified
# exactly by this session's own rules.
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI_PCT = 95

# ── Minimum-n warning threshold ──────────────────────────────────────
MIN_N_FOR_RELIABLE_CI = 30
