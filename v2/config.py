"""
v2/config.py — single source of truth for every v2 tunable.

Nothing in v2/ hardcodes a numeric constant that belongs here. The old
late-fusion pipeline hit the same failure class three times before this
rule was enforced: radar's dyn_prop/velocity field mixups, the
["points"] unwrap silently breaking across notebooks after a format
change, and SENSOR_TRUST_WEIGHT copy-pasted into three separate cells
before being consolidated into one shared config value.

SENSOR_NOISE_VAR convention — read this before touching anything below:

    SENSOR_NOISE_VAR stores MEASUREMENT NOISE VARIANCE (R), in metres^2.
    Bigger value = noisier sensor = LESS trusted.

    This is the OPPOSITE convention from the old pipeline's
    SENSOR_TRUST_WEIGHT, which stored a WEIGHT (~1/variance, bigger =
    MORE trusted). That ambiguity once cost real time in the old
    pipeline's audit: a claimed "reciprocal inversion bug" was raised
    against a weight-shaped value, checked directly against the code
    and a live numeric run, and found to be correct behaviour of a
    weight field being mistaken for a variance field. v2 avoids the
    ambiguity structurally: SENSOR_NOISE_VAR IS variance, used directly
    as R in ConstantVelocityKF.update() — no inversion, anywhere, ever.
"""

# ── Measurement noise ────────────────────────────────────────────────
# Variance (R), metres^2, per sensor. There is no "camera" key — only
# "lidar", "radar", "camera_mono". The old pipeline's LiDAR-assisted
# "camera" baseline (median LiDAR depth wearing a camera label) has no
# role in this design; see v2_architecture_brief.md.
#
# LiDAR and radar are literature-typical ASSUMPTIONS, NOT independently
# measured. A direct measurement was attempted in the prior late-fusion
# audit (matched-track position vs. sample_annotation ground truth,
# MAD-based robust sigma), but LiDAR/radar's own track fragmentation
# (4.1x / 2.4x vs. ~695 real objects in that pipeline) meant enough
# short tracks locked onto the WRONG nearby vehicle in dense traffic
# that even the MEDIAN came out ~2.0m — physically impossible for
# LiDAR — because only ~86% of matched points were real inliers, well
# below the >97% a robust estimator needs to correct for. Do not
# re-attempt this measurement for LiDAR/radar without first fixing
# whatever produces that fragmentation; record the reason rather than
# quietly re-deriving a bad number.
#
# camera_mono WAS measured this way successfully: 95.0% inlier rate
# (far less fragmented, high own ground-truth-lock rate), so its value
# below is real data, not a guess.
SENSOR_NOISE_VAR = {
    "lidar": 0.0225,       # assumed (literature-typical ~15cm std) — NOT independently measured
    "radar": 0.25,         # assumed (~50cm std) — NOT independently measured
    "camera_mono": 1.70,   # MEASURED from GT residuals, 95.0% inlier rate
}

# Ego vehicle's own position "measurement" noise (adapters.py's
# ego_pose_lookup(), which runs ego's own consecutive global positions
# through a ConstantVelocityKF to derive smoothed velocity -- see that
# function's docstring for why a raw 2-point finite difference is not
# used here). nuScenes' ego_pose comes from a fused localization stack
# (GPS/IMU/wheel odometry), far more accurate than any of the three
# tracked sensors above -- assumed (literature-typical for automotive
# localization, ~10cm std), NOT independently measured, same honesty
# convention as SENSOR_NOISE_VAR's lidar/radar entries.
EGO_POSITION_NOISE_VAR = 0.01

# ── Process noise ────────────────────────────────────────────────────
# White-noise-acceleration model: std of unmodeled acceleration, m/s^2.
# This represents uncertainty in the TRACKED OBJECT's own motion (how
# much it might accelerate/turn beyond constant velocity) -- a property
# of the object, not of whichever sensor happens to observe it. It is
# deliberately a single sensor-independent constant, not indexed by
# sensor: predict() runs against every active track before gating even
# decides which detection (if any), or which sensor, matches this
# event, so there is no sensor to attribute this value to at predict
# time. Measurement noise (R, which genuinely does vary by sensor) is
# SENSOR_NOISE_VAR above, applied only in update() -- keep these two
# separate; conflating them was an earlier draft's mistake here.
PROCESS_NOISE_SIGMA_A = 2.0

# ── Gating ───────────────────────────────────────────────────────────
# Chi-square confidence level for Mahalanobis gating (see gating.py).
# Replaces the old pipeline's fixed 3.0m Euclidean ASSOC_DIST_THRESHOLD,
# which created a hard speed ceiling at 2Hz keyframes and fragmented
# fast-moving objects — the gate here scales with each track's own
# predicted-position covariance instead of a flat radius.
GATE_CONFIDENCE_LEVEL = 0.99

# ── Track lifecycle ──────────────────────────────────────────────────
# A track is evicted once this many SECONDS pass with no update — time-
# based, not frame-count-based, since events arrive asynchronously at
# each sensor's own native rate (no shared "frame" exists in this
# design).
MAX_MISSED_SECONDS = 1.5

# ── Velocity deadband (R8) ───────────────────────────────────────────
# Below this estimated speed (m/s), ConstantVelocityKF.trusted_velocity()
# reports zero rather than detector/clustering jitter on a stationary
# object read as false motion.
MIN_TRUSTED_SPEED = 1.0

# A track needs at least this many real measurement updates before its
# velocity estimate is trusted at all, regardless of estimated speed —
# not explicitly listed as a tunable in v2_architecture_brief.md's
# config block, but R8's prose specifies it ("fewer than 3 updates"),
# so it lives here rather than as a hardcoded literal in kalman_track.py.
MIN_UPDATES_FOR_TRUSTED_VELOCITY = 3

# ── TTC closing-speed deadband ──────────────────────────────────────
# ttc.py's R10 formula already returns inf for closing_speed <= 0, but
# a closing speed that is technically positive yet negligibly small
# (extremely common for objects moving roughly tangentially to the ego
# vehicle at any given instant) still divides distance by that tiny
# number -- producing an enormous but FINITE ttc (observed: up to ~190
# MILLION seconds on the real dataset) that math.isfinite() doesn't
# catch, silently dominating MAE/RMSE with numbers that carry no real
# collision-relevant meaning. This is exactly the "raw division
# artifact" R10 already says never to produce; the literal
# closing_speed <= 0 boundary just didn't anticipate this near-zero
# case as a distinct failure mode. Same value as the old pipeline's
# Step 6 MIN_CLOSING_SPEED, for direct comparability.
#
# NOT the same thing as MIN_TRUSTED_SPEED above: that gates a track's
# raw velocity MAGNITUDE (is this object moving at all, vs. detector/
# clustering jitter on something stationary). This gates the CLOSING
# (radial-to-ego) COMPONENT specifically -- an object can have plenty
# of raw speed, well above MIN_TRUSTED_SPEED, while still having
# near-zero closing speed if its motion is roughly tangential to the
# ego vehicle's line of sight. The two constants are not redundant.
MIN_CLOSING_SPEED = 0.3

# ── Radar per-scan clustering (adapters.py) ─────────────────────────
# One real object triggers multiple radar echoes (3-8 blips off
# different parts of the same car) within a single channel's single
# scan. Unlike lidar (already pre-clustered by Step 2.1's DBSCAN) and
# camera_mono (one YOLO box per object already), raw per-scan radar
# points are NOT already one-point-per-object -- left unclustered, this
# single-event-at-a-time design would feed near-simultaneous same-scan
# blips into CentralTracker as repeated noisy measurements of one
# position, when they are actually different points on one object's
# surface. This was a plausible, previously-unexplained contributor to
# radar's 2.4x fragmentation ratio in the old pipeline's audit. Same
# value and same min_samples=1 convention as the old pipeline's
# RADAR_CLUSTER_EPS (an isolated blip still becomes its own detection).
RADAR_CLUSTER_EPS_M = 1.5

# ── Reporting ────────────────────────────────────────────────────────
# TTC at or below this many seconds is flagged as the "danger zone" in
# evaluate.py's reports.
TTC_DANGER_ZONE_SECONDS = 2.0

# (min_inclusive, max_exclusive) track-length buckets for the dilution-
# anomaly breakdown that reappeared unexplained in the old pipeline
# (medium tracks best, long tracks worst, driven by a handful of
# sparsely-matched long tracks) — evaluate.py checks whether it
# reappears here. `None` as a max means unbounded.
TRACK_LENGTH_BUCKETS = [(0, 5), (5, 20), (20, None)]
