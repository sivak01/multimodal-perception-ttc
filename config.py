# config.py — single source of truth for all paths
# Place this file at the root of your repository.
# Every notebook imports from here — change paths only once.

from pathlib import Path

# ── Anchor to config.py's own folder, NOT the Jupyter working directory ────
# This makes all relative paths below work regardless of where Jupyter
# was launched from.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR   # alias — same thing, either name works in notebooks

# ── Dataset ────────────────────────────────────────────────
DATA_ROOT = Path(r"F:\Sensor fusion Research\DATA SET\archive")   # ← change this once
NUSCENES_VERSION = "v1.0-mini"

# ── Output root (anchored to BASE_DIR, not "./") ────────────
OUTPUT_ROOT = BASE_DIR / "output"
ARCHIVE_ROOT = BASE_DIR / "archive"

# ── Per-step output folders (auto-derived) ─────────────────
STEP0_DIR  = OUTPUT_ROOT / "step_0"
STEP1_DIR  = OUTPUT_ROOT / "step_1"
STEP2_DIR  = OUTPUT_ROOT / "step_2"
STEP3_DIR  = OUTPUT_ROOT / "step_3"
STEP4_DIR  = OUTPUT_ROOT / "step_4"
STEP5_DIR  = OUTPUT_ROOT / "step_5"
STEP6_DIR  = OUTPUT_ROOT / "step_6"
STEP8_DIR  = OUTPUT_ROOT / "step_8"

# Create all output dirs on import
for d in [STEP0_DIR, STEP1_DIR, STEP2_DIR,
          STEP3_DIR, STEP4_DIR, STEP5_DIR, STEP6_DIR, STEP8_DIR, ARCHIVE_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Shared sensor trust weights ─────────────────────────────
# Inverse-variance-style trust weights (weight ~ 1/sigma^2) from each sensor's
# approximate positional accuracy. LiDAR (0.15m) and radar (0.5m) are still
# textbook engineering estimates, NOT measured from this dataset -- a direct
# measurement (matched-track position vs. sample_annotation ground truth,
# robust MAD-based sigma) was attempted, but for these two sensors the
# measurement itself is unusable: LiDAR/radar's much higher track
# fragmentation (4.1x / 2.4x vs. ~695 real objects) means enough short
# tracks lock onto the WRONG nearby vehicle in dense traffic that even a
# robust (MAD) estimator doesn't recover a plausible number (median position
# "error" came out ~2.0m for both, physically implausible for LiDAR) --
# only ~86% of matched points are real inliers, not the >97% robust
# statistics assume. Using that measurement would make LiDAR/radar LESS
# trusted than this hand-picked estimate, contradicting known sensor
# physics. Camera and camera-mono do NOT have this problem (94-95% inlier
# rate -- far less fragmented, high own GT-lock rate) so their values below
# ARE measured: camera 0.90m (var 0.81, weight 1.23 -- was 1.0m/1.0/1.0,
# a real but modest correction) and camera-mono 1.30m (var 1.70, weight
# 0.59 -- previously not in this dict at all, silently defaulting to
# camera's trust level via a fallback and understating its true noise).
# Shared by Step 4 (per-sensor UKF smoothing + cross-sensor merge weighting)
# and Step 5 (single-sensor UKF measurement noise) so the two can't silently
# drift apart -- this pipeline has already hit that failure class twice
# (radar's dyn_prop/velocity field mixups, the ["points"] unwrap breaking
# across notebooks after a format change).
SENSOR_TRUST_WEIGHT = {"lidar": 44.0, "radar": 4.0, "camera": 1.23, "camera_mono": 0.59}

# Radar's own ego-motion-compensated Doppler velocity (vx_comp/vy_comp) --
# parsed and stored back in Step 2.2 but never consumed anywhere until Step
# 3.2/Step 4 were fixed to use it. Assumed (~1.0 m/s std, literature-typical
# for automotive radar Doppler-derived vx/vy), NOT independently measured --
# same honesty convention as SENSOR_NOISE_VAR/SENSOR_TRUST_WEIGHT's own
# lidar/radar position entries. Used as the UKF's velocity-measurement R
# when smoothing a radar track (Step 4's smooth_track_states()).
RADAR_VELOCITY_NOISE_VAR = 1.0   # (m/s)^2

# Decoupled position vs velocity trust, for the cross-sensor MERGE step
# specifically (Step 4's merge_frame_observations()). SENSOR_TRUST_WEIGHT
# above is a POSITION weight -- using it for velocity too (the pipeline's
# original behaviour) silently assumes a sensor's velocity accuracy always
# tracks its position accuracy, which is false for radar: it measures
# velocity directly (Doppler) but position only indirectly (angle+range),
# the opposite of lidar/camera_mono, whose velocity is entirely DERIVED
# from differencing their own position measurements over time (no separate
# velocity sensor of their own).
#
# lidar/camera_mono's velocity weight is back-of-envelope-derived from
# their own position variance (1/SENSOR_TRUST_WEIGHT) propagated through a
# 2-point finite difference at this pipeline's ~0.5s (2Hz) sample cadence
# (var_v ~= 2*var_pos/dt^2) -- NOT independently measured, and the real
# UKF-converged velocity variance is typically tighter than this raw
# 2-point estimate, so treat this as an approximate, order-of-magnitude
# starting point, not a calibrated number. radar's velocity weight is
# simply 1/RADAR_VELOCITY_NOISE_VAR, its own direct measurement's assumed
# noise -- not derived from position at all, which is the whole point.
#
# Result, worth stating plainly since it's not the naive assumption:
# lidar's own DERIVED velocity (var ~0.18 (m/s)^2) comes out MORE trusted
# than radar's DIRECT Doppler measurement (var 1.0) at this sample rate,
# precisely because lidar's position is so accurate that differencing it
# still beats an imperfect direct measurement -- radar's velocity
# advantage over camera_mono's derived velocity (var ~13.6) is real and
# large, just not automatically "better than everything" by virtue of
# being direct.
SENSOR_VELOCITY_TRUST_WEIGHT = {"lidar": 5.5, "radar": 1.0, "camera_mono": 0.074}

print(f"config.py loaded. PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATA_ROOT   = {DATA_ROOT}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")