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

# Create all output dirs on import
for d in [STEP0_DIR, STEP1_DIR, STEP2_DIR,
          STEP3_DIR, STEP4_DIR, STEP5_DIR, STEP6_DIR, ARCHIVE_ROOT]:
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

print(f"config.py loaded. PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATA_ROOT   = {DATA_ROOT}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")