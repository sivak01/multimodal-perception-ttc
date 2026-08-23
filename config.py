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

print(f"config.py loaded. PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DATA_ROOT   = {DATA_ROOT}")
print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")