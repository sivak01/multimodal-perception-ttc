# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A multi-sensor (camera + radar + LiDAR) fusion research pipeline built on the **nuScenes v1.0-mini** dataset. It reconstructs, from raw sensor logs, per-object tracks in a global frame, fuses them across sensors, and computes Time-To-Collision (TTC) estimates for evaluation against ground-truth annotations. There is no application code — the entire pipeline is a sequence of numbered Jupyter notebooks, each a self-contained pipeline stage that reads the previous stage's output from disk and writes its own.

There is no test suite, linter, or build system — correctness is checked via the notebooks themselves (see Step 7 below) and by re-running stages.

## Running the pipeline

Notebooks must be run **in numeric order**, top to bottom, since each stage's outputs are the next stage's inputs (all paths flow through `config.py`, see below). There's no CLI entry point — open notebooks in Jupyter/VS Code and run all cells, or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace "Step_0_Dataset_Preparation.ipynb"
```

Pipeline order:

| Stage | Notebook | Purpose |
|---|---|---|
| 0 | `Step_0_Dataset_Preparation.ipynb` | Indexes nuScenes samples; writes `samples_index.json`, the master index every downstream step reads |
| 1.1 | `Step_1_1_Camera_Preprocessing.ipynb` | Camera calibration/intrinsics per sample; imports `project_point_to_camera()` from `src/geometry.py` |
| 1.2 | `Step_1_2_Radar_Parsing.ipynb` | Parses raw radar point clouds per channel |
| 1.3 | `Step_1_3_LiDAR_Parsing.ipynb` | Parses raw LiDAR point clouds (N×7 arrays: point + derived fields) |
| 2.1 | `Step_2_1_LiDAR_Processing.ipynb` | Ground-plane removal (Open3D `segment_plane`) + DBSCAN clustering |
| 2.2 | `Step_2_2_Radar_Processing.ipynb` | Radar filtering (dynamic-property filter) + feature extraction (raw + ego-compensated velocity) |
| 2.3 | `Step_2_3_YOLOv5.ipynb` | YOLO 2D object detection on camera frames (uses `ultralytics`, weights in `yolov8n.pt`) |
| 2.3.1 | `Step_2_3_1_YOLO_Global_Projection.ipynb` | Lifts each 2D YOLO box to a 3D global-frame position by fusing with LiDAR |
| 3.1 | `Step_3_1_LiDAR_Fusion.ipynb` | Multi-frame LiDAR tracking (nearest-neighbor / Hungarian assignment via `scipy.optimize.linear_sum_assignment`), global frame, no filter yet |
| 3.2 | `Step_3_2_Radar_Fusion.ipynb` | Multi-frame radar tracking, same family as 3.1 |
| 3.3 | `Step_3_3_Camera_Tracker.ipynb` | Multi-frame camera tracking, global 3D nearest-neighbor across all 6 cameras |
| 4 | `Step_4_Fusion.ipynb` | Fuses LiDAR/radar/camera tracks into unified tracks |
| 5 | `Step_5_TTC.ipynb` | UKF + CTRV motion model → TTC estimate per track, per sensor and fused |
| 6 | `Step_6_Visualization.ipynb` | Trend plots + evaluation metrics against nuScenes `sample_annotation` ground truth |
| 7 | `Step_7_Pipeline_Audit.ipynb` | Read-only audit: walks Steps 0–6 checking input/output counts and known-bug regressions at every stage; does not modify any output |
| — | `Validate_Step_2_3_1.ipynb` | Standalone validator for Step 2.3.1's 3D projections |

Each notebook's first markdown cell documents its own **Input / Outputs / Used by** table — read that cell before modifying a stage, since output schemas are contracts downstream notebooks depend on exactly (e.g. LiDAR/radar tracks are tuple-format, camera tracks are dict-format — Step 4 handles both explicitly).

## Architecture

**`config.py` is the single source of truth for all paths.** Every notebook starts by importing from it (e.g. `from config import STEP0_DIR, STEP1_DIR, ...`). It anchors `PROJECT_ROOT`/`BASE_DIR` to its own file location (not the Jupyter CWD), points `DATA_ROOT` at the nuScenes dataset, and derives one output subdirectory per stage (`STEP0_DIR` … `STEP6_DIR`) under `output/`. **Change dataset/output locations only in `config.py`** — never hardcode paths inside a notebook.

**`src/geometry.py`** holds the sensor-to-global and global-to-camera coordinate transform math (`transform_matrix`, `point_to_global`, `points_to_global`, `project_point_to_camera`, `global_points_to_camera`). This used to be copy-pasted independently into Steps 1.1, 2.3.1, 3.1, and 3.2; those notebooks now `from src.geometry import ...` instead. Step 3.1 imports `point_to_global` under the alias `centroid_to_global` to match its existing call sites. If you touch this file, all four notebooks are affected — re-run them to confirm. `GlobalFrameTracker` (Step 3.1 vs 3.2) and `compute_ttc` (Step 5 vs Step 6) are similarly duplicated but were *not* unified — their implementations have diverged enough (radar-specific tuning, ground-truth-specific lookups) that merging them needs a closer read than a mechanical extraction; treat that as a known follow-up, not an oversight. Verified by executing all four notebooks end-to-end (`python -m nbconvert --to notebook --execute ...` — note `python -m jupyter nbconvert` silently exits 1 with no output on at least one Windows setup seen in this repo's history; call `nbconvert` as a module directly instead); Step 3.1's per-run diagnostic numbers matched the pre-refactor baked-in output exactly.

**Step 3.1 and Step 3.2 are not idempotent.** `GlobalFrameTracker.save_tracks()` writes `track_<uuid4>.json` with a fresh random id every run and never clears `output/step_3/{lidar,radar}/` first, so re-running either notebook without deleting that directory first accumulates duplicate track files on top of the previous run's (confirmed: re-running Step 3.1 once turned 4930 LiDAR track files into 9860). Delete `output/step_3/lidar/` / `output/step_3/radar/` before re-running if you need a clean single-run result.

Data flow is entirely file-based: each stage reads JSON/CSV/NPY files written by the prior stage under `output/step_N/...` and writes its own under `output/step_(N+1)/...`; there is no in-memory hand-off between notebooks. Key artifacts:
- `output/step_0/samples_index.json` — master sample index, read by nearly every later stage.
- `output/step_1/lidar/<sample>/lidar_raw.npy` — per-point LiDAR array (N×7).
- `output/step_2/yolo_global/<sample>/<camera>.json` — YOLO detections lifted to 3D global coordinates.
- `output/step_3/{lidar,radar,camera}/track_<id>.json` — per-sensor multi-frame tracks in the global frame.
- `output/step_4/fused/track_<id>.json` and `fused_tracks_all.csv` — fused multi-sensor tracks.
- `output/step_5/ttc_{lidar,radar,camera,fused}.csv` — same schema across sensors, directly comparable.

`html/` holds rendered HTML exports of each notebook (for viewing without Jupyter) and is regenerated from the notebooks, not hand-edited. `archive/` and `Old Codes/` hold prior notebook revisions and an old zipped copy of the project kept for reference — do not treat them as current; the root-level `Step_*.ipynb` files are canonical. `assets/` holds a small set of result images copied from `output/` and committed for the README — it's curated by hand, not regenerated automatically, so re-copy into it manually after a pipeline run if you want the README images refreshed.

`output/`, `DATA SET/`, `html/`, `archive/`, and `Old Codes/` are gitignored (see `.gitignore`); `assets/` is intentionally not.

## Dependencies

`requirements.txt` pins the working set of libraries: `nuscenes-devkit` (`nuscenes`, `pyquaternion`), `open3d`, `ultralytics` (YOLO), `torch`/`torchvision`, `opencv-python` (`cv2`), `scikit-learn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `Pillow`. Install with `pip install -r requirements.txt`.
