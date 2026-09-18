# Implementation brief: asynchronous measurement-level fusion (v2 pipeline)

You are working in an existing repository (`multimodal-perception-ttc`) that estimates Time-to-Collision (TTC) on the nuScenes v1.0-mini dataset by fusing LiDAR, radar and camera. The current implementation is 17 sequential Jupyter notebooks using **late fusion**. Your task is to build a **new, parallel implementation** using **asynchronous measurement-level (early) fusion**, in a new `v2/` package, without modifying or deleting any existing notebook.

**This design uses exactly three sensors: LiDAR, radar, and monocular camera (`camera_mono`).** The existing repository also has a `camera` baseline (Step 2.3.1) that estimates depth by taking the median LiDAR point falling inside each YOLO box. That baseline is **deliberately excluded from this v2 design** — it is not a real single-sensor signal, since its depth is LiDAR's own measurement wearing a camera label. Do not port it, reference it, or use its noise value anywhere in v2. `camera_mono` (Step 2.3.2 — similar-triangles depth, genuinely LiDAR-free) is the only camera path, and it is a full participant in fusion, not a side comparison.

---

## Context you need (do not re-derive this — it is the result of a long prior audit)

The existing late-fusion pipeline works but underperforms. Root causes already diagnosed and proven:

1. **Only ~35% of "fused" output was ever an actual multi-sensor merge.** The other ~65% was a single sensor's detection passed through unchanged. That unmerged 65% had MAE 4.11 — worse than LiDAR alone (3.54). This is the single biggest problem, and it exists because there is a *separate* cross-sensor association pass that can silently fail to notice two tracks were the same object.
2. **A fixed 3.0m Euclidean association gate** created a hard ~6.0 m/s speed ceiling at nuScenes' 2Hz keyframes, fragmenting fast-moving objects into many short tracks.
3. **2-point finite-difference velocity estimation** was independently implemented (and independently buggy) in four separate trackers plus the fusion step.
4. **A flat `R = diag([0.5, 0.5])`** was used for all three sensors in one place, while correctly-tuned per-sensor values existed unused elsewhere. LiDAR was under-trusted ~22x.
5. Ego-motion contamination, missing track eviction, and missing scene-boundary resets each had to be fixed separately in each of the four trackers.

Baseline single-sensor metrics for reference (from the existing pipeline; do not try to reproduce these — they are what v2's single-sensor runs should land near, as a sanity check):

| Sensor | MAE | RMSE | Good% |
|---|---|---|---|
| lidar | 3.54 | 7.41 | 50.2 |
| radar | 5.53 | 10.06 | 34.5 |
| camera_mono (true monocular, no LiDAR) | 3.73 | 6.98 | 42.3 |

There is **no valid prior fused number to compare against** for this design. The old pipeline's fused MAE (3.81) was computed using the LiDAR-assisted `camera`, not `camera_mono` — it is not a like-for-like baseline here, because that fusion had access to a stronger (if methodologically questionable) camera signal. **The target to beat is LiDAR alone (MAE 3.54)**, and this run will be the first time a fused-with-genuinely-independent-camera number exists at all. Report whatever comes out plainly; do not treat 3.81 as a target.

---

## What to build

A new `v2/` package implementing **one shared track set that every sensor writes into directly**, replacing "track each sensor separately, then merge." There is no separate fusion step in this design — that is the entire point.

### Files to create

```
v2/
  __init__.py
  config.py           # all tunables, single source of truth
  kalman_track.py     # one constant-velocity KF class, used everywhere
  gating.py           # Mahalanobis gate + Hungarian assignment
  event_stream.py     # merges per-sensor streams into one chronological timeline
  central_tracker.py  # THE shared track set — replaces per-sensor trackers + fusion
  ttc.py              # one TTC formula
  evaluate.py         # metrics, including built-in diagnostics
  adapters.py         # bridges existing notebook outputs -> SensorEvent streams
  run_pipeline.py     # orchestrator
  README.md           # architecture notes
```

### Core design rules (these are requirements, not suggestions)

**R1 — One tracker class, instantiated four times.**
`CentralTracker` is the only tracker. Single-sensor baselines (lidar, radar, camera_mono) are produced by running the *same class* fed only one sensor's event stream. The fused result is the same class fed the merged lidar+radar+camera_mono stream. Do not write per-sensor tracker subclasses or branches. This guarantees single-sensor and fused numbers come from identical code, making the comparison clean.

**R2 — Asynchronous event processing, not synchronized frames.**
Each detection is a `SensorEvent` carrying its own real sensor timestamp. `event_stream.merge_streams()` uses `heapq.merge` to produce one globally time-ordered stream across sensors. In `CentralTracker.process_event()`, predict each active track forward using **that track's own** `last_predict_time` to this event's timestamp — never a shared `dt`, never a nominal 0.5s sample interval.

**R3 — Mahalanobis gating, not a fixed distance.**
Gate on squared Mahalanobis distance against each track's own predicted position covariance, with a chi-square threshold (`scipy.stats.chi2.ppf(confidence, df=2)`). This makes the gate widen automatically for fast objects and for tracks that have coasted through missed updates — no hand-tuned speed constant.

**R4 — Hungarian assignment, never greedy nearest-match.**
Use `scipy.optimize.linear_sum_assignment`. Pairs outside the gate must be unassignable (substitute a large finite cost for `inf`, then reject any accepted pair whose true cost was infeasible).

**R5 — Time-based lifecycle, not frame counts.**
Evict a track when `(now - last_update_time) > MAX_MISSED_SECONDS`. Frame counting is meaningless in an asynchronous design.

**R6 — Multi-sensor status is inherent, not tagged.**
A track has a `sensors_seen: set`. `is_multi_sensor` is `len(sensors_seen) > 1`. There is no `is_merged` flag applied by a separate pass, because there is no separate pass.

**R7 — Single source of truth for all constants.**
Every tunable lives in `v2/config.py` and is imported, never copy-pasted. `SENSOR_NOISE_VAR` stores **variance (R), in meters², where bigger = noisier = less trusted**. Document this convention in a module docstring — a prior audit wasted time on a weight-vs-variance ambiguity. There is no `"camera"` key — only `"lidar"`, `"radar"`, `"camera_mono"`.

**R8 — Velocity deadband.**
`ConstantVelocityKF.trusted_velocity(min_speed)` returns `(0.0, 0.0)` if the filter has fewer than 3 updates or if speed is below `MIN_TRUSTED_SPEED` (1.0 m/s). DBSCAN centroid jitter on stationary objects otherwise reads as false motion.

**R9 — Scene-boundary reset.**
On `scene_token` change, finalize all active tracks before continuing. After implementing, add an assertion in `run_pipeline.py` that no track's history spans two scene tokens, and confirm it passes on the real 10-scene dataset.

**R10 — TTC sign convention derived once.**
Closing speed is the radial component of the track's velocity toward the ego vehicle: `(vx*dx + vy*dy)/distance` where `(dx,dy) = (ego - track)`. Positive = closing. If `closing_speed <= 0`, TTC is `float('inf')` — never negative, never a raw division artifact. Never use a raw sensor-reported velocity field (e.g. radar Doppler) for TTC; always use the KF's own velocity state.

### Config values to use

```python
SENSOR_NOISE_VAR = {
    "lidar": 0.0225,       # assumed (literature-typical ~15cm std) — NOT independently measured
    "radar": 0.25,         # assumed (~50cm std) — NOT independently measured
    "camera_mono": 1.70,   # MEASURED from GT residuals, 95.0% inlier rate
}
PROCESS_NOISE_SIGMA_A = {"lidar": 2.0, "radar": 2.0, "camera_mono": 3.0}
GATE_CONFIDENCE_LEVEL = 0.99
MAX_MISSED_SECONDS = 1.5
MIN_TRUSTED_SPEED = 1.0
TTC_DANGER_ZONE_SECONDS = 2.0
TRACK_LENGTH_BUCKETS = [(0, 5), (5, 20), (20, None)]
```

Keep the "assumed vs measured" comments verbatim in the file. LiDAR/radar could not be empirically measured because their own track fragmentation drove the ground-truth-match inlier rate to ~86%, below the ~97% a robust estimator needs — record this reason in the docstring so nobody re-attempts it blindly.

### Metrics that must be reported on every run (not as optional diagnostics)

- MAE, RMSE, match rate, total tracks, GT-matched tracks — per sensor (lidar, radar, camera_mono) and for fused
- **Track-length bucket breakdown** (short/medium/long): a prior audit found a non-monotonic result (medium tracks best, long tracks *worst*) that remains unexplained on the old pipeline. Check whether it reappears here.
- **Multi-sensor composition breakdown** for the fused run: what fraction of matched tracks ever saw >1 sensor, and MAE for multi-sensor vs single-sensor tracks. Since camera_mono is now a full fusion participant (unlike in the old design), this number is genuinely new information, not a repeat of a prior finding.

---

## Integration with existing code

**Do not rewrite dataset parsing.** Steps 0–2 of the existing pipeline (nuScenes loading, calibration, RANSAC+DBSCAN clustering, YOLOv5 inference, radar field extraction) were audited and found correct. Reuse them. **Do not touch or reference Step 2.3.1 (`yolo_global`, the LiDAR-assisted camera path) at all** — it has no role in v2.

Write `v2/adapters.py` with three generator functions that read the existing notebooks' on-disk outputs and yield `SensorEvent` objects **sorted by timestamp**:

```python
def lidar_events():        # from output/step_2/lidar/<sample>/lidar_clusters.json + step_1 lidar_meta.json
def radar_events():        # from output/step_2/radar/<sample>/<channel>.json
def camera_mono_events():  # from output/step_2/yolo_mono/<sample>/<camera>.json
```

There is no `camera_events()` in this design.

Critical requirements for the adapters:
- Every `SensorEvent` position must already be in the **global frame**. Use the existing `src/geometry.py` transforms (`point_to_global` / `centroid_to_global`) — do not reimplement them.
- Use each detection's **real sensor timestamp** from the nuScenes `sample_data` record, not the parent sample's timestamp. They differ, and that difference is the point of this architecture.
- Carry `sample_id` on every event so ground-truth matching can still look up `sample_annotation` later.
- Carry `scene_token` on every event.

**Sweeps vs samples — read this carefully.** nuScenes `samples/` contains only 2Hz keyframes; `sweeps/` contains each sensor's full native rate (LiDAR ~20Hz, camera ~12Hz, radar ~13Hz). This architecture's benefit comes specifically from processing at native rates. First implement the adapters against the existing `samples/`-derived outputs so the pipeline runs end to end. Then report to me how much additional work it would be to extend the adapters to `sweeps/`, and wait for my decision before doing it. Note that camera_mono's monocular depth-from-height estimation may need to be re-derived per sweep frame rather than reused from a samples-only run — flag this if it looks like extra work, don't just assume it's free.

**Ground-truth matching needs care.** `sample_annotation` boxes exist only at 2Hz sample timestamps. A track updated many times between two annotated samples must have its state read **at the annotation's exact timestamp** (predict the KF forward to that instant), not by taking the nearest update. Implement this explicitly in the GT-matching function and note it in the README.

---

## Deliverables and order of work

1. `config.py`, `kalman_track.py`, `gating.py` — pure, testable, no I/O. Write a small unit test for each.
2. `event_stream.py`, `central_tracker.py` — core logic. Unit-test `CentralTracker` with synthetic events (two sensors observing one moving object; assert one track results, not two).
3. `ttc.py`, `evaluate.py`.
4. `adapters.py` — the only part touching existing on-disk outputs.
5. `run_pipeline.py` — runs 3 single-sensor baselines (lidar, radar, camera_mono) + 1 fused run (lidar+radar+camera_mono), prints the full metrics table with all three breakdowns.
6. `README.md` — architecture rationale, the design rules above, and stated limitations.

Run the full pipeline and report the metrics table. **Do not tune anything to make fusion look better.** If fused MAE is still above LiDAR's 3.54, report that plainly along with the multi-sensor composition breakdown, which should explain why.

## Stated scope limits (implement as-is; do not expand without asking)

- `ConstantVelocityKF` is deliberately **linear constant-velocity, not UKF/CTRV**. Turn-rate modeling is out of scope. If residual error analysis later suggests turning-heavy scenes dominate, upgrading this one class is a contained change.
- The LiDAR-assisted `camera` baseline (Step 2.3.1) is **out of scope entirely** — not excluded-but-referenced, simply not part of this design.
- Do not modify, delete, or refactor any existing notebook or `src/` file. `v2/` is additive.

## Working style

- Ask before making architectural decisions not covered above.
- After each numbered deliverable, stop and report what you did before continuing.
- If you find a bug in the existing Step 0–2 output while writing adapters, report it — do not silently work around it.
