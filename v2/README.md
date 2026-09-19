# v2 — asynchronous measurement-level fusion

A second, independent implementation of the TTC pipeline, built alongside the
original 17-notebook late-fusion pipeline (never modifying or depending on
it — see `v2_architecture_brief.md` for the full brief this was built from).
Where the original pipeline tracks each sensor separately on a shared 2Hz
frame clock and then merges the resulting tracks afterward, v2 uses **one
shared track set that every sensor writes into directly**, with each
detection carrying its own real sensor timestamp. There is no separate
fusion step — that is the whole point of this design.

## Why this exists

The original pipeline's fused output underperformed its own best single
sensor. Diagnosed root causes (see `v2_architecture_brief.md` for the full
list): only ~35% of "fused" points were ever a genuine multi-sensor merge;
a fixed 3.0m Euclidean association gate fragmented fast-moving objects; four
independent, independently-buggy 2-point finite-difference velocity
estimators; a flat measurement-noise `R` that under-trusted LiDAR by ~22x.

v2 addresses each of these structurally rather than patching the old
design: async per-sensor timestamps instead of a shared frame clock,
Mahalanobis gating instead of a flat radius, one shared Kalman filter class
instead of four duplicated trackers, and a single sensor-noise config.

## Architecture

**One `CentralTracker` class, instantiated four times** — three
single-sensor baselines (`lidar`, `radar`, `camera_mono`) plus one fused run
fed the merged stream of all three. Same class, same code path, no
per-sensor branching or subclassing: this is what makes the single-sensor
vs. fused comparison genuinely apples-to-apples.

**Async event processing.** Each detection is a `SensorEvent` carrying its
own real timestamp (`event_stream.py`). `merge_streams()` uses
`heapq.merge` to fold multiple already-sorted per-sensor streams into one
globally chronological iterator, without ever materializing or re-sorting a
combined list. `CentralTracker.process_event()` predicts each active track
forward using *that track's own* last-predicted time, never a shared or
nominal frame interval.

**Mahalanobis gating + Hungarian assignment** (`gating.py`), not a fixed
distance threshold. The gate scales with each track's own predicted
**innovation covariance** — critically, `P_pred + R` (the incoming
detection's own measurement noise), not `P_pred` alone; see "What we found
while tuning" below for why this specific detail mattered enormously in
practice. Hungarian assignment (`scipy.optimize.linear_sum_assignment`)
finds the optimal pairing; pairs outside the gate are never assignable even
as a last resort (a large finite `INFEASIBLE_COST` sentinel, not `inf`,
keeps the assignment solver well-defined).

**One Kalman filter class, sensor-agnostic** (`kalman_track.py`):
`ConstantVelocityKF` knows nothing about sensor identity or config — it
does Kalman math only. `Track` (`central_tracker.py`) wraps one KF with the
track-level bookkeeping that does NOT belong in a sensor-agnostic filter:
`sensors_seen` (a track is multi-sensor the instant a second sensor's event
matches it — no separate "merge" flag or pass anywhere), and `history` (one
snapshot per real update, since `evaluate.py` needs a track's full
trajectory, not just its final state).

**Time-based lifecycle**, not frame counts: a track is evicted once real
elapsed seconds since its last update exceed `MAX_MISSED_SECONDS` — frame
counting is meaningless when events arrive asynchronously at each sensor's
own native rate.

**Scene-boundary reset (R9):** on every `scene_token` change, every active
track is finalized before continuing; no ID, state, or position carries
across. Verified two ways on the real 404-sample/10-scene dataset: an
in-run assertion inside `CentralTracker` itself, and an independent
external check in `run_pipeline.py` that cross-references every track's
`history` sample_ids against `samples_index.json`'s own `scene_token` —
**0 violations** across all four runs (lidar/radar/camera_mono/fused).

**Ground-truth matching with proper timestamp extrapolation
(`evaluate.py`):** `sample_annotation` boxes exist only at fixed 2Hz
keyframes, while a track updates at its own sensor's native rate and
generally will not land on that exact instant. `position_at()` extrapolates
a track's most recent snapshot forward to the annotation's exact timestamp
using that snapshot's own `(vx, vy)` — no KF re-instantiation needed, since
velocity is constant between real updates under this constant-velocity
model. Bounded on both ends: an annotation before a track's first snapshot
is a clean no-match (no backward extrapolation, no index wraparound), and
one more than `MAX_MISSED_SECONDS` past a track's last snapshot is also a
no-match (the track would already be evicted by then) — reusing that one
constant rather than a second, possibly-inconsistent cutoff.

## Files

| File | Purpose |
|---|---|
| `config.py` | Single source of truth for every tunable. |
| `kalman_track.py` | `ConstantVelocityKF` — sensor-agnostic constant-velocity filter. |
| `gating.py` | Mahalanobis gate (on the innovation covariance) + Hungarian assignment. |
| `event_stream.py` | `SensorEvent` + `merge_streams()`. |
| `central_tracker.py` | `Track` + `CentralTracker` — the one shared tracker. |
| `ttc.py` | The one TTC formula, used for both predicted and ground-truth TTC. |
| `evaluate.py` | Metrics: MAE/RMSE/match-rate, track-length buckets, multi-sensor composition. |
| `adapters.py` | The only file touching existing on-disk pipeline output / the NuScenes devkit. |
| `diagnostics.py` | Post-hoc track-identity safety checks (see below). |
| `run_pipeline.py` | Orchestrator — runs all four trackers and prints the full metrics table. |

## Fixes made while building this, not called out in the original brief

- **Radar's native per-channel timestamp and ego_pose were never captured
  on disk.** Step 1.2 (the old pipeline's radar parser) recorded neither a
  per-channel `sample_data` timestamp nor confirmed it was using that
  channel's own `ego_pose` (as opposed to some other sensor's). Verified
  directly against the NuScenes devkit that the channel's own `ego_pose`
  *was* already correct on disk, but the timestamp was missing entirely —
  `adapters.py` looks both up via the devkit directly
  (`sample['data'][channel] → sample_data['timestamp'/'ego_pose_token']`),
  applied uniformly to lidar/radar/camera_mono rather than trusting one
  sensor's on-disk field and reinventing the lookup for another.
- **Radar needs per-scan clustering, unlike lidar/camera_mono.** One real
  object triggers 3-8 separate radar echoes within a single channel's
  single scan; lidar arrives pre-clustered (Step 2.1's DBSCAN) and
  camera_mono arrives one-box-per-object (YOLO) already. Added per-channel-
  per-scan DBSCAN clustering (`config.RADAR_CLUSTER_EPS_M`, same value as
  the old pipeline's), applied per-channel rather than pooled across all 5
  radar channels (pooling would mix detections from slightly different
  real timestamps).
- **Ego velocity must not be a raw two-point finite difference.**
  `nusc.ego_pose` carries no velocity field. The old pipeline derived one
  with `(pos[i] - pos[i-1]) / dt` — exactly the noise-amplifying pattern
  this whole rebuild exists to move away from, and ego velocity feeds every
  single TTC computation (predicted and ground-truth alike), so its noise
  propagates everywhere. `adapters.ego_pose_lookup()` instead runs ego's
  own consecutive positions through a `ConstantVelocityKF` — the same
  primitive already used for tracked objects — and reads the smoothed
  velocity back out. Resets to a fresh KF on every scene change for the
  same reason `CentralTracker` does (R9).
- **TTC needed a closing-speed deadband, not just `closing_speed <= 0`.**
  A closing speed that is technically positive but negligibly small
  (extremely common for an object moving roughly tangentially to the ego
  vehicle) still divides `distance` by that tiny number — observed on the
  real dataset producing ground-truth TTCs up to **~190 million seconds**,
  silently dominating MAE. `config.MIN_CLOSING_SPEED` (0.3 m/s, same value
  as the old pipeline's own guard) closes this gap directly in
  `ttc.compute_ttc()` — the root cause (the unstable division itself), not
  a post-hoc cap on the output, since a post-hoc cap can't distinguish
  "large because closing speed is near-zero" (broken) from "large but
  legitimate, because the object is far away and closing slowly" (fine).
  Applies identically to predicted and ground-truth TTC, since both call
  this same function.
- **Gating must use the innovation covariance, not the predicted covariance
  alone.** This was the most consequential fix found after the pipeline
  first ran end-to-end — see the next section.

## What we found while tuning: gating covariance, and its limits

**The bug.** Every single-sensor baseline's track count came out far higher
than the old pipeline's own — lidar +99%, radar +58%, camera_mono +1% — and
that ordering tracked *exactly* with each sensor's measurement noise `R`
(smallest-`R` sensors fragmenting worst). Root cause: `gating.py` gated on
a track's predicted covariance `P_pred` alone, when the textbook-correct
gate uses the **innovation covariance** `S = P_pred + R`. A track just
updated by a low-noise sensor collapses `P` so tight that the very next
real detection of the same object — ordinary motion, nothing anomalous —
could fall outside the gate. Fixed by threading the current event's own
sensor variance into `mahalanobis_gate_and_assign()` (the same value
already looked up for that event's `update()` call).

**Gate radii, concretely** (99% confidence, `PROCESS_NOISE_SIGMA_A=2.0`,
after one `predict(dt=0.5s)` step — the samples-only 2Hz cadence this
pipeline currently runs at): lidar ≈1.5m, radar ≈3.0m, camera_mono ≈6.1m.
At typical urban speeds (5–10 m/s), an object moves 2.5–5m between samples
— comfortably outside lidar's own gate, explaining its worst fragmentation.

**The sigma_a tuning attempt, and why it was abandoned.**
`PROCESS_NOISE_SIGMA_A` is deliberately one sensor-independent constant
(it models the tracked *object's* own motion uncertainty, not sensor
noise — predict() runs before gating even knows which sensor, if any, will
match). Raising it (tested at 5.0 and 8.0, against the baseline 2.0)
widens every sensor's gate and did reduce lidar/radar fragmentation
substantially (lidar tracks: 5509 → 4559 → 3672; radar: 2489 → 2218 →
1904) — but MAE did not improve (lidar 5.353 → 5.451 → 5.913), and tracing
individual camera_mono tracks (position **and** the underlying raw YOLO
detection's class label, not proximity alone — proximity alone was already
shown untrustworthy in dense scenes during the original LiDAR
fragmentation investigation) found genuine, class-verified cross-object
over-merges that got worse with a wider gate: a track absorbing a real car
for several updates, then silently switching onto a nearby pedestrian
group for the rest of its life (97 detections at sigma_a=8.0; a comparable
merge exists at every sigma_a value tested — see below). **For a TTC
system, a merged track (confidently wrong) is a worse failure mode than a
fragmented one (honestly incomplete)** — this was the deciding principle,
not MAE or track count alone. `PROCESS_NOISE_SIGMA_A` is retained at its
original value, **2.0**; a single shared constant cannot simultaneously
widen lidar/radar's too-tight gate and avoid over-widening camera_mono's,
and no value tested resolved this cleanly. Systematically scanning every
camera_mono track for a raw-detection class change
(`diagnostics.flag_class_inconsistent_tracks()`, kept permanently in the
codebase — see below) found comparable flag rates at **all three** values
tested (2.0: 247/812, 30.4%; 5.0: 238/686, 34.7%), and manually tracing
several of sigma_a=2.0's own flagged tracks confirmed most were genuine
merges, not classification jitter — so 2.0 does not represent a
merge-free choice either, only the default with the least additional
complexity, matching the original design intent. Further tuning of this
one parameter was stopped here as a diminishing-returns use of project
time, not because a solution was found.

**A separate, sigma_a-*independent* structural finding.** The single worst
over-merge found across the entire investigation — a **120-detection**
camera_mono track (nearly a third of the whole scene), every one of its
raw nearest-detections correctly labeled "person" — occurs at
`sigma_a=2.0`, not at the higher values. Comparable-severity merges (118,
97 detections) exist at 5.0 and 8.0 too. This particular failure is driven
by `camera_mono`'s own measurement noise (`R=1.70`) alone: even with
*zero* process-noise contribution, `sqrt(gate_threshold * R) ≈ 4m`, already
wider than the ~1-2m spacing between people standing in a real group.
**No value of `PROCESS_NOISE_SIGMA_A` fixes this** — it is a structural
limit of gating on isotropic position covariance alone when real, distinct
objects are spaced closer together than the sensor's own noise floor.
Fixing it needs information this design does not have: per-detection
appearance/re-identification, or image-space IoU tracking — genuinely out
of scope here, not a gap in the gating math. Flagged as future work, not
pursued.

**Why not a class-aware gate (rejecting cross-class detections during
association)?** It would catch the car/person-style cross-category merges
above, but explicitly **not** the same-class group-merge (multiple
distinct pedestrians, all correctly labeled "person") — that needs
appearance/re-identification information regardless, so a class-aware gate
would be a partial fix for real implementation cost, this close to project
completion. Noted as future work rather than implemented.

**`diagnostics.py`, kept permanently, not as a throwaway script.**
`flag_class_inconsistent_tracks()` finds, for every track, the nearest raw
detection to each of its own snapshots and flags any track whose nearest
detection's class label changes anywhere across its life — an
unambiguous, cheap, automatable signal for cross-object over-merging that
gating/covariance alone can never produce (position math has no notion of
semantic class). Its own documented blind spot: a same-class merge (the
120-detection pedestrian-cluster case above) produces no class-label
signal at all and will not be flagged — this function complements, not
replaces, a manual position/GT-instance-switch trace for that case. Only
implemented for `camera_mono` (the only sensor whose raw detections carry
a class label at all); lidar clusters and radar points carry none.

## Closing baseline results

Full 404-sample / 10-scene dataset, `PROCESS_NOISE_SIGMA_A=2.0` (final),
all fixes above applied:

| sensor | MAE | RMSE | match% | n_tracks | n_gt_matched | n_pairs |
|---|---|---|---|---|---|---|
| lidar | 5.353 | 10.146 | 62.13% | 5509 | 3423 | 1633 |
| radar | 6.784 | 14.391 | 48.90% | 2489 | 1217 | 1318 |
| camera_mono | 5.465 | 11.657 | 82.27% | 812 | 668 | 1292 |
| fused | 6.639 | 13.184 | 55.80% | 7346 | 4099 | 3598 |

Track-length bucket breakdown (MAE / RMSE):

| bucket | lidar | radar | camera_mono | fused |
|---|---|---|---|---|
| [0,5) | 5.406 / 10.408 | 8.065 / 15.235 | 7.207 / 14.608 | 7.767 / 15.17 |
| [5,20) | 5.096 / 9.087 | 5.669 / 14.359 | 5.027 / 10.531 | 7.02 / 13.24 |
| [20,∞) | 6.384 / 8.032 | 4.718 / 10.199 | 4.977 / 11.052 | 3.825 / 8.899 |

Multi-sensor composition (fused run): 3357 multi-sensor tracks (45.7%) vs.
3989 single-sensor. MAE multi-sensor **6.871** vs. single-sensor **5.978**
— multi-sensor tracks remain modestly worse, not better, than
single-sensor ones. `compute_multi_sensor_composition()` only gives this
binary split, not a per-sensor-combination breakdown, so this cannot yet
be attributed to any one sensor specifically (e.g. "radar dilutes
accuracy") without further work.

**Fused MAE (6.639) is above lidar's own MAE (5.353) and above the
brief's reference value (3.54, from the old pipeline's different sensor
mix)** — reported plainly, not tuned toward. This is the first run with a
genuinely independent (LiDAR-free) camera signal as a full fusion
participant, so there is no valid prior "fused" number to compare against
on a like-for-like basis; the honest reading of this run is that
measurement-level fusion, as built, has not yet closed the gap to LiDAR
alone, and the multi-sensor composition table above is the most direct
evidence of why.

## Stated scope limits

- `ConstantVelocityKF` is deliberately linear constant-velocity, not
  UKF/CTRV. Turn-rate modeling is out of scope; if residual analysis later
  suggests turning-heavy scenes dominate the error, upgrading this one
  class is a contained change.
- The old pipeline's LiDAR-assisted `camera` baseline (Step 2.3.1) has no
  role anywhere in v2 — not excluded-but-referenced, simply not part of
  this design. Only `lidar`, `radar`, `camera_mono`.
- Adapters currently read only `samples/` (2Hz keyframes), not `sweeps/`
  (each sensor's full native rate) — the architecture's async design is
  built to benefit from native-rate data, but that extension was
  deliberately deferred pending a decision on its cost (particularly for
  camera_mono, whose monocular depth-from-height estimation may need
  re-deriving per sweep frame rather than reused from a samples-only run).
  Not yet estimated or scoped in detail.
- No existing notebook, `src/` file, or the old pipeline's own output was
  modified, deleted, or depended on anywhere in this package.
