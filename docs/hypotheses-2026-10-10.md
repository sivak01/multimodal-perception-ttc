# Fusion evaluation: open hypotheses and checks (from session 9-10 Oct 2026)

## Where things stand
- New file `v2/evaluate_matched.py` + `v2/tests/test_evaluate_matched.py` + 10-line change to `v2/run_pipeline.py`.
- Committed locally in the cloud sandbox as `1feef7b`, NOT pushed (GitHub not linked). Copy files by hand or link GitHub and push.
- Tests: 9/9 new, 72/72 v2 suite. Never run on real nuScenes data yet.
- Caveat: the 2.36 vs 4.04 numbers come from the v1 notebooks (Step 6). The new evaluator plugs into v2 only.

## Hypotheses
| ID | Hypothesis | Status | How to test | Pass / fail signal |
|---|---|---|---|---|
| H1 | Merged (2.36) vs single-sensor (4.04) MAE is not apples-to-apples: different object populations | Confirmed by design reading | Run evaluate_matched; read paired_by_coverage | Gap shrinks a lot on identical keys = population effect |
| H2 | Single-sensor dominance is a pipeline artifact (gate too tight) | Mostly refuted by repo notes | Already measured: same-object offset median <1.3 m, 5-7% exceed 2.5 m gate; 3.5 m gate made MAE worse | No action unless new evidence |
| H3 | Per-sensor ROI (coverage model), union for the suite, is the fair evaluation domain | **Done 2026-10-17 — see "H3 findings" below** | Per GT key: expected coverage (FOV, range, num_lidar_pts, num_radar_pts) vs actual coverage | Large expected-but-missed share = pipeline misses, not physics |
| H4 | Camera depth error blocks merging; bearing gating would help | Low priority | v2 gate uses isotropic R; camera error is along line of sight | Revisit only if camera k=2 recall is low in matched report |
| H5 | Radar's 2.77 MAE is flattered by easy parked cars | Open | by_motion split in matched report | Radar moving MAE much worse than parked = confirmed |
| H6 | Fusion beats each sensor on the same objects | Open (the real question) | paired_by_coverage, k=2 and k=3 | Negative fused_minus_sensor = fusion wins |
| H7 | A few huge TTC errors dominate MAE | Open | Compare MAE vs median vs capped MAE | MAE >> median = outlier-driven |
| H8 | Lock-once GT matching in evaluate.py biases results | Open | Compare old v2 table vs matched table, same run | Big differences = old method biased |
| H9 | Excluding non-closing objects (inf GT TTC) changes the picture — revised to: precision/recall/false alarms per km at warn<=2.5s, brake<=1.5s | **Done 2026-10-18 — see "H9 findings" below** | Count excluded keys | Report the excluded share in the dissertation |

## H3 findings (2026-10-17)

H3's original framing ("per-sensor ROI, expected vs. actual coverage") was revised before building: run against **nuScenes v1.0-trainval** (not mini — mini's own result was flagged as inflated/underpowered, same statistical-power problem that motivated the separate v1-notebook trainval migration elsewhere in this project), tighten the expected-coverage thresholds (lidar needs `num_lidar_pts >= 10`, not just `>0`; radar `>= 1`; camera added via nuScenes' own `visibility_token >= 40%`), and build a 4-stage **miss funnel** — `expected -> detected -> tracked -> fused` — instead of a flat 2-stage expected/actual check, so a miss can be attributed to a specific pipeline stage.

**Built additively**: `v2/miss_funnel.py`, `v2/run_miss_funnel.py`, `v2/tests/test_miss_funnel.py` — does not modify `v2/coverage_model.py`, `central_tracker.py`, `adapters.py`, or any other existing v2 file. Stage definitions:
- **detected**: that sensor's own raw per-frame `SensorEvent` (before any tracking) lands within `MATCH_DIST_THRESHOLD` (3.0m) of the real GT position at that exact sample.
- **tracked**: that key is covered by one of that sensor's solo `CentralTracker` tracks that survived with `length>=2` (real, re-confirmed — not a single transient hit). Needed an explicit length filter: `CentralTracker.all_tracks()` includes every track ever created, even 1-hit ones immediately evicted, with no length filter anywhere in `central_tracker.py`.
- **fused**: that key is covered by the fused tracker's own output (`merge_streams(lidar, radar, camera_mono)` through one `CentralTracker`), same length filter. One shared fused candidate set across all three sensor rows (the fused tracker covers the whole scene, not per-sensor).

**Results** (`v2_miss_funnel.csv`, 107,206 total GT keys x 3 sensors, trainval):

| Sensor | n expected | detected | tracked | fused | median range_m of detection-stage misses |
|---|---|---|---|---|---|
| lidar | 34,305 | 60.52% | 66.45% | 89.12% | 25.58m |
| radar | 30,204 | 77.54% | 70.80% | 84.40% | 29.86m |
| camera_mono | 77,069 | 25.08% | 26.28% | 77.06% | 30.81m |

**Critical correction, verified rather than assumed: the "tracked"/"fused" percentages are substantially inflated by cross-object contamination, not genuine detection-recovery.** `tracked% > detected%` for lidar and camera_mono, and `fused% >> tracked%` for all three, is logically suspicious on its face — nothing can be tracked without first being detected. Traced 40 randomly-sampled lidar cases (`expected=True, detected=False, tracked=True`) back to the actual covering track's own raw detection history (rebuilt the lidar tracker fresh, ~3 min, standalone scratchpad script — not committed), checking whether that track's own snapshots are genuinely closest to the GT instance in question or to a *different* nearby real object at the matching times.

**Result: 37/40 (92.5%) were CONTAMINATION — the covering track's own history belongs to a different real object. Only 3/40 (7.5%) were GENUINE** same-object track continuity (a Kalman prediction correctly bridging one missed frame). Mechanism: `candidates_from_tracker()` offers a track as a candidate at *every* sample it could plausibly still be alive for (extrapolated via its own velocity, up to a 1.5s tolerance), not just its exact detection frame — combined with the 3.0m match radius, this creates ample opportunity for a different, nearby real object's track to spuriously "cover" the key in question. Not independently re-verified for radar/camera_mono or the `fused` stage specifically — inferred to be at least as bad there (more pooled tracks in `fused` = more collision opportunities), but that's an inference, not a checked fact.

**Honest answer to H3, as actually supported by what was verified**: only the **detected** column above is directly trustworthy as a real per-frame coverage check. The pattern there is not even — camera_mono misses detection 75% of the time it's expected to be visible (25.08%), vs. lidar missing ~40% and radar missing ~22% — consistent with camera_mono being the weakest single-frame detector of the three, a pipeline limitation rather than pure physics (median miss range ~25-31m for all three isn't concentrated at extreme range, where a physical limit would be the obvious excuse). The "tracked"/"fused" columns should **not** be read as genuine recovery rates until the same contamination check is extended to them — this is the clear next step if H3 is revisited, not a finished number.

## H9 findings (2026-10-18)

H9's framing was revised before building, matching the user's own note: "How many false alarms? Precision, recall and false alarms per km for warn at 2.5s and brake at 1.5s. 57% of keys have infinite GT TTC, so this metric is needed." Thresholds reuse `v2/eval_l3/config_l3.py`'s `FCW_TTC_S=2.5` / `AEB_TTC_S=1.5` — both already present in that file, explicitly reserved as "for a future hypothesis, not used by H1 itself." This is that hypothesis.

**Cheap by design — confirmed before building, not assumed.** Checked what was already on disk from H1/H6 (2026-10-17) first: `output/eval_l3/h6_v1_keys.csv` / `h6_v2_keys.csv` already carry per-key `pred_ttc`/`gt_ttc` pairs for every sensor + fused, both matching bounds, on trainval — no tracker rebuild needed. Confirmed `gt_ttc==inf` is 60.8% of rows there (close to the user's own "57%" estimate). Only new cost: one GT-context build (`v2/eval_l3/gt_context.build_gt_context()`, a single NuScenes metadata load, ~144s) — no event materialization, no `CentralTracker` run. Total H9 run time: a few minutes, not H1/H3/H6's ~70-minute multi-tracker rebuilds.

**Built additively**: `v2/eval_l3/hypothesis_h9.py`, `v2/eval_l3/run_h9.py`, `v2/tests/test_eval_l3_h9.py` (10 tests) — does not modify any existing v2 file.

**A real methodology gap in the existing H6 CSVs was found and fixed, not glossed over.** Those CSVs only contain rows where the per-sample matcher found some real GT instance within `MATCH_DIST_THRESHOLD_M` of a predicted candidate — a real hazard the system had *zero* candidate anywhere near is silently absent, not represented with `pred_ttc=inf`. Using those rows alone for recall's denominator would systematically overstate recall by hiding every completely-undetected hazard. Fixed by building the real hazard-event population independently from `v2.evaluate.compute_ground_truth_ttc()` (every real annotated instance, matched or not — the same GT-only computation `gt_context.py` already shares between the v1/v2 H6 builders) and only using the H6 CSVs for the *predicted* side.

**Event-based, not per-frame.** A real hazard or a predicted alarm is one contiguous (by real timestamp, not sample_id string order) run of consecutive samples where TTC ≤ threshold for the same real instance — not one event per frame, which would massively inflate both the hazard count and the false-alarm count for a single slow approach spanning many frames. FN is split into `fn_covered` (something was tracked nearby during the hazard, it just never crossed threshold — an algorithm/filtering miss) vs. `fn_uncovered` (zero candidate anywhere near during the hazard at all — a detection/coverage miss, same distinction H3's miss funnel draws).

**Scope limit, stated plainly:** false alarms here are always tied to a real GT instance (every H6 row already is). A "phantom" false alarm — a predicted track with no real object anywhere nearby at all — is **not** counted; that needs the pipeline's own full, unmatched per-frame track output. Available on disk already for v1 (`output/step_4/fused_tracks_all.csv`); not yet saved for v2. Flagged as real further work, not assumed away.

**Total distance driven**: 9.99 km over the whole trainval dataset (85 scenes, within-scene consecutive ego-position deltas only, never summed across a scene cut). Plausible for nuScenes' short (~20s), low-speed urban scenes, not an obvious bug — worth a second look if it ever needs defending in the dissertation, but not re-verified beyond the scene-isolation unit test.

**v1 results (the notebook pipeline) — directly trustworthy, report these as the primary H9 numbers:**

| bound | run | warn precision | warn recall | warn FA/km | brake precision | brake recall | brake FA/km |
|---|---|---|---|---|---|---|---|
| hungarian | lidar | 0.793 | 0.500 | 21.3 | 0.713 | 0.376 | 13.7 |
| hungarian | radar | 0.830 | 0.527 | 17.7 | 0.793 | 0.361 | 8.5 |
| hungarian | camera_mono | 0.740 | 0.299 | 17.2 | 0.667 | 0.232 | 10.5 |
| hungarian | fused | 0.840 | 0.601 | 18.7 | 0.795 | 0.468 | 10.9 |
| lock_once | lidar | 0.677 | 0.393 | 30.7 | 0.603 | 0.308 | 18.3 |
| lock_once | radar | 0.754 | 0.453 | 24.2 | 0.668 | 0.370 | 16.6 |
| lock_once | camera_mono | 0.743 | 0.267 | 15.1 | 0.629 | 0.214 | 11.4 |
| lock_once | fused | 0.700 | 0.453 | 31.7 | 0.654 | 0.364 | 17.4 |

Fused is the best or tied-best on precision and clearly best on recall under the optimistic (hungarian) bound; the pessimistic (lock_once) bound tells a less flattering story for fused specifically (recall unchanged, but FA/km becomes the *worst* of the four at 31.7). Consistent with the project's existing finding that fusion's real advantage shows up more under optimistic matching than under strict lock-once identity.

**v2 results (the async rebuild) — NOT yet trustworthy at face value, a concrete problem found and traced, not just a suspicious number reported blind:**

| bound | run | warn precision | warn recall | warn FA/km | n_actual_hazards | n_predicted_events |
|---|---|---|---|---|---|---|
| hungarian | lidar | 0.331 | 0.516 | 170.3 | 1635 | 2621 |
| hungarian | radar | 0.176 | 0.696 | 534.5 | 1635 | 6533 |
| hungarian | camera_mono | 0.502 | 0.315 | 51.2 | 1635 | 1038 |
| hungarian | fused | 0.178 | 0.815 | 615.1 | 1635 | 7691 |

v2's own predicted-event counts run 1.6x (lidar) to 6.3x (radar) the real hazard count (1,635), vs. v1's near-1:1 ratio (664-1,184) for the same real dataset. **Traced to a concrete cause, not inferred from the aggregate number alone**: pulled one real v2 radar instance's raw `pred_ttc` sequence from `h6_v2_keys.csv` directly — it flickers between `inf` and small finite values (1.72s, 3.25s, 1.98s, 3.32s, 5.37s...) frame to frame while the real GT TTC for that same instance stays `inf` the entire time (this object genuinely never closes). Each isolated dip below 2.5s becomes its own 1-frame "event" under H9's event-collapsing logic — confirmed this is a real property of v2's own matched predictions, not an artifact of how H9 counts events. camera_mono is the one v2 sensor that does NOT show this blow-up (1,038 vs v1's 568-1,184, the same range) — plausibly because its velocity is derived by differencing position over time (inherently smoother) rather than radar's raw, noisier instantaneous Doppler estimate; `fused` inherits/amplifies the problem (7,691, the worst of all four) because it pools lidar+radar+camera_mono and any one noisy stream creates a fusion-level blip opportunity. This specific causal chain (per-sample "no-lock" Hungarian matching → different nearby candidate each frame → inconsistent instantaneous velocity → oscillating TTC) is plausible and consistent with `hungarian_match.py`'s own documented design ("no identity persisted between samples... still a 'no-lock' scheme"), but was not independently isolated with a controlled test — stated as the most likely mechanism, not a fully proven one.

**Update (2026-10-19): the mechanism above is now confirmed, pooled across the whole radar dataset, not inferred from one traced example.** Rebuilt the radar solo tracker fresh (same methodology `build_h6_dataset_v2.py` uses — `candidates_from_tracker()` on all tracks, no length filter — the only way to recover `track_id` per matched row, since `h6_common.KeyRow` strips it before writing the CSV). For every radar instance's hungarian-bound matched sequence, tested whether a `track_id` switch (a different track winning the per-frame Hungarian match) co-occurs with a TTC threshold-crossing (warn, ≤2.5s) more than chance would predict:

- **Transition-level** (41,186 real consecutive-sample transitions, pooled across all radar instances): P(crossing | track switched) = 30.5% vs. P(crossing | track did NOT switch) = 3.2% — a **9.4x** higher rate. χ²=2,360.8, p≈0.
- **Instance-level**: instances whose matched sequence used only 1 track_id throughout (n=103) average 0.0136 warn-events per matched row; instances that used 2+ distinct track_ids (n=4,866) average 0.140 — a **10.3x** higher rate.

Both angles agree, with large effect sizes and high significance. One honest caveat: this is a correlational test, not a fully isolated controlled experiment (e.g. rebuilding the matcher with identity-lock and re-measuring directly) — a genuinely fast/close object could independently be both harder to track *and* have noisier real TTC for an unrelated reason. But a 9-10x effect this consistent across two different angles is about as strong as a pure-data correlational test can get without that further intervention. Script: scratchpad `verify_q1_track_churn_vs_ttc.py` (not committed — standalone investigation, same pattern as the H3 contamination trace).

**Honest conclusion**: v1's numbers above are the real, trustworthy H9 answer for this dataset as it stands today. v2's raw numbers are not comparable to v1's at face value (rule 2 holds regardless) and should not be reported as "v2's false alarm rate" without first addressing the predicted-event fragmentation shown above — likely the same kind of identity-continuity fix v1's `FusedTracker` already went through (documented at length elsewhere in this project as the identity-inheritance/de-merge investigation), not yet applied or verified for v2's `CentralTracker`. Flagged as a real, open follow-up for v2 specifically, not glossed over as "fusion is just like that."

Full numbers: `output/eval_l3/h9_metrics.csv`, `output/eval_l3/h9_verdict.md`.

## ROI/range follow-up (2026-10-19): does per-sensor range asymmetry explain fusion's aggregate gap?

Follow-up question, not in the original H-list: does the fact that different sensors have different practical detection ranges (e.g. a far object only camera can see) create a structural, already-measurable drag on fusion's aggregate accuracy, independent of the composition effect already documented? Two checks, both pure aggregation over already-written CSVs — no tracker rebuild, no NuScenes load.

**Check 1 — range-stratified sensor coverage, from `v2_miss_funnel.csv` (321,618 rows, 107,206 keys, already on disk from H3):**

| range | 0 sensors expected | 1 sensor (single-sensor-only) | 2 sensors | 3 sensors | n keys |
|---|---|---|---|---|---|
| 0-20m | 9.7% | 25.5% | 39.3% | 25.5% | 33,658 |
| 20-40m | 24.4% | 43.5% | 19.2% | 12.8% | 42,406 |
| 40-60m | 30.9% | 46.7% | 17.5% | 4.9% | 21,811 |
| 60-80m | 30.1% | 48.1% | 19.2% | 2.6% | 6,828 |
| 80-100m | 29.3% | 54.1% | 15.4% | 1.1% | 2,188 |
| 100-150m | 21.0% | 73.0% | 6.0% | 0.0% | 267 |
| **150m+** | 0.0% | **100.0%** | 0.0% | 0.0% | 48 |

**Beyond 150m, zero keys have 2+ sensors expected — fusion is structurally impossible there, not just rare.** And of the single-sensor-only keys, the lone sensor is camera 77.8% of the time even at 0-20m, rising to 100.0% at 150m+ — confirmed with real data, not assumed. (The 100-150m/150m+ buckets have modest absolute n (267/48), but the trend is already unambiguous and monotonic from 60m onward across much larger buckets (6,828/2,188 keys), so this isn't a tiny-n artifact.)

**Check 2 — does fusion's aggregate gap to radar actually come from the keys radar can never see at all?** Using `h6_v1_keys.csv`/`h6_v2_keys.csv` (already on disk from H6): split fused's matched rows into keys radar also covers ("shared") vs. keys radar has zero candidate for at all ("radar-invisible"), and separately score radar's own MAE restricted to that exact same shared set (the genuinely apples-to-apples comparison).

**v1 — confirms the mechanism directly:**

| bound | fused (shared) | fused (radar-invisible) | radar's own full MAE | radar on the SAME shared keys | % of fused rows radar can't see |
|---|---|---|---|---|---|
| hungarian | 2.360 | **3.456** | 2.356 | 2.300 | 26.8% |
| lock_once | 4.124 | **5.612** | 3.686 | 3.721 | 12.3% |

Radar-invisible keys are clearly the hard ones. Even restricted to the exact shared population, fusion still doesn't beat radar (ties or loses slightly: 2.300 vs 2.360). So for v1, the aggregate gap to radar is substantially explained by the harder, radar-invisible population it has to carry — your hypothesis, confirmed with data.

**v2 — does NOT show the same pattern (reported honestly, not smoothed over to match v1):**

| bound | fused (shared) | fused (radar-invisible) | radar's own full MAE | radar on the SAME shared keys | % of fused rows radar can't see |
|---|---|---|---|---|---|
| hungarian | 5.502 | 4.476 (**better**) | 5.679 | 5.675 | 25.2% |
| lock_once | 6.828 | 6.437 (**better**) | 7.602 | 7.602 | 6.2% |

Radar-invisible keys are *not* harder for v2, and fused actually beats radar on the shared population too. This doesn't contradict the H6 verdict (REFUTED, fused doesn't beat the best sensor) — H6 is scoped to the narrow in-corridor + GT-TTC-0-5s safety-critical slice specifically; this is the unrestricted full-dataset comparison, a genuinely different population, and both results stand side by side rather than one overturning the other (rule 2: v1 and v2 are never compared against each other, and the full-dataset vs. safety-critical-slice distinction is not the same axis as v1-vs-v2 either).

**Conclusion**: the range/ROI-asymmetry mechanism is real (Check 1, unambiguous) and does explain a meaningful share of v1's aggregate gap to radar (Check 2). It does not generalize to v2's current architecture on the unrestricted dataset — though v2 still loses to the best sensor in the safety-critical zone specifically (H6), for reasons this check doesn't speak to. Not committed; analysis run ad hoc from the terminal, no new script file.

## Tomorrow's order
1. Get code into local repo (copy files + `git apply run_pipeline.patch`, or link GitHub and push).
2. `pip install pytest pyquaternion`, then `python -m pytest v2/tests -q`.
3. `python -m v2.run_pipeline | Tee-Object v2_matched_report.txt`.
4. Answer H1, H5, H6, H7, H8, H9 from the report.
5. Only then decide whether to build the coverage model (H3).
