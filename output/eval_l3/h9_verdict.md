# H9 (revised): precision / recall / false alarms per km

warn <= 2.5s (config_l3.FCW_TTC_S), brake <= 1.5s (config_l3.AEB_TTC_S).

Total distance driven (whole trainval dataset, within-scene consecutive deltas): **9.99 km**.

Methodology: event-based, not per-frame -- a real hazard or a predicted alarm is one contiguous (by real timestamp) run of consecutive samples below threshold for the same real instance, not one event per frame (counting every frame separately would massively inflate both counts for one slow approach spanning many samples). Recall's denominator (n_actual_hazard_events) comes from v2.evaluate.compute_ground_truth_ttc()'s FULL real-hazard population -- every real annotated instance, matched or not -- NOT just the H6 CSV's matched subset, which silently omits any hazard the system had zero candidate anywhere near (see hypothesis_h9.py's module docstring).

**Scope limit**: false alarms here are always tied to a real GT instance (every H6 row already is). A "phantom" false alarm -- a predicted track with no real object anywhere nearby -- is NOT counted; that needs the pipeline's own full, unmatched per-frame track output, available on disk for v1 (output/step_4/fused_tracks_all.csv) but not yet saved for v2.

| pipeline | bound | run | event | threshold_s | TP | FP | FN (covered/uncovered) | precision | recall | FA/km |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | hungarian | lidar | warn | 2.5 | 818 | 213 | 198/619 | 0.7934 | 0.5003 | 21.3126 |
| v1 | hungarian | lidar | brake | 1.5 | 340 | 137 | 153/411 | 0.7128 | 0.3761 | 13.7081 |
| v1 | hungarian | radar | warn | 2.5 | 862 | 177 | 121/652 | 0.8296 | 0.5272 | 17.7105 |
| v1 | hungarian | radar | brake | 1.5 | 326 | 85 | 77/501 | 0.7932 | 0.3606 | 8.505 |
| v1 | hungarian | camera_mono | warn | 2.5 | 489 | 172 | 88/1058 | 0.7398 | 0.2991 | 17.2102 |
| v1 | hungarian | camera_mono | brake | 1.5 | 210 | 105 | 58/636 | 0.6667 | 0.2323 | 10.5062 |
| v1 | hungarian | fused | warn | 2.5 | 982 | 187 | 193/460 | 0.84 | 0.6006 | 18.7111 |
| v1 | hungarian | fused | brake | 1.5 | 423 | 109 | 156/325 | 0.7951 | 0.4679 | 10.9065 |
| v1 | lock_once | lidar | warn | 2.5 | 643 | 307 | 154/838 | 0.6768 | 0.3933 | 30.7182 |
| v1 | lock_once | lidar | brake | 1.5 | 278 | 183 | 131/495 | 0.603 | 0.3075 | 18.3109 |
| v1 | lock_once | radar | warn | 2.5 | 740 | 242 | 163/732 | 0.7536 | 0.4526 | 24.2144 |
| v1 | lock_once | radar | brake | 1.5 | 334 | 166 | 113/457 | 0.668 | 0.3695 | 16.6099 |
| v1 | lock_once | camera_mono | warn | 2.5 | 437 | 151 | 110/1088 | 0.7432 | 0.2673 | 15.109 |
| v1 | lock_once | camera_mono | brake | 1.5 | 193 | 114 | 74/637 | 0.6287 | 0.2135 | 11.4068 |
| v1 | lock_once | fused | warn | 2.5 | 740 | 317 | 216/679 | 0.7001 | 0.4526 | 31.7188 |
| v1 | lock_once | fused | brake | 1.5 | 329 | 174 | 164/411 | 0.6541 | 0.3639 | 17.4103 |
| v2 | hungarian | lidar | warn | 2.5 | 843 | 1702 | 464/328 | 0.3312 | 0.5156 | 170.3011 |
| v2 | hungarian | lidar | brake | 1.5 | 340 | 1402 | 328/236 | 0.1952 | 0.3761 | 140.2832 |
| v2 | hungarian | radar | warn | 2.5 | 1138 | 5342 | 149/348 | 0.1756 | 0.696 | 534.5172 |
| v2 | hungarian | radar | brake | 1.5 | 499 | 3885 | 113/292 | 0.1138 | 0.552 | 388.7307 |
| v2 | hungarian | camera_mono | warn | 2.5 | 515 | 512 | 105/1015 | 0.5015 | 0.315 | 51.2304 |
| v2 | hungarian | camera_mono | brake | 1.5 | 201 | 391 | 88/615 | 0.3395 | 0.2223 | 39.1232 |
| v2 | hungarian | fused | warn | 2.5 | 1333 | 6147 | 214/88 | 0.1782 | 0.8153 | 615.065 |
| v2 | hungarian | fused | brake | 1.5 | 614 | 4331 | 224/66 | 0.1242 | 0.6792 | 433.3571 |
| v2 | lock_once | lidar | warn | 2.5 | 719 | 1888 | 344/572 | 0.2758 | 0.4398 | 188.9121 |
| v2 | lock_once | lidar | brake | 1.5 | 274 | 1514 | 294/336 | 0.1532 | 0.3031 | 151.4899 |
| v2 | lock_once | radar | warn | 2.5 | 898 | 4742 | 321/416 | 0.1592 | 0.5492 | 474.4815 |
| v2 | lock_once | radar | brake | 1.5 | 325 | 3800 | 315/264 | 0.0788 | 0.3595 | 380.2256 |
| v2 | lock_once | camera_mono | warn | 2.5 | 393 | 538 | 86/1156 | 0.4221 | 0.2404 | 53.8319 |
| v2 | lock_once | camera_mono | brake | 1.5 | 161 | 395 | 84/659 | 0.2896 | 0.1781 | 39.5235 |
| v2 | lock_once | fused | warn | 2.5 | 1033 | 5339 | 379/223 | 0.1621 | 0.6318 | 534.217 |
| v2 | lock_once | fused | brake | 1.5 | 388 | 4258 | 384/132 | 0.0835 | 0.4292 | 426.0528 |