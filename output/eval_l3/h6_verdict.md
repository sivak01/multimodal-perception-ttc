# H6: does fused beat each single sensor, on the same GT keys?

Run separately for v1 and v2 (rule 2: never compared against each other). Paired diff = |fused_err| - |sensor_err| per shared key; negative = fused wins.

Pass/fail rule (given verbatim): **CONFIRMED** only if fused beats the BEST single sensor in the in-corridor, GT-TTC-0-5s band, with a scene-bootstrap 95% CI (of the paired diff) that excludes 0. Otherwise **REFUTED** (CI includes 0, or favours the sensor) or **INCONCLUSIVE** (n < 30).

## Pass/fail determination


### v2

**per-frame Hungarian (optimistic)**

- Best single sensor in this slice (lowest own mean error): **camera_mono** (mean_sensor_err=1.3972, n=158)
- Fused mean error: 1.6071; diff (fused-best): 0.2099; 95% CI [-0.0378, 0.4746]; fused wins 23.42% of keys
- paired sign test: {'n': 67, 'n_positive': 37, 'n_negative': 30, 'n_ties': 91, 'p_value': 0.463818}
- **VERDICT: REFUTED** -- 95% CI [-0.0378, 0.4746] includes 0

  Other sensors in this same slice, for context:
  - lidar: mean_sensor_err=1.4481, n=262
  - radar: mean_sensor_err=1.5274, n=646

**lock-once (pessimistic)**

- Best single sensor in this slice (lowest own mean error): **lidar** (mean_sensor_err=1.2743, n=176)
- Fused mean error: 1.7817; diff (fused-best): 0.5074; 95% CI [-0.2423, 1.5176]; fused wins 19.89% of keys
- paired sign test: {'n': 111, 'n_positive': 35, 'n_negative': 76, 'n_ties': 65, 'p_value': 0.000125}
- **VERDICT: REFUTED** -- 95% CI [-0.2423, 1.5176] includes 0

  Other sensors in this same slice, for context:
  - radar: mean_sensor_err=1.9483, n=563
  - camera_mono: mean_sensor_err=2.3606, n=117


### v1

**per-frame Hungarian (optimistic)**

- Best single sensor in this slice (lowest own mean error): **radar** (mean_sensor_err=0.3959, n=553)
- Fused mean error: 0.3918; diff (fused-best): -0.0041; 95% CI [-0.1195, 0.1079]; fused wins 61.66% of keys
- paired sign test: {'n': 544, 'n_positive': 341, 'n_negative': 203, 'n_ties': 9, 'p_value': 0.0}
- **VERDICT: REFUTED** -- 95% CI [-0.1195, 0.1079] includes 0

  Other sensors in this same slice, for context:
  - lidar: mean_sensor_err=1.209, n=197
  - camera_mono: mean_sensor_err=0.9542, n=121

**lock-once (pessimistic)**

- Best single sensor in this slice (lowest own mean error): **radar** (mean_sensor_err=1.0444, n=585)
- Fused mean error: 1.0926; diff (fused-best): 0.0481; 95% CI [-0.54, 0.6026]; fused wins 55.04% of keys
- paired sign test: {'n': 579, 'n_positive': 322, 'n_negative': 257, 'n_ties': 6, 'p_value': 0.007766}
- **VERDICT: REFUTED** -- 95% CI [-0.54, 0.6026] includes 0

  Other sensors in this same slice, for context:
  - lidar: mean_sensor_err=1.1776, n=209
  - camera_mono: mean_sensor_err=2.02, n=151


## Full results table (all slices, both bounds, all 3 sensors, both pipelines)

| pipeline | bound | sensor | slice | n | fused mean err | sensor mean err | mean diff | median diff | % fused wins | 95% CI | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | per-frame Hungarian (optimistic) | lidar | overall | 5566 | 3.2052 | 3.0739 | 0.1313 | 0.0 | 15.49 | [0.0217, 0.2377] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | lidar | k2 | 2735 | 3.0894 | 2.8417 | 0.2477 | 0.0 | 14.81 | [0.0643, 0.4345] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | lidar | k3 | 1872 | 3.115 | 3.1094 | 0.0056 | 0.0 | 23.61 | [-0.166, 0.1723] | 0.223002 |
| v2 | per-frame Hungarian (optimistic) | lidar | in_corridor | 379 | 2.7032 | 2.768 | -0.0647 | 0.0 | 20.84 | [-0.6977, 0.5489] | 0.321574 |
| v2 | per-frame Hungarian (optimistic) | lidar | ttc_0_3s | 2177 | 1.2112 | 1.269 | -0.0578 | 0.0 | 15.71 | [-0.1491, 0.0286] | 0.03632 |
| v2 | per-frame Hungarian (optimistic) | lidar | ttc_3_5s | 1386 | 2.0232 | 1.9401 | 0.0831 | 0.0 | 15.15 | [-0.1222, 0.2718] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | lidar | vru | 1920 | 3.3071 | 3.2004 | 0.1067 | 0.0 | 15.89 | [-0.0502, 0.2717] | 0.001109 |
| v2 | per-frame Hungarian (optimistic) | lidar | vehicle | 2091 | 3.2207 | 3.2431 | -0.0223 | 0.0 | 18.36 | [-0.2704, 0.2088] | 0.007324 |
| v2 | per-frame Hungarian (optimistic) | lidar | day | 5566 | 3.2052 | 3.0739 | 0.1313 | 0.0 | 15.49 | [0.0217, 0.2377] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | lidar | night | 0 | None | None | None | None | None | [None, None] | None |
| v2 | per-frame Hungarian (optimistic) | lidar | in_corridor_ttc_0_5s | 262 | 1.3279 | 1.4481 | -0.1202 | 0.0 | 18.32 | [-0.4788, 0.2266] | 0.554601 |
| v2 | per-frame Hungarian (optimistic) | radar | overall | 14083 | 5.5682 | 5.674 | -0.1058 | 0.0 | 14.32 | [-0.1708, -0.0405] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | k2 | 4271 | 4.5947 | 4.719 | -0.1243 | 0.0 | 22.59 | [-0.2649, 0.0413] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | k3 | 2252 | 3.6333 | 4.0311 | -0.3979 | 0.0 | 42.5 | [-0.6398, -0.1367] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | in_corridor | 1180 | 4.8134 | 4.8917 | -0.0783 | 0.0 | 13.98 | [-0.3766, 0.176] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | ttc_0_3s | 2886 | 1.2859 | 1.3324 | -0.0465 | 0.0 | 24.15 | [-0.1662, 0.066] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | ttc_3_5s | 3234 | 2.0866 | 2.1366 | -0.0501 | 0.0 | 16.76 | [-0.1343, 0.0527] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | vru | 4319 | 6.0891 | 6.416 | -0.327 | 0.0 | 18.18 | [-0.4509, -0.2156] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | vehicle | 6102 | 6.0751 | 6.0899 | -0.0148 | 0.0 | 14.01 | [-0.1426, 0.1145] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | day | 14083 | 5.5682 | 5.674 | -0.1058 | 0.0 | 14.32 | [-0.1708, -0.0405] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | radar | night | 0 | None | None | None | None | None | [None, None] | None |
| v2 | per-frame Hungarian (optimistic) | radar | in_corridor_ttc_0_5s | 646 | 1.5598 | 1.5274 | 0.0323 | 0.0 | 13.93 | [-0.0991, 0.1377] | 0.000441 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | overall | 4830 | 4.6129 | 4.3468 | 0.2662 | 0.0 | 16.96 | [0.1251, 0.4166] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | k2 | 2004 | 4.7731 | 4.6177 | 0.1554 | 0.0 | 17.42 | [-0.1073, 0.4258] | 5.4e-05 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | k3 | 2038 | 3.4435 | 2.9582 | 0.4853 | 0.0 | 22.72 | [0.2864, 0.6912] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | in_corridor | 346 | 5.9438 | 5.1695 | 0.7743 | 0.0 | 17.05 | [0.2027, 1.3143] | 0.075551 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | ttc_0_3s | 1381 | 1.0685 | 0.999 | 0.0695 | 0.0 | 23.39 | [-0.078, 0.242] | 0.010938 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | ttc_3_5s | 1106 | 2.005 | 1.8778 | 0.1272 | 0.0 | 16.27 | [-0.0598, 0.3114] | 0.001375 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | vru | 1920 | 4.6108 | 4.2888 | 0.322 | 0.0 | 15.62 | [0.1279, 0.5187] | 1e-06 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | vehicle | 2668 | 4.6299 | 4.3475 | 0.2823 | 0.0 | 18.07 | [0.0852, 0.4879] | 3e-06 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | day | 4830 | 4.6129 | 4.3468 | 0.2662 | 0.0 | 16.96 | [0.1251, 0.4166] | 0.0 |
| v2 | per-frame Hungarian (optimistic) | camera_mono | night | 0 | None | None | None | None | None | [None, None] | None |
| v2 | per-frame Hungarian (optimistic) | camera_mono | in_corridor_ttc_0_5s | 158 | 1.6071 | 1.3972 | 0.2099 | 0.0 | 23.42 | [-0.0378, 0.4746] | 0.463818 |
| v2 | lock-once (pessimistic) | lidar | overall | 3879 | 4.2331 | 3.6781 | 0.555 | 0.0 | 24.54 | [0.3476, 0.776] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | k2 | 1890 | 3.8586 | 3.3109 | 0.5477 | 0.0 | 23.81 | [0.286, 0.8312] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | k3 | 1229 | 3.7049 | 3.0845 | 0.6204 | 0.0436 | 31.24 | [0.2082, 0.9958] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | in_corridor | 265 | 3.3554 | 2.7511 | 0.6042 | 0.0 | 23.77 | [-0.1577, 1.4626] | 4e-05 |
| v2 | lock-once (pessimistic) | lidar | ttc_0_3s | 1340 | 1.2796 | 1.2229 | 0.0567 | 0.0 | 26.94 | [-0.1082, 0.2201] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | ttc_3_5s | 1003 | 2.6264 | 1.93 | 0.6964 | 0.1188 | 23.63 | [0.3569, 1.0388] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | vru | 1342 | 5.1407 | 4.1814 | 0.9593 | 0.0 | 19.45 | [0.6297, 1.305] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | vehicle | 1548 | 4.0658 | 3.7671 | 0.2986 | 0.0 | 34.82 | [-0.0799, 0.6485] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | day | 3879 | 4.2331 | 3.6781 | 0.555 | 0.0 | 24.54 | [0.3476, 0.776] | 0.0 |
| v2 | lock-once (pessimistic) | lidar | night | 0 | None | None | None | None | None | [None, None] | None |
| v2 | lock-once (pessimistic) | lidar | in_corridor_ttc_0_5s | 176 | 1.7817 | 1.2743 | 0.5074 | 0.0 | 19.89 | [-0.2423, 1.5176] | 0.000125 |
| v2 | lock-once (pessimistic) | radar | overall | 17110 | 7.111 | 7.1359 | -0.0249 | 0.0 | 1.24 | [-0.0705, 0.0026] | 0.000194 |
| v2 | lock-once (pessimistic) | radar | k2 | 4460 | 5.4239 | 5.4524 | -0.0285 | 0.0 | 1.55 | [-0.066, 0.0] | 0.005024 |
| v2 | lock-once (pessimistic) | radar | k3 | 2137 | 4.4344 | 4.4269 | 0.0075 | 0.0 | 1.22 | [-0.0307, 0.0614] | 1.0 |
| v2 | lock-once (pessimistic) | radar | in_corridor | 1243 | 6.3538 | 6.3961 | -0.0423 | 0.0 | 2.49 | [-0.1257, 0.0018] | 0.005402 |
| v2 | lock-once (pessimistic) | radar | ttc_0_3s | 2872 | 1.5289 | 1.5185 | 0.0104 | 0.0 | 0.49 | [-0.0077, 0.0422] | 0.48685 |
| v2 | lock-once (pessimistic) | radar | ttc_3_5s | 3775 | 2.4485 | 2.4497 | -0.0012 | 0.0 | 0.98 | [-0.0056, 0.0018] | 0.067445 |
| v2 | lock-once (pessimistic) | radar | vru | 5414 | 7.7334 | 7.7876 | -0.0543 | 0.0 | 1.39 | [-0.1613, 0.0] | 0.000107 |
| v2 | lock-once (pessimistic) | radar | vehicle | 7961 | 7.5899 | 7.6078 | -0.0179 | 0.0 | 1.24 | [-0.0509, 0.006] | 0.046464 |
| v2 | lock-once (pessimistic) | radar | day | 17110 | 7.111 | 7.1359 | -0.0249 | 0.0 | 1.24 | [-0.0705, 0.0026] | 0.000194 |
| v2 | lock-once (pessimistic) | radar | night | 0 | None | None | None | None | None | [None, None] | None |
| v2 | lock-once (pessimistic) | radar | in_corridor_ttc_0_5s | 563 | 1.9338 | 1.9483 | -0.0145 | 0.0 | 1.6 | [-0.042, 0.0] | 0.021484 |
| v2 | lock-once (pessimistic) | camera_mono | overall | 4045 | 6.1274 | 5.7786 | 0.3487 | 0.0 | 33.15 | [-0.0828, 0.7559] | 0.0 |
| v2 | lock-once (pessimistic) | camera_mono | k2 | 1480 | 5.4863 | 5.2057 | 0.2807 | 0.0196 | 36.89 | [-0.5055, 0.9722] | 0.0 |
| v2 | lock-once (pessimistic) | camera_mono | k3 | 1377 | 3.9404 | 3.233 | 0.7074 | 0.2901 | 36.38 | [0.2332, 1.1339] | 0.0 |
| v2 | lock-once (pessimistic) | camera_mono | in_corridor | 299 | 7.3622 | 6.0307 | 1.3315 | 0.0 | 32.78 | [0.0692, 2.7305] | 0.002102 |
| v2 | lock-once (pessimistic) | camera_mono | ttc_0_3s | 932 | 1.3607 | 1.1313 | 0.2294 | 0.0666 | 36.37 | [-0.1647, 0.5753] | 0.0 |
| v2 | lock-once (pessimistic) | camera_mono | ttc_3_5s | 972 | 2.6032 | 2.7288 | -0.1256 | 0.0 | 34.05 | [-1.0816, 0.6078] | 0.0 |
| v2 | lock-once (pessimistic) | camera_mono | vru | 1321 | 6.3963 | 5.6683 | 0.728 | 0.0723 | 24.98 | [0.1894, 1.2479] | 0.0 |
| v2 | lock-once (pessimistic) | camera_mono | vehicle | 2489 | 6.1352 | 5.8976 | 0.2376 | 0.0 | 38.69 | [-0.3315, 0.7994] | 1e-06 |
| v2 | lock-once (pessimistic) | camera_mono | day | 4045 | 6.1274 | 5.7786 | 0.3487 | 0.0 | 33.15 | [-0.0828, 0.7559] | 0.0 |
| v2 | lock-once (pessimistic) | camera_mono | night | 0 | None | None | None | None | None | [None, None] | None |
| v2 | lock-once (pessimistic) | camera_mono | in_corridor_ttc_0_5s | 117 | 3.0558 | 2.3606 | 0.6952 | 0.0 | 29.91 | [-0.8712, 2.137] | 0.069346 |
| v1 | per-frame Hungarian (optimistic) | lidar | overall | 5785 | 1.9039 | 2.3617 | -0.4578 | -0.0353 | 57.61 | [-0.5686, -0.3529] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | k2 | 2288 | 1.5287 | 2.1434 | -0.6146 | -0.0644 | 59.35 | [-0.7997, -0.4263] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | k3 | 1349 | 1.4264 | 2.1113 | -0.6849 | -0.0574 | 58.56 | [-0.9604, -0.4135] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | in_corridor | 310 | 1.3667 | 2.0307 | -0.664 | -0.059 | 61.61 | [-1.0353, -0.3092] | 9e-06 |
| v1 | per-frame Hungarian (optimistic) | lidar | ttc_0_3s | 2151 | 0.4605 | 0.8259 | -0.3654 | -0.0198 | 58.02 | [-0.5, -0.2446] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | ttc_3_5s | 1529 | 0.7926 | 1.5519 | -0.7594 | -0.0363 | 57.03 | [-0.975, -0.5706] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | vru | 1989 | 2.3048 | 2.7418 | -0.437 | -0.0193 | 54.9 | [-0.6259, -0.2296] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | vehicle | 2374 | 1.7785 | 2.1685 | -0.39 | -0.0391 | 57.37 | [-0.5712, -0.2265] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | day | 5785 | 1.9039 | 2.3617 | -0.4578 | -0.0353 | 57.61 | [-0.5686, -0.3529] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | lidar | night | 0 | None | None | None | None | None | [None, None] | None |
| v1 | per-frame Hungarian (optimistic) | lidar | in_corridor_ttc_0_5s | 197 | 0.4619 | 1.209 | -0.7471 | -0.0384 | 61.42 | [-1.5753, -0.1734] | 0.000699 |
| v1 | per-frame Hungarian (optimistic) | radar | overall | 12130 | 2.1661 | 2.136 | 0.0301 | -0.0114 | 54.78 | [-0.0422, 0.0976] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | k2 | 2411 | 1.6989 | 1.9407 | -0.2418 | -0.037 | 57.36 | [-0.38, -0.1022] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | k3 | 1327 | 1.2153 | 1.657 | -0.4417 | -0.0954 | 63.75 | [-0.6496, -0.2544] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | in_corridor | 1045 | 1.9548 | 1.9386 | 0.0163 | -0.0233 | 59.71 | [-0.1698, 0.2077] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | ttc_0_3s | 2386 | 0.3916 | 0.415 | -0.0234 | -0.0108 | 55.32 | [-0.0783, 0.0342] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | ttc_3_5s | 3138 | 0.6815 | 0.6954 | -0.0139 | -0.0104 | 54.97 | [-0.0966, 0.0692] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | vru | 3549 | 2.4728 | 2.5185 | -0.0457 | -0.0123 | 54.02 | [-0.1538, 0.0523] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | vehicle | 5742 | 2.3605 | 2.2655 | 0.095 | -0.0125 | 54.68 | [-0.0141, 0.2] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | day | 12130 | 2.1661 | 2.136 | 0.0301 | -0.0114 | 54.78 | [-0.0422, 0.0976] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | radar | night | 0 | None | None | None | None | None | [None, None] | None |
| v1 | per-frame Hungarian (optimistic) | radar | in_corridor_ttc_0_5s | 553 | 0.3918 | 0.3959 | -0.0041 | -0.0226 | 61.66 | [-0.1195, 0.1079] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | overall | 4315 | 2.5713 | 3.492 | -0.9207 | -0.1789 | 65.84 | [-1.1227, -0.7255] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | k2 | 1626 | 1.9192 | 3.1234 | -1.2042 | -0.3206 | 71.09 | [-1.5384, -0.8875] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | k3 | 1228 | 1.2559 | 2.5281 | -1.2722 | -0.2933 | 71.25 | [-1.6532, -0.9379] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | in_corridor | 250 | 2.8634 | 4.6995 | -1.8361 | -0.2966 | 69.6 | [-2.6834, -1.0704] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | ttc_0_3s | 1341 | 0.4413 | 0.7107 | -0.2694 | -0.0881 | 63.98 | [-0.3705, -0.1789] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | ttc_3_5s | 1082 | 0.8647 | 1.4569 | -0.5922 | -0.2129 | 69.32 | [-0.816, -0.3974] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | vru | 1808 | 2.9284 | 3.6913 | -0.763 | -0.1248 | 62.39 | [-1.1056, -0.4171] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | vehicle | 2402 | 2.3234 | 3.3622 | -1.0389 | -0.2263 | 68.82 | [-1.3468, -0.7591] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | day | 4315 | 2.5713 | 3.492 | -0.9207 | -0.1789 | 65.84 | [-1.1227, -0.7255] | 0.0 |
| v1 | per-frame Hungarian (optimistic) | camera_mono | night | 0 | None | None | None | None | None | [None, None] | None |
| v1 | per-frame Hungarian (optimistic) | camera_mono | in_corridor_ttc_0_5s | 121 | 0.7154 | 0.9542 | -0.2388 | -0.1303 | 64.46 | [-0.5043, 0.0265] | 0.000888 |
| v1 | lock-once (pessimistic) | lidar | overall | 5774 | 3.0138 | 2.9461 | 0.0677 | -0.0042 | 51.59 | [-0.1771, 0.3216] | 3.9e-05 |
| v1 | lock-once (pessimistic) | lidar | k2 | 1475 | 1.9113 | 2.577 | -0.6656 | -0.0722 | 58.1 | [-0.98, -0.3944] | 0.0 |
| v1 | lock-once (pessimistic) | lidar | k3 | 790 | 2.0616 | 2.6184 | -0.5568 | -0.0489 | 56.46 | [-0.9339, -0.1707] | 0.000321 |
| v1 | lock-once (pessimistic) | lidar | in_corridor | 332 | 3.1507 | 2.9828 | 0.168 | 0.0 | 48.49 | [-0.5813, 1.1077] | 0.782538 |
| v1 | lock-once (pessimistic) | lidar | ttc_0_3s | 2127 | 0.8819 | 1.0537 | -0.1718 | -0.0016 | 51.15 | [-0.4248, 0.0762] | 0.031503 |
| v1 | lock-once (pessimistic) | lidar | ttc_3_5s | 1532 | 1.7927 | 1.7479 | 0.0448 | 0.0 | 49.74 | [-0.3326, 0.46] | 0.587932 |
| v1 | lock-once (pessimistic) | lidar | vru | 2103 | 3.7411 | 3.8027 | -0.0616 | -0.0104 | 52.07 | [-0.4837, 0.4297] | 0.004798 |
| v1 | lock-once (pessimistic) | lidar | vehicle | 2383 | 2.6968 | 2.552 | 0.1448 | 0.0 | 49.6 | [-0.2322, 0.5206] | 0.885335 |
| v1 | lock-once (pessimistic) | lidar | day | 5774 | 3.0138 | 2.9461 | 0.0677 | -0.0042 | 51.59 | [-0.1771, 0.3216] | 3.9e-05 |
| v1 | lock-once (pessimistic) | lidar | night | 0 | None | None | None | None | None | [None, None] | None |
| v1 | lock-once (pessimistic) | lidar | in_corridor_ttc_0_5s | 209 | 1.3526 | 1.1776 | 0.175 | -0.0015 | 51.2 | [-0.782, 1.451] | 0.67676 |
| v1 | lock-once (pessimistic) | radar | overall | 16293 | 3.6057 | 3.2898 | 0.3159 | -0.0077 | 52.29 | [0.1438, 0.5173] | 0.0 |
| v1 | lock-once (pessimistic) | radar | k2 | 2113 | 2.0087 | 2.3495 | -0.3407 | -0.064 | 58.31 | [-0.5268, -0.1463] | 0.0 |
| v1 | lock-once (pessimistic) | radar | k3 | 808 | 1.7705 | 2.3579 | -0.5874 | -0.103 | 64.36 | [-1.071, -0.1425] | 0.0 |
| v1 | lock-once (pessimistic) | radar | in_corridor | 1188 | 3.1513 | 2.8104 | 0.3409 | -0.0173 | 54.71 | [-0.1553, 0.8869] | 0.000181 |
| v1 | lock-once (pessimistic) | radar | ttc_0_3s | 3347 | 0.8355 | 0.8715 | -0.036 | -0.0132 | 54.08 | [-0.1667, 0.0959] | 0.0 |
| v1 | lock-once (pessimistic) | radar | ttc_3_5s | 3924 | 1.2821 | 1.1729 | 0.1092 | -0.0076 | 52.65 | [-0.0539, 0.2963] | 6e-06 |
| v1 | lock-once (pessimistic) | radar | vru | 5093 | 4.5084 | 4.2346 | 0.2738 | -0.0063 | 51.36 | [-0.1735, 0.74] | 0.003182 |
| v1 | lock-once (pessimistic) | radar | vehicle | 7493 | 3.7364 | 3.3154 | 0.4211 | -0.0065 | 52.08 | [0.1263, 0.7137] | 2e-06 |
| v1 | lock-once (pessimistic) | radar | day | 16293 | 3.6057 | 3.2898 | 0.3159 | -0.0077 | 52.29 | [0.1438, 0.5173] | 0.0 |
| v1 | lock-once (pessimistic) | radar | night | 0 | None | None | None | None | None | [None, None] | None |
| v1 | lock-once (pessimistic) | radar | in_corridor_ttc_0_5s | 585 | 1.0926 | 1.0444 | 0.0481 | -0.0144 | 55.04 | [-0.54, 0.6026] | 0.007766 |
| v1 | lock-once (pessimistic) | camera_mono | overall | 5080 | 3.7108 | 4.3194 | -0.6086 | -0.0993 | 58.62 | [-0.9108, -0.2993] | 0.0 |
| v1 | lock-once (pessimistic) | camera_mono | k2 | 1196 | 2.0571 | 3.3549 | -1.2978 | -0.2696 | 66.22 | [-1.7985, -0.8352] | 0.0 |
| v1 | lock-once (pessimistic) | camera_mono | k3 | 701 | 1.6438 | 3.0194 | -1.3756 | -0.3012 | 71.33 | [-1.9201, -0.8949] | 0.0 |
| v1 | lock-once (pessimistic) | camera_mono | in_corridor | 328 | 4.4719 | 5.7424 | -1.2705 | -0.2904 | 63.41 | [-3.1011, 0.2007] | 0.0 |
| v1 | lock-once (pessimistic) | camera_mono | ttc_0_3s | 1468 | 1.0181 | 0.9006 | 0.1175 | -0.0227 | 54.63 | [-0.0617, 0.3331] | 1e-06 |
| v1 | lock-once (pessimistic) | camera_mono | ttc_3_5s | 1252 | 1.9047 | 1.8247 | 0.08 | -0.1244 | 59.74 | [-0.3544, 0.5431] | 0.0 |
| v1 | lock-once (pessimistic) | camera_mono | vru | 1994 | 4.2902 | 4.2042 | 0.086 | -0.0224 | 53.36 | [-0.4923, 0.6766] | 3e-05 |
| v1 | lock-once (pessimistic) | camera_mono | vehicle | 2933 | 3.3682 | 4.4125 | -1.0443 | -0.1731 | 62.46 | [-1.4135, -0.7021] | 0.0 |
| v1 | lock-once (pessimistic) | camera_mono | day | 5080 | 3.7108 | 4.3194 | -0.6086 | -0.0993 | 58.62 | [-0.9108, -0.2993] | 0.0 |
| v1 | lock-once (pessimistic) | camera_mono | night | 0 | None | None | None | None | None | [None, None] | None |
| v1 | lock-once (pessimistic) | camera_mono | in_corridor_ttc_0_5s | 151 | 2.4242 | 2.02 | 0.4042 | -0.0992 | 55.63 | [-1.38, 2.1503] | 0.140055 |

## Limitations

- v2's fused composition (n_sensors, for the k2/k3 slices) is independently derived from the three solo-sensor trackers' own candidates (see build_h6_dataset_v2.py / build_fused_dataset.py's docstring) -- v2's CentralTracker.Track carries no per-snapshot sensor field. v1's composition is read directly from its own output/step_4/fused_tracks_all.csv 'sensors' column -- v1's real, already-recorded composition, not re-derived. This is a genuine methodological difference between the two pipelines' k2/k3 slices, stated here rather than hidden by matching labels.
- v1's lock-once bound reuses v2.evaluate.match_track_to_ground_truth() unchanged, fed v1's own track-id ('fused_id' column) groupings -- v1 rows sit exactly on the real GT sample grid already, so no extrapolation is needed (unlike a genuinely async v2 track).
- "Best single sensor" (for the pass/fail determination) is judged on EACH sensor's own paired-with-fused population in the in-corridor+0-5s slice, not an unconditioned standalone accuracy number -- consistent with rule 2's same-GT-keys fairness requirement, but it does mean each sensor's own n can differ.
- The GT-side enrichment (range/corridor/class/motion/odd) is identical code shared by both pipelines (gt_context.py) -- a slice named the same way means the same thing in both pipelines' tables, even though the two tables are never directly compared.
- Scene-level bootstrap: 2000 resamples, seed 42, 95% CI, resampling the real scenes with replacement, per rule 4.
- All thresholds (config_l3.py) were fixed before any H6 result was computed and were not tuned afterward.