# Why did "Fusion" lose to Radar alone?

*A plain-language explanation, backed by real numbers, with the exact file each number came from.*

---

## The short version (read this first)

Imagine three friends standing by the road, each trying to spot cars and guess how fast they're coming:

- **Radar** — throws out radio waves and listens for the echo. Not super precise, but works in rain/fog, and is good at telling how fast something moves.
- **Lidar** — sweeps a laser beam around. Very precise about *where* something is.
- **Camera** — just looks with its eyes. Sees far away and sees detail, but can't tell distance well on its own.

You'd expect that if all three friends **work together as a team** ("fusion") and combine their guesses, the team should beat any one friend working alone. **That is not what happened.** In this project, Radar working *alone* was more accurate than the whole team combined.

This document explains **why**, using only things we actually measured — not guesses. Every number below says exactly which file it came from, so you (or anyone else) can check it.

**The one-sentence answer**: the "team" mostly isn't really a team. Most of the time, only *one* friend ever sees a given car — so "the team's answer" is usually really just one friend's answer, relabeled. And the cars that only one friend can see tend to be the hardest ones (far away, or only visible to one kind of sensor) — so the team's average score gets dragged down by hard cases that a single sensor's own "report card" never had to include.

---

## The headline numbers

On the full test (3,376 real driving moments, called "trainval"), here's how each one did — lower number = better (it's the average mistake, in seconds, on "when will this object reach me?"):

| Who | Average mistake (MAE) |
|---|---|
| **Radar (best alone)** | **2.39** |
| Camera (with laser help) | 2.87 |
| **Fusion (the whole team)** | **2.88** |
| Lidar | 3.08 |
| Camera (alone, no help) | 3.63 |

**Source**: `output/step_6/evaluation_metrics.csv` (v1 notebook pipeline, trainval dataset).

Fusion is basically *tied* with "Camera + laser help," but **Radar alone beats both.**

---

## Evidence 1 — Most of "the team's answer" is really just one friend talking

**Simple explanation**: Imagine the "team" reports 100 car-sightings. You'd hope most of those are cases where 2 or 3 friends agreed. Instead, only about **1 in 4** are a real team effort. The other **3 in 4** are just one friend's guess, stamped with the team's name.

| What happened | How often | How accurate |
|---|---|---|
| **Real teamwork** (2+ friends agreed) | 24.8% of the time | **2.06** (best score of anyone!) |
| Just Radar alone (passed through as "team") | 55.5% of the time | 2.75 |
| Just Lidar alone (passed through) | 13.3% of the time | 3.98 |
| Just Camera alone (passed through) | 6.4% of the time | 4.90 |

**The important part**: when the friends *actually* work together, they do the best job of anyone — 2.06 beats even Radar's own 2.39. **Teamwork itself is not the problem.** The problem is teamwork almost never happens (only 24.8% of the time).

**Source**: this exact breakdown was computed and double-checked against the official numbers before trusting it (the math `0.248×2.06 + 0.752×3.15 = 2.88` exactly reproduces the real team score). Investigation script: `composition_investigation.py` (kept in scratchpad, not part of the permanent project); verified against `output/step_6/evaluation_metrics.csv` and `output/step_4/fused_tracks_all.csv` (v1 pipeline, trainval dataset).

---

## Evidence 2 — The friends usually can't even see the same car at the same time

**Simple explanation**: For teamwork to happen, at least 2 friends need to be looking at the *same car* at the *same moment*. How often does that actually happen in real life?

| How many friends see the same real car | How often |
|---|---|
| **Zero** friends see it at all | ~61-63% |
| **Exactly one** friend sees it | ~21-23% |
| **Two or more** friends see it (teamwork is even possible) | ~14-25%† |

†Depends on a fix described in Evidence 5 below — before the fix it was ~14%, after it was ~25%.

**This is a hard ceiling.** Even if the "team" were perfect at combining opinions, it can only ever do real teamwork on 14-25% of real cars — the rest are automatically "pass the one friend's answer through," which brings us back to Evidence 1.

**Source**: measured directly against real ground-truth car positions. Documented in `CLAUDE.md`, "second audit round" and "Radar declutter filter relaxed" sections (v1 pipeline, **mini dataset** — this was measured before the project moved to the bigger trainval dataset).

---

## Evidence 3 — Far-away cars can usually *only* be seen by Camera, making teamwork impossible there

This is something we checked specifically because it matches a real, sensible worry: *"what if a car 150 meters away is only visible to the camera, and that unfairly drags down the team's score?"*

**Simple explanation**: Radar and Lidar have a "comfortable reach" of roughly 60-80 meters in this dataset. Camera can "see" further. So for far-away cars, it's not that the team disagrees — it's that **only one friend is even in the room.**

| How far away the car is | % of cars NO friend can see | % of cars where ONLY ONE friend can see it | Of those "only one" cases, how often is it the Camera |
|---|---|---|---|
| 0-20m (close) | 9.7% | 25.5% | 77.8% |
| 40-60m | 30.9% | 46.7% | 86.4% |
| 100-150m | 21.0% | 73.0% | 96.9% |
| **150m or further** | 0.0% | **100.0%** | **100.0%** |

**At 150 meters and beyond, every single car is a "camera-only" case. Teamwork is not rare there — it is impossible, every time.** This confirms the worry directly, with real data, not a guess.

**Source**: `v2_miss_funnel.csv` (107,206 real car-sightings checked), from the v2 (newer, rebuilt) pipeline, trainval dataset. Full writeup: `docs/hypotheses-2026-10-10.md`, "ROI/range follow-up (2026-10-19)" section.

---

## Evidence 4 — Do those "Radar-can't-see-it" cars actually make the team's score worse?

Evidence 3 shows far cars are camera-only. This check asks the next, sharper question: **does that actually hurt the team's final score, or not?**

**Simple explanation**: take every car the "team" reported on. Split them into two piles — (A) cars Radar could *also* see, and (B) cars Radar *never* saw at all. Compare the team's accuracy on each pile.

**Using the older (v1) pipeline — this pile-split confirms the worry:**

| | Team's score on "Radar also saw it" pile | Team's score on "Radar never saw it" pile |
|---|---|---|
| Quick check | 2.36 | **3.46** (much worse) |
| Careful check | 4.12 | **5.61** (much worse) |

Cars Radar can't see really are the hard ones — they drag the team average down. And even on the *fair* pile (cars Radar *also* saw), the team still doesn't beat Radar's own score (2.30 vs 2.36) — so teamwork isn't adding value even where it's possible, in this pipeline.

**Using the newer (v2) pipeline — the SAME check does NOT show this pattern:**

| | Team's score on "Radar also saw it" pile | Team's score on "Radar never saw it" pile |
|---|---|---|
| Quick check | 5.50 | 4.48 (actually **better**, not worse) |
| Careful check | 6.83 | 6.44 (slightly better) |

**We are reporting this honestly even though it doesn't match the first pipeline's story** — in the newer pipeline, the "Radar can't see it" cars are *not* the harder ones. This tells us the mechanism in Evidence 3/4 is real, but it isn't the *whole* explanation for every version of the pipeline — something else going on in the newer pipeline cancels it out (see Evidence 7 below).

**Source**: `output/eval_l3/h6_v1_keys.csv` (older/v1 pipeline) and `output/eval_l3/h6_v2_keys.csv` (newer/v2 pipeline), both trainval dataset. Full numbers: `docs/hypotheses-2026-10-10.md`, same "ROI/range follow-up" section.

---

## Evidence 5 — We found and fixed two real "teamwork" bugs — they helped, but not enough to win

**Simple explanation**: While investigating, two actual mistakes were found in how the "team's" answers were being combined — not in how each friend saw things individually, but in the step where their answers get merged.

1. **Bug A**: the team-combining code kept "forgetting who's who" every single moment, instead of remembering "oh, this is the same car I was tracking a second ago." This caused one real car to get counted as up to **10 different cars** in the team's records.
2. **Bug B**: when two friends' opinions drifted apart for just one instant, the system sometimes created a brand-new, fake "extra car" out of thin air.

Both were fixed. Here's what happened to the team's score:

| Stage | Team's average mistake | How many "duplicate car" mix-ups |
|---|---|---|
| Before either fix | 4.08 | 64.8% of cars were being double/triple-counted |
| After Bug A fixed | 3.90 | 14.3% still duplicated |
| After Bug B fixed too | 3.91 | **0.0%** — fully fixed |

**The duplicate-counting problem was completely solved — but the team's score barely moved (4.08 → 3.91, about 4% better).** This told us that duplicate-counting, while a real bug worth fixing, was never the main reason the team was losing.

**Source**: `CLAUDE.md`, "Fix: FusedTracker identity inheritance" and "Further tuning and analysis" sections (v1 pipeline, **mini dataset**).

---

## Evidence 6 — Even with no bugs left, "lone car" sightings are just naturally harder

**Simple explanation**: once the duplicate-counting bugs were fully fixed (Evidence 5), we checked again: are the "only one friend saw it" cars still worse than they should be? Yes — but now for a *real*, not-a-bug reason.

| | How long the car stays "on the books" (average) |
|---|---|
| Cars that get seen by 2+ friends at some point | 9.38 "moments" |
| Cars that are ONLY ever seen by 1 friend, their whole life | 3.68 "moments" — **less than half as long** |

Cars that nobody else ever corroborates tend to be the fleeting, hard-to-pin-down ones to begin with — a parked car in a weird spot, a pedestrian half-hidden behind a pole, a car at the very edge of what a sensor can see. It's not that the team mis-handles them; it's that **they were always going to be harder**, and the team's overall score has to include them while a single sensor's own report card is allowed to just be about whatever it manages to see.

**Source**: `CLAUDE.md`, "Further tuning and analysis" section (v1 pipeline, mini dataset).

---

## Evidence 7 — The newer pipeline (v2) has its own, separate problem: a "jittery" Radar

This is specific to the brand-new rebuild of the system (not the original one), and it's a different kind of problem from everything above.

**Simple explanation**: imagine asking Radar every single second "is the car dangerously close RIGHT NOW?" If Radar's attention keeps jumping between different nearby cars (instead of steadily watching ONE car), its answer can flicker — "danger! ...no wait, fine... no wait, danger!" — even when nothing is actually changing.

We checked this directly: we picked one real car and looked at Radar's own raw predictions over time:

| Moment | Radar's prediction | What was actually happening |
|---|---|---|
| 1 | 1.72 seconds (danger!) | car is NOT actually closing in |
| 2 | — (no danger) | " |
| 3 | 3.25 seconds (danger!) | " |
| 4 | 1.98 seconds (danger!) | " |
| 5 | — (no danger) | " |

The real car was never actually approaching — but Radar's prediction kept flickering into "danger" territory anyway. We then tested this across **every** radar car-sighting, not just this one example:

| | How often a "false alarm" moment happens |
|---|---|
| When Radar is tracking the SAME identified car the whole time | 3.2% of moments |
| When Radar's "who am I even looking at" switches car-to-car | **30.5%** of moments — **9.4 times more often** |

This is a strong, clear, statistically confirmed pattern (not a one-off coincidence — tested over 41,186 real moments, and the chance of this being a fluke is essentially zero).

**Source**: `docs/hypotheses-2026-10-10.md`, "H9 findings" and the "Update (2026-10-19)" confirmation inside it. Script: `verify_q1_track_churn_vs_ttc.py` (scratchpad, not part of the permanent project). v2 pipeline, trainval dataset.

---

## Evidence 8 — With real, big data (not a tiny test), the answer is confirmed: fusion still loses where it matters most

Early in this project, this whole investigation was run on a *small* practice dataset (only 404 car-moments, called "mini"). The small dataset meant some of the most important checks — specifically, "does the team beat the best single friend in a true emergency (car about to hit you in under 5 seconds)?" — had too few real examples to trust (as few as 12-45 cases).

So the entire dataset was switched to a much bigger one (3,376 car-moments, 8x more) and every check was re-run.

| Question | Old (small) dataset | New (big) dataset |
|---|---|---|
| Does the team beat a single friend in general? | Too little data to be sure | **Yes** — confirmed, with 2,000+ real examples |
| Does the team beat the BEST friend specifically in a true emergency (car about to hit you)? | Too little data to be sure (12-45 examples) | **No** — confirmed REFUTED, with 67-579 real examples, in every version checked (both pipelines, both scoring methods) |

**This is the final, most trustworthy answer in this whole document**: with enough real data to be confident, teamwork does NOT beat the single best sensor specifically in the moment that matters most — right before a collision.

**Source**: `output/eval_l3/h1_refined_verdict.md` and `output/eval_l3/h6_verdict.md` (both v1 and v2 pipelines, trainval dataset).

---

## What did NOT turn out to be the reason (we checked, and ruled these out)

It's just as important to say what we looked at and found **not guilty** — these are not guesses, they were tested directly:

| Suspected reason | What we found |
|---|---|
| The math formula for "how long until collision" is wrong | **Ruled out.** Ran the exact same formula two different ways on the same data — got byte-for-byte identical answers. |
| The friends report slightly different spots for the same car | **Ruled out.** When we know for certain it's the same real car, the friends' reported positions are usually within about 1 meter of each other — not enough to explain the gap. |
| The team is too strict about what counts as "the same car" (needs a looser rulebook) | **Tested, made things WORSE**, not better. Reverted. |
| (v2 only) Radar-invisible cars are the hard ones dragging the score down | **Does NOT hold for the newer pipeline** (it held for the older one — see Evidence 4). |

---

## What we do NOT have solid proof for yet (you asked us to be honest about this)

- **Evidence 7's exact cause.** We showed a very strong *statistical link* (9.4x) between "Radar's attention switching cars" and "false danger alarms" — but we have not yet done the stronger test of actually fixing that specific switching behavior and re-measuring to *prove* it's the full cause (versus, say, fast/close cars just naturally being both harder to track *and* noisier for some unrelated reason).
- **"Phantom" false alarms.** All our false-alarm checks (Evidence 7, and the earlier false-alarm-rate work) only count a false alarm when it's tied to a *real* car that just wasn't actually dangerous. We have not yet measured the case where the system raises an alarm with **no real car anywhere nearby at all** — that would need different data we haven't built yet.
- **Evidence 1's "passing through" explanation, double-checked only for one friend.** We proved (with real data, tracing specific examples) that a related earlier worry — "maybe the system just *thinks* it recovered a missed car when really it grabbed a different nearby car by mistake" — was true 92.5% of the time **for Lidar specifically**. We have not yet re-run that same careful check for Radar, Camera, or the full team's combined output — so treat that specific number as Lidar-only for now, not proven for everyone.
- **Why the newer pipeline (v2) behaves oppositely to the older one in Evidence 4.** We reported this honestly as a real difference, but we don't yet know exactly *why* it flips.

---

## The honest final answer

Fusion loses to Radar alone mainly because:

1. **Real teamwork almost never happens** — only ~1 in 4 "team answers" are genuinely 2+ sensors agreeing; the rest is one sensor's own guess wearing a team jersey (Evidence 1).
2. **Most real cars can only ever be seen by one sensor to begin with** — this is a hard physical/practical limit, not something teamwork can fix (Evidence 2, 3).
3. **The lone-sensor cars tend to be the harder cases** (far away, fleeting, edge-of-range) — so the team's average has to include hard cases that each sensor's own personal "report card" never had to (Evidence 4, 6).
4. **When it's checked on enough real data to be sure, the team specifically loses in the moment that matters most** — right before a possible collision (Evidence 8).

When teamwork *does* happen, it's genuinely the best performer of all (Evidence 1's 2.06 beats everyone). **The problem was never that combining sensors is a bad idea — it's that in this system, real combining is rare, and everything else gets dragged down by the hard, lone-sensor cases riding along with it.**
