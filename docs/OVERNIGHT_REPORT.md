# Overnight run — full pipeline on fresh data (seed 2025)

Generated a brand-new synthetic dataset and re-ran **every engine at its best-learned
configuration**, graded against the same fresh sealed truth. Also wired in two new things first:
Meta Robyn's **experiment calibration** (`calibration_input` anchored to the spend-ladder readout)
and a one-command **orchestrator** (`scripts/run_all_engines.sh`). All engines succeeded; the
frequentist NLS diverged (flagged below). Live: the [engine leaderboard](engines/index.html) and
the [run tracker](runs/index.html) (this is run `seed2025`).

Fresh dataset: national media 960 conv/wk · spend↔season confound 0.60 · geo confound 0.78 ·
geo-world media 800.

## 1. Fresh-data leaderboard (seed 2025)

| # | engine | world | MAE/ch ↓ | media bias | CIs |
|--:|--------|------|--------:|-----------:|:---:|
| 1 | Meridian (national, Fourier) | nat'l | **24** | −12% | 6/6 |
| 2 | Spend ladder (curve fit) | nat'l | **25** | −14% | 6/6 |
| 3 | DraftZone PyMC (obs) | nat'l | 44 | **−2%** | 6/6 |
| 4 | Meridian (national, AKS) | nat'l | 47 | −26% | 6/6 |
| 5 | Naive OLS | nat'l | 59 | −23% | — |
| 6 | **Meta Robyn (experiment-calibrated)** | nat'l | 77 | −48% | — |
| 7 | Meridian (geo + proxy 0.98) | geo | 91 | +68% | 1/6 |
| 8 | Robyn-style (Python reimpl.) | nat'l | 91 | −57% | — |
| 9 | DraftZone PyMC (anchored) | nat'l | 101 | −48% | 5/6 |
| 10 | Meta Robyn (real, plain) | nat'l | 102 | −64% | — |
| 11 | Meridian (naive lift→prior) | nat'l | 150 | +19% | 3/6 |
| 12 | Meridian (geo + proxy 0.78) | geo | 153 | +114% | 0/6 |
| 13 | Meridian (geo panel, no control) | geo | 164 | +123% | 0/6 |
| — | Frequentist NLS | nat'l | ⚠ diverged | — | — |

## 2. The headline: rankings are NOT stable across datasets

This is the single most important result, and exactly why the project runs multiple seeds.

| engine | seed 77 MAE | seed 2025 MAE | verdict |
|--------|-----------:|-------------:|---------|
| Spend ladder | 39 | **25** | robust — top-2 both times |
| PyMC (obs) | 50 | **44** | robust — best-calibrated both (bias −2%/+20%) |
| Meridian Fourier | 36 | **24** | strong both, **best on 2025** |
| Meridian AKS | **35** (best-cal) | 47 | **advantage did NOT hold** — over-corrected to −26% |
| Robyn-style | **32** (best) | 91 | **collapsed** — dataset-dependent |
| Meta Robyn (real) | 53 | 102 | much worse on 2025 |
| PyMC (anchored) | 79 | 101 | anchor hurts on both |

On seed 77 the "winner" was Robyn-style (32) and the best-calibrated was Meridian AKS. On
seed 2025 the winner is Meridian Fourier (24), AKS over-corrected, and Robyn-style fell to the
middle. **"Which MMM is best" is a property of the dataset, not the method.** The only engines
that were *consistently* good are the **spend ladder** (the experiment that measures the curve) and
**PyMC obs** (well-calibrated, bias near zero both times). The honest takeaway of the whole project,
reproduced: no single observational fit is trustworthy on its own; you triangulate, and you check
robustness across realisations.

## 3. Did experiment calibration fix Robyn? Partly — and it's capped.

Robyn under-credits the media *level* because Prophet's flexible trend absorbs the
seasonal-confounded variance. Feeding it the spend-ladder lifts via `calibration_input` was the
direct "apply the experiment-anchoring lesson" move:

| channel | truth | Robyn plain | Robyn + calibration | ladder (the target) |
|---------|------:|-----------:|-------------------:|--------------------:|
| paid_social | 286 | 117 | 177 | 258 |
| paid_search | 194 | 64 | 93 | 171 |
| programmatic | 129 | 36 | 64 | 123 |
| influencer | 51 | 33 | 41 | 36 |
| dooh | 61 | 20 | 24 | 69 |
| tv_ctv | 240 | 77 | 102 | 170 |
| **total** | **960** | **346 (−64%)** | **501 (−48%)** | **827 (−14%)** |

Calibration **lifted every channel** toward the experiment and improved Robyn from MAE 102 → 77,
bias −64% → −48%. But two honest limits: (a) it only got **halfway** to the ladder targets —
Prophet's trend keeps dragging even against the calibration objective; and (b) it is **capped by
the experiment readout's own accuracy** — the ladder itself runs −14%, so calibrating to it can't
beat that. Calibration helps the level honestly, but it is not magic, and a better experiment (or a
less greedy baseline) would be needed to close the rest.

## 4. The geo confounder reproduces cleanly

The hardened geo panel (latent demand confounder, corr 0.78) again **destroys** geo Meridian, and
the control-quality spectrum holds monotonically — graded against the geo world's own answer key:

| geo config | MAE | bias |
|-----------|----:|-----:|
| no control | 164 | +123% |
| + imperfect proxy (0.78) | 153 | +114% |
| + near-perfect proxy (0.98) | 91 | +68% |

Same story as seed 77: geo data buys cross-sectional power, **not immunity from an unobserved
confounder**, and even a near-perfect proxy only claws back part of the bias. The cure remains the
randomised experiment, not more observational geo data.

## 5. Frequentist NLS diverged (and that's a finding)

Unregularised non-linear least squares has no prior to keep it in bounds; on this data realisation
it ran to a degenerate fit (paid_social β exploded to ~6,600, bias +3,378%). It is kept in the
leaderboard table flagged `⚠ diverged` but excluded from the bar chart so it doesn't crush the
scale. This is a clean illustration that the Bayesian/regularised engines' priors are not only about
uncertainty — they are what keep a thin-data, many-parameter MMM numerically sane.

## 6. What was built / changed this run

- **Robyn experiment calibration** (`fit_meta_robyn.R --calibrate`): real `calibration_input`
  anchored to the spend-ladder readout; new engine `meta_robyn_calibrated`.
- **Orchestrator** (`scripts/run_all_engines.sh`): one command runs data → every engine at its best
  config → leaderboard → reports, with per-engine failure isolation, timeouts, and checkpoint
  commits so an unattended run survives a container reclaim.
- **Leaderboard divergence guard**: flag + exclude engines whose fit blew up.
- All committed and pushed across the run; 25 tests pass; no-truth-leak guard green.

## Suggested next steps

- **Run a 3rd and 4th seed** to turn the robustness table into a distribution (the multi-run index
  is built for exactly this) — the real deliverable is "engine A beats B in k of N datasets," not a
  single leaderboard.
- **Average-rank leaderboard** across seeds, so the headline is stability, not one lucky fit.
- **Better experiment for Robyn calibration**: a less greedy baseline or a higher-power ladder so
  the calibration ceiling rises above −14%.

---

# Robustness across 6 seeds (follow-up)

Ran the fast national engines (best config) across **6 independent datasets** (seeds 2025, 101, 202,
303, 404, 505) and aggregated by **average rank, win-rate, and MAE stability** — the only fair way to
compare methods. Page: [robustness across seeds](robustness/index.html). This reframes the single-seed
story substantially.

## National engines, ranked by average rank (6 seeds)

| engine | avg rank ↓ | wins | median MAE | spread (±std) | mean bias | ESS / R̂ |
|--------|-----------:|-----:|-----------:|--------------:|----------:|:-------:|
| **Meridian (AKS)** | **2.0** | 3 | 40 | **±9** | −14% | — |
| Meridian (Fourier) | 2.5 | 3 | 49 | ±15 | +10% | — |
| Robyn-style | 3.7 | 0 | 46 | ±19 | −28% | — |
| Spend ladder (after fix) | 4.3 | 0 | 74 | ±25 | +25% | — |
| PyMC (obs) | 4.5 | 0 | 51 | ±23 | +11% | 592 / 1.001 |
| PyMC (anchored) | 5.3 | 0 | 77 | ±14 | −37% | **230 / 1.013** |
| Naive OLS | 6.0 | 0 | 78 | ±10 | −19% | — |
| Frequentist NLS | 8.0 | 0 | 193 | ±52 (diverged 2/6) | +42% | — |

## What the sweep changed vs. the single-seed view

1. **Meridian (AKS) is the robust champion** — best average rank, 3 wins, tightest spread (±9).
   Meridian (AKS *or* Fourier) won all 6 seeds. On seed-2025 alone AKS looked mediocre; across seeds
   it edges Fourier. **The single-seed "winner" is noise; AKS is the stable choice.**
2. **The spend ladder was *not* robust — and we fixed it.** Its seed-2025 MAE-25 was luck: on ~half
   the seeds the unregularised Hill curve-fit blew up (max MAE 360). We added a ridge on the ceiling
   (`BETA_REG`) + a tighter half-sat bound; the blow-ups vanished (max 360→98, no regressions), and
   the ladder settled into stable mid-pack (rank 4.3, spread ±25 vs ±116). A real optimisation, found
   only because we looked across seeds.
3. **The experiment anchor reliably hurts** — worse rank than obs (5.3 vs 4.5) *and* worse sampling
   (ESS 230 vs 592, R̂ 1.013 vs 1.001). The ESS diagnostic shows it's a worse *model* (more posterior
   tension), not bad luck. The curve-aware anchor on replica geos needs rethinking.
4. **Bias direction is realization-dependent** (e.g. ladder +25% mean but ranges negative-to-positive
   across seeds); only MAE *magnitude* is stable. No engine has a dependable bias sign.
5. **Frequentist NLS is unstable** (diverged 2/6, median MAE 193) — same disease as the un-fixed
   ladder (unregularised least squares), same cure available.

## Note on reuse (answering the question that prompted this)

Spend-ladder results are **recomputed every run**, never reused across seeds — each seed is a new
synthetic world with its own randomized experiments. *Within* a run they are computed once and reused
(e.g. by Robyn's calibration). The slow deep-dives (real Meta Robyn 2000×5, the geo control spectrum)
remain single-seed documented results; Robyn's *method* is represented in the sweep by the Robyn-style
reimplementation.
