# DraftZone MMM

An end-to-end, **honest** Media Mix Modeling project on a synthetic DFS / fantasy-sports
subscription app. It demonstrates not just how to *build* an MMM, but how to know **when to
trust it** — by grading every estimate against a sealed ground truth and using randomized
geo-experiments to produce **confident, uncertainty-aware budget recommendations**.

> **The thesis:** an observational MMM is biased by the correlation between ad spend and
> seasonal demand. Bayesian priors bound the damage but don't cure it. Randomized
> geo-experiments give confound-immune causal anchors that, fed back into the model, repair the
> decomposition. Decisions are made on **marginal ROI under uncertainty** — acting only on
> **robust** moves and routing the rest to experiments.

## What makes this different
- **Sealed answer key.** Every true parameter lives in `data_sealed/ground_truth.json`, which
  the modeling pipeline is forbidden to read (enforced by `tests/test_no_truth_leak.py`). Only
  the evaluation step may open it — so recovery results are earned, not peeked.
- **Full experimentation path.** The dataset ships a *rotating geo-experiment calendar* — one
  randomized test per channel — so every channel can be causally anchored, turning "cut search,
  test everything else" into confident moves across the board.
- **Uncertainty everywhere.** No bare point estimates. ROI and recovered parameters carry
  credible intervals; the optimizer separates confident vs test-first recommendations.
- **Interactive dashboard** on GitHub Pages illustrates the process and findings.

## Repository map
```
CLAUDE.md                 orientation for Claude Code (read first)
docs/
  DATA_SPEC.md            new dataset design (incl. rotating experiments)
  DASHBOARD_SPEC.md       interactive site spec + data contract
  INFRA.md                VM-fits / Actions-site split, Makefile targets
  PROTOTYPE_FINDINGS.md   distilled results & lessons from the prototype
  prototype_src/          reference scripts from the prototype (refactor, don't copy)
src/draftzone_mmm/        the package (transforms.py ported; rest are stubs to implement)
notebooks/                01–08 re-runnable narrative (see notebooks/README.md)
dashboard/                React/Vite interactive site (reads docs/data/*)
data/ , data_sealed/      public data ; SEALED truth
tests/                    incl. the no-truth-leak guard
scripts/run_fits.sh       heavy VM pipeline -> produces docs/data/*
.github/workflows/        ci.yml (tests) , pages.yml (build & deploy)
```

## Quick start (for Claude Code)
1. Read `CLAUDE.md`, then the four `docs/*.md` specs.
2. Implement `src/draftzone_mmm/*` (start with `datagen.py` per `DATA_SPEC.md`; `transforms.py`
   is already ported and tested).
3. `make data && make fit && make experiments && make anchored && make evaluate && make figures`
   on a VM (see `docs/INFRA.md`).
4. Build the dashboard + notebooks; let Actions deploy Pages.
5. Confirm the dashboard shows the recovery scorecard, the per-channel experiment repair, and
   the interactive optimizer with confident/test-first verdicts.

## Status
Prototype complete (see `docs/PROTOTYPE_FINDINGS.md`). This repo is the productionization:
new dataset + reproducible pipeline + published interactive findings.
