# CLAUDE.md — Orientation for Claude Code

You are picking up a **Media Mix Modeling (MMM)** project mid-stream. A working prototype
was built in a chat session (see `docs/PROTOTYPE_FINDINGS.md`). Your job is to turn it into
a **reproducible, cloud-runnable repository** that publishes an **interactive GitHub Pages
dashboard** illustrating the process and findings, built on a **new synthetic dataset** that
supports the *full experimentation path* so we can produce **confident, robust budget
recommendations across all channels** — not just one.

## The one-paragraph premise
MMM estimates how much each marketing channel drives conversions, from aggregate weekly data.
It is plagued by (a) weak identifiability (many nonlinear params, thin data), and (b)
confounding — spend correlates with seasonal demand, so models over/under-credit channels.
The honest resolution is **triangulation**: a Bayesian MMM for the broad picture + **randomized
geo-experiments** that give confound-immune causal anchors, fed back as priors. Decisions are
made on **marginal ROI under uncertainty**, acting only on **robust** moves and routing the
rest to experiments.

## Non-negotiable principles (do not violate)
1. **The pipeline must NEVER read `data_sealed/ground_truth.json`.** Only `src/.../evaluate.py`
   may open it, and only to score recovery. This keeps the project honest. Enforce with a test
   (`tests/test_no_truth_leak.py`) that greps the pipeline for forbidden imports/paths.
2. **Report uncertainty everywhere.** No bare point estimates for ROI or recovered params.
   Every channel number carries a credible interval.
3. **Separate confident moves from test-first moves** in any recommendation output.
4. **Average ROI ≠ marginal ROI.** Decisions use marginal ROI.

## Build order (suggested)
1. Package the prototype code under `src/draftzone_mmm/` (transforms, datagen, fit_freq,
   fit_bayes, evaluate, experiment, revenue, optimize). Prototype scripts are in
   `docs/prototype_src/` for reference — refactor, don't just copy.
2. Implement the **new dataset generator** per `docs/DATA_SPEC.md` (adds a rotating
   per-channel experiment calendar). Write truth to `data_sealed/`, public data to `data/`.
3. Build the **notebooks** in `notebooks/` (01–08) that re-run the full path and emit figures
   to `docs/assets/`. These are the re-runnable record.
4. Build the **interactive dashboard** in `dashboard/` (see `docs/DASHBOARD_SPEC.md`).
5. Wire **CI/CD** per `docs/INFRA.md`: a VM/long-job path for the Bayesian fits (artifacts
   cached), and a GitHub Actions workflow that builds the dashboard + notebook HTML to Pages.
6. Run `evaluate.py` to produce the scorecard; surface it on the dashboard.

## Environment
- Python ≥3.11. Core deps: numpy, pandas, scipy, statsmodels, pymc>=5, arviz, matplotlib.
- PyMC sampling is the expensive step. **Run multi-core locally/VM, NOT in Actions** (Actions
  runners segfault on multi-core compiled pytensor graphs in our experience — sample with
  `cores=1` if you must run there, or better, cache the InferenceData artifact).
- Save InferenceData as NetCDF if `h5netcdf` is available, else pickle (prototype used pickle).

## Definition of done
- `make all` (or documented equivalent) regenerates data → fits → evaluates → figures.
- GitHub Pages shows: the narrative, the recovery scorecard, before/after-experiment repair,
  and an interactive budget optimizer with confident-vs-test-first verdicts.
- `tests/` pass, including the no-truth-leak guard.
- A reader with no context understands *what MMM can and cannot be trusted to do.*
