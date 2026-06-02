# notebooks/ — Re-runnable record of the full path

Build these as Jupyter notebooks (`.ipynb`). Each imports from `draftzone_mmm`, runs a stage,
and writes figures to `docs/assets/` and any data contracts to `docs/data/`. They double as the
narrative source and (via nbconvert) as HTML pages on the site.

- **01_data.ipynb** — generate/load data; visualize national series, spend, impressions; show
  the realized confound; describe the DGP at a high level (no truth leak — use config.json).
- **02_naive.ipynb** — EDA correlations; fit naive raw-spend OLS; contrast with the evaluation
  scorecard later. Message: good fit ≠ good attribution.
- **03_transforms.ipynb** — interactive-style adstock & Hill demos; show transformed media
  tracking contribution shape.
- **04_fit.ipynb** — frequentist (fast) + load the VM-produced Bayesian InferenceData; posterior
  summaries. (Do NOT sample heavy chains inside the notebook on CI; load cached artifacts.)
- **05_evaluate.ipynb** — render the recovery scorecard + calibration + coverage from
  scorecard.json. This is where ground truth appears, clearly labeled as the sealed key.
- **06_experiments.ipynb** — the rotating geo calendar; per-channel DiD; the repair plots
  (before/after anchor) for every channel.
- **07_revenue.ipynb** — LTV, revenue contribution, average vs marginal ROI with intervals.
- **08_optimize.ipynb** — robust optimization; confident vs test-first verdicts; the final
  recommendation narrative.

Keep heavy computation out of notebook execution on CI: notebooks should *load* artifacts
produced by `scripts/run_fits.sh`, not re-sample. A `PAPERMILL`/parameters cell can point at
`artifacts/`.
