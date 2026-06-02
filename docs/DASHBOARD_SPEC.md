# DASHBOARD_SPEC.md — Interactive GitHub Pages Dashboard

A single-page interactive dashboard that tells the story AND lets the reader poke at it.
Framework: React + Vite (static build to `docs/` or a `gh-pages` artifact). Charts: Recharts
or similar. All data is **precomputed** (the heavy Bayesian fits run on the VM; the dashboard
loads their JSON/CSV outputs — no live PyMC in the browser).

## Sections (top to bottom)

1. **Hero / premise** — one paragraph: what MMM is, the trust problem, the triangulation answer.

2. **The data** — interactive time-series of conversions with a toggle to overlay the hidden
   baseline+season (revealed from the *evaluation* output, clearly labeled "answer key").
   Slider/legend to show/hide channels' spend & impressions.

3. **Why naive fails** — the naive-vs-truth attribution bars; a short callout that good R²
   hides bad attribution.

4. **The transforms** — interactive adstock (θ slider) and Hill (half_sat, slope sliders)
   widgets so the reader *feels* carryover and diminishing returns. Pure JS, no backend.

5. **Model recovery scorecard** — table + calibration plot from `evaluate.py`: per-channel
   recovered contribution & θ with credible intervals vs. truth, HIT/MISS flags, and the
   **interval coverage** metric (the overconfidence check).

6. **Experiment repair (the centerpiece)** — for EACH channel, a before/after-anchor
   distribution showing the estimate sliding toward truth once its geo-experiment is fed in.
   A control to "turn on" experiments one at a time (or all) and watch the whole decomposition
   tighten. This is the payoff of the rotating-experiment dataset.

7. **ROI** — average vs marginal ROI bars with intervals; the "average is a trap" message.

8. **Interactive budget optimizer** — sliders for total budget and per-channel bounds; reads
   precomputed posterior-draw allocations to render the **robust recommendation** (median Δ +
   interval) with **confident vs test-first** color coding. Let the reader change the
   confidence threshold and see verdicts update.

9. **Honest limitations** — explicit list (see PROTOTYPE_FINDINGS.md). Non-negotiable section.

## Data contract (the dashboard reads these, produced by the pipeline)
- `docs/data/decomposition.json` — per-week, per-channel posterior-mean contribution + bands.
- `docs/data/scorecard.json` — recovery table + coverage metric.
- `docs/data/repair.json` — per-channel before/after-anchor contribution distributions.
- `docs/data/roi.json` — avg & marginal ROI with intervals.
- `docs/data/optim_draws.json` — per-draw optimal allocations (for the interactive optimizer).
- `docs/data/timeseries.csv` — the public national dataset for the charts.

Keep the contract stable; the notebooks/pipeline write these, the dashboard only reads them.
