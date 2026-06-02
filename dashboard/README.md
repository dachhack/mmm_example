# Dashboard (React + Vite)
Interactive GitHub Pages site. Reads precomputed `docs/data/*` (produced by scripts/run_fits.sh
on the VM) — NO live PyMC. Build with `npm ci && npm run build` (outputs to ../docs for Pages).
Implement sections per ../docs/DASHBOARD_SPEC.md. The adstock/Hill widgets are pure-JS and
interactive; everything else renders precomputed JSON with credible intervals.
