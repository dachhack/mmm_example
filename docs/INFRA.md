# INFRA.md — Cloud Compute & CI/CD

Two-track design: **heavy fits on a VM** (cached as artifacts), **site build in GitHub Actions**.

## Why split
PyMC sampling is slow and multi-core compiled pytensor graphs are known to segfault on
hosted Actions runners. So we do NOT sample in Actions. We sample on a VM (or any long-running
job), commit/upload the resulting InferenceData + derived JSON as artifacts, and Actions only
transforms those into the published site.

## Track A — VM (the fits)
- `scripts/run_fits.sh`: provision deps (`pip install -e .[fit]`), then:
  1. `python -m draftzone_mmm.datagen` (writes data/ + data_sealed/)
  2. `python -m draftzone_mmm.fit_bayes --out artifacts/idata.nc` (multi-core OK here)
  3. `python -m draftzone_mmm.experiment --all-channels --out artifacts/anchors.json`
  4. `python -m draftzone_mmm.fit_bayes --anchors artifacts/anchors.json --out artifacts/idata_anchored.nc`
  5. `python -m draftzone_mmm.evaluate --out docs/data/scorecard.json` (only step that reads data_sealed/)
  6. `python -m draftzone_mmm.export_dashboard_data --out docs/data/` (decomposition, repair, roi, optim_draws)
- Recommend a small GPU-less VM with ≥8GB RAM. Cache `~/.cache/pytensor`.
- Upload `docs/data/*` and `artifacts/*` (e.g. to the repo via a bot commit, or as a workflow
  artifact the Pages job downloads).

## Track B — GitHub Actions (the site)
- `.github/workflows/pages.yml`:
  - Trigger: push to main, or manual.
  - Steps: checkout → setup-node → `cd dashboard && npm ci && npm run build` (Vite builds to
    `../docs/` or `dist/`) → optionally render notebooks to HTML (jupyter nbconvert) into
    `docs/notebooks/` → upload-pages-artifact → deploy-pages.
  - It consumes the precomputed `docs/data/*` produced by Track A. If absent, fail with a clear
    message ("run scripts/run_fits.sh on the VM first").
- `.github/workflows/ci.yml`: lint + `pytest` (fast tests only; mock/skip sampling). MUST run
  `tests/test_no_truth_leak.py`.

## Makefile targets (document these)
- `make data` · `make fit` · `make experiments` · `make evaluate` · `make figures`
- `make dashboard` (npm build) · `make notebooks` (nbconvert) · `make all`

## Secrets / config
- No secrets needed for synthetic data. If a bot commit is used to push `docs/data/`, store a
  PAT in repo secrets and document it.

## Alternative engines (leaderboard)
The engine leaderboard (`scripts/engine_leaderboard.py` → `docs/engines/`) grades several MMM
engines against the same sealed truth. Two are external:

- **Google Meridian** (`scripts/fit_meridian.py`, `make meridian`): pip-installable
  (`pip install -e ".[meridian]"`). Heavy TensorFlow; run on the VM. National (Fourier/AKS) and
  geo-panel modes.
- **Meta Robyn** (`scripts/fit_meta_robyn.R`, `make robyn-real`): the REAL R package. No R is
  preinstalled here and CRAN is network-blocked, so `scripts/robyn/setup_robyn.sh` installs R and
  ~all deps from the Ubuntu archive (`apt r-cran-*`), plus a minimal **`lares` shim**
  (`scripts/robyn/lares_shim`) standing in for Robyn's heavy plotting/utility dependency (used only
  off the model-fit path), and applies two one-line Robyn patches (relax `patchwork` version pin;
  convert nevergrad's `ask()` value via `$tolist()` because reticulate 1.35 doesn't auto-convert
  numpy-2 arrays). The modeling path is unmodified genuine Robyn. Robyn's optimiser is Nevergrad,
  called via reticulate — keep `nevergrad` in the project venv. Run:
  `bash scripts/robyn/setup_robyn.sh && RETICULATE_PYTHON=$(which python) Rscript scripts/fit_meta_robyn.R`.
- **Robyn-style** (`scripts/fit_robyn_style.py`, `make robyn`): a pure-Python reimplementation of
  Robyn's method (ridge + Nevergrad + DECOMP.RSSD) for environments without R.

## Compute paths — running the heavy sweeps OFF the web sandbox
The web container is ephemeral (reclaimed on inactivity) and runs PyMC at `cores=1` with no BLAS, so
multi-hour sweeps are slow and fragile there. Two better targets:

### CI (GitHub Actions) — parallel, self-contained
`.github/workflows/sweep.yml` (manual: Actions ▸ "MMM sweep (parallel compute)" ▸ Run workflow).
Each (saturation × confound × seed) **cell is independent**, so the workflow fans them out as a
matrix — an N-cell sweep finishes in ~one cell's wall-clock instead of N× sequential. Flow:
- `setup` builds the matrix from the dispatch inputs (`scripts/ci_matrix.py`).
- `cell` (matrix, ≤20 parallel) installs `.[fit,meridian]` and runs `scripts/run_one_cell.sh`
  (datagen → PyMC obs+anchored → spend ladder → Robyn-style → Meridian Fourier+AKS → snapshot),
  uploading the graded snapshot as an artifact. PyMC samples `cores=1` (Actions segfaults on
  multi-core compiled pytensor — see CLAUDE.md), but each cell is its own runner so wall-clock is fine.
- `aggregate` downloads all snapshots, rebuilds `docs/conditional/` + `docs/robustness/`, and commits
  once (artifacts → single commit, so no parallel-git races).

### VM — multi-core, no time limit, and the home of the real R Robyn
`scripts/run_remote.sh` syncs the repo to a host you control, runs a command in a venv (PyMC can use
all cores there), and pulls `docs/` back:
```
MMM_VM_HOST=ubuntu@HOST bash scripts/run_remote.sh 'make all'
MMM_VM_HOST=ubuntu@HOST bash scripts/run_remote.sh \
  'bash scripts/run_conditional_sweep.sh "0.5 1.0 2.0" "11 22 33 44 55" 0.6'
```
The single-cell unit (`scripts/run_one_cell.sh`) is shared by the local sweep, the VM, and CI, so a
cell is defined in exactly one place. Design-for-interruption still applies: every sweep is
checkpoint-committed and skip-existing/resumable.
