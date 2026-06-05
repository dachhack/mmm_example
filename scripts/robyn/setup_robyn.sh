#!/usr/bin/env bash
# scripts/robyn/setup_robyn.sh — install the REAL Meta Robyn (R package) in this environment.
#
# Why this is non-trivial here: no R is preinstalled and CRAN is network-blocked. But the Ubuntu
# archive IS reachable, and it ships almost all of Robyn's dependencies as apt `r-cran-*` packages.
# The only gaps are (1) `lares` (Robyn's author's plotting/utility package — heavy tree, used only
# for plots/formatting/clustering, not the model fit) and (2) a couple of version/numpy frictions.
# We solve them surgically so the MODELING path is the genuine Robyn algorithm:
#   - install R + all available deps from apt,
#   - install a minimal `lares` SHIM (scripts/robyn/lares_shim) providing exactly the symbols Robyn
#     imports, with real implementations on the fit path and stubs for plots/clustering,
#   - fetch Robyn from GitHub and apply two one-line patches: relax patchwork's version pin (plots
#     only) and convert nevergrad's ask() value via $tolist() (reticulate 1.35 doesn't auto-convert
#     numpy 2.x arrays). Neither touches Robyn's model.
#
# nevergrad (Robyn's optimiser, called via reticulate) must be in the project venv: pip install nevergrad.
# Run:  bash scripts/robyn/setup_robyn.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBYN_REF="${ROBYN_REF:-main}"
TMP="$(mktemp -d)"

echo "== 1/4 apt: R + Robyn dependencies (from the Ubuntu archive, not CRAN) =="
apt-get install -y --no-install-recommends \
  r-base-core \
  r-cran-data.table r-cran-glmnet r-cran-foreach r-cran-doparallel r-cran-dorng \
  r-cran-future r-cran-dplyr r-cran-ggplot2 r-cran-jsonlite r-cran-lubridate \
  r-cran-nloptr r-cran-reticulate r-cran-stringr r-cran-tidyr r-cran-patchwork \
  r-cran-prophet r-cran-rlang r-cran-tibble r-cran-purrr r-cran-ggridges \
  r-cran-glue r-cran-magrittr

echo "== 2/4 install the minimal lares shim =="
R CMD INSTALL "$HERE/lares_shim"

echo "== 3/4 fetch + patch + install Robyn ($ROBYN_REF) =="
curl -sSL -o "$TMP/robyn.tar.gz" "https://github.com/facebookexperimental/Robyn/archive/refs/heads/${ROBYN_REF}.tar.gz"
tar xzf "$TMP/robyn.tar.gz" -C "$TMP"
PKG="$(echo "$TMP"/Robyn-*/R)"
# (a) relax patchwork version pin (used only for combining plots)
sed -i 's/patchwork (>= [0-9.]*)/patchwork/' "$PKG/DESCRIPTION"
# (b) numpy2 + reticulate 1.35: convert nevergrad ask() value via tolist()
sed -i 's/nevergrad_hp_val\[\[co\]\] <- nevergrad_hp\[\[co\]\]\$value$/nevergrad_hp_val[[co]] <- nevergrad_hp[[co]]$value$tolist()/' "$PKG/R/model.R"
R CMD INSTALL --no-docs --no-byte-compile "$PKG"

echo "== 4/4 verify =="
Rscript -e 'suppressMessages(library(Robyn)); cat("Robyn", as.character(packageVersion("Robyn")), "OK\n")'
rm -rf "$TMP"
echo "Done. Run: RETICULATE_PYTHON=\$(which python) Rscript scripts/fit_meta_robyn.R --iterations 1000 --trials 3"
