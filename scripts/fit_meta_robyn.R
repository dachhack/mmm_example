#!/usr/bin/env Rscript
# scripts/fit_meta_robyn.R — fit the REAL Meta Robyn (R package) and write the engine contract.
#
# This runs the actual facebookexperimental/Robyn package (not a reimplementation): Prophet
# trend/season decomposition + geometric adstock + Hill saturation + ridge (glmnet) with
# Nevergrad multi-objective hyperparameter search over NRMSE + DECOMP.RSSD. It writes
# artifacts/meta_robyn_results.json in the same engine contract the harness grades, so Meta Robyn
# sits on the leaderboard next to Meridian, our PyMC, and the Robyn-style Python reimplementation.
#
# Environment notes (this repo's setup): no R was preinstalled and CRAN is blocked, so R + all deps
# come from the Ubuntu archive (apt r-cran-*), a minimal `lares` shim stands in for Robyn's plotting
# utility, and Robyn's nevergrad value-extraction has a one-line numpy2/reticulate-1.35 fix. See
# docs/INFRA.md. Run with the project venv active so reticulate finds nevergrad:
#   RETICULATE_PYTHON=$(which python) Rscript scripts/fit_meta_robyn.R --iterations 800 --trials 2
#
# CONTRACT: a modeling engine — it reads only public data/, never data_sealed/.

suppressMessages({library(Robyn); library(data.table); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) { i <- which(args == flag); if (length(i)) args[i + 1] else default }
iterations <- as.integer(getarg("--iterations", "800"))
trials     <- as.integer(getarg("--trials", "2"))
cores      <- as.integer(getarg("--cores", "3"))
calibrate  <- "--calibrate" %in% args   # anchor Robyn to the spend-ladder experiment readout
out_path   <- getarg("--out", "")
repo       <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
if (length(repo) == 0 || is.na(repo)) repo <- normalizePath(".")

reticulate::use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
set.seed(1)

chans <- c("paid_social","paid_search","programmatic_display","influencer","dooh","tv_ctv")
spend_vars <- paste0(chans, "_spend")
expo_vars  <- paste0(chans, "_impressions")

dt <- fread(file.path(repo, "data", "national_weekly.csv"))
dt[, week := as.Date(week)]
data("dt_prophet_holidays")

# Prophet trend+season is Robyn's confound control (its analogue of the Fourier basis our other
# engines use); context vars are the observable controls. paid_media_vars = exposure (impressions).
InputCollect <- robyn_inputs(
  dt_input = dt, date_var = "week", dep_var = "conversions", dep_var_type = "conversion",
  prophet_vars = c("trend", "season"), prophet_country = "US", dt_holidays = dt_prophet_holidays,
  context_vars = c("promo_flag", "price_index", "competitor_pressure", "holiday_flag"),
  paid_media_spends = spend_vars, paid_media_vars = expo_vars,
  adstock = "geometric", window_start = min(dt$week), window_end = max(dt$week)
)
hyps <- list()
for (v in expo_vars) {
  hyps[[paste0(v, "_thetas")]] <- c(0, 0.8)
  hyps[[paste0(v, "_alphas")]] <- c(0.5, 3)
  hyps[[paste0(v, "_gammas")]] <- c(0.3, 1)
}

# EXPERIMENT CALIBRATION (Robyn's calibration_input). Applying the project's lesson that the
# confound-immune signal comes from experiments, we anchor each channel's TOTAL effect to the
# spend-ladder readout (artifacts/ladder_results.json) — our most accurate experiment estimate,
# which cracks the saturated channels. Robyn then adds a third objective (MAPE.LIFT) pulling the
# fitted contribution toward the experiment, which should lift the media LEVEL that Prophet's trend
# otherwise under-credits. This is the real-world workflow: feed your lift tests to your MMM.
cal_in <- NULL
engine_suffix <- ""
if (calibrate) {
  lad <- jsonlite::fromJSON(file.path(repo, "artifacts", "ladder_results.json"))$channels
  n_weeks <- nrow(dt)
  cal_in <- do.call(rbind, lapply(seq_along(chans), function(i) {
    est <- lad[[chans[i]]]$est_contrib
    data.frame(channel = spend_vars[i], liftStartDate = min(dt$week), liftEndDate = max(dt$week),
               liftAbs = est * n_weeks, spend = sum(dt[[spend_vars[i]]]),
               confidence = 0.90, metric = "conversions", calibration_scope = "total",
               stringsAsFactors = FALSE)
  }))
  engine_suffix <- "_calibrated"
  cat("Calibrating Robyn to spend-ladder lifts (conv/wk):",
      paste(sprintf("%s=%.0f", chans, vapply(chans, function(c) lad[[c]]$est_contrib, numeric(1))),
            collapse = " "), "\n")
}
InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyps,
                             calibration_input = cal_in)

cat(sprintf("Running Meta Robyn: %d iterations x %d trials on %d cores...\n", iterations, trials, cores))
OutputModels <- robyn_run(InputCollect = InputCollect, iterations = iterations, trials = trials,
                          ts_validation = FALSE, add_penalty_factor = FALSE, cores = cores,
                          outputs = FALSE)

# Gather every candidate model across trials, pick the balanced Pareto knee on (NRMSE, DECOMP.RSSD).
hp  <- rbindlist(lapply(OutputModels[grepl("^trial[0-9]+$", names(OutputModels))],
                        function(t) t$resultCollect$resultHypParam), fill = TRUE)
agg <- rbindlist(lapply(OutputModels[grepl("^trial[0-9]+$", names(OutputModels))],
                        function(t) t$resultCollect$xDecompAgg), fill = TRUE)
saveRDS(OutputModels, file.path(repo, "artifacts", paste0("meta_robyn", engine_suffix, "_models.rds")))
nz <- function(x) { r <- max(x) - min(x); if (r > 0) (x - min(x)) / r else rep(0, length(x)) }
hp[, score := nz(nrmse) + nz(decomp.rssd)]
best <- hp[which.min(score)]
best_sol <- best$solID
n_weeks <- InputCollect$rollingWindowLength
if (is.null(n_weeks) || !is.numeric(n_weeks) || length(n_weeks) != 1) n_weeks <- nrow(dt)

# media rows are named by the modelled variable (here the exposure/impression var); map each row
# back to its channel by prefix so we are robust to spend- vs exposure-naming.
chan_of <- function(x) { hit <- chans[vapply(chans, function(c) startsWith(x, c), logical(1))]; if (length(hit)) hit[1] else NA }
media <- agg[solID == best_sol & (rn %in% expo_vars | rn %in% spend_vars)]
media[, channel := vapply(rn, chan_of, character(1))]
est <- setNames(media$xDecompAgg / n_weeks, media$channel)

channels <- list()
for (i in seq_along(chans)) channels[[chans[i]]] <- list(est_contrib = unname(est[chans[i]]), ci = NULL)

results <- list(
  engine = paste0("meta_robyn", engine_suffix),
  label = if (calibrate) "Meta Robyn (experiment-calibrated)" else "Meta Robyn (R 3.12.1)",
  bayesian = FALSE,
  fit = list(r2 = best$rsq_train, nrmse = best$nrmse, decomp_rssd = best$decomp.rssd),
  selected_solID = best_sol, iterations = iterations, trials = trials,
  channels = channels,
  note = paste("REAL Meta Robyn (facebookexperimental/Robyn 3.12.1): prophet trend/season +",
               "geometric adstock + Hill + ridge, Nevergrad multi-objective (NRMSE + DECOMP.RSSD).",
               "Pareto-knee model. Point estimate (no credible interval).")
)
if (out_path == "") out_path <- file.path("artifacts", paste0("meta_robyn", engine_suffix, "_results.json"))
out_abs <- if (substr(out_path, 1, 1) == "/") out_path else file.path(repo, out_path)
dir.create(dirname(out_abs), showWarnings = FALSE, recursive = TRUE)
write(toJSON(results, auto_unbox = TRUE, pretty = TRUE, null = "null"), out_abs)

cat(sprintf("Selected model %s: NRMSE=%.4f  DECOMP.RSSD=%.3f  R2=%.3f\n",
            best_sol, best$nrmse, best$decomp.rssd, best$rsq_train))
for (i in seq_along(chans)) cat(sprintf("  %-22s est=%6.1f\n", chans[i], est[chans[i]]))
cat(sprintf("Wrote %s\n", out_abs))
