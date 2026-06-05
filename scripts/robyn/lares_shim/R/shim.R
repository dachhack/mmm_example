#' @importFrom magrittr %>%
#' @export
`%>%` <- magrittr::`%>%`

#' @export
check_opts <- function(inputs, opts, input_name = "value", stop = TRUE, ...) {
  bad <- setdiff(inputs, opts)
  if (length(bad) > 0 && stop) base::stop(sprintf("Invalid %s: %s. Allowed: %s",
      input_name, paste(bad, collapse = ", "), paste(opts, collapse = ", ")))
  invisible(length(bad) == 0)
}

#' @export
num_abbr <- function(x, ...) {
  vapply(x, function(v) {
    if (is.na(v)) return(NA_character_)
    a <- abs(v)
    if (a >= 1e9) paste0(round(v / 1e9, 1), "B")
    else if (a >= 1e6) paste0(round(v / 1e6, 1), "M")
    else if (a >= 1e3) paste0(round(v / 1e3, 1), "K")
    else as.character(round(v, 1))
  }, character(1))
}

#' @export
formatNum <- function(x, decimals = 2, abbr = FALSE, signif = NULL, ...) {
  if (isTRUE(abbr)) return(num_abbr(x))
  if (!is.null(signif)) x <- signif(x, signif)
  formatC(x, format = "f", big.mark = ",", digits = decimals)
}

#' @export
formatColoured <- function(x, ...) x

#' @export
glued <- function(..., .envir = parent.frame()) as.character(glue::glue(..., .envir = .envir))

#' @export
v2t <- function(vector, quotes = TRUE, sep = ", ", and = "") {
  if (quotes) vector <- paste0("'", vector, "'")
  paste(vector, collapse = sep)
}

#' @export
removenacols <- function(df, all = TRUE) df[, colSums(!is.na(df)) > 0, drop = FALSE]

#' @export
missingness <- function(df, summary = TRUE) {
  miss <- vapply(df, function(c) sum(is.na(c)), integer(1))
  miss <- miss[miss > 0]
  if (length(miss) == 0) return(NULL)
  data.frame(variable = names(miss), missing = as.integer(miss),
             missingness = round(100 * miss / nrow(df), 1), row.names = NULL)
}

#' @export
right <- function(string, n = 1) substr(string, nchar(string) - n + 1, nchar(string))

#' @export
winsorize <- function(x, thresh = c(0.01, 0.99), ...) {
  q <- stats::quantile(x, thresh, na.rm = TRUE)
  x[x < q[1]] <- q[1]; x[x > q[2]] <- q[2]; x
}

#' @export
normalize <- function(x, ...) {
  r <- range(x, na.rm = TRUE)
  if (diff(r) == 0) return(rep(0, length(x)))
  (x - r[1]) / diff(r)
}

#' @export
zerovar <- function(df) names(df)[vapply(df, function(c) length(unique(c[!is.na(c)])) <= 1, logical(1))]

#' @export
try_require <- function(package, stopper = TRUE, ...) {
  ok <- requireNamespace(package, quietly = TRUE)
  if (!ok && stopper) base::stop("Package '", package, "' required but not installed")
  invisible(ok)
}

#' @export
statusbar <- function(run = 1, max.run = 1, ...) invisible(NULL)

#' @export
freqs <- function(df, ..., result = TRUE) {
  df %>% dplyr::count(..., sort = TRUE)
}

#' @export
glued <- glued

#' @export
clusterKmeans <- function(df, k = NULL, ...) base::stop("lares shim: clusterKmeans unsupported; run Robyn with clusters = FALSE")

#' @export
ohse <- function(df, ...) df

# ---- plotting stubs (only invoked on the output/plot path, not the fit) ----
#' @export
theme_lares <- function(...) ggplot2::theme_minimal()
#' @export
noPlot <- function(message = "", ...) ggplot2::ggplot() + ggplot2::theme_void()
#' @export
plot_palette <- function(...) NULL
#' @export
lares_pal <- function(...) list(labels = NULL, palette = NULL, colors = NULL)
#' @export
scale_x_abbr <- function(...) ggplot2::scale_x_continuous(labels = num_abbr)
#' @export
scale_y_abbr <- function(...) ggplot2::scale_y_continuous(labels = num_abbr)
#' @export
scale_x_percent <- function(...) ggplot2::scale_x_continuous(labels = function(x) paste0(round(100 * x), "%"))
#' @export
scale_y_percent <- function(...) ggplot2::scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"))
