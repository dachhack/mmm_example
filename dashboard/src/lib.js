// Pure-JS MMM transforms (for the interactive widgets) + small data helpers.

export function geometricAdstock(x, theta, normalize = true, L = 12) {
  const w = [];
  for (let l = 0; l < L; l++) w.push(Math.pow(theta, l));
  const s = w.reduce((a, b) => a + b, 0);
  const wn = normalize ? w.map((v) => v / s) : w;
  const out = new Array(x.length).fill(0);
  for (let t = 0; t < x.length; t++) {
    let acc = 0;
    for (let l = 0; l < L; l++) {
      if (t - l >= 0) acc += x[t - l] * wn[l];
    }
    out[t] = acc;
  }
  return out;
}

export function hill(x, halfSat, slope) {
  return x.map((v) => {
    const a = Math.max(v, 0);
    return Math.pow(a, slope) / (Math.pow(a, slope) + Math.pow(halfSat, slope) + 1e-12);
  });
}

// Minimal CSV parser (numeric where possible). Assumes a header row, comma-separated.
export function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const head = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    head.forEach((h, i) => {
      const v = cells[i];
      const num = Number(v);
      row[h] = v !== "" && !Number.isNaN(num) ? num : v;
    });
    return row;
  });
}

export async function loadAll() {
  const base = "data/";
  const jget = (f) => fetch(base + f).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const cget = (f) => fetch(base + f).then((r) => (r.ok ? r.text() : null)).catch(() => null);
  const [scorecard, decomposition, repair, roi, optim, tsText] = await Promise.all([
    jget("scorecard.json"),
    jget("decomposition.json"),
    jget("repair.json"),
    jget("roi.json"),
    jget("optim_draws.json"),
    cget("timeseries.csv"),
  ]);
  return {
    scorecard,
    decomposition,
    repair,
    roi,
    optim,
    timeseries: tsText ? parseCSV(tsText) : null,
  };
}

export const CHANNEL_COLORS = {
  tv: "#d62728",
  search: "#1f77b4",
  social: "#2ca02c",
  affiliate: "#ff7f0e",
  brand: "#9467bd",
};

export const fmt = (v, d = 0) =>
  v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
