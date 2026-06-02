import React, { useEffect, useMemo, useState } from "react";
import {
  Area, ComposedChart, CartesianGrid, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { CHANNEL_COLORS, fmt, loadAll } from "./lib.js";
import TransformsWidget from "./TransformsWidget.jsx";
import OptimizerWidget from "./OptimizerWidget.jsx";

const CHANNELS = ["tv", "search", "social", "affiliate", "brand"];

export default function App() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    loadAll().then((d) => {
      if (!d.decomposition || !d.scorecard) setErr(true);
      setData(d);
    });
  }, []);

  if (err) {
    return (
      <div className="wrap loading">
        <h2>Dashboard data not found.</h2>
        <p className="small">Run <code>scripts/run_fits.sh</code> (or <code>make all</code>) to
          produce <code>docs/data/*</code> first — see <code>docs/INFRA.md</code>.</p>
      </div>
    );
  }
  if (!data) return <div className="wrap loading">Loading the model…</div>;

  return (
    <>
      <Hero />
      <main className="wrap">
        <DataSection data={data} />
        <NaiveSection sc={data.scorecard} />
        <TransformsSection />
        <ScorecardSection sc={data.scorecard} />
        <RepairSection repair={data.repair} sc={data.scorecard} />
        <RoiSection roi={data.roi} />
        <OptimizerSection optim={data.optim} />
        <LimitationsSection />
      </main>
      <footer className="wrap">
        DraftZone MMM — a synthetic, end-to-end Media Mix Model graded against a sealed answer key.
        All figures are precomputed by the pipeline; the dashboard only reads them. Built with
        React + Vite + Recharts.
      </footer>
    </>
  );
}

function Hero() {
  return (
    <header className="hero">
      <div className="wrap">
        <div className="kicker">A Skeptic's Guide to Marketing Mix Modeling</div>
        <h1>DraftZone MMM</h1>
        <p className="sub">How much can you actually trust a media mix model — and when?</p>
        <p>
          An MMM estimates how much each marketing channel drives conversions from aggregate weekly
          data. It is plagued by <b>confounding</b>: ad spend rises with seasonal demand, so models
          over- or under-credit channels. Bayesian priors bound the damage but don't cure it. The
          honest resolution is <b>triangulation</b> — a Bayesian MMM for the broad picture plus
          <b> randomized geo-experiments</b> that give confound-immune causal anchors. Decisions are
          made on <b>marginal ROI under uncertainty</b>: act only on <b>robust</b> moves, and route
          everything else to a test. Every number below was graded against a sealed ground truth.
        </p>
      </div>
    </header>
  );
}

function DataSection({ data }) {
  const { timeseries, decomposition } = data;
  const [showKey, setShowKey] = useState(false);
  const [shown, setShown] = useState(() => new Set(CHANNELS));
  const rows = useMemo(() => {
    if (!timeseries) return [];
    return timeseries.map((r, i) => {
      const o = { week: r.week, conversions: r.conversions };
      o.nonmedia = decomposition.nonmedia_mean ? decomposition.nonmedia_mean[i] : null;
      CHANNELS.forEach((c) => (o[`${c}_spend`] = r[`${c}_spend`]));
      return o;
    });
  }, [timeseries, decomposition]);

  return (
    <section>
      <h2>The data</h2>
      <p className="lead">
        140+ weeks of a fictional DFS subscription app: weekly conversions driven by five channels
        plus baseline, NFL-season seasonality, promos, price and competitor pressure. About
        <b> 43%</b> of conversions are organic baseline, and total spend correlates with season at
        <b> ρ≈0.6</b> — the confound that makes attribution hard.
      </p>
      <div className="card">
        <div className="controls">
          <button className={"toggle-btn" + (showKey ? " active" : "")}
            onClick={() => setShowKey((v) => !v)}>
            {showKey ? "Hide" : "Reveal"} the hidden baseline+season (answer key)
          </button>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#2a313c" vertical={false} />
            <XAxis dataKey="week" stroke="#9aa7b4" fontSize={10} minTickGap={48} />
            <YAxis stroke="#9aa7b4" fontSize={11} />
            <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }} />
            <Line dataKey="conversions" stroke="#e6edf3" dot={false} strokeWidth={1.4}
              name="conversions" />
            {showKey && (
              <Area dataKey="nonmedia" stroke="#d9a200" fill="rgba(217,162,0,0.18)"
                name="baseline+controls (answer key)" />
            )}
          </ComposedChart>
        </ResponsiveContainer>
        <p className="small">
          {showKey
            ? "Gold band = the true non-media floor (baseline + trend + season + controls). Everything above it is what the five channels must explain — and split among themselves."
            : "The observed conversions. Reveal the answer key to see how much is organic before a single ad runs."}
        </p>
      </div>
    </section>
  );
}

function NaiveSection({ sc }) {
  const naive = sc.naive;
  if (!naive) return null;
  const max = Math.max(...naive.channels.flatMap((c) => [Math.abs(c.naive_contrib), c.true_contrib]));
  const bar = (v) => `${(Math.abs(v) / max) * 100}%`;
  return (
    <section>
      <h2>Why the naive model fails</h2>
      <p className="lead">
        A raw-spend regression (no adstock, no saturation, no season) gets a respectable
        R²={naive.r2}, then attributes wildly — carryover-heavy channels get blinded and their
        effect is dumped into the intercept. <b>Good fit ≠ good attribution.</b>
      </p>
      <div className="card">
        <table>
          <thead><tr><th>channel</th><th>true contribution</th><th>naive implied</th><th>　</th></tr></thead>
          <tbody>
            {naive.channels.map((c) => (
              <tr key={c.channel}>
                <td style={{ color: CHANNEL_COLORS[c.channel] }}>{c.channel}</td>
                <td>{fmt(c.true_contrib, 0)}</td>
                <td className={Math.abs(c.naive_contrib - c.true_contrib) > 0.5 * c.true_contrib ? "miss" : ""}>
                  {fmt(c.naive_contrib, 0)}
                </td>
                <td style={{ minWidth: 220 }}>
                  <svg width="220" height="16">
                    <rect x="0" y="2" width={bar(c.true_contrib)} height="5" fill="#2ca02c" opacity="0.8" />
                    <rect x="0" y="9" width={bar(c.naive_contrib)} height="5"
                      fill={c.naive_contrib < 0 ? "#d62728" : "#888"} />
                  </svg>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small">green = true · grey/red = naive implied. Naive intercept ≈ {fmt(naive.intercept, 0)} absorbs the carryover it can't model.</p>
      </div>
    </section>
  );
}

function TransformsSection() {
  return (
    <section>
      <h2>The transforms that matter</h2>
      <p className="lead">
        Functional form is the lever. Two transforms turn raw media into the shape advertising
        actually acts in: <b>adstock</b> (carryover) and <b>Hill saturation</b> (diminishing
        returns). Order matters: accumulate exposure first, then model the response. Play with them.
      </p>
      <TransformsWidget />
    </section>
  );
}

function ScorecardSection({ sc }) {
  return (
    <section>
      <h2>Model recovery scorecard</h2>
      <p className="lead">
        The anchored Bayesian model, graded against the sealed truth. Aggregate fit is strong, θ
        recovers well, and every channel's 89% interval contains the truth — but the predictive
        intervals are <b>overconfident</b>.
      </p>
      <div className="card">
        <div className="metric">
          <div className="m"><b>{sc.fit.r2.toFixed(2)}</b><span>R² (posterior mean)</span></div>
          <div className="m"><b>{sc.fit.mape.toFixed(1)}%</b><span>MAPE</span></div>
          <div className="m"><b>{sc.fit.pp_interval_coverage}%</b>
            <span>of weeks in the 89% interval (want 89%)</span></div>
          <div className="m"><b>{sc.summary.contrib_ci_hits}/{sc.summary.n_channels}</b>
            <span>channel CIs contain truth</span></div>
        </div>
        {sc.summary.overconfident && (
          <div className="callout warn">
            <b>Overconfidence check:</b> the 89% predictive interval covers only
            {" "}{sc.fit.pp_interval_coverage}% of weeks. The model is more certain than it should be —
            a calibration failure you'd never see from R² alone.
          </div>
        )}
        <table>
          <thead>
            <tr><th>channel</th><th>true</th><th>estimate</th><th>89% CI</th><th>θ true→est</th><th></th></tr>
          </thead>
          <tbody>
            {sc.channels.map((c) => (
              <tr key={c.channel}>
                <td style={{ color: CHANNEL_COLORS[c.channel] }}>{c.channel}</td>
                <td>{fmt(c.true_contrib, 0)}</td>
                <td>{fmt(c.est_contrib, 0)}</td>
                <td>[{fmt(c.ci[0], 0)}, {fmt(c.ci[1], 0)}]</td>
                <td>{c.true_theta.toFixed(2)} → {c.est_theta.toFixed(2)}</td>
                <td className={c.hit ? "hit" : "miss"}>{c.hit ? "HIT" : "MISS"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small">
          Media total: true {fmt(sc.summary.media_total_true, 0)} vs est
          {" "}{fmt(sc.summary.media_total_est, 0)} ({sc.summary.media_under_credit_pct}% under-credit).
          θ recovered for {sc.summary.theta_hits}/{sc.summary.n_channels} channels.
        </p>
      </div>
    </section>
  );
}

function RepairSection({ repair, sc }) {
  const [on, setOn] = useState(() => new Set(CHANNELS));
  const truth = Object.fromEntries(sc.channels.map((c) => [c.channel, c.true_contrib]));

  const toggle = (c) =>
    setOn((prev) => {
      const n = new Set(prev);
      n.has(c) ? n.delete(c) : n.add(c);
      return n;
    });

  // MAE before vs after (using current toggle to choose which channels are anchored)
  const mae = useMemo(() => {
    let before = 0, after = 0;
    repair.channels.forEach((ch) => {
      const t = truth[ch.channel];
      before += Math.abs(ch.before.mean - t);
      after += Math.abs((on.has(ch.channel) ? ch.after.mean : ch.before.mean) - t);
    });
    return { before: before / repair.channels.length, after: after / repair.channels.length };
  }, [repair, on, truth]);

  const allMax = Math.max(...repair.channels.flatMap((c) => [c.before.hi, c.after.hi, truth[c.channel]]));
  const scale = (v) => `${(v / allMax) * 100}%`;

  return (
    <section>
      <h2>Experiment repair <span className="small">— the centerpiece</span></h2>
      <p className="lead">
        Each channel ships a randomized geo-experiment (a rotating, always-on testing calendar).
        Fed back as a confound-immune anchor on the channel ceiling, it slides the estimate toward
        truth and tightens the whole decomposition. Toggle experiments on and off:
      </p>
      <div className="card">
        <div className="legend">
          {repair.channels.map((c) => (
            <span key={c.channel} className={on.has(c.channel) ? "" : "off"}
              onClick={() => toggle(c.channel)}>
              <i style={{ background: CHANNEL_COLORS[c.channel] }} />
              {c.channel} {on.has(c.channel) ? "✓ anchored" : "off"}
            </span>
          ))}
        </div>
        <div className="metric">
          <div className="m"><b>{fmt(mae.before, 0)}</b><span>mean abs error — observational only</span></div>
          <div className="m"><b style={{ color: "var(--good)" }}>{fmt(mae.after, 0)}</b>
            <span>mean abs error — with selected experiments</span></div>
        </div>
        <table>
          <thead><tr><th>channel</th><th>before</th><th>after</th><th>truth</th><th style={{ width: 260 }}></th></tr></thead>
          <tbody>
            {repair.channels.map((c) => {
              const active = on.has(c.channel);
              const est = active ? c.after : c.before;
              return (
                <tr key={c.channel}>
                  <td style={{ color: CHANNEL_COLORS[c.channel] }}>{c.channel}</td>
                  <td>{fmt(c.before.mean, 0)}</td>
                  <td style={{ color: active ? CHANNEL_COLORS[c.channel] : "var(--muted)" }}>
                    {fmt(c.after.mean, 0)}</td>
                  <td>{fmt(truth[c.channel], 0)}</td>
                  <td>
                    <svg width="260" height="22">
                      <line x1={`${(est.lo / allMax) * 100}%`} y1="11" x2={`${(est.hi / allMax) * 100}%`}
                        y2="11" stroke={active ? CHANNEL_COLORS[c.channel] : "#5b6571"} strokeWidth="4" />
                      <circle cx={scale(est.mean)} cy="11" r="4"
                        fill={active ? CHANNEL_COLORS[c.channel] : "#9aa7b4"} />
                      <line x1={scale(truth[c.channel])} y1="2" x2={scale(truth[c.channel])} y2="20"
                        stroke="#d9a200" strokeWidth="2" />
                    </svg>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="small">
          dot = estimate · bar = 89% CI · gold line = sealed truth. The rotating calendar lets us
          anchor <b>every</b> channel — turning "test everything" into confident moves across the board.
        </p>
      </div>
    </section>
  );
}

function RoiSection({ roi }) {
  const max = Math.max(...roi.channels.flatMap((c) => [c.roi[2], c.mroi[2]]));
  const bar = (v) => `${Math.max(0, (v / max) * 100)}%`;
  return (
    <section>
      <h2>Revenue &amp; ROI — average is a trap</h2>
      <p className="lead">
        Blended LTV ${roi.ltv.mu} (89% ${roi.ltv.lo}–${roi.ltv.hi}); blended media ROI
        {" "}{roi.blended_roi}. But <b>average ROI and marginal ROI disagree</b>. Decisions ride on
        marginal ROI — the return on the <em>next</em> dollar, which collapses as a channel saturates.
      </p>
      <div className="card">
        <table>
          <thead><tr><th>channel</th><th>avg ROI</th><th>marginal ROI (next $)</th><th style={{ width: 200 }}></th></tr></thead>
          <tbody>
            {roi.channels.map((c) => (
              <tr key={c.channel}>
                <td style={{ color: CHANNEL_COLORS[c.channel] }}>{c.channel}</td>
                <td>{c.roi[0].toFixed(2)} <span className="small">[{c.roi[1].toFixed(1)},{c.roi[2].toFixed(1)}]</span></td>
                <td className={c.mroi[0] < 1 ? "miss" : ""}>
                  {c.mroi[0].toFixed(2)} <span className="small">[{c.mroi[1].toFixed(1)},{c.mroi[2].toFixed(1)}]</span>
                </td>
                <td>
                  <svg width="200" height="18">
                    <line x1={`${(1 / max) * 100}%`} y1="0" x2={`${(1 / max) * 100}%`} y2="18"
                      stroke="#d62728" strokeDasharray="2 2" />
                    <rect x="0" y="2" width={bar(c.roi[0])} height="5" fill={CHANNEL_COLORS[c.channel]} opacity="0.5" />
                    <rect x="0" y="9" width={bar(c.mroi[0])} height="5" fill={CHANNEL_COLORS[c.channel]} />
                  </svg>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small">
          faint = avg ROI · solid = marginal ROI · red line = break-even (ROI 1). A channel can look
          great on average yet lose money on its next dollar.
        </p>
      </div>
    </section>
  );
}

function OptimizerSection({ optim }) {
  return (
    <section>
      <h2>Budget optimizer — under uncertainty</h2>
      <p className="lead">
        A point-estimate optimizer will confidently reallocate the whole budget. But propagate the
        posterior and most of those moves can't be told apart from zero. Only act on
        <b> robust</b> moves; route the rest to an experiment.
      </p>
      <OptimizerWidget optim={optim} />
    </section>
  );
}

function LimitationsSection() {
  return (
    <section>
      <h2>Honest limitations</h2>
      <p className="lead">The non-negotiable section. What this model still can't be trusted to do.</p>
      <div className="card">
        <ul className="tight">
          <li><b>Translation is idealized.</b> The experiment→prior bridge assumes test markets sit
            near half-saturation and shares the model's saturation slope; it pins the channel ceiling
            but carries its own error (see social/affiliate, which the anchor moves least).</li>
          <li><b>Observational bias persists.</b> Even anchored, correlated channels can trade credit;
            priors bound the damage but don't cure it.</li>
          <li><b>Overconfident intervals.</b> The 89% predictive band covers far fewer than 89% of
            weeks. Treat the point estimates as ranges, and the ranges as optimistic.</li>
          <li><b>LTV is a single blended figure</b> with a range, not per-channel — deliberately,
            because per-channel LTV is usually a fiction early on.</li>
          <li><b>Light sampling.</b> Short chains for compute; a production run samples longer and
            checks convergence harder.</li>
        </ul>
        <div className="callout good">
          The credible deliverable to a CMO is not "here is the optimal budget." It is:
          <b> cut the saturated channel now — we're confident</b>; for everything else the model can't
          tell us reliably, so <b>run a geo-test before moving money</b>, starting with the highest-upside,
          highest-uncertainty channel.
        </div>
      </div>
    </section>
  );
}
