import React, { useMemo, useState } from "react";
import { CHANNEL_COLORS, fmt } from "./lib.js";

// Recompute the robust verdict for a channel at a user-chosen confidence level + dead-band.
function quantile(sorted, q) {
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

export default function OptimizerWidget({ optim }) {
  const [conf, setConf] = useState(89);
  const [band, setBand] = useState(5);
  if (!optim) return null;

  const tail = (100 - conf) / 2 / 100;
  const rows = useMemo(() => {
    return optim.channels.map((ch) => {
      const s = [...ch.draws].sort((a, b) => a - b);
      const lo = quantile(s, tail);
      const hi = quantile(s, 1 - tail);
      const med = quantile(s, 0.5);
      let verdict = "TEST FIRST";
      if (lo > band) verdict = "INCREASE";
      else if (hi < -band) verdict = "DECREASE";
      return { ...ch, lo, hi, med, verdict };
    });
  }, [optim, tail, band]);

  const span = 200; // -100%..+100% mapped across the bar
  const toX = (v) => 50 + (Math.max(-100, Math.min(100, v)) / 100) * 50;

  const pillClass = (v) => (v === "INCREASE" ? "inc" : v === "DECREASE" ? "dec" : "test");

  return (
    <div className="card">
      <div className="controls">
        <div className="control">
          <label>confidence level = <span className="val">{conf}%</span></label>
          <input type="range" min="50" max="99" step="1" value={conf}
            onChange={(e) => setConf(+e.target.value)} />
        </div>
        <div className="control">
          <label>dead-band (ignore moves smaller than) = <span className="val">±{band}%</span></label>
          <input type="range" min="0" max="25" step="1" value={band}
            onChange={(e) => setBand(+e.target.value)} />
        </div>
      </div>
      <p className="small">
        Point-estimate optimum promises <b>+{optim.point_estimate.lift_pct}%</b> conversions at the
        same budget. But a move is only <b>robust</b> if its whole {conf}% interval clears the
        dead-band. Drag the sliders: most "confident" moves dissolve into <b>test-first</b>.
      </p>
      <table>
        <thead>
          <tr><th>channel</th><th>median Δ</th><th>{conf}% interval</th><th style={{ width: span }}>　</th><th>verdict</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.channel}>
              <td style={{ color: CHANNEL_COLORS[r.channel] }}>{r.channel}</td>
              <td>{r.med >= 0 ? "+" : ""}{fmt(r.med, 0)}%</td>
              <td>[{fmt(r.lo, 0)}%, {fmt(r.hi, 0)}%]</td>
              <td>
                <svg width={span} height="20">
                  <line x1={toX(0)} y1="0" x2={toX(0)} y2="20" stroke="#5b6571" strokeDasharray="2 2" />
                  <line x1={toX(r.lo)} y1="10" x2={toX(r.hi)} y2="10"
                    stroke={CHANNEL_COLORS[r.channel]} strokeWidth="3" />
                  <circle cx={toX(r.med)} cy="10" r="4" fill={CHANNEL_COLORS[r.channel]} />
                </svg>
              </td>
              <td><span className={"pill " + pillClass(r.verdict)}>{r.verdict}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small">{optim.note}</p>
    </div>
  );
}
