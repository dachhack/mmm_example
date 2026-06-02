import React, { useMemo, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { geometricAdstock, hill } from "./lib.js";

// A representative bursty impression series (flighted) to feel carryover + saturation.
const RAW = (() => {
  const x = new Array(40).fill(0);
  [4, 5, 12, 13, 14, 24, 30, 31].forEach((i) => (x[i] = 100));
  for (let i = 0; i < 40; i++) x[i] += i % 7 === 0 ? 40 : 8;
  return x;
})();

export default function TransformsWidget() {
  const [theta, setTheta] = useState(0.6);
  const [halfSat, setHalfSat] = useState(120);
  const [slope, setSlope] = useState(1.6);

  const adData = useMemo(() => {
    const ad = geometricAdstock(RAW, theta);
    return RAW.map((v, i) => ({ week: i, raw: v, adstock: +ad[i].toFixed(1) }));
  }, [theta]);

  const hillData = useMemo(() => {
    const xs = [];
    for (let v = 0; v <= 400; v += 5) xs.push(v);
    const h = hill(xs, halfSat, slope);
    return xs.map((v, i) => ({ x: v, resp: +(h[i] * 100).toFixed(2) }));
  }, [halfSat, slope]);

  return (
    <div className="grid2">
      <div className="card">
        <b>Adstock — carryover (θ)</b>
        <p className="small">
          Advertising echoes forward: each week keeps a fraction θ of last week's effective exposure.
          Raise θ and the bursts smear into the following weeks.
        </p>
        <div className="control">
          <label>carryover θ = <span className="val">{theta.toFixed(2)}</span></label>
          <input type="range" min="0" max="0.95" step="0.05" value={theta}
            onChange={(e) => setTheta(+e.target.value)} />
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={adData} margin={{ top: 6, right: 10, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#2a313c" vertical={false} />
            <XAxis dataKey="week" stroke="#9aa7b4" fontSize={11} />
            <YAxis stroke="#9aa7b4" fontSize={11} />
            <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }} />
            <Area dataKey="raw" stroke="#5b6571" fill="#2a313c" name="raw" />
            <Area dataKey="adstock" stroke="#4da3ff" fill="rgba(77,163,255,0.25)" name="adstocked" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <b>Hill saturation — diminishing returns</b>
        <p className="small">
          Each extra unit of (adstocked) exposure buys less. half-sat is the bend point; slope
          sets how S-shaped the threshold is.
        </p>
        <div className="control">
          <label>half-saturation = <span className="val">{halfSat}</span></label>
          <input type="range" min="20" max="300" step="5" value={halfSat}
            onChange={(e) => setHalfSat(+e.target.value)} />
        </div>
        <div className="control">
          <label>slope = <span className="val">{slope.toFixed(1)}</span></label>
          <input type="range" min="0.6" max="3.5" step="0.1" value={slope}
            onChange={(e) => setSlope(+e.target.value)} />
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={hillData} margin={{ top: 6, right: 10, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#2a313c" vertical={false} />
            <XAxis dataKey="x" stroke="#9aa7b4" fontSize={11} />
            <YAxis stroke="#9aa7b4" fontSize={11} domain={[0, 100]} />
            <Tooltip contentStyle={{ background: "#161b22", border: "1px solid #2a313c" }}
              formatter={(v) => `${v}% of ceiling`} />
            <Line dataKey="resp" stroke="#2ca02c" dot={false} strokeWidth={2} name="response" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
