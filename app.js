"use strict";
/* Multi-scale battery twin — concept demo surrogate chain.
   Every number is computed here, live, from the six knob values.

   Revision (Nov 2026): the electrode scale now follows the time-dependent
   deep-learning manufacturing model of Galvez-Aranda et al., Adv. Energy
   Mater. 2024, 14, 2400376 (DOI 10.1002/aenm.202400376) — a CNN time-series
   surrogate trained on CGPD + DEM data (ARTISTIC project) that predicts the
   3D microstructure frame by frame during calendering from the compression
   degree (CD), including the spring-back (elastic recovery) when the roll
   pressure is released. This demo keeps stand-in physics, but adopts the
   paper's structure: CD as the calendering control variable, a process-time
   axis with a compression phase and a relaxation phase, the four phases
   AM / CBD / pores / void, functional metrics (porosity, tortuosity,
   diffusivity proxy, contact) derived from the microstructure rather than
   fitted, the RELAXED microstructure as what flows downstream, and a
   quasi-real-time inverse-design scan over CD. */

// ---------- deterministic RNG (stable pack sampling per knob set) ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) { // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// ---------- surrogate chain ----------
// knobs: adh (0.5-1.5), bin wt% (2-6), dry (0.5-2), cal = CD % (20-50), psd um (4-14), crate (0.5-2)
const NSTEPS = 28, PRESS_END = 20; // time frames: 0..20 compression ramp, 21..28 spring-back release

function driedState(k, epsJitter = 0) {
  // porosity of the dried (uncalendered) electrode — the state the calendering TS starts from
  const epsDry = clamp(0.62 - 0.012 * k.bin + 0.03 * (k.dry - 1) - 0.004 * (k.psd - 9) + epsJitter, 0.2, 0.68);
  const mig = clamp(0.22 + 0.38 * (k.dry - 0.5) / 1.5 - 0.18 * (k.adh - 0.5), 0, 1);
  return { epsDry, mig };
}
function calenderSeries(k, epsJitter = 0) {
  // Time series of the calendering step, structured after the paper:
  // thickness h(t) (relative to dried = 1) under a CD ramp, then spring-back.
  // Solid volume is conserved: (1 - eps) * h = const = (1 - epsDry).
  const { epsDry, mig } = driedState(k, epsJitter);
  const CD = k.cal / 100;
  // spring-back recovers a fraction of the imposed compression; stiffer,
  // better-bonded coatings (high adhesion, more binder) spring back less.
  const sbFrac = clamp(0.30 - 0.10 * (k.adh - 1) - 0.03 * (k.bin - 3.5), 0.10, 0.45);
  const steps = [];
  for (let t = 0; t <= NSTEPS; t++) {
    let h;
    if (t <= PRESS_END) h = 1 - CD * (t / PRESS_END);
    else h = (1 - CD) + sbFrac * CD * (1 - Math.exp(-(t - PRESS_END) / 3));
    const eps = clamp(1 - (1 - epsDry) / h, 0.05, 0.68);
    const tau = Math.pow(eps, -0.55) * (1 + 0.35 * mig);
    steps.push({ t, h, eps, tau });
  }
  const fin = steps[NSTEPS], load = steps[PRESS_END];
  return { steps, epsDry, mig, sbFrac, CD, hCal: fin.h, epsLoad: load.eps, eps: fin.eps, tau: fin.tau };
}
function electrode(k, epsJitter = 0) {
  // What flows downstream is the RELAXED microstructure (after spring-back),
  // exactly as in the paper: h^cal and its porosity/tortuosity.
  const s = calenderSeries(k, epsJitter);
  const sig = clamp(Math.pow(1 - s.eps, 1.4) * (0.55 + 0.45 * k.adh) * (1 - 0.25 * s.mig), 0.05, 1.2);
  return { eps: s.eps, mig: s.mig, tau: s.tau, sig, series: s };
}
function cellFrom(k, el) {
  const T = el.eps / el.tau;
  const U = T / (T + 0.06);
  const Q = 3.40 * Math.pow(U, 0.30) * (0.86 + 0.14 * el.sig);
  const rate2C = clamp(112 * Math.pow(U, 0.85) * (0.80 + 0.20 * el.sig) - 8, 40, 98);
  const res = 10 + 9 * Math.pow(el.tau, 1.2) / (el.eps * 2.4);
  return { T, U, Q, rate2C, res };
}
function rulYears(k, el, cell) {
  // years to 80% capacity for one cell
  const stress = Math.pow(1 / cell.U, 0.7) * Math.pow(k.crate, 0.9);
  const build = (0.80 + 0.20 * k.adh) * (1 - 0.30 * el.mig);
  return clamp(13.5 * build / stress, 1.5, 25);
}
function packSample(k, n = 96, seed = 20261113) {
  const rng = mulberry32(seed);
  const sigmaP = 0.010 + 0.014 * (k.dry - 0.5) / 1.5 + 0.008 * (1 - (k.cal - 20) / 30);
  const cells = [];
  for (let i = 0; i < n; i++) {
    const el = electrode(k, gauss(rng) * sigmaP);
    const c = cellFrom(k, el);
    cells.push({ el, c, rul: rulYears(k, el, c) });
  }
  return cells;
}
function retentionAt(cellRul, t) { // capacity % at year t for a cell with given RUL
  const fade = 20 * Math.pow(t / cellRul, 1.25);
  return clamp(100 - fade, 55, 100);
}
const pct = (arr, p) => {
  const s = [...arr].sort((a, b) => a - b);
  return s[clamp(Math.floor(p * s.length), 0, s.length - 1)];
};

// ---------- state ----------
const $ = id => document.getElementById(id);
let tIdx = NSTEPS; // current calendering time frame shown in the electrode view
let playing = false, playTimer = null;
function readKnobs() {
  return { adh: +$("adh").value, bin: +$("bin").value, dry: +$("dry").value,
           cal: +$("cal").value, psd: +$("psd").value, crate: +$("crate").value };
}
function fmtLabels(k) {
  $("v_adh").textContent = k.adh.toFixed(2) + "×";
  $("v_bin").textContent = k.bin.toFixed(1) + " wt%";
  $("v_dry").textContent = k.dry.toFixed(2) + "×";
  $("v_cal").textContent = k.cal.toFixed(0) + "%";
  $("v_psd").textContent = k.psd.toFixed(1) + " µm";
  $("v_crate").textContent = k.crate.toFixed(2) + "C";
  $("v_teps").textContent = (+$("teps").value).toFixed(3);
  $("v_tstep").textContent = "frame " + tIdx + " / " + NSTEPS +
    (tIdx <= PRESS_END ? " — compression" : " — spring-back (roll released)");
}

// ---------- canvas helpers ----------
function fitCanvas(cv, height = 240) {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(300, r.width) * dpr;
  cv.height = height * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W: Math.max(300, r.width), H: height };
}
function axes(ctx, W, H, pad = 34) {
  ctx.strokeStyle = "#22314f"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, 8); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 8, H - pad);
  ctx.stroke();
}

// ---------- view 1: electrode microstructure over calendering time ----------
// Phases colored as in the paper's Figure 1: AM blue, CBD red, pores yellow, void white.
function drawElectrode(k, el) {
  const { ctx, W, H } = fitCanvas($("electrode"));
  ctx.clearRect(0, 0, W, H);
  const s = el.series;
  const fr = s.steps[tIdx];
  const rng = mulberry32(777);
  const boxX = 18, boxW = W - 36;
  const frameH = H - 56;                 // fixed grid = dried electrode height (h = 1)
  const frameY = 14;
  const boxH = frameH * fr.h;            // coating height at this time frame
  const boxY = frameY + (frameH - boxH); // coating sits on the collector, void opens above
  // void region (paper: the empty region on top of the electrode in the grid)
  ctx.fillStyle = "rgba(226,232,240,0.85)";
  ctx.fillRect(boxX, frameY, boxW, frameH - boxH);
  ctx.fillStyle = "#5b6b82"; ctx.font = "10px sans-serif";
  if (frameH - boxH > 12) ctx.fillText("VOID", boxX + 6, frameY + 12);
  // pore space tint inside the coating (pores = yellow in the paper)
  ctx.fillStyle = "rgba(250,204,21,0.13)";
  ctx.fillRect(boxX, boxY, boxW, boxH);
  // current collector
  ctx.fillStyle = "#3b4a63";
  ctx.fillRect(boxX, frameY + frameH, boxW, 7);
  ctx.fillStyle = "#8fa6c4";
  ctx.fillText("CURRENT COLLECTOR", boxX, frameY + frameH + 19);
  // particles (AM) — positions fixed per particle, squeezed vertically with h(t)
  const n = 150;
  const parts = [];
  for (let i = 0; i < n; i++) {
    const r = clamp(k.psd * 0.62 * (0.55 + rng() * 0.95), 2.2, 11);
    const ux = rng(), uy = rng();
    parts.push({ x: boxX + r + ux * (boxW - 2 * r), y0: r + uy * (frameH - 2 * r), r });
  }
  const py = p => frameY + frameH - p.y0 * fr.h - p.r * (1 - fr.h) * 0;
  // binder (CBD) bridges — red, as in the paper
  ctx.lineWidth = 1;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const a = parts[i], b = parts[j];
    const d = Math.hypot(a.x - b.x, (a.y0 - b.y0) * fr.h);
    if (d < (a.r + b.r) * 1.25) {
      const topBias = 1 - a.y0 / frameH;
      const alpha = 0.10 + 0.5 * (k.adh - 0.5) * (0.4 + 0.6 * (el.mig * topBias + (1 - el.mig) * 0.5));
      ctx.strokeStyle = `rgba(239,68,68,${clamp(alpha, 0.05, 0.6)})`;
      ctx.beginPath(); ctx.moveTo(a.x, py(a)); ctx.lineTo(b.x, py(b)); ctx.stroke();
    }
  }
  for (const p of parts) {
    const y = py(p);
    if (y < boxY - p.r || y > frameY + frameH) continue;
    const g = ctx.createRadialGradient(p.x - p.r * 0.3, y - p.r * 0.3, p.r * 0.2, p.x, y, p.r);
    g.addColorStop(0, "#7dd3fc"); g.addColorStop(1, "#2563eb");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(p.x, y, p.r, 0, 7); ctx.fill();
  }
  ctx.strokeStyle = "#22314f"; ctx.strokeRect(boxX, frameY, boxW, frameH);
  ctx.fillStyle = "#8fa6c4";
  ctx.fillText(`h = ${fr.h.toFixed(2)} of dried · ε = ${fr.eps.toFixed(3)} · τ = ${fr.tau.toFixed(2)} at this frame — AM blue · CBD red · pores pale yellow · void white`, boxX, 12);
}

// ---------- view 1b: calendering time series (thickness, porosity, tortuosity) ----------
function drawSeries(el) {
  const { ctx, W, H } = fitCanvas($("tschart"), 118);
  ctx.clearRect(0, 0, W, H);
  const s = el.series;
  const pad = 30, iw = W - pad - 10;
  const rows = [
    { key: "h", label: "h(t)", color: "#38bdf8", fmt: v => v.toFixed(2) },
    { key: "eps", label: "ε(t)", color: "#fbbf24", fmt: v => v.toFixed(3) },
    { key: "tau", label: "τ(t)", color: "#a78bfa", fmt: v => v.toFixed(2) },
  ];
  const rh = (H - 8) / 3;
  rows.forEach((row, ri) => {
    const y0 = 4 + ri * rh, vals = s.steps.map(st => st[row.key]);
    const lo = Math.min(...vals), hi = Math.max(...vals), span = Math.max(hi - lo, 1e-9);
    const X = t => pad + (t / NSTEPS) * iw;
    const Y = v => y0 + rh - 4 - ((v - lo) / span) * (rh - 10);
    // spring-back region shading
    ctx.fillStyle = "rgba(52,211,153,0.08)";
    ctx.fillRect(X(PRESS_END), y0 + 1, X(NSTEPS) - X(PRESS_END), rh - 2);
    ctx.strokeStyle = "#22314f"; ctx.strokeRect(pad, y0 + 1, iw, rh - 2);
    ctx.beginPath();
    s.steps.forEach((st, i) => i === 0 ? ctx.moveTo(X(st.t), Y(st[row.key])) : ctx.lineTo(X(st.t), Y(st[row.key])));
    ctx.strokeStyle = row.color; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.fillStyle = row.color; ctx.font = "10px sans-serif";
    ctx.fillText(row.label, 2, y0 + rh / 2 + 3);
    ctx.fillStyle = "#dbe6f5";
    const lx = tIdx > NSTEPS - 7 ? X(tIdx) - 34 : X(tIdx) + 4;
    ctx.fillText(row.fmt(s.steps[tIdx][row.key]), lx, Y(s.steps[tIdx][row.key]) - 3);
    ctx.beginPath(); ctx.arc(X(tIdx), Y(s.steps[tIdx][row.key]), 3, 0, 7);
    ctx.fillStyle = "#fff"; ctx.fill();
  });
  ctx.fillStyle = "#34d399"; ctx.font = "10px sans-serif";
  ctx.fillText("spring-back", (30 + (PRESS_END / NSTEPS) * (W - 40)) + 4, H - 2);
}

// ---------- view 2: cell discharge curves ----------
function drawCell(k, cell) {
  const { ctx, W, H } = fitCanvas($("cell"));
  ctx.clearRect(0, 0, W, H);
  axes(ctx, W, H);
  const pad = 34, iw = W - pad - 12, ih = H - pad - 12;
  const qMax = 3.6, vMin = 2.9, vMax = 4.35;
  const curve = (crateFactor, sagMul) => {
    ctx.beginPath();
    for (let i = 0; i <= 60; i++) {
      const x = i / 60; // SOC discharged fraction of this curve's capacity
      const q = x * cell.Q * crateFactor;
      let v = 3.32 + 1.02 * Math.pow(1 - x, 1.12) + 0.10 * Math.exp(-9 * x);
      v -= (1 - cell.U) * sagMul * 0.38 * Math.pow(Math.sin(Math.PI * Math.min(x * 1.04, 1)), 1.5);
      v -= 0.16 * Math.pow(x, 9);
      const px = pad + (q / qMax) * iw, py = 8 + (1 - (v - vMin) / (vMax - vMin)) * ih;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  };
  ctx.lineWidth = 2; ctx.strokeStyle = "#38bdf8";
  curve(1, 1);
  ctx.lineWidth = 1.5; ctx.strokeStyle = "#fbbf24";
  curve(cell.rate2C / 100, 2.1);
  ctx.fillStyle = "#8fa6c4"; ctx.font = "10px sans-serif";
  ctx.fillText("0.2C", pad + (cell.Q / qMax) * iw - 26, 22);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText("2C", pad + ((cell.Q * cell.rate2C / 100) / qMax) * iw - 20, 40);
  ctx.fillStyle = "#8fa6c4";
  ctx.fillText("discharge capacity (Ah) →", pad + 4, H - 10);
  ctx.save(); ctx.rotate(-Math.PI / 2);
  ctx.fillText("voltage (V)", -H + 60, 12); ctx.restore();
}

// ---------- view 3: pack grid ----------
function drawPack(cells) {
  const { ctx, W, H } = fitCanvas($("pack"));
  ctx.clearRect(0, 0, W, H);
  const cols = 12, rows = 8, gap = 3;
  const cw = (W - 24 - (cols - 1) * gap) / cols, ch = (H - 30 - (rows - 1) * gap) / rows;
  const qs = cells.map(c => c.c.Q);
  const lo = Math.min(...qs), hi = Math.max(...qs);
  cells.forEach((c, i) => {
    const x = 12 + (i % cols) * (cw + gap), y = 8 + Math.floor(i / cols) * (ch + gap);
    const t = (c.c.Q - lo) / Math.max(hi - lo, 1e-9);
    ctx.fillStyle = `rgb(${Math.round(30 + 200 * (1 - t))},${Math.round(120 + 100 * t)},${Math.round(200 - 120 * t)})`;
    ctx.fillRect(x, y, cw, ch);
    if (c.c.Q === lo) { ctx.strokeStyle = "#f87171"; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, cw + 2, ch + 2); }
  });
  ctx.fillStyle = "#8fa6c4"; ctx.font = "10px sans-serif";
  ctx.fillText("each square = one cell, colored by capacity; red outline = weakest cell (limits the string)", 12, H - 6);
}

// ---------- view 4: service life ----------
function drawLife(k, cells) {
  const { ctx, W, H } = fitCanvas($("life"));
  ctx.clearRect(0, 0, W, H);
  const pad = 34, iw = W - pad - 12, ih = (H - pad - 12) / 2;
  // top: retention fan
  axes(ctx, W, H);
  const years = 15;
  const band = [];
  for (let t = 0; t <= years; t += 0.5) {
    const rets = cells.map(c => retentionAt(c.rul, t));
    band.push({ t, p10: pct(rets, 0.10), p50: pct(rets, 0.50), p90: pct(rets, 0.90) });
  }
  const X = t => pad + (t / years) * iw, Y = v => 8 + (1 - (v - 55) / 45) * ih;
  ctx.beginPath();
  band.forEach((b, i) => i === 0 ? ctx.moveTo(X(b.t), Y(b.p90)) : ctx.lineTo(X(b.t), Y(b.p90)));
  [...band].reverse().forEach(b => ctx.lineTo(X(b.t), Y(b.p10)));
  ctx.closePath(); ctx.fillStyle = "rgba(56,189,248,0.18)"; ctx.fill();
  ctx.beginPath();
  band.forEach((b, i) => i === 0 ? ctx.moveTo(X(b.t), Y(b.p50)) : ctx.lineTo(X(b.t), Y(b.p50)));
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2; ctx.stroke();
  ctx.setLineDash([4, 4]); ctx.strokeStyle = "#f87171"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, Y(80)); ctx.lineTo(pad + iw, Y(80)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#8fa6c4"; ctx.font = "10px sans-serif";
  ctx.fillText("capacity retention % (band = P10–P90 across the batch)", pad + 4, 16);
  ctx.fillStyle = "#f87171"; ctx.fillText("80% end-of-life line", pad + iw - 108, Y(80) - 4);
  // bottom: Weibull reliability
  const ruls = cells.map(c => c.rul);
  const med = pct(ruls, 0.5), beta = 2.4;
  const eta = med / Math.pow(Math.LN2, 1 / beta);
  const y0 = 8 + ih + 16, ih2 = H - pad - y0;
  ctx.strokeStyle = "#22314f";
  ctx.beginPath(); ctx.moveTo(pad, y0); ctx.lineTo(pad, y0 + ih2); ctx.lineTo(pad + iw, y0 + ih2); ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const t = (i / 60) * years, R = Math.exp(-Math.pow(t / eta, beta));
    const px = X(t), py = y0 + (1 - R) * ih2;
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = "#8fa6c4";
  ctx.fillText("pack reliability R(t) — probability of staying above 80%", pad + 4, y0 + 10);
  ctx.fillText("years →", pad + 4, H - 10);
  return Math.exp(-Math.pow(8 / eta, beta));
}

// ---------- memory meter (live sensitivity sweep) ----------
function lhsSamples(n, seed) {
  const rng = mulberry32(seed);
  const dims = ["adh", "bin", "dry", "cal", "psd", "crate"];
  const ranges = { adh: [0.5, 1.5], bin: [2, 6], dry: [0.5, 2], cal: [20, 50], psd: [4, 14], crate: [0.5, 2] };
  const cols = {};
  for (const d of dims) {
    const arr = Array.from({ length: n }, (_, i) => (i + rng()) / n);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
    cols[d] = arr.map(u => ranges[d][0] + u * (ranges[d][1] - ranges[d][0]));
  }
  return { dims, samples: Array.from({ length: n }, (_, i) => Object.fromEntries(dims.map(d => [d, cols[d][i]]))) };
}
function pearson(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
  return syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy);
}
function memoryShares() {
  const { dims, samples } = lhsSamples(240, 4242);
  const outs = { tau: [], cap: [], rul: [] };
  for (const k of samples) {
    const el = electrode(k), c = cellFrom(k, el);
    outs.tau.push(el.tau); outs.cap.push(c.Q); outs.rul.push(rulYears(k, el, c));
  }
  const shares = {};
  for (const d of dims) {
    const xs = samples.map(s => s[d]);
    shares[d] = Object.fromEntries(Object.entries(outs).map(([stage, ys]) => {
      const r = pearson(xs, ys); return [stage, r * r];
    }));
  }
  // normalize within each stage
  for (const stage of ["tau", "cap", "rul"]) {
    const tot = dims.reduce((a, d) => a + shares[d][stage], 0) || 1;
    for (const d of dims) shares[d][stage] /= tot;
  }
  return shares;
}
const KNOB_LABELS = { adh: "Binder adhesion (MD)", bin: "Binder content", dry: "Drying rate",
                      cal: "Compression degree CD", psd: "Particle size D50", crate: "Charge rate (use)" };
function drawMeter() {
  const shares = memoryShares();
  const colors = { adh: "#38bdf8", bin: "#7dd3fc", dry: "#fbbf24", cal: "#f59e0b", psd: "#a78bfa", crate: "#34d399" };
  $("meter").innerHTML = Object.keys(KNOB_LABELS).map(d => {
    const s = shares[d];
    const bar = (v, stage) => `<div class="bar"><i style="width:${(v * 100).toFixed(1)}%;background:${colors[d]}"></i><em>${(v * 100).toFixed(0)}% of ${stage} variance</em></div>`;
    return `<div class="meter-row"><div class="k">${KNOB_LABELS[d]}</div>${bar(s.tau, "τ")}${bar(s.cap, "capacity")}${bar(s.rul, "RUL")}</div>`;
  }).join("");
}

// ---------- inverse design: scan CD for a target relaxed porosity ----------
// The paper's point of the fast surrogate: quasi-real-time optimization loops
// that predict the calendering conditions to adopt for a desired electrode.
function recommendCD(k, targetEps) {
  let best = null;
  for (let cd = 20; cd <= 50.001; cd += 0.25) {
    const e = electrode({ ...k, cal: cd }).eps;
    const err = Math.abs(e - targetEps);
    if (!best || err < best.err) best = { cd, eps: e, err };
  }
  return best;
}

// ---------- main update ----------
function update() {
  const k = readKnobs();
  fmtLabels(k);
  const el = electrode(k);
  const cell = cellFrom(k, el);
  drawElectrode(k, el);
  drawSeries(el);
  drawCell(k, cell);
  const cells = packSample(k);
  drawPack(cells);
  const rel8 = drawLife(k, cells);
  const qs = cells.map(c => c.c.Q);
  const mean = qs.reduce((a, b) => a + b) / qs.length;
  const std = Math.sqrt(qs.reduce((a, b) => a + (b - mean) ** 2, 0) / qs.length);
  const ruls = cells.map(c => c.rul);
  const s = el.series;
  $("s_eps").textContent = el.eps.toFixed(3);
  $("s_tau").textContent = el.tau.toFixed(2);
  $("s_sb").textContent = (s.sbFrac * s.CD * 100).toFixed(1) + "% of h";
  $("s_mig").textContent = (el.mig * 100).toFixed(0) + "%";
  $("s_sig").textContent = el.sig.toFixed(2);
  $("loadnote").textContent =
    `Under load the coating reached ε = ${s.epsLoad.toFixed(3)} at h = ${(1 - s.CD).toFixed(2)}; ` +
    `after spring-back it relaxed to ε = ${el.eps.toFixed(3)} at h^cal = ${s.hCal.toFixed(2)}. ` +
    `The relaxed state is what the cell, pack and life views below are built from.`;
  const rec = recommendCD(k, +$("teps").value);
  $("s_cdrec").textContent = `CD ≈ ${rec.cd.toFixed(1)}% gives relaxed ε = ${rec.eps.toFixed(3)} (scan of CD 20–50%, computed live)`;
  $("s_cap").textContent = cell.Q.toFixed(2) + " Ah";
  $("s_rate").textContent = cell.rate2C.toFixed(0) + "%";
  $("s_res").textContent = cell.res.toFixed(0) + " mΩ";
  $("s_mean").textContent = mean.toFixed(2) + " Ah";
  $("s_std").textContent = std.toFixed(3) + " Ah";
  $("s_min").textContent = Math.min(...qs).toFixed(2) + " Ah";
  $("s_pack").textContent = (96 * pct(qs, 0.05)).toFixed(0) + " Ah";
  $("s_rul").textContent = pct(ruls, 0.5).toFixed(1) + " yr";
  $("s_rulband").textContent = pct(ruls, 0.10).toFixed(1) + " – " + pct(ruls, 0.90).toFixed(1) + " yr";
  $("s_rel").textContent = (rel8 * 100).toFixed(0) + "%";
}
for (const id of ["adh", "bin", "dry", "cal", "psd", "crate", "teps"])
  $(id).addEventListener("input", update);
$("tstep").addEventListener("input", () => {
  stopPlay();
  tIdx = +$("tstep").value;
  update();
});
function stopPlay() {
  playing = false;
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  $("play").textContent = "▶ Play calendering";
}
$("play").addEventListener("click", () => {
  if (playing) { stopPlay(); update(); return; }
  playing = true;
  $("play").textContent = "⏸ Pause";
  if (tIdx >= NSTEPS) tIdx = 0;
  playTimer = setInterval(() => {
    tIdx++;
    $("tstep").value = tIdx;
    update();
    if (tIdx >= NSTEPS) stopPlay();
  }, 130);
});
window.addEventListener("resize", update);
drawMeter();
update();
