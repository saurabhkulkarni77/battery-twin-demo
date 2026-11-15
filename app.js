"use strict";
/* Multi-scale battery twin — TRAINED-MODEL edition (Nov 2026).
   Every stage is now a genuinely trained model; weights live in models.js
   (exported by dl/scripts/train_models.py) and every forward pass runs here,
   live, in the browser. No server calls.

   Models:
   1. ELECTRODE  MLP trained on a simulation grid built from documented
      relations (volume conservation, Bruggeman tortuosity, drying-driven
      binder migration, partial spring-back). No open labeled electrode
      dataset was found — simulation-trained, labeled as such. Structure
      follows Galvez-Aranda et al. 2024 (CD control, time axis, spring-back,
      relaxed state downstream).
   2. SOH        Multi-fidelity fusion after Kohtz et al. 2022 (RaDA Lab): a
      physics TREND net (parabolic SEI growth) plus a DISCREPANCY net trained
      on NASA Ames PCoE residuals — the co-kriging structure. Drives cell
      capacity + service life; RUL = first crossing of SOH 0.80, as in the paper.
   3. SEGMENT    Gaussian Process Regression from partial-charging-segment
      features (3.8-4.1 V window, stretched vs the cell's own early-life
      reference) to SOH — Kohtz et al.'s core estimator, NASA data.
   4. RESISTANCE MLP on NASA EIS measurements (Re + Rct).
   Structural links between stages (electrode transport -> initial capacity
   Q0 and initial SEI offset L0) are simple stated mappings, not learned —
   see the model notes at the bottom of the page. */

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
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const pct = (arr, p) => {
  const s = [...arr].sort((a, b) => a - b);
  return s[clamp(Math.floor(p * s.length), 0, s.length - 1)];
};

// ---------- trained-model forward passes ----------
function mlpForward(net, x) {
  let h = x.map((v, i) => (v - net.x_mean[i]) / net.x_std[i]);
  for (const L of net.layers) {
    const out = new Array(L.b.length);
    for (let j = 0; j < L.b.length; j++) {
      let s = L.b[j];
      const W = L.W[j];
      for (let i = 0; i < h.length; i++) s += W[i] * h[i];
      out[j] = L.act === "relu" ? Math.max(0, s) : s;
    }
    h = out;
  }
  return h.map((v, i) => v * net.y_scale[i] + net.y_mean[i]);
}
function gprPredict(g, x) {
  const z = x.map((v, i) => (v - g.x_mean[i]) / g.x_std[i]);
  let s = 0;
  for (let j = 0; j < g.support_X.length; j++) {
    const Xj = g.support_X[j];
    let d2 = 0;
    for (let i = 0; i < z.length; i++) {
      const d = (z[i] - Xj[i]) / g.length_scales[i];
      d2 += d * d;
    }
    s += g.alpha[j] * g.const * Math.exp(-0.5 * d2);
  }
  return g.y_mean + g.y_std * s; // sklearn normalize_y: y was standardized by mean AND std
}

// ---------- stage 1: electrode (simulation-trained MLP) ----------
// knobs: adh (0.5-1.5), bin wt% (2-6), dry (0.5-2), cal = CD % (20-50), psd um (4-14), crate (0.5-2)
const NSTEPS = 28, PRESS_END = 20;
function electrode(k) {
  const y = mlpForward(MODELS.electrode, [k.adh, k.bin, k.dry, k.cal, k.psd]);
  const o = {};
  MODELS.electrode.outputs.forEach((nm, i) => o[nm] = y[i]);
  return o; // {epsDry, epsRelaxed, tau, hCal, sbFrac, mig, transport}
}
function calSeries(k, el) {
  // time axis: compression ramp, then spring-back release ending exactly at
  // the model's relaxed thickness hCal; porosity from volume conservation.
  const CD = k.cal / 100, hLoad = 1 - CD;
  const norm = 1 - Math.exp(-(NSTEPS - PRESS_END) / 3);
  const steps = [];
  for (let t = 0; t <= NSTEPS; t++) {
    let h;
    if (t <= PRESS_END) h = 1 - CD * (t / PRESS_END);
    else h = hLoad + (el.hCal - hLoad) * (1 - Math.exp(-(t - PRESS_END) / 3)) / norm;
    const eps = clamp(1 - (1 - el.epsDry) / h, 0.05, 0.68);
    const tau = Math.pow(eps, -0.5) * (1 + 0.35 * el.mig);
    steps.push({ t, h, eps, tau });
  }
  return { steps, epsLoad: steps[PRESS_END].eps, hLoad };
}
// structural link (stated, not learned): electrode transport quality sets the
// cell's initial capacity and its initial SEI offset (pre-age) in the life model
function qualityOf(el) {
  const e = MODELS.electrode;
  return clamp((el.transport - e.transport_min) / (e.transport_max - e.transport_min), 0, 1);
}
const q0Of = el => 1.80 + 0.25 * qualityOf(el);       // Ah, NASA-frame initial capacity
const l0Of = el => (1 - qualityOf(el)) * 60;          // equivalent pre-age cycles

// ---------- stage 2+4: SOH (multi-fidelity, Kohtz-style) + resistance ----------
const T_OP = 24, I_DIS_OP = 2.0; // operating point of the twin's reference cell (NASA set-1 conditions)
function sohAt(N, crate, L0) {
  // multi-fidelity fusion: physics TREND net + NASA DISCREPANCY net (co-kriging spirit)
  const x = [Math.log1p(Math.max(N, 0)), T_OP, crate * 2.0, I_DIS_OP, L0];
  const y = mlpForward(MODELS.soh.trend, x)[0] +
            mlpForward(MODELS.soh.discrepancy, [x[0], x[1], x[3]])[0];
  return clamp(y, 0.45, 1.0);
}
function capacityAt(N, crate, el) { return sohAt(N, crate, l0Of(el)) * q0Of(el); }
function resistanceAt(N) {
  const y = mlpForward(MODELS.resistance, [Math.log1p(Math.max(N, 0)), T_OP, I_DIS_OP]);
  return Math.pow(10, y[0]); // mOhm
}
function rulCycles(crate, L0) {
  // first crossing of SOH 0.80 — coarse scan + bisection refine
  let prevN = 0, prevS = sohAt(0, crate, L0);
  for (let N = 10; N <= 800; N += 10) {
    const s = sohAt(N, crate, L0);
    if (s < 0.80 && prevS >= 0.80) {
      let lo = prevN, hi = N;
      for (let i = 0; i < 6; i++) {
        const mid = (lo + hi) / 2;
        (sohAt(mid, crate, L0) < 0.80) ? hi = mid : lo = mid;
      }
      return hi;
    }
    prevN = N; prevS = s;
  }
  return 800; // beyond model horizon
}
function cellFrom(k, el) {
  return { Q0: q0Of(el), Q100: capacityAt(100, k.crate, el),
           Q500: capacityAt(500, k.crate, el), res: resistanceAt(100),
           SOH500: sohAt(500, k.crate, l0Of(el)) };
}

// ---------- stage 3: pack sampling (spread = process jitter through the models) ----------
function packSample(k, n = 96, seed = 20261113) {
  const rng = mulberry32(seed);
  const cells = [];
  for (let i = 0; i < n; i++) {
    const kj = { adh: k.adh * (1 + gauss(rng) * 0.03), bin: k.bin, dry: k.dry * (1 + gauss(rng) * 0.04),
                 cal: k.cal + gauss(rng) * 1.2, psd: k.psd * (1 + gauss(rng) * 0.03), crate: k.crate * (1 + gauss(rng) * 0.05) };
    kj.cal = clamp(kj.cal, 20, 50);
    const el = electrode(kj);
    // batch spread beyond the knobs: per-cell initial-capacity jitter (±1.5%)
    // and initial-SEI jitter (σ = 12 cycles) — identical-condition NASA cells
    // (B0005/6/7) lived 106/61/125 cycles, spread the knobs alone cannot make
    const Q0j = q0Of(el) * (1 + gauss(rng) * 0.015);
    const L0j = clamp(l0Of(el) + gauss(rng) * 12, 0, 120);
    cells.push({ el, crate: kj.crate, L0: L0j, Q0: Q0j,
                 cap10: sohAt(10, kj.crate, L0j) * Q0j,
                 life: rulCycles(kj.crate, L0j) });
  }
  return cells;
}

// ---------- state ----------
const $ = id => document.getElementById(id);
let tIdx = NSTEPS, playing = false, playTimer = null;
let curK = null, curEl = null, curCell = null, curCells = null;
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
// Phases colored as in Galvez-Aranda et al. Fig. 1: AM blue, CBD red, pores yellow, void white.
function drawElectrode(k, el) {
  const { ctx, W, H } = fitCanvas($("electrode"));
  ctx.clearRect(0, 0, W, H);
  const s = calSeries(k, el);
  const fr = s.steps[tIdx];
  const rng = mulberry32(777);
  const boxX = 18, boxW = W - 36;
  const frameH = H - 56;
  const frameY = 14;
  const boxH = frameH * fr.h;
  const boxY = frameY + (frameH - boxH);
  ctx.fillStyle = "rgba(226,232,240,0.85)";
  ctx.fillRect(boxX, frameY, boxW, frameH - boxH);
  ctx.fillStyle = "#5b6b82"; ctx.font = "10px sans-serif";
  if (frameH - boxH > 12) ctx.fillText("VOID", boxX + 6, frameY + 12);
  ctx.fillStyle = "rgba(250,204,21,0.13)";
  ctx.fillRect(boxX, boxY, boxW, boxH);
  ctx.fillStyle = "#3b4a63";
  ctx.fillRect(boxX, frameY + frameH, boxW, 7);
  ctx.fillStyle = "#8fa6c4";
  ctx.fillText("CURRENT COLLECTOR", boxX, frameY + frameH + 19);
  const n = 150;
  const parts = [];
  for (let i = 0; i < n; i++) {
    const r = clamp(k.psd * 0.62 * (0.55 + rng() * 0.95), 2.2, 11);
    const ux = rng(), uy = rng();
    parts.push({ x: boxX + r + ux * (boxW - 2 * r), y0: r + uy * (frameH - 2 * r), r });
  }
  const py = p => frameY + frameH - p.y0 * fr.h;
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

// ---------- view 1b: calendering time series ----------
function drawSeries(k, el) {
  const { ctx, W, H } = fitCanvas($("tschart"), 118);
  ctx.clearRect(0, 0, W, H);
  const s = calSeries(k, el);
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

// ---------- view 2: cell (capacity levels from the SOH model; curve shape schematic) ----------
function drawCell(k, cell) {
  const { ctx, W, H } = fitCanvas($("cell"));
  ctx.clearRect(0, 0, W, H);
  axes(ctx, W, H);
  const pad = 34, iw = W - pad - 12, ih = H - pad - 12;
  const qMax = 2.2, vMin = 2.9, vMax = 4.35;
  const curve = (Q, soh, sagMul) => {
    ctx.beginPath();
    for (let i = 0; i <= 60; i++) {
      const x = i / 60;
      const q = x * Q;
      let v = 3.32 + 1.02 * Math.pow(1 - x, 1.12) + 0.10 * Math.exp(-9 * x);
      v -= (1 - soh) * sagMul * 0.38 * Math.pow(Math.sin(Math.PI * Math.min(x * 1.04, 1)), 1.5);
      v -= 0.16 * Math.pow(x, 9);
      const px = pad + (q / qMax) * iw, py = 8 + (1 - (v - vMin) / (vMax - vMin)) * ih;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  };
  ctx.lineWidth = 2; ctx.strokeStyle = "#38bdf8";
  curve(cell.Q100, sohAt(100, k.crate, l0Of(curEl)), 1);
  ctx.lineWidth = 1.5; ctx.strokeStyle = "#fbbf24";
  curve(cell.Q500, cell.SOH500, 2.1);
  ctx.fillStyle = "#8fa6c4"; ctx.font = "10px sans-serif";
  ctx.fillText("cycle 100", pad + (cell.Q100 / qMax) * iw - 52, 22);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText("cycle 500", pad + (cell.Q500 / qMax) * iw - 52, 40);
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
  const qs = cells.map(c => c.cap10);
  const lo = Math.min(...qs), hi = Math.max(...qs);
  cells.forEach((c, i) => {
    const x = 12 + (i % cols) * (cw + gap), y = 8 + Math.floor(i / cols) * (ch + gap);
    const t = (c.cap10 - lo) / Math.max(hi - lo, 1e-9);
    ctx.fillStyle = `rgb(${Math.round(30 + 200 * (1 - t))},${Math.round(120 + 100 * t)},${Math.round(200 - 120 * t)})`;
    ctx.fillRect(x, y, cw, ch);
    if (c.cap10 === lo) { ctx.strokeStyle = "#f87171"; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, cw + 2, ch + 2); }
  });
  ctx.fillStyle = "#8fa6c4"; ctx.font = "10px sans-serif";
  ctx.fillText("each square = one cell, colored by capacity at cycle 10; red outline = weakest cell (limits the string)", 12, H - 6);
}

// ---------- view 4: service life (retention band + empirical reliability) ----------
const CYC_PER_YEAR = () => MODELS.meta.cycles_per_year;
function drawLife(k, cells) {
  const { ctx, W, H } = fitCanvas($("life"));
  ctx.clearRect(0, 0, W, H);
  const pad = 34, iw = W - pad - 12, ih = (H - pad - 12) / 2;
  axes(ctx, W, H);
  const years = 4, cpy = CYC_PER_YEAR();
  const band = [];
  for (let t = 0; t <= years; t += 0.1) {
    const N = t * cpy;
    const rets = cells.map(c => clamp(sohAt(N, c.crate, c.L0) * 100, 45, 100));
    band.push({ t, p10: pct(rets, 0.10), p50: pct(rets, 0.50), p90: pct(rets, 0.90) });
  }
  const X = t => pad + (t / years) * iw, Y = v => 8 + (1 - (v - 45) / 55) * ih;
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
  ctx.fillText("capacity retention % from the SOH model (band = P10–P90 across the batch)", pad + 4, 16);
  ctx.fillStyle = "#f87171"; ctx.fillText("80% end-of-life line", pad + iw - 108, Y(80) - 4);
  // bottom: empirical reliability = share of cells whose predicted life exceeds t
  const y0 = 8 + ih + 16, ih2 = H - pad - y0;
  ctx.strokeStyle = "#22314f";
  ctx.beginPath(); ctx.moveTo(pad, y0); ctx.lineTo(pad, y0 + ih2); ctx.lineTo(pad + iw, y0 + ih2); ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const t = (i / 60) * years, N = t * cpy;
    const R = cells.filter(c => c.life > N).length / cells.length;
    const px = X(t), py = y0 + (1 - R) * ih2;
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = "#8fa6c4";
  ctx.fillText("pack reliability R(t) — share of cells still above 80% (empirical, from predicted lives)", pad + 4, y0 + 10);
  ctx.fillText("years → (1 year = " + cpy + " cycles, stated assumption)", pad + 4, H - 10);
  return cells.filter(c => c.life > 2 * cpy).length / cells.length;
}

// ---------- segment lab (Kohtz et al.): GPR SOH from a partial charge segment ----------
function drawSegmentLab() {
  const ex = MODELS.segment_examples[+$("seg_sel").value];
  const pred = gprPredict(MODELS.segment_gpr, [ex.lr_t, ex.lr_q, ex.log_t, ex.T_C, ex.I_dis_A]);
  const err = Math.abs(pred - ex.soh_true);
  $("seg_out").textContent =
    `window 3.8–4.1 V took ${ex.t_win_s.toFixed(0)} s and ${ex.q_win_Ah.toFixed(2)} Ah of charge ` +
    `(stretch vs this cell's early-life reference: ${(Math.exp(ex.lr_t)).toFixed(2)}×) → ` +
    `GPR estimate SOH = ${pred.toFixed(3)} · measured SOH = ${ex.soh_true.toFixed(3)} · |error| = ${err.toFixed(3)}`;
}

// ---------- memory meter (live sensitivity sweep over the trained models) ----------
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
    const el = electrode(k);
    outs.tau.push(el.tau);
    outs.cap.push(capacityAt(100, k.crate, el));
    outs.rul.push(rulCycles(k.crate, l0Of(el)) / CYC_PER_YEAR());
  }
  const shares = {};
  for (const d of dims) {
    const xs = samples.map(s => s[d]);
    shares[d] = Object.fromEntries(Object.entries(outs).map(([stage, ys]) => {
      const r = pearson(xs, ys); return [stage, r * r];
    }));
  }
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

// ---------- inverse design: scan CD for a target relaxed porosity (electrode MLP) ----------
function recommendCD(k, targetEps) {
  let best = null;
  for (let cd = 20; cd <= 50.001; cd += 0.25) {
    const e = electrode({ ...k, cal: cd }).epsRelaxed;
    const err = Math.abs(e - targetEps);
    if (!best || err < best.err) best = { cd, eps: e, err };
  }
  return best;
}

// ---------- model notes (numbers injected from the training artifact) ----------
function fillNotes() {
  const M = MODELS.meta.metrics, meta = MODELS.meta;
  const f = (x, d = 3) => (+x).toFixed(d);
  $("modelnotes").textContent =
`TRAINED-MODEL EDITION. Every number on this page is computed live in your browser
by running the trained models below (weights in models.js, exported from the
training run — no server, no stored results). Metrics quoted here were measured
in that training run on held-out data; they are read from the training artifact.

ELECTRODE — neural network (MLP 5→64→32→7), SIMULATION-TRAINED.
  No open labeled dataset of (process settings -> porosity/tortuosity) was found
  (Zenodo searched Nov 2026: ARTISTIC/Franco records are papers, reports or
  multi-GB tomography without tabular labels). The net is therefore trained on a
  20,000-point grid generated by stated literature relations: CBD volume
  fraction from wt% and densities; dried porosity = packing base minus CBD pore
  filling (weak drying/adhesion/D50 terms are stated assumptions); binder
  migration rising with drying rate, falling with adhesion; compression by
  volume conservation; partial spring-back (assumed form, falling with
  adhesion/binder); tortuosity by the Bruggeman relation tau = eps^-0.5 with a
  migration factor. Holdout fit vs the simulator: R²(tau) = ${f(M.electrode_sim_fit.tau.R2)}, R²(eps) = ${f(M.electrode_sim_fit.epsRelaxed.R2)}
  — this measures fit to those relations, NOT to measured electrodes.
  Structure (CD control, time axis, spring-back, relaxed state downstream)
  follows Galvez-Aranda et al. 2024 (see reference below).

SOH / LIFE — multi-fidelity fusion of two neural networks, after Kohtz et al. 2022.
  TREND net (MLP 5→64→32→1) trained on a reduced-order SEI physics model —
  parabolic (diffusion-limited) SEI growth, loss = a(T,I)·(sqrt(L0+N) − sqrt(L0))
  + b(T)·N, Arrhenius in T, power laws in charge/discharge current, L0 = initial
  SEI offset (pre-age); ${M.soh.physics_rows} physics trajectories. DISCREPANCY net
  (MLP 3→32→16→1) trained on the residuals of NASA Ames PCoE battery aging data
  (sets 1+3; ${meta.nasa_cells_used} cells / ${meta.nasa_rows_used} discharge cycles after cleaning; 4 cells with
  multi-level discharge profiles excluded — their stored capacity reference
  breaks, SOH up to 1.7) against the trend net — the trend + discrepancy
  structure of co-kriging. The discrepancy net excludes charge current and L0
  (NASA charges are all 1.5 A; pre-age is unobserved), so those trends come
  from the physics layer. Validation of the FUSED model on 5 HELD-OUT NASA cells (${meta.test_cells.join(", ")}):
  SOH R² = ${f(M.soh.test_SOH_R2, 2)}, RMSE = ${f(M.soh.test_SOH_RMSE)}, MAE = ${f(M.soh.test_SOH_MAE)}. RUL (cycles to SOH 0.80) on held-out cells
  with a defined life: ${Object.entries(M.soh.RUL_by_cell).map(([c, r]) => `${c} true ${r.true} / pred ${r.pred}`).join("; ")}.
  Residual band: ±${f(M.soh.conformal_band90)} SOH (90th pct of train-cell residuals) covered
  ${(M.soh.conformal_coverage_test * 100).toFixed(0)}% of held-out points — honest under-coverage, reported as measured
  (conformal calibration in the spirit of the RaDA Lab's DACP work, not claimed exact).
  The twin's reference cell runs at 24 °C, 2 A discharge (NASA set-1 conditions);
  charge current = charge-rate knob × 2 A. Years assume ${meta.cycles_per_year} cycles/year (stated).

SEGMENT ESTIMATOR — Gaussian Process Regression, Kohtz et al.'s core method.
  From a PARTIAL charging segment only (the 3.8–4.1 V window of the CC charge):
  window duration and charged Ah, each as a log-ratio vs the same cell's
  early-life reference segment (cycles 3–8 median), plus T and discharge
  current → SOH. Trained on NASA train cells (${M.segment.train_rows} segments), validated on the
  held-out cells (${M.segment.test_rows} segments): R² = ${f(M.segment.test_R2, 2)}, RMSE = ${f(M.segment.test_RMSE)}. The lab above the meter runs
  this GPR live on real held-out NASA examples. (Kohtz et al. used modeled
  curves from a 1D FE SEI model as the reference family; here the reference is
  the cell's own early-life segment — a stated simplification.)

RESISTANCE — neural network (MLP 3→32→16→1) on NASA EIS measurements
  (Re + Rct, ${M.resistance.train_rows} train / ${M.resistance.test_rows} held-out rows): R² = ${f(M.resistance.test_R2_on_mOhm, 2)}, RMSE = ${f(M.resistance.test_RMSE_mOhm, 0)} mΩ on held-out cells.

STRUCTURAL LINKS (stated mappings, NOT learned): electrode transport eps/tau,
  normalized over the knob space, sets the cell's initial capacity
  Q0 = 1.80 + 0.25·q Ah and initial SEI offset L0 = (1 − q)·60 cycles. Pack
  spread = per-cell jitter of the process knobs, plus initial-capacity jitter
  (±1.5%) and initial-SEI jitter (σ = 12 cycles), pushed through the same
  models — identical-condition NASA cells B0005/6/7 lived 106/61/125 cycles,
  a spread the process knobs alone cannot produce.
  Discharge-curve SHAPES are schematic; their capacity extents come from the
  SOH model. Binder adhesion remains an input to the electrode stage only —
  no open dataset links it directly to cell aging.

REFERENCES
  Kohtz, S., Xu, Y., Zheng, Z. & Wang, P. (2022). Physics-informed machine
  learning model for battery state of health prognostics using partial
  charging segments. Mech. Syst. Signal Process. 172, 109002.
  DOI 10.1016/j.ymssp.2022.109002. (RaDA Lab, UT Dallas.)
  Galvez-Aranda et al. (2024). Time-Dependent Deep Learning Manufacturing
  Process Model for Battery Electrode Microstructure Prediction. Adv. Energy
  Mater. 14, 2400376. DOI 10.1002/aenm.202400376.
  NASA Ames Prognostics Center of Excellence, Battery Data Set (sets 1+3),
  NASA Prognostics Data Repository. SEI physics form: parabolic growth law as
  in the reduced-order SEI literature behind Kohtz et al.'s FE model.
  In the full proposed system the stand-ins are replaced by LAMMPS (MD),
  Yade/LIGGGHTS (DEM), FEniCSx + TauFactor (FEM/transport), PyBaMM (cell) and
  physics-informed neural operators exported to ONNX.`;
}

// ---------- main update ----------
function drawElectrodeTexts() {
  const k = curK, el = curEl, s = calSeries(k, el);
  $("s_eps").textContent = el.epsRelaxed.toFixed(3);
  $("s_tau").textContent = el.tau.toFixed(2);
  $("s_sb").textContent = (el.sbFrac * (k.cal / 100) * 100).toFixed(1) + "% of h";
  $("s_mig").textContent = (el.mig * 100).toFixed(0) + "%";
  $("s_sig").textContent = el.transport.toFixed(3);
  $("loadnote").textContent =
    `Under load the coating reached ε = ${s.epsLoad.toFixed(3)} at h = ${s.hLoad.toFixed(2)}; ` +
    `after spring-back it relaxed to ε = ${el.epsRelaxed.toFixed(3)} at h^cal = ${el.hCal.toFixed(2)} (electrode model). ` +
    `Structural link: this electrode's transport sets Q0 = ${q0Of(el).toFixed(2)} Ah and initial SEI offset L0 = ${l0Of(el).toFixed(0)} cycles in the life model.`;
  const rec = recommendCD(k, +$("teps").value);
  $("s_cdrec").textContent = `CD ≈ ${rec.cd.toFixed(1)}% gives relaxed ε = ${rec.eps.toFixed(3)} (scan of CD 20–50% through the electrode model, computed live)`;
}
function update() {
  const k = readKnobs();
  curK = k;
  fmtLabels(k);
  curEl = electrode(k);
  curCell = cellFrom(k, curEl);
  curCells = packSample(k);
  drawElectrode(k, curEl);
  drawSeries(k, curEl);
  drawCell(k, curCell);
  drawPack(curCells);
  const rel2 = drawLife(k, curCells);
  drawElectrodeTexts();
  const qs = curCells.map(c => c.cap10);
  const mean = qs.reduce((a, b) => a + b) / qs.length;
  const std = Math.sqrt(qs.reduce((a, b) => a + (b - mean) ** 2, 0) / qs.length);
  const lives = curCells.map(c => c.life / CYC_PER_YEAR());
  $("s_cap").textContent = curCell.Q100.toFixed(2) + " Ah";
  $("s_rate").textContent = curCell.Q500.toFixed(2) + " Ah";
  $("s_res").textContent = curCell.res.toFixed(0) + " mΩ";
  $("s_mean").textContent = mean.toFixed(2) + " Ah";
  $("s_std").textContent = std.toFixed(3) + " Ah";
  $("s_min").textContent = Math.min(...qs).toFixed(2) + " Ah";
  $("s_pack").textContent = (96 * pct(qs, 0.05)).toFixed(0) + " Ah";
  $("s_rul").textContent = pct(lives, 0.5).toFixed(1) + " yr";
  $("s_rulband").textContent = pct(lives, 0.10).toFixed(1) + " – " + pct(lives, 0.90).toFixed(1) + " yr";
  $("s_rel").textContent = (rel2 * 100).toFixed(0) + "%";
}
function redrawElectrodeOnly() {
  if (!curK) return;
  fmtLabels(curK);
  drawElectrode(curK, curEl);
  drawSeries(curK, curEl);
}
for (const id of ["adh", "bin", "dry", "cal", "psd", "crate", "teps"])
  $(id).addEventListener("input", update);
$("tstep").addEventListener("input", () => {
  stopPlay();
  tIdx = +$("tstep").value;
  redrawElectrodeOnly();
});
function stopPlay() {
  playing = false;
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  $("play").textContent = "▶ Play calendering";
}
$("play").addEventListener("click", () => {
  if (playing) { stopPlay(); return; }
  playing = true;
  $("play").textContent = "⏸ Pause";
  if (tIdx >= NSTEPS) tIdx = 0;
  playTimer = setInterval(() => {
    tIdx++;
    $("tstep").value = tIdx;
    redrawElectrodeOnly();
    if (tIdx >= NSTEPS) stopPlay();
  }, 130);
});
$("seg_sel").innerHTML = MODELS.segment_examples.map((ex, i) =>
  `<option value="${i}">${ex.cell} · cycle ${ex.cycle} · ${ex.T_C.toFixed(0)} °C · ${ex.I_dis_A.toFixed(0)} A discharge</option>`).join("");
$("seg_sel").addEventListener("change", drawSegmentLab);
window.addEventListener("resize", update);
fillNotes();
drawMeter();
update();
drawSegmentLab();
