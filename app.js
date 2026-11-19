/* MLP parameter-learning lab — 2-2-1, sigmoid, BCE, Adam */
(() => {
  'use strict';

  const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const Y = [0, 1, 1, 0];
  const EPOCHS = 300;
  const LR0 = 0.03;
  const KEYS = ['w00', 'w01', 'w10', 'w11', 'bh0', 'bh1', 'v0', 'v1', 'bo'];
  const ADAM = { b1: 0.9, b2: 0.999, eps: 1e-8 };
  const INIT = { w00: 2, w01: -2, w10: -2, w11: 2, bh0: -1, bh1: -1, v0: 3, v1: 3, bo: -2.5 };
  const REDUCE = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  function sig(z) {
    return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }
  function bce(z, y) {
    return Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
  }
  function zeros() {
    const o = {};
    for (const k of KEYS) o[k] = 0;
    return o;
  }

  function createModel() {
    return {
      p: Object.assign({}, INIT),
      epoch: 0,
      t: 0,
      loss: null,
      acc: 0,
      firstFull: null,
      preds: [0, 0, 0, 0],
      grads: zeros(),
      history: [],
      m: zeros(),
      s: zeros(),
    };
  }

  function forward(p, x) {
    const h0 = sig(x[0] * p.w00 + x[1] * p.w10 + p.bh0);
    const h1 = sig(x[0] * p.w01 + x[1] * p.w11 + p.bh1);
    const z = h0 * p.v0 + h1 * p.v1 + p.bo;
    return { h0, h1, z, p: sig(z) };
  }

  function computeBatch(model) {
    const p = model.p;
    const g = zeros();
    let loss = 0;
    let acc = 0;
    const preds = [];
    for (let i = 0; i < 4; i++) {
      const f = forward(p, X[i]);
      const y = Y[i];
      loss += bce(f.z, y) / 4;
      preds.push(f.p);
      if ((f.p >= 0.5) === (y === 1)) acc += 1;
      const d = (f.p - y) / 4;
      g.v0 += f.h0 * d;
      g.v1 += f.h1 * d;
      g.bo += d;
      const d0 = d * p.v0 * f.h0 * (1 - f.h0);
      const d1 = d * p.v1 * f.h1 * (1 - f.h1);
      g.w00 += X[i][0] * d0;
      g.bh0 += d0;
      g.w10 += X[i][1] * d0;
      g.w01 += X[i][0] * d1;
      g.bh1 += d1;
      g.w11 += X[i][1] * d1;
    }
    return { loss, preds, acc, grads: g };
  }

  function evaluate(model) {
    const b = computeBatch(model);
    model.loss = b.loss;
    model.preds = b.preds;
    model.acc = b.acc;
    model.grads = b.grads;
  }

  function trainStep(model, lr) {
    const b = computeBatch(model);
    model.t += 1;
    for (const k of KEYS) {
      model.m[k] = ADAM.b1 * model.m[k] + (1 - ADAM.b1) * b.grads[k];
      model.s[k] = ADAM.b2 * model.s[k] + (1 - ADAM.b2) * b.grads[k] * b.grads[k];
      const mh = model.m[k] / (1 - Math.pow(ADAM.b1, model.t));
      const sh = model.s[k] / (1 - Math.pow(ADAM.b2, model.t));
      model.p[k] -= lr * mh / (Math.sqrt(sh) + ADAM.eps);
    }
    model.epoch += 1;
    model.loss = b.loss;
    model.preds = b.preds;
    model.acc = b.acc;
    if (b.acc === 4 && model.firstFull === null) model.firstFull = model.epoch;
    model.grads = b.grads;
    model.history.push(b.loss);
  }

  /* ---------- canvases ---------- */

  const PAD = 0.16;
  const offs = new WeakMap();

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(rect.width);
    const h = Math.round(rect.height);
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function colorAt(p) {
    const t = clamp(Math.abs(p - 0.5) / 0.5, 0, 1);
    const a = 0.04 + 0.32 * t;
    const base = [250, 247, 238];
    const tint = p >= 0.5 ? [198, 67, 43] : [31, 91, 87];
    return [
      Math.round(base[0] * (1 - a) + tint[0] * a),
      Math.round(base[1] * (1 - a) + tint[1] * a),
      Math.round(base[2] * (1 - a) + tint[2] * a),
    ];
  }

  function drawSurface(canvas, params) {
    const s = setupCanvas(canvas);
    if (!s) return;
    const { ctx, w, h } = s;

    const gw = clamp(Math.round(w / 5.2), 64, 144);
    const gh = clamp(Math.round(gw * (h / w)), 48, 120);

    let store = offs.get(canvas);
    if (!store || store.width !== gw || store.height !== gh) {
      const oc = document.createElement('canvas');
      oc.width = gw;
      oc.height = gh;
      store = { canvas: oc, ctx: oc.getContext('2d'), image: oc.getContext('2d').createImageData(gw, gh), vals: new Float32Array(gw * gh), width: gw, height: gh };
      offs.set(canvas, store);
    }

    const data = store.image.data;
    const vals = store.vals;
    for (let j = 0; j < gh; j++) {
      const y = (1 + PAD) - (j / (gh - 1)) * (1 + PAD * 2);
      for (let i = 0; i < gw; i++) {
        const x = -PAD + (i / (gw - 1)) * (1 + PAD * 2);
        const f = forward(params, [x, y]);
        vals[j * gw + i] = f.p;
        const c = colorAt(f.p);
        const k = (j * gw + i) * 4;
        data[k] = c[0];
        data[k + 1] = c[1];
        data[k + 2] = c[2];
        data[k + 3] = 255;
      }
    }
    store.ctx.putImageData(store.image, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(store.canvas, 0, 0, w, h);

    // contour p = 0.5 (marching squares, pairs of crossings per cell)
    ctx.save();
    ctx.strokeStyle = 'rgba(21,24,26,.5)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    const sx = (gx) => (gx / (gw - 1)) * w;
    const sy = (gy) => (gy / (gh - 1)) * h;
    for (let j = 0; j < gh - 1; j++) {
      for (let i = 0; i < gw - 1; i++) {
        const a = vals[j * gw + i];
        const b = vals[j * gw + i + 1];
        const c = vals[(j + 1) * gw + i + 1];
        const d = vals[(j + 1) * gw + i];
        const pts = [];
        const cross = (v1, v2, x1, y1, x2, y2) => {
          if ((v1 - 0.5) * (v2 - 0.5) > 0) return;
          if (Math.abs(v2 - v1) < 1e-9) return;
          const t = (0.5 - v1) / (v2 - v1);
          pts.push([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]);
        };
        cross(a, b, i, j, i + 1, j);
        cross(b, c, i + 1, j, i + 1, j + 1);
        cross(d, c, i, j + 1, i + 1, j + 1);
        cross(a, d, i, j, i, j + 1);
        if (pts.length >= 2) {
          ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
          ctx.lineTo(sx(pts[1][0]), sy(pts[1][1]));
          if (pts.length === 4) {
            ctx.moveTo(sx(pts[2][0]), sy(pts[2][1]));
            ctx.lineTo(sx(pts[3][0]), sy(pts[3][1]));
          }
        }
      }
    }
    ctx.stroke();
    ctx.restore();

    // axes ticks 0/1
    ctx.save();
    ctx.fillStyle = 'rgba(107,109,100,.85)';
    ctx.font = `${Math.max(9, w / 62)}px ${getMono()}`;
    ctx.textAlign = 'center';
    const x0 = sx(((0 + PAD) / (1 + PAD * 2)) * (gw - 1));
    const x1 = sx(((1 + PAD) / (1 + PAD * 2)) * (gw - 1));
    ctx.fillText('0', x0, h - 5);
    ctx.fillText('1', x1, h - 5);
    ctx.textAlign = 'left';
    const y0 = sy((1 - (0 + PAD) / (1 + PAD * 2)) * (gh - 1));
    const y1 = sy((1 - (1 + PAD) / (1 + PAD * 2)) * (gh - 1));
    ctx.fillText('0', 4, y0 + 8);
    ctx.fillText('1', 4, y1 + 3);
    ctx.textAlign = 'right';
    ctx.fillText('x₁', w - 5, h - 5);
    ctx.textAlign = 'left';
    ctx.fillText('x₂', 4, 12);
    ctx.restore();

    drawPoints(ctx, w, h, (x, y) => [sx(((x + PAD) / (1 + PAD * 2)) * (gw - 1)), sy((1 - (y + PAD) / (1 + PAD * 2)) * (gh - 1))]);
  }

  function drawPoints(ctx, w, h, toPx) {
    const r = clamp(w * 0.017, 7, 12);
    const font = Math.max(9, r * 1.05);
    X.forEach((pt, i) => {
      const [px, py] = toPx(pt[0], pt[1]);
      const one = Y[i] === 1;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = one ? '#C6432B' : '#1F5B57';
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#FAF7EE';
      ctx.stroke();
      ctx.fillStyle = '#FAF7EE';
      ctx.font = `600 ${font}px ${getMono()}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(Y[i]), px, py + 0.5);
    });
  }

  function getMono() {
    return '"SFMono-Regular","JetBrains Mono",Consolas,monospace';
  }

  function drawHidden(canvas, params) {
    const s = setupCanvas(canvas);
    if (!s) return;
    const { ctx, w, h } = s;
    const padX = Math.max(46, w * 0.13);
    const padY = Math.max(40, h * 0.13);
    const x = (v) => padX + v * (w - padX - 26);
    const y = (v) => h - padY - v * (h - padY - 24);

    ctx.fillStyle = '#FAF7EE';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#E7E0D1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = 0; t <= 1; t += 0.25) {
      ctx.moveTo(x(0), y(t));
      ctx.lineTo(x(1), y(t));
      ctx.moveTo(x(t), y(0));
      ctx.lineTo(x(t), y(1));
    }
    ctx.stroke();

    ctx.strokeStyle = '#C7BEAA';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x(0), y(0));
    ctx.lineTo(x(1), y(0));
    ctx.moveTo(x(0), y(0));
    ctx.lineTo(x(0), y(1));
    ctx.stroke();

    ctx.fillStyle = '#6B6D64';
    ctx.font = `10px ${getMono()}`;
    ctx.textAlign = 'center';
    ctx.fillText('0', x(0), y(0) + 16);
    ctx.fillText('1', x(1), y(0) + 16);
    ctx.textAlign = 'right';
    ctx.fillText('0', x(0) - 8, y(0) + 4);
    ctx.fillText('1', x(0) - 8, y(1) + 4);
    ctx.textAlign = 'left';
    ctx.fillText('h₁', x(1) + 8, y(0) + 4);
    ctx.fillText('h₂', x(0) - 2, y(1) - 14);

    // output-layer decision line v0*h0 + v1*h1 + bo = 0
    ctx.save();
    ctx.beginPath();
    ctx.rect(x(0), y(1), x(1) - x(0), y(0) - y(1));
    ctx.clip();
    ctx.strokeStyle = '#15181A';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    if (Math.abs(params.v1) > 1e-7) {
      const yOf = (h0) => (-params.bo - params.v0 * h0) / params.v1;
      for (let i = 0; i <= 120; i++) {
        const h0 = -1.5 + (i / 120) * 4;
        const h1 = yOf(h0);
        if (h1 > 1.6 || h1 < -1.6) { started = false; continue; }
        if (!started) { ctx.moveTo(x(h0), y(h1)); started = true; }
        else ctx.lineTo(x(h0), y(h1));
      }
    } else if (Math.abs(params.v0) > 1e-7) {
      const h0 = -params.bo / params.v0;
      ctx.moveTo(x(h0), y(-0.5));
      ctx.lineTo(x(h0), y(1.5));
    }
    ctx.stroke();
    ctx.restore();

    const r = clamp(w * 0.022, 7, 11);
    const offsets = [[-14, -12], [12, -10], [12, 14], [-8, 18]];
    X.forEach((pt, i) => {
      const f = forward(params, pt);
      const px = x(f.h0);
      const py = y(f.h1);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = Y[i] === 1 ? '#C6432B' : '#1F5B57';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#FAF7EE';
      ctx.stroke();
      ctx.fillStyle = '#15181A';
      ctx.font = `10px ${getMono()}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${pt[0]}${pt[1]}`, px + offsets[i][0], py + offsets[i][1]);
    });
  }

  function drawLoss(canvas, model) {
    const s = setupCanvas(canvas);
    if (!s) return;
    const { ctx, w, h } = s;
    const L = 40;
    const R = 16;
    const T = 12;
    const B = 34;
    const maxY = Math.max(0.75, Math.max.apply(null, model.history.concat([0])) * 1.05);
    const x = (i) => L + (i / EPOCHS) * (w - L - R);
    const y = (v) => h - B - (v / maxY) * (h - B - T);

    ctx.strokeStyle = '#E7E0D1';
    ctx.fillStyle = '#8A8B80';
    ctx.font = `9.5px ${getMono()}`;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    for (let k = 0; k <= 3; k++) {
      const val = (maxY * k) / 3;
      const py = y(val);
      ctx.beginPath();
      ctx.moveTo(L, py);
      ctx.lineTo(w - R, py);
      ctx.stroke();
      ctx.fillText(val.toFixed(2), L - 6, py + 3);
    }
    ctx.textAlign = 'center';
    [0, 150, 300].forEach((e) => {
      ctx.textAlign = e === 0 ? 'left' : e === 300 ? 'right' : 'center';
      ctx.fillText(String(e), x(e), h - 19);
    });
    ctx.textAlign = 'center';
    ctx.fillStyle = '#B3B4A8';
    ctx.fillText('epoch', (L + w - R) / 2, h - 5);

    if (model.history.length > 1) {
      ctx.beginPath();
      model.history.forEach((v, i) => {
        const px = x(i);
        const py = y(v);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = '#C6432B';
      ctx.lineWidth = 1.8;
      ctx.lineJoin = 'round';
      ctx.stroke();

      const lastX = x(model.history.length - 1);
      const lastY = y(model.history[model.history.length - 1]);
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#C6432B';
      ctx.fill();
    }
  }

  /* ---------- linear perceptron demo ---------- */

  const lin = { w1: 1, w2: 1, b: -0.5 };
  const linCanvas = $('#linCanvas');
  const linScore = $('#linScore');
  const linNote = $('#linNote');

  function drawLinear() {
    const s = setupCanvas(linCanvas);
    if (!s) return;
    const { ctx, w, h } = s;
    const pad = 0.16;
    const sx = (v) => ((v + pad) / (1 + pad * 2)) * w;
    const sy = (v) => (1 - (v + pad) / (1 + pad * 2)) * h;

    const gw = clamp(Math.round(w / 8), 48, 110);
    const gh = clamp(Math.round(gw * (h / w)), 36, 90);
    if (!lin.store || lin.store.width !== gw || lin.store.height !== gh) {
      const oc = document.createElement('canvas');
      oc.width = gw;
      oc.height = gh;
      lin.store = { width: gw, height: gh, canvas: oc, ctx: oc.getContext('2d'), image: oc.getContext('2d').createImageData(gw, gh) };
    }
    const img = lin.store.image;
    const data = img.data;
    const posC = [Math.round(250 * 0.9 + 198 * 0.1), Math.round(247 * 0.9 + 67 * 0.1), Math.round(238 * 0.9 + 43 * 0.1)];
    const negC = [Math.round(250 * 0.92 + 31 * 0.08), Math.round(247 * 0.92 + 91 * 0.08), Math.round(238 * 0.92 + 87 * 0.08)];
    for (let j = 0; j < gh; j++) {
      const x2 = (1 - j / (gh - 1)) * (1 + pad * 2) - pad;
      for (let i = 0; i < gw; i++) {
        const x1 = (i / (gw - 1)) * (1 + pad * 2) - pad;
        const c = lin.w1 * x1 + lin.w2 * x2 + lin.b >= 0 ? posC : negC;
        const k = (j * gw + i) * 4;
        data[k] = c[0];
        data[k + 1] = c[1];
        data[k + 2] = c[2];
        data[k + 3] = 255;
      }
    }
    lin.store.ctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(lin.store.canvas, 0, 0, w, h);

    // boundary w1*x + w2*y + b = 0
    ctx.strokeStyle = '#15181A';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    if (Math.abs(lin.w2) > 1e-6) {
      const yOf = (x1) => -(lin.w1 * x1 + lin.b) / lin.w2;
      const xa = -0.6;
      const xb = 1.6;
      let started = false;
      for (let i = 0; i <= 40; i++) {
        const x1 = xa + ((xb - xa) * i) / 40;
        const x2 = yOf(x1);
        if (x2 < -1.2 || x2 > 2.2) { started = false; continue; }
        if (!started) { ctx.moveTo(sx(x1), sy(x2)); started = true; }
        else ctx.lineTo(sx(x1), sy(x2));
      }
    } else if (Math.abs(lin.w1) > 1e-6) {
      const x0 = -lin.b / lin.w1;
      ctx.moveTo(sx(x0), sy(-0.5));
      ctx.lineTo(sx(x0), sy(1.5));
    }
    ctx.stroke();

    ctx.strokeStyle = 'rgba(21,24,26,.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx(0), sy(0));
    ctx.lineTo(sx(1), sy(0));
    ctx.moveTo(sx(0), sy(0));
    ctx.lineTo(sx(0), sy(1));
    ctx.stroke();

    X.forEach((pt, i) => {
      const px = sx(pt[0]);
      const py = sy(pt[1]);
      const r = clamp(w * 0.018, 8, 12);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = Y[i] === 1 ? '#C6432B' : '#1F5B57';
      ctx.fill();
      ctx.lineWidth = 2.4;
      ctx.strokeStyle = '#FAF7EE';
      ctx.stroke();
      ctx.fillStyle = '#FAF7EE';
      ctx.font = `600 ${Math.max(9, r)}px ${getMono()}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(Y[i]), px, py + 0.5);
    });

    let score = 0;
    for (let i = 0; i < 4; i++) {
      const pred = lin.w1 * X[i][0] + lin.w2 * X[i][1] + lin.b >= 0 ? 1 : 0;
      if (pred === Y[i]) score += 1;
    }
    linScore.textContent = `${score} / 4`;
    linNote.textContent = `判定式 z = w₁x₁ + w₂x₂ + b；z ≥ 0 输出 1。当前 ${score} / 4，直线方案的上限是 3 / 4。`;
  }

  ['w1', 'w2', 'bias'].forEach((id) => {
    const el = document.getElementById(id);
    const out = document.getElementById(id + 'v');
    el.addEventListener('input', () => {
      lin[id === 'bias' ? 'b' : id] = parseFloat(el.value);
      out.textContent = parseFloat(el.value).toFixed(1).replace('-', '−');
      drawLinear();
    });
  });

  /* ---------- the lab ---------- */

  const lab = {
    model: createModel(),
    running: false,
    raf: 0,
    lr: LR0,
    sample: 1,
    lastPaint: 0,
  };
  evaluate(lab.model);
  lab.model.history = [lab.model.loss];

  const labCanvas = $('#labCanvas');
  const heroCanvas = $('#heroCanvas');
  const hiddenCanvas = $('#hiddenCanvas');
  const lossCanvas = $('#lossCanvas');
  const net = $('#net');
  const playLabel = $('#playLabel');
  const btnPlay = $('#btnPlay');
  const btnStep = $('#btnStep');
  const btnReset = $('#btnReset');
  const lrInput = $('#lr');
  const lrValue = $('#lrValue');
  const epochText = $('#epochText');
  const epochBar = $('#epochBar');
  const labEpoch = $('#labEpoch');
  const lossNow = $('#lossNow');
  const accNow = $('#accNow');

  const fmt = (v, d = 2) => v.toFixed(d).replace('-', '−');

  function updateParams() {
    const g = lab.model.grads;
    let maxAbs = 0;
    for (const k of KEYS) maxAbs = Math.max(maxAbs, Math.abs(g[k]));
    maxAbs = maxAbs || 1;
    for (const k of KEYS) {
      const v = document.querySelector(`[data-p="${k}"]`);
      const bar = document.querySelector(`[data-g="${k}"]`);
      if (v) v.textContent = fmt(lab.model.p[k], 2);
      if (bar) {
        bar.style.width = `${clamp((Math.abs(g[k]) / maxAbs) * 100, 0, 100)}%`;
        bar.classList.toggle('neg', g[k] < 0);
      }
    }
  }

  function updateReadout() {
    const i = lab.sample;
    const f = forward(lab.model.p, X[i]);
    $('#roH0').textContent = fmt(f.h0, 2);
    $('#roH1').textContent = fmt(f.h1, 2);
    $('#roZ').textContent = fmt(f.z, 2);
    $('#roP').textContent = fmt(f.p, 2);
    $('#roSample').textContent = `x₁=${X[i][0]}, x₂=${X[i][1]}`;
    $('#roY').textContent = String(Y[i]);
    document.querySelectorAll('[data-in="0"]').forEach((el) => { el.textContent = String(X[i][0]); });
    document.querySelectorAll('[data-in="1"]').forEach((el) => { el.textContent = String(X[i][1]); });
  }

  function updateNetwork() {
    const f = forward(lab.model.p, X[lab.sample]);
    document.querySelector('[data-h="0"]').textContent = fmt(f.h0, 2);
    document.querySelector('[data-h="1"]').textContent = fmt(f.h1, 2);
    document.querySelector('[data-out]').textContent = fmt(f.p, 2);

    const hidden = net.querySelectorAll('.node--h');
    hidden[0].style.fill = `rgba(31,91,87,${0.05 + 0.4 * f.h0})`;
    hidden[1].style.fill = `rgba(31,91,87,${0.05 + 0.4 * f.h1})`;
    net.querySelector('.node--out').style.fill = `rgba(198,67,43,${0.05 + 0.4 * f.p})`;

    net.querySelectorAll('line[data-w]').forEach((line) => {
      const k = line.getAttribute('data-w');
      const v = lab.model.p[k];
      const t = clamp(Math.abs(v) / 9, 0, 1);
      line.style.stroke = v >= 0 ? '#C6432B' : '#1F5B57';
      line.style.strokeWidth = `${1 + 4.4 * t}`;
      line.style.opacity = `${0.28 + 0.62 * t}`;
    });
    net.querySelectorAll('text[data-wl]').forEach((t) => {
      const k = t.getAttribute('data-wl');
      const v = lab.model.p[k];
      t.textContent = v.toFixed(1).replace('-', '−');
      t.style.opacity = `${0.3 + 0.7 * clamp(Math.abs(v) / 8, 0, 1)}`;
    });
  }

  function updateProgress() {
    const e = lab.model.epoch;
    epochText.textContent = `${e} / ${EPOCHS}`;
    epochBar.style.width = `${(e / EPOCHS) * 100}%`;
    labEpoch.textContent = `EPOCH ${String(e).padStart(3, '0')}`;
    lossNow.textContent = fmt(lab.model.loss, 3);
    accNow.textContent = `${lab.model.acc} / 4`;
    const first = document.getElementById('firstFull');
    if (first) first.textContent = lab.model.firstFull === null ? '—' : String(lab.model.firstFull);
  }

  function paintLab(force) {
    drawSurface(labCanvas, lab.model.p);
    drawLoss(lossCanvas, lab.model);
    updateNetwork();
    updateReadout();
    updateParams();
    updateProgress();
  }

  function setPlayState() {
    const done = lab.model.epoch >= EPOCHS;
    if (lab.running) {
      playLabel.textContent = '暂停';
      btnPlay.setAttribute('aria-label', '暂停训练');
    } else if (done) {
      playLabel.textContent = '再跑 300 轮';
      btnPlay.setAttribute('aria-label', '重置并重新训练');
    } else {
      playLabel.textContent = lab.model.epoch === 0 ? '播放训练' : '继续训练';
      btnPlay.setAttribute('aria-label', lab.model.epoch === 0 ? '播放训练' : '继续训练');
    }
    btnStep.disabled = lab.running;
    btnStep.style.opacity = lab.running ? '.45' : '1';
  }

  function loop(ts) {
    if (!lab.running) return;
    for (let i = 0; i < 2 && lab.model.epoch < EPOCHS; i++) trainStep(lab.model, lab.lr);
    updateProgress();
    if (ts - lab.lastPaint > 70 || lab.model.epoch >= EPOCHS) {
      paintLab();
      lab.lastPaint = ts;
    }
    if (lab.model.epoch >= EPOCHS) {
      lab.running = false;
      paintLab();
      setPlayState();
      return;
    }
    lab.raf = requestAnimationFrame(loop);
  }

  function resetLab() {
    cancelAnimationFrame(lab.raf);
    lab.running = false;
    lab.model = createModel();
    evaluate(lab.model);
    lab.model.history = [lab.model.loss];
    lab.lastPaint = 0;
    paintLab();
    setPlayState();
  }

  btnPlay.addEventListener('click', () => {
    if (lab.running) {
      lab.running = false;
      cancelAnimationFrame(lab.raf);
      setPlayState();
      return;
    }
    if (lab.model.epoch >= EPOCHS) resetLab();
    lab.running = true;
    setPlayState();
    lab.lastPaint = 0;
    lab.raf = requestAnimationFrame(loop);
  });

  btnStep.addEventListener('click', () => {
    if (lab.running) return;
    if (lab.model.epoch >= EPOCHS) return;
    trainStep(lab.model, lab.lr);
    paintLab();
    setPlayState();
  });

  btnReset.addEventListener('click', resetLab);

  lrInput.addEventListener('input', () => {
    lab.lr = parseFloat(lrInput.value);
    lrValue.textContent = lab.lr.toFixed(3);
  });

  $$('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      lab.sample = parseInt(chip.getAttribute('data-i'), 10);
      $$('.chip').forEach((c) => {
        const on = c === chip;
        c.classList.toggle('is-on', on);
        c.setAttribute('aria-pressed', String(on));
      });
      updateNetwork();
      updateReadout();
    });
  });

  /* ---------- hero: one training animation ---------- */

  let heroModel = null;

  function runHero() {
    const model = createModel();
    heroModel = model;
    evaluate(model);
    const epochEl = $('#heroEpoch');
    const lossEl = $('#heroLoss');
    const accEl = $('#heroAcc');

    const paint = () => {
      epochEl.textContent = String(model.epoch).padStart(3, '0');
      lossEl.textContent = fmt(model.loss, 3);
      accEl.textContent = `${model.acc} / 4`;
      drawSurface(heroCanvas, model.p);
    };

    if (REDUCE) {
      while (model.epoch < EPOCHS) trainStep(model, LR0);
      paint();
      return;
    }

    let start = 0;
    let last = 0;
    const DURATION = 1900;
    function frame(ts) {
      if (!start) start = ts;
      const target = Math.min(EPOCHS, Math.floor(((ts - start) / DURATION) * EPOCHS));
      while (model.epoch < target) trainStep(model, LR0);
      if (ts - last > 66 || model.epoch >= EPOCHS) {
        paint();
        last = ts;
      }
      if (model.epoch < EPOCHS) requestAnimationFrame(frame);
      else paint();
    }
    requestAnimationFrame(frame);
  }

  /* ---------- hidden space, frozen after 300 epochs ---------- */

  let frozenModel = null;

  function drawFrozen() {
    const model = createModel();
    while (model.epoch < EPOCHS) trainStep(model, LR0);
    frozenModel = model;
    drawHidden(hiddenCanvas, model.p);
    const rows = $$('#hiddenTable tr');
    X.forEach((pt, i) => {
      const f = forward(model.p, pt);
      const cells = rows[i].children;
      cells[1].textContent = fmt(f.h0, 2);
      cells[2].textContent = fmt(f.h1, 2);
    });
  }

  /* ---------- resize ---------- */

  let rq = 0;
  function repaintAll() {
    cancelAnimationFrame(rq);
    rq = requestAnimationFrame(() => {
      drawLinear();
      drawSurface(labCanvas, lab.model.p);
      drawLoss(lossCanvas, lab.model);
      if (frozenModel) drawHidden(hiddenCanvas, frozenModel.p);
      if (heroModel) drawSurface(heroCanvas, heroModel.p);
    });
  }

  if ('ResizeObserver' in window) {
    const ro = new ResizeObserver(repaintAll);
    [linCanvas, labCanvas, heroCanvas, hiddenCanvas, lossCanvas].forEach((c) => ro.observe(c));
  } else {
    window.addEventListener('resize', repaintAll);
  }

  /* ---------- boot ---------- */

  drawLinear();
  paintLab();
  drawFrozen();
  runHero();
  setPlayState();
})();
