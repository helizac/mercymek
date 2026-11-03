export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const lerp = (a, b, t) => a + (b - a) * t;

export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (k === 'html') n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}

export const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const fmtT = t => (t < 0 ? '−' : '') + Math.abs(t).toFixed(2) + 's';
export const fmtN = (v, d = 1) => (v == null || isNaN(v)) ? '—' : Number(v).toFixed(d);
export const hexRgb = h => { const x = parseInt(h.slice(1), 16); return [(x >> 16) & 255, (x >> 8) & 255, x & 255]; };
export const rgba = (h, a) => { const [r, g, b] = hexRgb(h); return `rgba(${r},${g},${b},${a})`; };
export const prettyBytes = b => b > 1e6 ? (b / 1e6).toFixed(1) + ' MB' : b > 1e3 ? (b / 1e3).toFixed(0) + ' KB' : b + ' B';

export class Emitter {
  constructor() { this.h = {}; }
  on(t, f) { (this.h[t] ||= []).push(f); return this; }
  emit(t, d) { (this.h[t] || []).forEach(f => { try { f(d); } catch (e) { console.error(t, e); } }); }
}

/** Index into a sorted array of times; returns the nearest index. */
export function nearestIdx(ts, t) {
  let lo = 0, hi = ts.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; ts[m] < t ? (lo = m) : (hi = m); }
  return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
}

/** Linear interpolation of an [N][k] array sampled at sorted times ts. */
export function interp(ts, rows, t) {
  if (!ts.length) return null;
  if (t <= ts[0]) return rows[0];
  if (t >= ts[ts.length - 1]) return rows[rows.length - 1];
  let i = nearestIdx(ts, t); if (ts[i] > t) i--;
  const a = ts[i], b = ts[i + 1], u = (t - a) / (b - a || 1);
  return rows[i].map((v, j) => v + (rows[i + 1][j] - v) * u);
}
