/* Ring-style event timeline: speed curves, event pills (impact / brake / detection), 3D moments, draggable playhead. */
import { $, el, clamp, rgba } from './util.js';

export class Timeline {
  constructor(onSeek, onMoment) {
    this.root = $('#tl'); this.cv = $('#tlCanvas'); this.ctx = this.cv.getContext('2d');
    this.bar = $('#tlBar'); this.played = $('#tlPlayed'); this.head = $('#tlHead'); this.pills = $('#tlPills');
    this.onSeek = onSeek; this.onMoment = onMoment; this.case = null;
    const seek = e => { const r = this.bar.getBoundingClientRect(); this.onSeek(clamp((e.clientX - r.left) / r.width, 0, 1) * this.dur); };
    let drag = false;
    for (const t of [this.cv, this.bar]) t.addEventListener('pointerdown', e => { drag = true; seek(e); t.setPointerCapture(e.pointerId); });
    addEventListener('pointermove', e => drag && seek(e));
    addEventListener('pointerup', () => drag = false);
    new ResizeObserver(() => this.resize()).observe(this.root);
  }

  load(c) {
    this.case = c; this.dur = c.duration;

    this.pills.innerHTML = '';
    for (const e of c.events) {
      const p = el('div', { class: 'pill ' + e.kind, title: `${e.t.toFixed(2)}s · ${e.label}`, style: { left: (e.t / this.dur) * 100 + '%' },
        onclick: () => this.onSeek(Math.max(0, e.kind === 'impact' ? e.t - 0.5 : e.t)) },
        e.kind === 'scene' ? '' : e.label.slice(0, 22));
      if (e.kind === 'impact') p.style.zIndex = 3;
      this.pills.append(p);
    }
    for (const m of c.moments || []) {                      // the key instants frozen in 3D: open one to turn around it
      this.pills.append(el('button', { class: 'moment', title: `${m.t.toFixed(2)}s · ${m.label} · 3D`,
        style: { left: (m.t / this.dur) * 100 + '%' }, onclick: e => { e.stopPropagation(); this.onMoment(m); } }));
    }
    this.resize();
  }

  /** Lift labelled pills that would overlap their left neighbour onto a second row. */
  stack() {
    let right = -1e9;
    for (const p of [...this.pills.children].filter(p => !p.classList.contains('scene')).sort((a, b) => a.offsetLeft - b.offsetLeft)) {
      p.classList.remove('up');
      if (p.offsetLeft < right + 6 && !p.classList.contains('impact')) p.classList.add('up');
      else right = p.offsetLeft + p.offsetWidth;
    }
  }

  resize() {
    const r = this.cv.getBoundingClientRect(), d = devicePixelRatio || 1;
    this.cv.width = r.width * d; this.cv.height = r.height * d; this.ctx.setTransform(d, 0, 0, d, 0, 0);
    this.w = r.width; this.h = r.height;
    requestAnimationFrame(() => this.stack());
  }

  draw(S) {
    if (!this.case || !this.w) return;
    const c = this.ctx, w = this.w, h = this.h, x = t => (t / this.dur) * w;
    c.clearRect(0, 0, w, h);
    // speed curves of the involved parties, scaled to the max speed
    const series = this.case.series?.objects || {};
    const inv = this.case.objects.filter(o => o.role !== 'other');
    let vmax = 10;
    const edge = s => s.t.length > 10 ? 3 : 0;                 // drop Savitzky-Golay edge samples
    for (const o of inv) { const s = series[o.id]; if (s) for (const v of s.speed_kmh.slice(edge(s), s.t.length - edge(s))) vmax = Math.max(vmax, v); }
    for (const o of inv) {
      const s = series[o.id]; if (!s) continue;
      const e = edge(s);
      c.beginPath();
      s.t.forEach((t, i) => { if (i < e || i >= s.t.length - e) return; const px = x(t), py = h - 6 - (s.speed_kmh[i] / vmax) * (h - 22); i > e ? c.lineTo(px, py) : c.moveTo(px, py); });
      c.strokeStyle = rgba(o.color, .75); c.lineWidth = 1.5; c.stroke();
      const i = s.t.findIndex(t => t >= S.t);
      if (i >= e && i < s.t.length - e) {
        const py = h - 6 - (s.speed_kmh[i] / vmax) * (h - 22);
        c.fillStyle = o.color; c.beginPath(); c.arc(x(s.t[i]), py, 2.5, 0, 7); c.fill();
      }
    }
    c.strokeStyle = 'rgba(230,61,92,.6)'; c.setLineDash([3, 3]);
    c.beginPath(); c.moveTo(x(this.case.impact_t), 0); c.lineTo(x(this.case.impact_t), h); c.stroke(); c.setLineDash([]);
    c.fillStyle = 'rgba(255,255,255,.55)';
    for (let t = 0; t <= this.dur; t += 1) c.fillRect(x(t), h - 3, 1, 3);
    const f = S.t / this.dur * 100;
    this.played.style.width = f + '%'; this.head.style.left = f + '%';
  }
}
