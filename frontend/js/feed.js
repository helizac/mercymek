/* Evidence camera: redacted dashcam video + SAM 3.1 mask overlays + motion trails + AR labels + zoom. */
import { $, el, clamp, rgba, lerp } from './util.js';
import { VideoFrameClock } from './video-frame-clock.js';

const VEHICLE = /car|truck|bus|motorcycle|van|suv|hatchback|sedan|pickup|vehicle/i;
const HIDDEN = new Set(['license plate', 'face']);

export class Feed {
  constructor() {
    this.video = $('#video'); this.cv = $('#overlay'); this.ctx = this.cv.getContext('2d');
    this.clock = new VideoFrameClock(this.video);
    this.videoHome = this.video.parentElement;
    this.videoNext = this.video.nextSibling;
    this.stage = $('#camStage'); this.frame = $('#camFrame'); this.arLayer = $('#arLayer');
    this.ov = {}; this.case = null; this.live = null; this.ar = {}; this.zoomNow = { s: 1, x: 0, y: 0 };
    this.verdict = null;
    new ResizeObserver(() => this.resize()).observe(this.frame);
  }

  async load(c) {
    const generation = this.loadGeneration = (this.loadGeneration || 0) + 1;
    this.case = c; this.ov = {}; this.live = null;
    this.W = c.meta.width; this.H = c.meta.height; this.fps = c.fps;
    this.objs = Object.fromEntries(c.objects.map(o => [o.id, o]));
    this.hide = new Set([...(c.hide_concepts || []), ...HIDDEN]);
    this.arLayer.innerHTML = ''; this.ar = {};
    this.video.src = c.urls.video; this.video.load();
    this.clock.reset(c.meta.src_fps || c.fps);
    $('#camId').textContent = c.meta.source_clip.replace(/.(mp4|webm)$/, '').slice(0, 24);
    const overlays = c.urls.overlays ? await (await fetch(c.urls.overlays)).json() : {};
    if (generation !== this.loadGeneration) return;
    this.ov = overlays;
    this.paths = this.buildPaths();
    this.resize();
  }

  /** Per-object 2D paths (box bottom-centre) for trails/trajectory: key -> [[t, x, y], ...]. */
  buildPaths() {
    const out = {};
    const frames = Object.keys(this.ov).map(Number).sort((a, b) => a - b);
    for (const f of frames) for (const [k, o] of Object.entries(this.ov[f])) {
      if (!this.objs[k]) continue;
      (out[k] ||= []).push([f / this.fps, (o.box[0] + o.box[2]) / 2, o.box[3]]);
    }
    return out;
  }

  resize() {
    const r = this.frame.getBoundingClientRect(), d = devicePixelRatio || 1;
    this.cv.width = r.width * d; this.cv.height = r.height * d; this.ctx.setTransform(d, 0, 0, d, 0, 0);
    this.cw = r.width; this.ch = r.height;
    if (this.W) { this.s = Math.min(r.width / this.W, r.height / this.H); this.ox = (r.width - this.W * this.s) / 2; this.oy = (r.height - this.H * this.s) / 2; }
  }

  P(x, y) { return [this.ox + x * this.s, this.oy + y * this.s]; }

  frameObjs(t) {
    const f = Math.round(t * this.fps);
    for (const d of [0, -1, 1, -2, 2]) if (this.ov[f + d]) return this.ov[f + d];
    return {};
  }

  syncVideo(S) {
    const v = this.video;
    if (!v.duration) return;
    v.playbackRate = S.speed;
    if (S.playing) {
      if (v.paused) v.play().catch(() => {});
      if (Math.abs(v.currentTime - S.t) > 0.35) v.currentTime = S.t;
    } else {
      if (!v.paused) v.pause();
      if (Math.abs(v.currentTime - S.t) > 0.02) v.currentTime = S.t;
    }
  }

  presentInScene(scene) {
    // rVFC needs a composited video surface, including while seeking a paused clip.
    // Keep the SAME decoder in the scene's comparison window, rather than a hidden second player.
    if(scene){
      if(this.video.parentElement!==scene)scene.append(this.video);
      this.video.classList.add('scene-source-video');
    }else{
      if(this.video.parentElement!==this.videoHome)this.videoHome.insertBefore(this.video,this.videoNext);
      this.video.classList.remove('scene-source-video');
    }
  }

  poly(p) {
    const c = this.ctx; c.beginPath();
    for (let i = 0; i < p.length; i += 2) { const [x, y] = this.P(p[i], p[i + 1]); i ? c.lineTo(x, y) : c.moveTo(x, y); }
    c.closePath();
  }

  draw(S) {
    if (!this.case || !this.s) return;
    const c = this.ctx, objs = this.frameObjs(S.t);
    c.clearRect(0, 0, this.cw, this.ch);
    const iso = new Set(S.isolate || []);
    const isoMode = iso.size > 0;
    const keys = Object.keys(objs).filter(k => !this.hide.has(objs[k].concept));
    // isolation veil: darken everything except the isolated masks
    if (isoMode && S.layers.masks) {
      c.save(); c.fillStyle = 'rgba(0,0,0,.62)'; c.fillRect(0, 0, this.cw, this.ch);
      c.globalCompositeOperation = 'destination-out';
      for (const k of keys) if (iso.has(k)) for (const p of objs[k].poly) { this.poly(p); c.fill(); }
      c.restore();
    }
    if (S.layers.masks) {
      for (const k of keys) {
        const o = objs[k], meta = this.objs[k];
        const isV = VEHICLE.test(o.concept);
        const strong = iso.has(k) || (!isoMode && meta && meta.role !== 'other');
        if (isoMode && !iso.has(k)) continue;
        if (!strong && !isV && !/traffic light/.test(o.concept)) continue;
        const col = meta?.color || o.color;
        c.lineWidth = strong ? 2 : 1;
        c.strokeStyle = rgba(col, strong ? .95 : .45);
        c.fillStyle = rgba(col, strong ? .26 : .08);
        if (strong) { c.shadowColor = col; c.shadowBlur = 14; }
        for (const p of o.poly) { this.poly(p); c.fill(); c.stroke(); }
        c.shadowBlur = 0;
        if (strong || S.selected === k) this.cornerBox(o.box, col);
      }
      if (iso.has('ego')) this.egoHighlight();
    }
    // trails (last 2.5 s) and full trajectories
    const trailKeys = new Set([...(S.trajectory || []), ...(this.case.involved || []).filter(k => k !== 'ego'), ...iso]);
    if (S.layers.tracks) for (const k of trailKeys) {
      const path = this.paths[k]; if (!path) continue;
      const col = this.objs[k]?.color || '#fff', full = (S.trajectory || []).includes(k);
      c.lineWidth = full ? 2.5 : 2; c.lineCap = 'round';
      let prev = null;
      for (const [t, x, y] of path) {
        if (!full && (t > S.t || t < S.t - 2.5)) { prev = null; continue; }
        const [px, py] = this.P(x, y);
        if (prev) {
          const a = full ? (t <= S.t ? .9 : .35) : clamp(1 - (S.t - t) / 2.5, 0, 1) * .9;
          c.strokeStyle = rgba(col, a); c.setLineDash(full && t > S.t ? [5, 6] : []);
          c.beginPath(); c.moveTo(prev[0], prev[1]); c.lineTo(px, py); c.stroke();
        }
        prev = [px, py];
      }
      c.setLineDash([]);
    }
    // live SAM result on the paused frame
    if (this.live && Math.abs(S.t - this.live.t) < 0.4) {
      c.save(); c.fillStyle = 'rgba(0,0,0,.45)'; c.fillRect(0, 0, this.cw, this.ch);
      c.globalCompositeOperation = 'destination-out';
      for (const o of this.live.objects) for (const p of o.poly) { this.poly(p); c.fill(); }
      c.restore();
      for (const o of this.live.objects) {
        c.strokeStyle = this.live.color; c.fillStyle = rgba(this.live.color, .25); c.lineWidth = 2; c.shadowColor = this.live.color; c.shadowBlur = 16;
        for (const p of o.poly) { this.poly(p); c.fill(); c.stroke(); }
        c.shadowBlur = 0; this.cornerBox(o.box, this.live.color);
        const [x, y] = this.P(o.box[0], o.box[1]);
        c.font = '600 11px Inter, sans-serif'; c.fillStyle = '#fff';
        c.fillText(`LIVE · SAM 3.1 · ${this.live.concept}${o.confidence != null ? ' · ' + (o.confidence * 100).toFixed(0) + '%' : ''}`, x, y - 8);
      }
    }
    this.drawAR(S, objs);
    this.applyZoom(S, objs);
  }

  cornerBox(b, col) {
    const c = this.ctx, [x1, y1] = this.P(b[0], b[1]), [x2, y2] = this.P(b[2], b[3]), L = Math.min(14, (x2 - x1) / 3, (y2 - y1) / 3);
    c.strokeStyle = col; c.lineWidth = 2; c.beginPath();
    for (const [x, y, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
      c.moveTo(x + dx * L, y); c.lineTo(x, y); c.lineTo(x, y + dy * L);
    }
    c.stroke();
  }

  egoHighlight() {
    const c = this.ctx, [x1, y1] = this.P(this.W * .12, this.H * .86), [x2, y2] = this.P(this.W * .88, this.H);
    const col = this.objs.ego?.color || '#7fd1c7';
    const g = c.createLinearGradient(0, y1, 0, y2); g.addColorStop(0, rgba(col, 0)); g.addColorStop(1, rgba(col, .35));
    c.fillStyle = g; c.fillRect(x1, y1, x2 - x1, y2 - y1);
  }

  /** Evidence label per involved object, anchored above its box (screenshot-style AR callout). */
  drawAR(S, objs) {
    const want = new Set();
    if (S.layers.hud) {
      for (const o of this.case.objects) {
        if (o.role === 'other' && !(S.isolate || []).includes(o.id) && S.selected !== o.id) continue;
        if ((S.isolate || []).length && !(S.isolate || []).includes(o.id)) continue;
        want.add(o.id);
      }
    }
    for (const id of Object.keys(this.ar)) if (!want.has(id)) { this.ar[id].el.remove(); delete this.ar[id]; }
    for (const id of want) {
      const o = this.objs[id];
      let ax, ay;
      if (id === 'ego') { [ax, ay] = this.P(this.W * .5, this.H * .84); }
      else {
        const b = objs[id]?.box; if (!b) { if (this.ar[id]) this.ar[id].el.style.opacity = 0; continue; }
        [ax, ay] = this.P((b[0] + b[2]) / 2, b[1]);
      }
      let a = this.ar[id];
      if (!a) { a = this.ar[id] = { el: el('div', { class: 'ar' }), x: ax, y: ay }; this.arLayer.append(a.el); }
      const html = this.arHtml(o);
      if (html !== a.html) { a.el.innerHTML = html; a.html = html; }
      a.x = lerp(a.x, ax, .35); a.y = lerp(a.y, ay, .35);
      const lx = clamp(a.x, 8, this.cw - 230), ly = clamp(a.y - 72, 150, this.ch - 40);
      a.el.style.opacity = 1;
      a.el.style.transform = `translate(${lx}px, ${ly}px) translateY(-100%)`;
      a.el.querySelector('.stem').style.height = Math.max(20, a.y - ly) + 'px';
      a.el.querySelector('.stem').style.bottom = -Math.max(20, a.y - ly) + 'px';
      a.el.querySelector('.dot').style.bottom = -Math.max(20, a.y - ly) - 8 + 'px';
      a.el.querySelector('.stem').style.left = a.el.querySelector('.dot').style.left = (a.x - lx) + 'px';
      a.el.querySelector('.dot').style.marginLeft = '-7px';
    }
  }

  arHtml(o) {
    const c = this.case, m = id => c.measurements.find(x => x.id === id);
    const sp = o.vid ? m(`${o.vid}.pre_impact_speed`) : null;
    const lines = [];
    if (sp) lines.push(`speed ${sp.lo.toFixed(0)}–${sp.hi.toFixed(0)} km/h`);
    if (o.vid === 'v2') { const vis = m('v2.visible_before_impact'); if (vis) lines.push(`visible ${vis.value.toFixed(1)} s before impact`); }
    if (o.vid === 'v1') { const br = m('v1.brake_onset'); if (br) lines.push(br.value > 0 ? `braking ${br.value.toFixed(1)} s before impact` : 'no braking detected'); }
    if (!o.vid) lines.push(o.concept);
    const plate = (c.plates?.plates || []).find(p => p.vehicle === o.id && p.legible);
    if (plate) lines.push(`plate ${plate.text}`);
    const f = this.verdict?.fault?.find(x => o.label && (x.party.toLowerCase().includes(o.label.toLowerCase().split(' #')[0]) || (o.id === 'ego' && /camera/i.test(x.party))));
    const tag = `${o.label.split(' #')[0].toUpperCase()} - #${c.meta.source_clip.replace(/.(mp4|webm)$/, '').slice(-3)}${o.id === 'ego' ? '-01' : '-' + (o.id.split(':')[1] || '0').padStart(2, '0')}`;
    const status = f ? `<div class="status ok">complete · fault ${f.lo}–${f.hi}%</div>` : `<div class="status">incomplete</div>`;
    return `<div class="box"><div class="h">${tag}</div><div class="rule"></div>${lines.map(l => `<div class="v">${l}</div>`).join('')}
      <div class="rule"></div><div class="small">Analysis of impact vector:</div>${status}</div><div class="stem"></div><div class="dot"></div>`;
  }

  applyZoom(S, objs) {
    let ts = 1, tx = 0, ty = 0;
    const native = S.zoom && this.case?.plates?.plates?.find(p => p.key === S.zoom && p.source_verified && Math.abs(S.t - p.t) < .06);
    const b = S.zoom && (objs[S.zoom]?.box || native?.box);
    if (b) {
      const [x1, y1] = this.P(b[0], b[1]), [x2, y2] = this.P(b[2], b[3]);
      ts = clamp(Math.min(this.cw / (x2 - x1), this.ch / (y2 - y1)) * .45, 1.4, 3.2);
      const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
      tx = clamp(this.cw / 2 - cx * ts, this.cw * (1 - ts), 0); ty = clamp(this.ch / 2 - cy * ts, this.ch * (1 - ts), 0);
    }
    const z = this.zoomNow;
    if (Math.abs(z.s - ts) < 1e-3 && Math.abs(z.x - tx) < .5 && Math.abs(z.y - ty) < .5) return;
    z.s = lerp(z.s, ts, .08); z.x = lerp(z.x, tx, .08); z.y = lerp(z.y, ty, .08);
    this.stage.style.transition = 'none';
    this.stage.style.transformOrigin = '0 0';
    this.stage.style.transform = `translate(${z.x}px, ${z.y}px) scale(${z.s})`;
    this.arLayer.style.opacity = z.s > 1.05 ? 0 : 1;
  }
}
