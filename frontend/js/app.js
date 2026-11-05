/* mercymek HUD: one client state, server patches, views, voice. The video is the main screen; every other screen
   (3D moments, map, report, evidence) is one sentence away. */
import { $, $$, el, esc } from './util.js';
import { Net } from './net.js';
import { Feed } from './feed.js';
import { Timeline } from './timeline.js';
import { Moments } from './moments.js';
import { MapView } from './map.js';
import { Panels } from './panels.js';

const S = { t: 0, playing: false, speed: 1, selected: null, isolate: [], trajectory: [], zoom: null, camera: 'orbit', view: 'video', street: false,
  moment: null,
  layers: { masks: true, tracks: true, hud: true, cloud: true } };
const net = new Net();
const feed = new Feed();
const moments = new Moments(m => app.local({ t: m.replay_start, playing: true, speed: .5 }));
const map = new MapView();
let C = null, cases = [], loading = null;

const app = {
  local(p) { patch(p); net.send({ type: 'ui', patch: p }); },
};
const panels = new Panels(app);
const timeline = new Timeline(t => app.local({ t: Math.max(0, Math.min(C?.duration ?? 0, t)), playing: false }),
  m => app.local({ view: 'scene', moment: m.id, t: m.t, playing: false }));

/* ---------------- boot ---------------- */
const bootLines = [];
function boot(line, prog) {
  if (line) { bootLines.push(line); $('#bootLines').innerHTML = bootLines.slice(-3).map((l, i, a) => i === a.length - 1 ? `<b>${esc(l)}</b>` : esc(l)).join('<br>'); }
  if (prog != null) $('#bootProg').style.width = (prog * 100) + '%';
}
const bootDone = () => setTimeout(() => $('#boot').classList.add('out'), 300);
$('#boot').addEventListener('click', bootDone);
(function bokeh() {
  const b = $('#bokeh');
  [['#5b8f5f', 260, 78, 12], ['#c98a3c', 200, 90, 64], ['#3c5fa8', 300, 6, 80], ['#8a5fd0', 160, 60, 90]].forEach(([c, s, x, y], i) =>
    b.append(el('i', { style: { background: c, width: s + 'px', height: s + 'px', left: x + '%', top: y + '%', animationDelay: -i * 7 + 's' } })));
})();

/* ---------------- views ---------------- */
const VIEWS = ['video', 'scene', 'map', 'report', 'evidence'];
function setView(v) {
  if (!VIEWS.includes(v)) v = 'video';
  $$('.view').forEach(n => n.classList.toggle('on', n.id === 'v-' + v));
  feed.presentInScene(v==='scene'?moments.root:null);
  $$('#viewNav button').forEach(b => b.classList.toggle('on', b.dataset.view === v));
  if (v === 'scene') { moments.resize(); const m = moments.find(S.moment) || moments.nearest(S.t); if (m) moments.show(m.id); }
  if (v === 'map') map.show();
}
$('#viewNav').onclick = e => { const b = e.target.closest('button'); if (b) app.local({ view: b.dataset.view }); };

/* ---------------- state ---------------- */
function patch(p) {
  if (!p) return;
  if (p.layers) Object.assign(S.layers, p.layers);
  for (const [k, v] of Object.entries(p)) if (k !== 'layers' && k in S) S[k] = v;
  if ('view' in p) { setView(S.view); if (S.view !== 'video') hideReplay(); }
  if ('street' in p || 'view' in p) map.setStreet(S.view === 'map' && S.street);
  if ('camera' in p && p.camera) moments.setMode(p.camera);
  if ('moment' in p && p.moment && S.view === 'scene') moments.show(p.moment);
  if ('t' in p && feed.video.duration && Math.abs(feed.video.currentTime-S.t)>.02) feed.video.currentTime = S.t;
  if ('zoom' in p && !p.zoom && 'isolate' in p && !(p.isolate || []).length) hideEnhance();
  $('#icoPlay').innerHTML = S.playing ? '<path d="M3 2h4v12H3zM9 2h4v12H9z"/>' : '<path d="M4 2l10 6-10 6z"/>';
  panels.markSelected(S);
}

async function loadCase(c) {
  const token = Symbol(); loading = token;
  C = null;
  $('#boot').classList.remove('out'); bootLines.length = 0;
  boot(c.meta.title, .1);
  document.title = `mercymek · ${c.meta.title.replace(/^Nexar \d+ · /, '')}`;
  await feed.load(c);
  if (loading !== token) return;
  boot('evidence', .6);
  moments.setCase(c); map.setCase(c); timeline.load(c); panels.load(c);
  const loc = c.location;
  $('#sideLoc').textContent = loc?.place_name ? loc.place_name : 'unknown';
  $('#sideLoc').title = loc?.address || '';
  $('#sideCond').textContent = [c.meta.light !== 'unknown' ? c.meta.light + ' light' : null, c.scene?.weather, c.scene?.road_type].filter(Boolean).join(' · ') || '—';
  hideEnhance(); hideReplay(); showWhatif(null);
  C = c;
  boot('ready', 1);
  feed.verdict = null;
  patch({ t: S.t, view: S.view, camera: S.camera || 'orbit' });
  bootDone();
  say('', 'sys');
}

/* ---------------- conversation ---------------- */
let capTimer;
function say(text, kind = 'user', ms = 6000) {
  const cap = $('#caption');
  cap.classList.remove('fade');
  cap.innerHTML = !text ? '' : kind === 'user' ? `“${esc(text)}”` : `<span class="sys">${esc(text)}</span>`;
  clearTimeout(capTimer); capTimer = setTimeout(() => cap.classList.add('fade'), ms);
}
function act(text, err = false) {
  const box = $('#action');
  box.prepend(el('span', { class: err ? 'err' : '' }, text));
  while (box.children.length > 3) box.lastChild.remove();
  clearTimeout(act.t); act.t = setTimeout(() => box.innerHTML = '', 7000);
}
const label = id => C?.objects.find(o => o.id === id)?.label || (id === 'ego' ? 'camera vehicle' : id);
function describe(intent, s = {}) {
  const obj = s.sam_key ? label(s.sam_key) : s.object_id === 0 ? 'both vehicles' : s.object_id ? `vehicle ${s.object_id}` : s.concept || '';
  switch (intent) {
    case 'seek': return s.anchor === 'absolute' ? `go to ${s.t_seconds} s` : s.anchor === 'start' ? 'back to the start'
      : `${s.t_seconds ? Math.abs(s.t_seconds) + ' s ' + (s.t_seconds < 0 ? 'before' : 'after') : 'at'} ${s.anchor === 'current' ? 'from here' : s.anchor === 'brake' ? 'braking' : 'impact'}`;
    case 'playback': return s.playing === false ? 'pause' : s.speed ? `play ${s.speed}×` : 'play';
    case 'camera': return `${s.camera_preset} view`;
    case 'layers': return `${s.visible === false ? 'hide' : 'show'} ${s.layer}`;
    case 'view': return { video: 'video', scene: '3D moment', map: 'map', report: 'report', evidence: 'evidence', street: 'street view' }[s.view] || 'view';
    case 'ask': return 'question';
    case 'zoom': return s.enhance ? `enhance ${s.concept === 'license plate' ? 'licence plate' : obj}` : `zoom ${obj}`;
    case 'isolate': return s.live ? `find “${s.concept}”` : `isolate ${obj}`;
    case 'trajectory': return `trajectory of ${obj}`;
    case 'analyze': return 'analysing fault';
    case 'diagram': return 'blueprint';
    case 'note': return 'note saved';
    case 'reset': return 'reset';
    case 'whatif': return 'what if';
    case 'replay': return s.mode === 'bullet' ? 'impact in 3D' : 'impact replay ×4';
    case 'claim': return 'weighing the argument';
    case 'verify': return 'verifying the seal';
    case 'statement': return 'accident statement';
  }
  return intent;
}

/* ---------------- spoken replies ---------------- */
const voice = new Audio();
voice.addEventListener('playing', () => $('#mic').classList.add('talking'));
for (const ev of ['ended', 'pause', 'error']) voice.addEventListener(ev, () => $('#mic').classList.remove('talking'));
let actx = null, analyser = null, abuf = null;
voice.addEventListener('play', () => {       // the judge's voice drives the glow around the mic
  try {
    if (!actx) {
      actx = new AudioContext();
      const src = actx.createMediaElementSource(voice);
      analyser = actx.createAnalyser(); analyser.fftSize = 512; abuf = new Uint8Array(analyser.fftSize);
      src.connect(analyser); analyser.connect(actx.destination);
    }
    actx.resume();
  } catch { analyser = null; }
});
function voiceLevel() {
  if (!analyser || voice.paused) return 0;
  analyser.getByteTimeDomainData(abuf);
  let e = 0; for (const v of abuf) e += (v - 128) * (v - 128);
  return Math.min(1, Math.sqrt(e / abuf.length) / 40);
}
let replyTimer;
function reply(text, url) {
  const r = $('#reply'); r.classList.remove('fade'); r.textContent = text;
  clearTimeout(replyTimer); replyTimer = setTimeout(() => r.classList.add('fade'), 15000);
  if (url) { voice.pause(); voice.src = url; voice.play().catch(() => {}); }
}

/* ---------------- what if ---------------- */
function showWhatif(r) {
  const b = $('#whatif');
  if (!r) { b.hidden = true; return; }
  const cls = { avoided: 'avoided', borderline: 'borderline' }[r.verdict] || 'not';
  const near = C?.impact?.kind === 'near-miss';
  const kmh = r.impact_kmh[1] < 1 ? 'stops' : `${r.impact_kmh[0].toFixed(0)}–${r.impact_kmh[1].toFixed(0)} km/h`;
  b.hidden = false;
  b.innerHTML = `<div class="h">What if<b class="${cls}">${r.verdict === 'not avoided' ? (near ? 'still reaches it' : 'still crashes') : r.verdict}</b></div>
    <div class="what">${esc(r.what.charAt(0).toUpperCase() + r.what.slice(1))}</div>
    <div class="nums"><span>${near ? 'at the conflict point' : 'impact speed'}</span><b>${kmh} <em>as driven ${r.real.impact_kmh}</em></b>
      <span>stop margin</span><b>${r.margin_m[0].toFixed(0)} to ${r.margin_m[1].toFixed(0)} m</b>
      <span>reaction · μ</span><b>${r.reaction_s[0] === r.reaction_s[1] ? r.reaction_s[0] : r.reaction_s[0] + '–' + r.reaction_s[1]} s · ${r.mu[0]}–${r.mu[1]}</b></div>`;
}

/* ---------------- impact replay (RAFT slow motion) ---------------- */
let replayInfo = null;
function hideReplay() { const b = $('#replay'); if (b.hidden) return; b.hidden = true; $('#replayVid').pause(); replayInfo = null; }
function showReplay(m) {
  replayInfo = m;
  const b = $('#replay'), v = $('#replayVid'); b.hidden = false;
  v.loop = false;
  if (!v.src.endsWith(m.url)) v.src = m.url;
  v.currentTime = 0; v.play().catch(() => {});
  $('#rpImpact').style.left = ((m.impact_t - m.t0) / Math.max(.01, m.t1 - m.t0) * 100) + '%';
}
$('#replay').addEventListener('click', hideReplay);
function replayTick() {
  if (!replayInfo) return;
  const v = $('#replayVid'), m = replayInfo, ts = m.t0 + v.currentTime / m.factor, d = ts - m.impact_t;
  $('#rpT').textContent = `${d >= 0 ? 'T+' : 'T−'}${Math.abs(d).toFixed(2)} s`;
  $('#rpBar').style.width = ((ts - m.t0) / Math.max(.01, m.t1 - m.t0) * 100) + '%';
}

/* ---------------- enhance card ---------------- */
function hideEnhance() { $('#enh').hidden = true; }
function showEnhance(m) {
  const box = $('#enh'); box.hidden = false;
  if (m.status === 'failed') { box.innerHTML = `<div class="head"><b>Enhance</b></div><div class="note">${esc(m.message)}</div>`; return; }
  if (m.status === 'running') { box.innerHTML = `<div class="head"><b>${esc(m.label || 'enhancing')}</b></div><div class="note"><span class="spin"></span> aligning neighbouring frames and upscaling…</div>`; return; }
  const ar = m.output_px ? (m.output_px[0] / m.output_px[1]).toFixed(3) : 2.4;
  const plate = m.plate;
  box.innerHTML = `<div class="head"><b>${esc(m.label)}</b><span>t ${m.t}s · ${m.frames_used} frames fused · ×4</span></div>
    <div class="compare" style="--ar:${ar}"><img class="sr" src="${m.images.enhanced}"><img class="raw" src="${m.images.raw}"><div class="cut"></div>
      <span class="lab" style="left:8px">original ${m.source_px?.join('×') || ''}</span><span class="lab" style="right:8px">enhanced</span></div>
    ${m.status === 'reading' ? '<div class="note"><span class="spin"></span> reading the plate…</div>' : ''}
    ${plate ? `<div class="plate-read"><span class="txt">${esc(plate.text || '—')}</span><div class="meta">${plate.legible ? 'legible' : 'not legible'} · confidence ${(plate.confidence * 100).toFixed(0)}%${plate.region ? '<br>' + esc(plate.region) : ''}${plate.notes ? '<br>' + esc(plate.notes) : ''}</div></div>` : ''}`;
  const cmp = box.querySelector('.compare');
  const move = e => { const r = cmp.getBoundingClientRect(); cmp.style.setProperty('--cut', Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100)) + '%'); };
  cmp.addEventListener('pointermove', move); cmp.addEventListener('pointerdown', move);
}

/* ---------------- server messages ---------------- */
net.on('conn', s => { $('#conn').classList.toggle('ok', s === 'open'); if (s !== 'open') boot('reconnecting…'); });
net.on('cases', d => { cases = d; });
// deep links: ?case=<id>&view=<video|scene|map|report|evidence>&t=<seconds>
const deep = new URLSearchParams(location.search);
net.on('case', c => {
  const requestedCase = deep.get('case');
  deep.delete('case');
  if (requestedCase && requestedCase !== c.id) { net.send({ type: 'load_case', id: requestedCase }); return; }
  loadCase(c).then(() => {
    const p = {};
    if (deep.get('view')) p.view = deep.get('view');
    if (deep.get('t')) { p.t = +deep.get('t'); p.playing = false; }
    if (Object.keys(p).length) { app.local(p); deep.delete('view'); deep.delete('t'); }
  });
});
net.on('state', p => patch(p));
net.on('status', st => {
  if (st.mic) net.micSource = st.mic;
  if (st.note) say(st.note, 'sys', 4000);
});
net.on('transcript', m => {
  $('#mic').classList.toggle('busy', !!m.final && !pttOn);
  if (!m.typed) say(m.text, 'user', m.final ? 6000 : 60000);
});
net.on('intent', m => { $('#mic').classList.remove('busy'); act(describe(m.intent, m.slots)); });
net.on('verdict', m => {
  if (m.status === 'running') { if (panels.verdict) panels.verdictStage(m.stage); else panels.verdictRunning(m.stage); return; }
  feed.verdict = m.data; panels.verdictReady(m);
});
net.on('image', m => panels.imageMsg(m));
net.on('say', m => reply(m.text, m.url));
net.on('enhance', m => showEnhance(m));
net.on('whatif', r => showWhatif(r));
net.on('replay', m => showReplay(m));
net.on('meter', m => panels.meterMsg(m));
net.on('seal', v => panels.sealMsg(v));
net.on('statement', st => panels.statement(st));
net.on('overlay', m => { feed.live = m; if (m) act(`found ${m.objects.length} × ${m.concept}`); });
net.on('error', m => { $('#mic').classList.remove('busy'); act(m.message, true); });

/* ---------------- voice: mic button (hold) or Space ---------------- */
let pttOn = false;
async function ptt(down) {
  if (pttOn === down) return;
  pttOn = down;
  $('#mic').classList.toggle('hot', down);
  if (down) { voice.pause(); say('listening…', 'sys', 60000); await net.pttDown(); }
  else { net.pttUp(); $('#mic').classList.add('busy'); }
}
const mic = $('#mic');
mic.addEventListener('pointerdown', e => { e.preventDefault(); ptt(true); });
mic.addEventListener('pointerup', () => ptt(false));
mic.addEventListener('pointerleave', () => pttOn && ptt(false));
const typing = () => ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
addEventListener('keydown', e => {
  if (e.defaultPrevented) return;
  if (e.code === 'Space' && !typing()) { e.preventDefault(); if (!e.repeat) ptt(true); return; }
  if (typing()) { if (e.key === 'Escape') closePalette(); return; }
  if (!C) return;
  if (e.key === 'Escape') { hideEnhance(); hideReplay(); app.local({ zoom: null }); }
  else if (e.key === 'ArrowLeft') app.local({ t: Math.max(0, S.t - (e.shiftKey ? 1 / 30 : 1)), playing: false });
  else if (e.key === 'ArrowRight') app.local({ t: Math.min(C.duration, S.t + (e.shiftKey ? 1 / 30 : 1)), playing: false });
  else if (e.key.toLowerCase() === 'p') app.local({ playing: !S.playing });
  else if (e.key === '/') { e.preventDefault(); $('#palette').hidden = false; $('#cmdInput').focus(); }
  else if (/^[1-5]$/.test(e.key)) app.local({ view: VIEWS[+e.key - 1] });
});
addEventListener('keyup', e => { if (e.code === 'Space') ptt(false); });
function closePalette() { $('#palette').hidden = true; $('#cmdInput').value = ''; $('#cmdInput').blur(); }
$('#commandHint').onclick = () => { $('#palette').hidden = false; $('#cmdInput').focus(); };
$('#cmdInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.target.value.trim()) { const v = e.target.value.trim(); say(v); net.send({ type: 'cmd', text: v }); closePalette(); }
});

/* ---------------- transport ---------------- */
$('#btnPlay').onclick = () => app.local({ playing: !S.playing });
$('#caseBtn').onclick = () => panels.caseSelector(cases, C?.id, id => net.send({ type: 'load_case', id }));

/* ---------------- loop ---------------- */
let last = performance.now(), lastSent = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (C) {
    if (S.playing) {
      const v = feed.video;
      if (v.readyState >= 2 && !v.paused && !v.seeking) S.t = v.currentTime;
      if (S.t >= C.duration - 0.04) { S.t = C.duration - 0.04; app.local({ playing: false }); }
      if (now - lastSent > 250) { lastSent = now; net.send({ type: 'ui', patch: { t: +S.t.toFixed(3) } }); }
    }
    feed.syncVideo(S);
    const presented = feed.clock.snapshot();
    const shown = presented ? {...S, t: presented.time, videoFrameRevision: presented.revision} : S;
    if (S.view === 'video') feed.draw(shown);
    if (S.view === 'scene') {
      if (S.playing && moments.cur?.reconstruction && S.t >= moments.cur.replay_end) app.local({ playing: false });
      if (presented && (moments.cameraSeeded || Math.abs(presented.time-S.t)<.1)) moments.draw(now, presented.image, shown);
    }
    replayTick();
    const displayTime = replayInfo ? Math.min(replayInfo.t1, replayInfo.t0 + $('#replayVid').currentTime / replayInfo.factor) : shown.t;
    timeline.draw(replayInfo ? { ...S, t: displayTime } : S);
    $('#mic').style.setProperty('--amp', voiceLevel().toFixed(3));
    const d = displayTime - C.impact_t;
    $('#camTs').textContent = `T+${displayTime.toFixed(2)}s`;
    $('#camDelta').textContent = `${C.impact?.kind === 'near-miss' ? 'EVENT' : 'IMPACT'} ${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(2)}s`;
  }
  requestAnimationFrame(frame);
}
boot('connecting…', .02);
net.start();
requestAnimationFrame(frame);
window.MERCY = { S, net, feed, moments, panels, map, app };
