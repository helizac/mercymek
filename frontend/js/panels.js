/* Report, measurements, blueprint, evidence (dossiers, plates, entities, files) and the case selector. */
import { $, $$, el, esc, fmtN, prettyBytes, rgba, clamp } from './util.js';

const CIRC = ['parked / stationary', 'leaving a parking space / opening a door', 'entering a parking space',
  'emerging from a car park or private ground', 'entering a car park or private ground', 'entering a roundabout',
  'circulating in a roundabout', 'rear-ending the other vehicle, same direction and lane', 'same direction, different lane',
  'changing lanes', 'overtaking', 'turning right', 'turning left', 'reversing', 'encroaching on the opposite lane',
  'coming from the right at a junction', 'not observing a sign or a red light'];

export class Panels {
  constructor(app) { this.app = app; this.case = null; this.verdict = null; this.image = null; }

  load(c) {
    this.case = c; this.verdict = null; this.image = null; this._mv = undefined;
    this.measurements(c.measurements);
    this.reportIdle(); this.faultIdle(); this.blueprintIdle(); this.files(); this.dossiers(); this.plates(); this.entities();
    $('#meter').hidden = true; $('#ledger').innerHTML = ''; $('#meterDelta').className = 'delta';
    if (c.claims?.meter?.length) this.setMeter(c.claims.meter, c.claims.start, c.claims.ledger, false);
    this.statement(c.statement); this.sealState = null;
    $('#caseId').textContent = c.meta.source_clip.replace(/\.(mp4|webm)$/, '').slice(0, 14);
    $('#caseTitle').textContent = c.meta.title.replace(/^Nexar \d+ · /, '');
  }

  /* ---------------- measurements ---------------- */
  measurements(rows) {
    const box = $('#tab-measure'); box.innerHTML = '<h3>Measurements · every value is a range</h3>';
    for (const m of rows) {
      const span = Math.max(m.hi * 1.15, m.value * 1.3, 1e-6);
      box.append(el('div', { class: 'mrow', title: m.method },
        el('div', { class: 'top1' }, el('span', { class: 'lab' }, m.label),
          el('span', { class: 'val', html: `${fmtN(m.value, m.unit === 's' ? 2 : 1)}<small>${esc(m.unit)}</small>` })),
        el('div', { class: 'rbar', html: `<i style="left:${m.lo / span * 100}%;width:${Math.max(.6, (m.hi - m.lo) / span * 100)}%"></i><b style="left:${m.value / span * 100}%"></b>` }),
        el('div', { class: 'meth' }, `${fmtN(m.lo, 1)} – ${fmtN(m.hi, 1)} ${m.unit} · ${m.method}`)));
    }
    // ground truth printed by the dashcam itself (GPS speed overlay), against the reconstruction
    const gt = this.case?.location?.overlay_speed_kmh, ego = this.case?.series?.objects?.ego;
    if (gt?.length && ego) {
      box.append(el('h3', { style: { marginTop: '14px' } }, 'Check against the dashcam’s own GPS speed'));
      for (const [t, v] of gt) {
        let i = 0; ego.t.forEach((tt, j) => { if (Math.abs(tt - t) < Math.abs(ego.t[i] - t)) i = j; });
        const r = ego.speed_kmh[i], err = (r - v) / v * 100;
        box.append(el('div', { class: 'mrow' }, el('div', { class: 'top1' }, el('span', { class: 'lab' }, `t ${t.toFixed(1)} s · overlay ${v} km/h`),
          el('span', { class: 'val', html: `${r.toFixed(0)}<small>km/h (${err >= 0 ? '+' : ''}${err.toFixed(0)}%)</small>` }))));
      }
    }
  }

  /* ---------------- report (examiner document) ---------------- */
  docShell(c, inner, status) {
    const m = c.meta, sc = c.scene || {};
    const ang = c.measurements.find(x => x.id === 'impact.angle')?.value;
    const kind = ang == null ? null : ang < 35 ? 'rear' : ang > 145 ? 'head' : 'angle';
    const box = (k, label) => `<label><i>${kind === k ? '✓' : ''}</i>${label}</label>`;
    const vehicles = (sc.vehicles || []).map(v => v.label).join(', ');
    const loc = c.location?.place_name;
    return `<div class="doc">
      <div class="hdr"><div class="seal"><span class="ring"></span></div>
        <div><h2>Office of Collision Reconstruction</h2><div class="sub2">mercymek · report of investigation</div></div>
        <div class="caseno">Case number<b>${esc(m.source_clip.replace(/\.(mp4|webm)$/, '').slice(0, 14))}-${String(Math.round(c.impact_t * 100)).padStart(4, '0')}</b>${status || ''}</div></div>
      <div class="box"><h4>Case information</h4><div class="kv">
        ${loc ? `<span>Location</span><span>${esc(loc)}</span>` : ''}
        <span>Scene</span><span>${esc(sc.road_type || m.scene)}</span>
        <span>Conditions</span><span>${esc(m.light)} light · ${esc(m.weather)} · road ${esc(sc.weather || '?')}</span>
        <span>Controls</span><span>${esc((sc.traffic_controls || []).join(', ') || 'none visible')}</span>
        <span>Vehicles</span><span>${esc(vehicles)}</span>
        <span>Event</span><span>${c.impact?.kind === 'near-miss' ? 'Near-miss' : 'Collision'} at ${c.impact_t.toFixed(2)} s</span></div>
        <div class="checks">${box('rear', 'Rear-end')}${box('angle', 'Angle / side')}${box('head', 'Head-on / turning')}
          <label><i>${sc.camera_vehicle_involved ? '✓' : ''}</i>Camera vehicle involved</label></div></div>
      ${inner}</div>`;
  }

  reportIdle() {
    const c = this.case;
    $('#tab-report').innerHTML = this.docShell(c, `<div class="box"><h4>Summary</h4><div class="typed">${esc(c.scene?.summary || '')}</div></div>
      <div class="box"><h4>Fault assessment</h4><div class="caveat">Not analysed yet. Ask <b>“who is at fault?”</b></div></div>`);
  }

  faultIdle() { $('#faultCard').innerHTML = '<h3>Fault</h3><div class="caveat">Pending analysis.</div>'; }

  verdictRunning(stage) {
    $('#tab-report').innerHTML = this.docShell(this.case, `<div class="box"><h4>Analysis in progress</h4>
      <div class="stage-line"><span class="spin"></span>${esc(stage || 'Working…')}</div>
      <div class="skeleton" style="width:92%"></div><div class="skeleton" style="width:76%"></div><div class="skeleton" style="width:84%"></div></div>`);
    $('#faultCard').innerHTML = `<h3>Fault</h3><div class="stage-line"><span class="spin"></span>${esc(stage || 'Working…')}</div>`;
  }

  verdictStage(stage) {
    const doc = $('#tab-report .doc'); if (!doc) return;
    let line = doc.querySelector('.live-stage');
    if (!line) { line = el('div', { class: 'stage-line live-stage', style: { marginTop: '10px' } }); doc.querySelector('.hdr').after(line); }
    line.innerHTML = `<span class="spin"></span>${esc(stage || 'updating…')}`;
  }

  verdictReady(msg) {
    const d = msg.data, c = this.case; this.verdict = d;
    const fault = d.fault.map(f => `<div class="row"><span class="name">${esc(f.party)}</span><span class="pc">${f.pct}% <small style="color:var(--ink-3)">(${f.lo}–${f.hi})</small></span>
      <div class="bar"><i style="left:${f.lo}%;width:${f.hi - f.lo}%"></i><b style="left:${f.pct}%"></b></div><div class="why">${esc(f.rationale || '')}</div></div>`).join('');
    const integ = d.integrity || {};
    const checks = (integ.checks || []).map(k => `<span>${k.ok ? '✓' : '✗'} ${esc(k.name)}</span>`).join('');
    const rules = (d.rules || []).map(r => `<div class="factor"><span class="w">rule</span><span>${esc(r.rule)} ${r.source_url?.startsWith('http') ? `<a href="${esc(r.source_url)}" target="_blank" rel="noopener" style="color:#9cc3ff">↗</a>` : ''}</span></div>`).join('');
    const inner = `
      <div class="box"><h4>Findings</h4><div class="typed" id="typed"></div></div>
      <div class="box"><h4>Fault share (estimate, range)</h4><div class="fault">${fault}</div></div>
      <div class="box"><h4>Key measurements</h4>${d.factors.map(f => `<div class="factor"><span class="w">${f.weight}</span><span>${esc(f.text)}</span></div>`).join('')}</div>
      ${rules ? `<div class="box"><h4>Applicable rules</h4>${rules}</div>` : ''}
      <div class="box cites"><h4>Sources</h4>${d.citations.map(x => `<a href="${esc(x.url)}" target="_blank" rel="noopener">↗ ${esc(x.title)}</a>`).join('') || '<span class="caveat">none</span>'}</div>
      <div class="integrity ${integ.review_required ? 'review' : ''}"><div><b>${integ.review_required ? 'REVIEW REQUIRED' : 'VERDICT INTEGRITY'} · ${integ.passed}/${integ.total}</b>
        <div class="ck">${checks}</div></div></div>
      ${(d.caveats || []).map(x => `<div class="caveat">· ${esc(x)}</div>`).join('')}
      <div class="sig-line"><div>Signature of examiner<br><svg width="120" height="34" viewBox="0 0 120 34"><path d="M4 26c10-18 16-20 18-8s8 10 14-4 10-10 12 2 6 10 14-2 12-12 16 0 10 6 18-6 12-6 20-2" fill="none" stroke="#e9ecea" stroke-width="1.6"/></svg><br>mercymek · ${new Date().toLocaleDateString()}</div></div>`;
    $('#tab-report').innerHTML = this.docShell(c, inner);
    this.typeOut($('#typed'), d.headline);
    $('#faultCard').innerHTML = `<h3>Fault</h3><div class="bigfault">${d.fault.map(f => `<div class="p"><span class="n">${esc(f.party)}</span><span class="v">${f.pct}%</span>
      <div class="bar"><i style="left:${f.lo}%;width:${f.hi - f.lo}%"></i><b style="left:${f.pct}%"></b></div><span class="r">range ${f.lo}–${f.hi}% · ${esc(d.confidence)} confidence</span></div>`).join('')}</div>`;
    this.meter(d);
    this.dossiers();
  }

  typeOut(node, text) {
    if (!node) return;
    let i = 0; clearInterval(this._typer);
    this._typer = setInterval(() => { i += 2; node.textContent = text.slice(0, i); if (i >= text.length) clearInterval(this._typer); }, 18);
  }

  meter(d) {
    const top = d.fault[0]; if (!top) return;
    if (this.case.claims?.meter?.length) return;          // the argued meter is already showing
    const second = d.fault[1];
    const meter = [{ party: top.party, pct: top.pct }, ...(second ? [{ party: second.party, pct: 100 - top.pct }] : [])];
    this.setMeter(meter, meter, [], true, `${top.lo}–${top.hi}%`);
  }

  /** Mercy mode liability meter: the leading party's share, animated from its previous value. */
  setMeter(meter, start, ledger, count = true, range = null) {
    if (!meter?.length) return;
    const top = meter.reduce((a, b) => (b.pct > a.pct ? b : a));
    const from = this._mv ?? (count ? 0 : top.pct);
    $('#meter').hidden = false; $('#meter').classList.remove('weighing');
    $('#meterWho').textContent = top.party;
    const st = (start || []).find(x => x.party === top.party);
    $('#meterRange').textContent = range || (st && Math.abs(st.pct - top.pct) > .05 ? `started at ${st.pct.toFixed(0)}%` : 'from the verdict');
    $('#meterBar').style.width = top.pct + '%';
    $('#meterStart').style.left = (st ? st.pct : top.pct) + '%';
    const pct = $('#meterPct'); let v = from; const to = top.pct;
    clearInterval(this._m);
    this._m = setInterval(() => { v += Math.sign(to - v) * Math.min(Math.abs(to - v), .6); pct.textContent = v.toFixed(1) + '%'; if (Math.abs(to - v) < .01) clearInterval(this._m); }, 16);
    this._mv = to;
    const L = $('#ledger'); L.innerHTML = '';
    for (const e of (ledger || []).slice(-3).reverse()) {
      const k = e.assessment === 'supported' ? ['ok', '✓'] : e.assessment === 'contradicted' ? ['no', '✕'] : ['un', '?'];
      L.append(el('div', { title: e.reply }, el('b', { class: k[0] }, k[1]), el('span', {}, e.claim.replace(/^(objection[,:]?\s*|i (argue|claim) (that )?)/i, '')),
        el('em', {}, e.moved ? (e.moved > 0 ? '+' : '') + e.moved.toFixed(1) : '·')));
    }
  }

  meterMsg(m) {
    if (m.status === 'weighing') { $('#meter').hidden = !this._mv && !this.case.claims?.meter?.length; $('#meter').classList.add('weighing'); return; }
    $('#meter').classList.remove('weighing');
    if (m.status !== 'ready') return;
    this.case.claims = { meter: m.meter, start: m.start, ledger: m.ledger };
    const d = $('#meterDelta'), e = m.entry, top = m.meter.reduce((a, b) => (b.pct > a.pct ? b : a));
    const mv = e.moved ? (e.party === top.party || !m.meter.find(x => x.party === e.party) ? e.moved : -e.moved) : 0;
    d.className = 'delta' + (mv > 0 ? ' up' : mv < 0 ? ' down' : '');
    d.textContent = mv ? (mv > 0 ? '+' : '') + mv.toFixed(1) : '';
    clearTimeout(this._dt); this._dt = setTimeout(() => { d.className = 'delta'; }, 6000);
    this.setMeter(m.meter, m.start, m.ledger);
  }

  /* ---------------- European Accident Statement ---------------- */
  statement(st) {
    const box = $('#stmtCard');
    if (!st) { box.innerHTML = '<h3>Accident statement</h3><div class="caveat">Say “fill in the accident statement”.</div>'; return; }
    this.case.statement = st;
    const A = new Set(st.A.circumstances), B = new Set(st.B.circumstances);
    const rows = CIRC.map((c, i) => {
      const n = i + 1, on = A.has(n) || B.has(n);
      return on ? `<span class="c on">${n}. ${esc(c)}</span><i class="${A.has(n) ? 'on' : ''}"></i><i class="${B.has(n) ? 'on' : ''}"></i>` : '';
    }).join('');
    box.innerHTML = `<h3>Accident statement · section 12</h3>
      <div class="stmt"><span></span><span class="hd" title="${esc(st.A.label)}">A</span><span class="hd" title="${esc(st.B.label)}">B</span>
      ${rows || '<span class="c">No circumstance box applies.</span><span></span><span></span>'}</div>
      <div class="stmt-foot">A · ${esc(st.A.label)} — first impact ${esc(st.A.initial_impact)}<br>B · ${esc(st.B.label)}${st.B.plate ? ' (' + esc(st.B.plate) + ')' : ''} — first impact ${esc(st.B.initial_impact)}<br>${esc(st.remarks)}</div>`;
  }

  sealMsg(v) { this.sealState = v; this.files(); }

  /* ---------------- blueprint ---------------- */
  blueprintIdle() {
    const c = this.case;
    $('#tab-blueprint').innerHTML = `<h3>Overhead diagram</h3>${c.urls.diagram ? `<div class="bp"><img src="${c.urls.diagram}" alt="overhead diagram"></div>` : ''}
      <div class="caveat" style="margin-top:8px">Ask for “the blueprint” to restyle it.</div>`;
  }

  imageMsg(m) {
    const box = $('#tab-blueprint');
    if (m.status === 'running') {
      box.innerHTML = `<h3>Blueprint</h3><div class="stage-line"><span class="spin"></span>${esc(m.stage || 'Rendering…')}</div>
        ${m.diagram ? `<div class="bp" style="margin-top:10px"><img src="${m.diagram}"></div>` : '<div class="skeleton" style="height:180px"></div>'}`;
    } else if (m.url) {
      this.image = m;
      box.innerHTML = `<h3>Blueprint</h3><div class="bp"><img src="${m.url}" alt="blueprint"><div class="wm"><span>${esc(m.watermark || 'AI ILLUSTRATION · NOT EVIDENCE')}</span></div></div>
        <div class="caveat" style="margin-top:8px">${esc(m.caption || '')}</div>`;
    }
  }

  /* ---------------- evidence ---------------- */
  files() {
    const c = this.case, box = $('#tab-files');
    const img = /\.(jpg|png|webp)$/i;
    const items = c.files.filter(f => !/_mask\.png$|_(raw|fused)\.png$/.test(f.name) && !f.name.startsWith('live/') && !f.name.startsWith('geo/intake'));
    box.innerHTML = `<h3>Evidence files · ${items.length}</h3>`;
    const source = c.meta.source_attribution;
    if (source) {
      const credit = el('div', { class: 'caveat' }, `${source.title || 'Source video'} · ${source.author || 'Creator unspecified'} · `);
      if (/^https?:\/\//i.test(source.source_url || '')) credit.append(el('a', { href: source.source_url, target: '_blank', rel: 'noopener noreferrer' }, 'Original video'));
      credit.append(document.createTextNode(' · '));
      if (/^https?:\/\//i.test(source.license_url || '')) credit.append(el('a', { href: source.license_url, target: '_blank', rel: 'noopener noreferrer' }, source.license));
      credit.append(el('div', {}, source.modifications || ''));
      box.append(credit);
    }
    const s = c.seal, v = this.sealState;
    if (s) {
      const bad = v && v.sealed && !v.intact;
      box.append(el('div', { class: 'evseal' + (bad ? ' bad' : ''), title: 'SHA-256 manifest of the source footage and every artifact, signed with Ed25519' },
        el('div', {}, el('b', {}, bad ? `Seal broken · ${v.changed.length + v.missing.length} file(s) changed` : v?.intact ? `Seal verified · ${v.files} files intact` : `Sealed · ${s.files} files`),
          el('div', { class: 'mono' }, `root ${s.root.slice(0, 16)}…${s.source_sha256 ? ' · source ' + s.source_sha256.slice(0, 12) + '…' : ''} · ${new Date(s.created * 1000).toISOString().slice(0, 16).replace('T', ' ')}`))));
    }
    const grid = el('div', { class: 'fgrid' });
    for (const f of items) {
      const isImg = img.test(f.name) && f.url;
      const th = el('div', { class: 'th' + (isImg ? '' : ' dark'), style: isImg ? { backgroundImage: `url("${f.url}")` } : {} },
        isImg ? '' : (f.name.split('.').pop() || 'dir').slice(0, 5));
      grid.append(el(f.url ? 'a' : 'div', { class: 'file', href: f.url, target: '_blank', title: `${f.name} · ${prettyBytes(f.bytes)}` },
        th, el('span', {}, f.name.split('/').pop())));
    }
    box.append(grid);
  }

  plates() {
    const c = this.case, box = $('#platesCard'), P = c.plates || {};
    box.innerHTML = '<h3>Licence plates</h3>';
    if (P.blurred_at_source) { box.append(el('div', { class: 'caveat' }, P.note || 'Plates were blurred at the source.')); return; }
    if (!(P.plates || []).length) { box.append(el('div', { class: 'caveat' }, 'No readable plate was tracked in this clip. Ask to “zoom in on the plate” at any moment.')); return; }
    for (const p of P.plates) {
      const owner = c.objects.find(o => o.id === p.vehicle);
      box.append(el('div', { class: 'plate-item', onclick: () => this.app.local({ t: p.t, playing: false, view: 'video', zoom: p.key }) },
        el('img', { src: `/cases/${c.id}/${p.images.enhanced || p.images.raw}` }),
        el('div', {}, el('div', { class: 'txt' }, p.legible ? p.text : (p.text || 'illegible')),
          el('div', { class: 'm' }, `${owner ? owner.label + ' · ' : ''}t ${p.t}s · confidence ${(p.confidence * 100).toFixed(0)}% · ${p.source_verified ? 'original frame' : p.frames_used + ' frames fused'}`))));
    }
  }

  plateOf(id) { return (this.case.plates?.plates || []).find(p => p.vehicle === id && p.legible); }

  faultFor(o) {
    return this.verdict?.fault?.find(x => x.party.toLowerCase().includes(o.label.toLowerCase().split(' #')[0]) || (o.id === 'ego' && /camera/i.test(x.party)));
  }

  dossiers() {
    const c = this.case, inv = c.objects.filter(o => o.vid).sort((a, b) => a.vid.localeCompare(b.vid));
    [$('#dossier1'), $('#dossier2')].forEach((node, i) => {
      const o = inv[i];
      node.innerHTML = '';
      if (!o) { node.hidden = true; return; }
      node.hidden = false; node.dataset.id = o.id;
      const ev = c.evidence.find(e => e.object === o.id);
      const photo = o.id === 'ego' ? (c.evidence.find(e => e.label === 'pre impact') || c.evidence[0]) : ev;
      const m = id => c.measurements.find(x => x.id === id);
      const sp = m(`${o.vid}.pre_impact_speed`);
      const sv = (c.scene?.vehicles || []).find(v => o.id === 'ego' ? v.role === 'camera_vehicle' : (c.aliases?.[o.id] === v.label));
      const f = this.faultFor(o), plate = this.plateOf(o.id);
      const extra = o.vid === 'v1' ? m('v1.brake_onset') : m('v2.visible_before_impact');
      const fields = [
        ['Vehicle', `${o.vid.toUpperCase()} · ${o.id === 'ego' ? 'camera vehicle' : o.label}`],
        ['Body / colour', sv ? `${sv.body_type}${sv.color && sv.color !== 'unknown' ? ' · ' + sv.color : ''}` : o.concept],
        ['Speed before impact', sp ? `${sp.lo.toFixed(0)}–${sp.hi.toFixed(0)} km/h` : '—'],
        [o.vid === 'v1' ? 'Braking began' : 'Visible before impact', extra ? (o.vid === 'v1' && !extra.value ? 'not detected' : `${extra.value.toFixed(1)} s before`) : '—'],
        ['Plate', plate ? plate.text : '—'],
        ['Fault share', f ? `${f.lo}–${f.hi}%` : 'pending'],
      ];
      node.append(el('div', { class: 'stripe', style: { background: o.color } }));
      node.append(el('div', { class: 'photo', style: { backgroundImage: photo?.url ? `url("${photo.url}")` : 'none' } }, el('span', { class: 'who' }, o.label)));
      const body = el('div', { class: 'body' });
      for (const [l, v] of fields) body.append(el('div', { class: 'f' }, el('div', { class: 'l' }, l),
        el('div', { class: 'v' + (l === 'Plate' && plate ? ' plate' : /km\/h|s before|%/.test(v) ? ' mono' : '') }, v)));
      const s = c.series?.objects?.[o.id], isEgo = o.id === 'ego';
      const mprint = isEgo ? el('div', { class: 'print' }, el('canvas'), el('span', {}, 'Speed print'))
        : el('div', { class: 'print', style: { backgroundImage: ev?.mask_url ? `url("${ev.mask_url}")` : 'none' } }, el('span', {}, 'Silhouette'));
      const tprint = el('div', { class: 'print' }, el('canvas'), el('span', {}, 'Motion print'));
      body.append(el('div', { class: 'prints' }, mprint, tprint));
      node.append(body);
      node.onclick = () => this.app.local({ isolate: [o.id], selected: o.id, view: 'video' });
      requestAnimationFrame(() => {
        if (s) this.ridges(tprint.querySelector('canvas'), s.xz.map(p => [p[1], p[0]]), o.color);
        if (s && isEgo) { const e = s.t.length > 10 ? 3 : 0; this.ridges(mprint.querySelector('canvas'), s.t.slice(e, s.t.length - e).map((t, k) => [t * 6, -s.speed_kmh[k + e]]), o.color); }
      });
    });
  }

  /** A curve drawn as fingerprint-like ridges. pts = [[x, y], ...]. */
  ridges(cv, pts, color) {
    if (!cv || !pts || pts.length < 3) return;
    const r = cv.getBoundingClientRect(); if (!r.width) return;
    const d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d;
    const c = cv.getContext('2d'); c.scale(d, d);
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sc = Math.min((r.width - 16) / Math.max(1, x1 - x0), (r.height - 16) / Math.max(1, y1 - y0));
    const ox = (r.width - (x1 - x0) * sc) / 2, oy = (r.height - (y1 - y0) * sc) / 2;
    for (let k = -7; k <= 7; k++) {
      c.beginPath();
      pts.forEach((p, i) => { const x = ox + (p[0] - x0) * sc, y = oy + (p[1] - y0) * sc + k * 2.6; i ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.strokeStyle = k === 0 ? color : rgba('#e9ecea', .55 - Math.abs(k) * .06); c.lineWidth = k === 0 ? 1.8 : .8; c.stroke();
    }
  }

  entities() {
    const c = this.case, box = $('#entCloud'); box.innerHTML = '';
    const items = c.evidence.filter(e => e.kind === 'object');
    $('#entCount').textContent = `${items.length} tracked`;
    const v2 = c.objects.find(o => o.vid === 'v2')?.id;
    items.sort((a, b) => (b.object === v2) - (a.object === v2));
    const W = 560, H = 240;
    items.forEach((e, i) => {
      const main = i === 0, ang = i * 2.39996, rad = main ? 0 : 34 + 30 * Math.sqrt(i);
      const size = main ? 110 : clamp(70 - i * 5, 38, 64);
      const x = W / 2 + Math.cos(ang) * rad * 1.9, y = H / 2 + Math.sin(ang) * rad * .8;
      const o = c.objects.find(q => q.id === e.object);
      box.append(el('div', { class: 'ent' + (i > 4 ? ' blur' : ''), title: e.label,
        style: { left: `${clamp(x / W * 100, 8, 92)}%`, top: clamp(y, size / 2, H - size / 2) + 'px', width: size + 'px', height: size + 'px',
          backgroundImage: `url("${e.url}")`, borderColor: o && o.role !== 'other' ? o.color : undefined },
        onclick: () => this.app.local({ isolate: [e.object], selected: e.object, view: 'video' }) },
        main ? el('span', { class: 'n' }, e.label) : null));
    });
  }

  markSelected(S) { $$('.dossier').forEach(d => d.classList.toggle('sel', (S.isolate || []).includes(d.dataset.id) || S.selected === d.dataset.id)); }

  caseSelector(cases, current, onPick) {
    const box = $('#casesel'); box.hidden = false; box.innerHTML = '';
    const previous = document.activeElement;
    const close = () => { box.hidden = true; previous?.focus(); };
    const grid = el('div', { class: 'case-grid' });
    const dialog = el('section', { class: 'case-dialog', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'caseHeading' },
      el('h2', { id: 'caseHeading', class: 'case-sr' }, 'Choose a case'),
      el('button', { class: 'case-close', type: 'button', 'aria-label': 'Close case selector', onclick: close }, '×'), grid);
    const satellites = cases.filter(c => c.id !== current);
    const selected = cases.find(c => c.id === current);
    const ordered = selected ? [selected, ...satellites] : cases;
    let satellite = 0;
    ordered.forEach(c => {
      const main = c.id === current;
      const event = c.event_kind === 'near-miss' ? 'Near miss' : c.event_kind === 'collision' ? 'Collision' : c.event_kind;
      const button = el('button', { class: 'c' + (main ? ' selected' : ''), type: 'button', 'data-case-id': c.id,
        title: [c.title, event].filter(Boolean).join(' · '),
        'aria-current': main ? 'true' : null, onclick: () => { close(); onPick(c.id); } },
        el('span', { class: 'case-poster', 'aria-hidden': 'true', style: { backgroundImage: c.poster ? `url("${c.poster}")` : 'none' } }),
        el('span', { class: 'cap' }, el('b', {}, c.title.replace(/^Nexar \d+ · /, '')),
          el('span', {}, c.id)),
        event ? el('span', { class: 'case-event case-sr' }, event) : null);
      if (!main) {
        const angle = -Math.PI / 2 + satellite++ * Math.PI * 2 / Math.max(1, satellites.length);
        button.style.setProperty('--orbit-x', Math.cos(angle).toFixed(6));
        button.style.setProperty('--orbit-y', Math.sin(angle).toFixed(6));
      }
      grid.append(button);
    });
    box.append(dialog);
    box.onclick = e => { if (e.target === box || e.target === dialog || e.target === grid) close(); };
    box.onkeydown = e => {
      e.stopPropagation();
      if (e.key === 'Escape') { e.preventDefault(); close(); }
      if (e.key === 'Tab') {
        const buttons = [...dialog.querySelectorAll('button')], first = buttons[0], last = buttons.at(-1);
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    (grid.querySelector('.selected') || dialog.querySelector('button')).focus();
  }
}
