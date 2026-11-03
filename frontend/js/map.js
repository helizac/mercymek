/* Where it happened: OpenStreetMap (Leaflet) with the crash location, its uncertainty, heading and evidence. */
import { $, esc } from './util.js';

const TIER = { high: 'confirmed', medium: 'likely', low: 'tentative' };

export class MapView {
  constructor() { this.map = null; this.layer = null; this.loc = null; }

  ensure() {
    if (this.map || !window.L) return;
    this.map = L.map('leaflet', { zoomControl: false, attributionControl: true }).setView([20, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(this.map);
    L.control.zoom({ position: 'topleft' }).addTo(this.map);
    this.layer = L.layerGroup().addTo(this.map);
  }

  setCase(c) { this.case = c; this.loc = c.location; this.drawn = false; this.card(); this.mlyViewer?.remove?.(); this.mlyViewer = null; $('#street').hidden = true; }

  /** Called when the map view becomes visible (Leaflet needs a visible container to size itself). */
  show() {
    this.ensure();
    if (!this.map) return;
    setTimeout(() => { this.map.invalidateSize(); if (!this.drawn) this.draw(); }, 80);
  }

  draw() {
    this.drawn = true;
    this.layer.clearLayers();
    const l = this.loc;
    if (!l || l.lat == null) { this.map.setView([20, 0], 2); return; }
    const ll = [l.lat, l.lon];
    L.circle(ll, { radius: Math.max(15, l.radius_m || 50), color: '#ff5a5f', weight: 1.5, fillColor: '#ff5a5f', fillOpacity: .12 }).addTo(this.layer);
    if (l.gps_track?.length > 1) L.polyline(l.gps_track.map(g => [g[1], g[2]]), { color: '#7fd1c7', weight: 4, opacity: .9 }).addTo(this.layer);
    if (l.camera_heading_deg != null) {       // heading cone: where the camera was pointing
      const h = l.camera_heading_deg * Math.PI / 180, d = Math.max(60, (l.radius_m || 50) * 1.6) / 111320;
      const k = 1 / Math.cos(l.lat * Math.PI / 180);
      const tip = a => [l.lat + Math.cos(a) * d, l.lon + Math.sin(a) * d * k];
      L.polygon([ll, tip(h - .35), tip(h + .35)], { color: '#eef1f2', weight: 1, fillColor: '#eef1f2', fillOpacity: .18 }).addTo(this.layer);
    }
    L.marker(ll, { icon: L.divIcon({ className: '', html: '<div class="crash-pin"></div>', iconSize: [26, 26], iconAnchor: [13, 13] }) }).addTo(this.layer);
    for (const a of l.alternatives || []) {
      if (a.lat != null) L.circleMarker([a.lat, a.lon], { radius: 6, color: '#f2d28b', weight: 2, fillOpacity: .3 })
        .bindTooltip(esc(a.place_name)).addTo(this.layer);
    }
    const z = (l.radius_m || 50) < 100 ? 17 : (l.radius_m || 50) < 1000 ? 15 : (l.radius_m || 50) < 10000 ? 12 : 9;
    this.map.flyTo(ll, z, { duration: 1.4 });
  }

  /** Street-level imagery near the scene (Mapillary when a token is configured, Panoramax otherwise). */
  setStreet(open) {
    const box = $('#street');
    box.hidden = !open;
    if (!open) { this.streetLayer?.clearLayers(); return; }
    this.ensure();
    const sv = this.case?.streetview || { items: [] }, items = sv.items || [];
    this.streetLayer ||= L.layerGroup().addTo(this.map);
    this.streetLayer.clearLayers();
    for (const it of items) {
      L.marker([it.lat, it.lon], { icon: L.divIcon({ className: '', html: '<div class="cam-pin"></div>', iconSize: [12, 12], iconAnchor: [6, 6] }) })
        .bindTooltip(`${it.provider} · ${it.distance_m} m · ${it.date}`).addTo(this.streetLayer);
    }
    if (!items.length) {
      box.innerHTML = `<div class="sh"><b>Street level</b></div><div class="none">No open street-level photos were found within 1.5 km of this place.${
        this.case?.mapillary_token ? '' : ' Adding a Mapillary token gives much wider coverage.'}</div>`;
      return;
    }
    const mly = this.case.mapillary_token && items.find(i => i.provider === 'Mapillary');
    box.innerHTML = `<div class="sh"><b>Street level</b><span>${items.length} photos near the scene · ${esc([...new Set(items.map(i => i.provider))].join(', '))}</span></div>` +
      (mly ? '<div class="mly" id="mly"></div>' : '') +
      `<div class="strip">${items.map((it, i) => `<div class="shot" data-i="${i}"><img src="${esc(it.url)}" alt="" loading="lazy">
        <div>${it.distance_m} m away · ${esc(it.date || '')}</div></div>`).join('')}</div>`;
    box.querySelectorAll('.shot').forEach(n => n.onclick = () => {
      const it = items[+n.dataset.i];
      if (mly && this.mlyViewer && it.provider === 'Mapillary') { this.mlyViewer.moveTo(it.id).catch(() => {}); return; }
      const lb = $('#lightbox'); lb.hidden = false;
      lb.innerHTML = `<img src="${esc(it.url)}" alt=""><div>${esc(it.provider)} · ${it.distance_m} m from the scene · ${esc(it.date || '')} · ${esc(it.license || '')}</div>`;
      lb.onclick = () => { lb.hidden = true; };
      this.map.flyTo([it.lat, it.lon], 18, { duration: .8 });
    });
    if (mly) this.mapillary(mly.id);
    this.map.fitBounds(L.latLngBounds([...items.map(i => [i.lat, i.lon]), [this.loc.lat, this.loc.lon]]).pad(.3), { maxZoom: 18 });
  }

  /** Interactive Mapillary viewer (MapillaryJS, loaded on demand; client tokens are meant for browsers). */
  async mapillary(id) {
    if (!window.mapillary) {
      const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = 'https://unpkg.com/mapillary-js@4.1.2/dist/mapillary.css';
      document.head.append(css);
      await new Promise((ok, no) => { const s = document.createElement('script'); s.src = 'https://unpkg.com/mapillary-js@4.1.2/dist/mapillary.js'; s.onload = ok; s.onerror = no; document.head.append(s); });
    }
    this.mlyViewer?.remove?.();
    this.mlyViewer = new window.mapillary.Viewer({ accessToken: this.case.mapillary_token, container: 'mly', imageId: id });
  }

  card() {
    const l = this.loc, box = $('#locCard');
    if (!l || l.lat == null) {
      box.innerHTML = `<h2>Location unknown</h2><div class="addr">${esc(l?.analysis || 'No location evidence was found in this footage.')}</div>`;
      return;
    }
    box.innerHTML = `
      <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3)">Scene location</div>
      <h2>${esc(l.place_name || 'Located')}</h2>
      <div class="addr">${esc(l.address || '')}</div>
      <div class="coords">${l.lat.toFixed(5)}, ${l.lon.toFixed(5)} · ±${Math.round(l.radius_m || 0)} m</div>
      <span class="tier ${esc(l.confidence)}">${TIER[l.confidence] || l.confidence}</span>
      ${l.road ? `<div class="addr" style="margin-top:8px">Road: <b style="color:var(--ink)">${esc(l.road.road || '')}</b>${l.road.ref ? ' (' + esc(l.road.ref) + ')' : ''}</div>` : ''}
      <ul>${(l.evidence || []).slice(0, 7).map(e => `<li>${esc(e)}</li>`).join('')}</ul>
      <div class="kf">${(l.keyframes || []).map(k => `<img src="/cases/${this.case.id}/${k}" alt="">`).join('')}</div>`;
  }
}
