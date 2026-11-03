/* The static world and persistent vehicle tracks share one coordinate frame. The default camera is manual;
   video-camera following and approach replay are explicit, optional views. */
import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '../vendor/spark/spark.module.js';
import { $, clamp, esc } from './util.js';
import { createVehicle } from './vehicle-models.js';
import { SceneNavigation } from './scene-navigation.js';
import { keyAt, browserForward, actorVisible } from './replay-pose.js';

const T = p => new THREE.Vector3(p[0], p[1], -p[2]);     // road frame (z forward) -> three.js (-z forward)
const UP = new THREE.Vector3(0, 1, 0);
const HOLD = 1.5;                                         // seconds the replay rests on the instant before looping
const NEAR_CAM = 2.5;                                     // a vehicle layer this close to the camera is hidden (m)

/** Screen backdrop fading from the sky's colour to the road's: what the camera never saw reads as sky or road. */
function backdrop(sky, ground) {
  const hex = c => c ? `rgb(${c.map(v => Math.round(v * 255)).join(',')})` : '#0d1014';
  const cv = document.createElement('canvas'); cv.width = 2; cv.height = 256;
  const g = cv.getContext('2d'), grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, hex(sky)); grad.addColorStop(0.45, hex(sky)); grad.addColorStop(0.62, hex(ground || sky)); grad.addColorStop(1, hex(ground || sky));
  g.fillStyle = grad; g.fillRect(0, 0, 2, 256);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class Moments {
  constructor(onReplay = () => {}) {
    this.root = $('#v-scene'); this.canvas = $('#gl'); this.pip = $('#pip');
    this.r = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false });
    this.r.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    this.r.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.add(new SparkRenderer({ renderer: this.r }));
    this.scene.add(new THREE.HemisphereLight(0xddeaff, 0x6c6860, 2.4));
    const sun = new THREE.DirectionalLight(0xfff5e6, 2.0); sun.position.set(-8, 15, -5); this.scene.add(sun);
    this.replay = document.createElement('button'); this.replay.className = 'scene-replay';
    this.replay.textContent = 'Replay collision · ½ speed'; this.replay.hidden = true;
    this.replay.onclick = () => { if (this.cur) onReplay(this.cur); };
    this.root.append(this.replay);
    this.side = document.createElement('button'); this.side.className = 'scene-side';
    this.side.textContent = 'Dashcam view'; this.side.hidden = true;
    this.side.onclick = () => this.setMode(this.mode === 'driver' ? 'free' : 'driver');
    this.root.append(this.side);
    this.approach = document.createElement('button'); this.approach.className = 'scene-approach';
    this.approach.textContent = 'Replay approach'; this.approach.hidden = true;
    this.approach.onclick = () => { if (this.cur?.approach) { this.setMode('approach'); onReplay(this.cur); } };
    this.root.append(this.approach);
    this.reset = document.createElement('button'); this.reset.className = 'scene-reset';
    this.reset.textContent = 'Reset camera'; this.reset.hidden = true;
    this.reset.onclick = () => { this.cameraSeeded = false; this.setMode('free'); this.canvas.focus(); };
    this.root.append(this.reset);
    this.help = document.createElement('div'); this.help.className = 'scene-controls';
    this.help.textContent = 'Drag: look · Wheel / W S: move · A D: strafe · Q E: height · Shift: faster · Right drag: pan';
    this.root.append(this.help);
    this.comparison = document.createElement('div'); this.comparison.className = 'scene-comparison';
    this.root.append(this.comparison);
    this.cam = new THREE.PerspectiveCamera(40, 1, 0.05, 800);
    this.list = []; this.meshes = new Map(); this.layers = new Map(); this.worlds = new Map(); this.cur = null; this.mode = 'free';
    this.cameraSeeded = false; this.t0 = performance.now();
    this.nav = new SceneNavigation(this.canvas, this.cam, () => this.setMode('free'));
    new ResizeObserver(() => this.resize()).observe(this.root);
  }

  setCase(c) {
    for (const m of new Set(this.meshes.values())) { this.scene.remove(m); m.dispose?.(); }
    for (const ls of new Set(this.layers.values())) for (const l of ls) { this.scene.remove(l.mesh); l.mesh.dispose?.(); }
    this.meshes.clear(); this.layers.clear(); this.worlds.clear(); this.cur = null;
    this.replay.hidden = true;
    this.side.hidden = true;
    this.approach.hidden = this.reset.hidden = true; this.cameraSeeded = false; this.mode = 'free';
    this.nav.keys.clear();
    this.list = c.moments || [];
    $('#hud3').textContent = '';
  }

  nearest(t) {
    let best = null;
    for (const m of this.list) if (!best || Math.abs(m.t - t) < Math.abs(best.t - t)) best = m;
    return best;
  }

  find(id) { return this.list.find(m => m.id === id) || null; }

  async show(id) {
    const m = this.find(id); if (!m) return;
    if (this.cur?.id === id) return;
    if (!this.cur || this.cur.url !== m.url) this.cameraSeeded = false;
    this.cur = m; this.t0 = performance.now();
    this.span = Math.max(0, ...(m.vehicles || []).map(v => v.keys.length ? -v.keys[0][0] : 0));
    this.replay.hidden = !m.reconstruction;
    this.replay.textContent = m.event_kind === 'near-miss' || /closest/i.test(m.label) ? 'Replay event · ½ speed' : 'Replay collision · ½ speed';
    this.side.hidden = !m.reconstruction;
    this.approach.hidden = !m.approach;
    this.reset.hidden = false;
    $('#hud3').innerHTML = `<b>${esc(m.label)}</b> · ${m.t.toFixed(2)} s<span id="hud3t"></span>` +
      (m.reconstruction ? '<div class="ai">AI-generated environment' + (m.approach ? ' and approach road' : '') + ' · estimated 3D vehicle bodies · ' + (m.road_motion ? 'estimated traffic motion' : 'tracked motion') + '</div>' :
        m.clean ? '<div class="ai">background behind the vehicles is AI-filled</div>' : '');
    this.scene.background?.dispose?.();
    this.scene.background = backdrop(m.cam.sky, m.cam.ground);
    const worldKey = m.reconstruction ? m.url : id;
    let world = this.worlds.get(worldKey);
    if (!world) {
      const mesh = new SplatMesh({ url: m.url });
      const ls = (m.vehicles || []).map(v => ({ v, mesh: v.model ? createVehicle(v.model) : new SplatMesh({ url: v.url }) }));
      world = {mesh,ls,ready:Promise.all([mesh.initialized,...ls.map(l=>l.mesh.initialized)])};
      this.worlds.set(worldKey,world); this.meshes.set(worldKey,mesh);
    }
    const {mesh,ls}=world, vehicles=new Map((m.vehicles||[]).map(v=>[v.key,v]));
    // Each physical actor keeps one mesh across all moments in the same world. Only its time origin changes.
    for(const layer of ls)if(vehicles.has(layer.v.key))layer.v=vehicles.get(layer.v.key);
    this.layers.set(id,ls);
    for(const [key,other] of this.worlds){
      other.mesh.visible=key===worldKey;
      for(const l of other.ls)l.mesh.visible=key===worldKey;
    }
    $('#sceneLoad').textContent='loading the moment…';$('#sceneLoad').hidden=false;
    try{await world.ready;}
    catch(error){
      if(this.worlds.get(worldKey)===world){
        this.worlds.delete(worldKey);this.meshes.delete(worldKey);this.scene.remove(mesh);mesh.dispose?.();
        for(const [key,value] of this.layers)if(value===ls)this.layers.delete(key);
        ls.forEach(l=>l.mesh.dispose?.());
        if(this.cur===m){this.cur=null;$('#sceneLoad').textContent='Could not load the 3D environment. Reopen to retry.';}
      }
      return;
    }
    if(this.worlds.get(worldKey)!==world)return;
    mesh.position.copy(T(m.origin||[0,0,0]));
    if(!mesh.parent)this.scene.add(mesh);
    for(const l of ls)if(!l.mesh.parent)this.scene.add(l.mesh);
    if(this.cur===m)$('#sceneLoad').hidden=true;
  }

  /** Vehicle layers along their paths: replay into the instant, hold, loop. */
  animate(now, state) {
    const ls = this.layers.get(this.cur.id); if (!ls) return;
    const cycle = this.span + HOLD;
    const tr = this.cur.reconstruction ? (state?.t ?? this.cur.t_frame) - this.cur.t_frame :
      this.span ? Math.min((now - this.t0) / 1000 % cycle - this.span, 0) : 0;
    const q = new THREE.Quaternion(), p = new THREE.Vector3(), origin = T(this.cur.origin || [0, 0, 0]);
    for (const { v, mesh } of ls) {
      const k = keyAt(v.keys, tr, { clampBefore: !!v.persistent }), pivot = T(v.pivot);
      if (!k) { mesh.visible = false; continue; }
      const supported = actorVisible(v,tr);
      if (v.model && !v.persistent && tr > v.keys.at(-1)[0] + .15) { mesh.visible = false; continue; }
      q.setFromUnitVectors(new THREE.Vector3(0,0,-1),new THREE.Vector3(...browserForward(k[2])));
      p.copy(pivot).add(new THREE.Vector3(k[0], 0, -k[1]));
      // hide a vehicle passing through the camera, never one that is this close at the instant itself
      mesh.visible = supported && (!!v.model || p.distanceTo(this.cam.position) > Math.min(NEAR_CAM, 0.7 * pivot.distanceTo(this.cam.position)));
      if (v.model && v.key === 'ego' && this.mode === 'driver') mesh.visible = false;
      if (v.model && v.key === 'ego' && this.mode === 'free') {
        const local = this.cam.position.clone().sub(p).applyQuaternion(q.clone().invert());
        const [w,,l] = v.model.dimensions_m;
        if (Math.abs(local.x)<w/2+.15 && Math.abs(local.z)<l/2+.15) mesh.visible=false;
      }
      // splats are stored relative to the origin: turn them about the pivot, then move the pivot along the path
      mesh.quaternion.copy(q);
      if (v.model) {
        mesh.position.copy(p); mesh.position.y = 0;
      } else mesh.position.copy(origin).sub(pivot).applyQuaternion(q).add(p);
    }
    const el = $('#hud3t');
    const txt = this.cur.reconstruction ? ` · T+${(tr + this.cur.t_frame).toFixed(2)} s` :
      this.span && tr < -0.05 ? ` · ${tr.toFixed(1)} s` : '';
    if (el && el.textContent !== txt) el.textContent = txt;
  }

  setMode(m) {
    this.mode = !m || m === 'orbit' ? 'free' : m;
    this.side.textContent = this.mode === 'driver' ? 'Free camera' : 'Dashcam view';
    this.help.hidden = this.mode !== 'free';
  }

  resize() {
    const r = this.root.getBoundingClientRect();
    if (!r.width) return;
    this.r.setSize(r.width, r.height, false);
    this.cam.aspect = r.width / r.height; this.cam.updateProjectionMatrix();
  }

  sourceCamera(m, t) {
    const poses = m.camera_poses;
    if (poses?.length) {
      let i = 1; while (i < poses.length-1 && poses[i][0] < t) i++;
      const a = poses[Math.max(0,i-1)], b = poses[Math.min(i,poses.length-1)];
      const f = clamp((t-a[0])/Math.max(1e-6,b[0]-a[0]),0,1);
      const position = new THREE.Vector3(...a[1]).lerp(new THREE.Vector3(...b[1]),f);
      if(a[3]&&b[3]){
        const h=b[0]-a[0],f2=f*f,f3=f2*f;
        position.set(0,0,0).addScaledVector(new THREE.Vector3(...a[1]),2*f3-3*f2+1)
          .addScaledVector(new THREE.Vector3(...a[3]),h*(f3-2*f2+f))
          .addScaledVector(new THREE.Vector3(...b[1]),-2*f3+3*f2)
          .addScaledVector(new THREE.Vector3(...b[3]),h*(f3-f2));
        const edge=t<poses[0][0]?poses[0]:t>poses.at(-1)[0]?poses.at(-1):null;
        if(edge)position.fromArray(edge[1]).addScaledVector(new THREE.Vector3(...edge[3]),t-edge[0]);
      }
      const quaternion = new THREE.Quaternion(...a[2]).slerp(new THREE.Quaternion(...b[2]),f);
      return {position,quaternion};
    }
    const c = m.cam, pose = keyAt(m.camera_keys, t, {clampBefore:true});
    const position = pose ? new THREE.Vector3(pose[0],c.pos[1],-pose[1]) : T(c.pos);
    const forward = pose ? new THREE.Vector3(Math.sin(THREE.MathUtils.degToRad(pose[2])),c.fwd[1],-Math.cos(THREE.MathUtils.degToRad(pose[2]))).normalize() : T(c.fwd).normalize();
    const temp = new THREE.PerspectiveCamera(); temp.position.copy(position); temp.lookAt(position.clone().add(forward));
    return {position,quaternion:temp.quaternion};
  }

  sourceFov(m,t) {
    const keys=m.camera_lenses;
    if(!keys?.length)return m.cam.fov_y;
    let i=1;while(i<keys.length-1&&keys[i][0]<t)i++;
    const a=keys[Math.max(0,i-1)],b=keys[Math.min(i,keys.length-1)];
    return a[1]+(b[1]-a[1])*clamp((t-a[0])/Math.max(1e-6,b[0]-a[0]),0,1);
  }

  sourceIntrinsics(m,t) {
    const keys=m.camera_intrinsics;
    if(!keys?.length)return null;
    let i=1;while(i<keys.length-1&&keys[i][0]<t)i++;
    const a=keys[Math.max(0,i-1)],b=keys[Math.min(i,keys.length-1)];
    const f=clamp((t-a[0])/Math.max(1e-6,b[0]-a[0]),0,1);
    return a.slice(1).map((v,j)=>v+(b[j+1]-v)*f);
  }

  /** Manual pose survives time changes. Following a source pose is only enabled by an explicit control. */
  place(now, state) {
    const m = this.cur, c = m.cam;
    const t = state?.t ?? m.t_frame;
    if (!this.cameraSeeded || this.mode === 'driver' || this.mode === 'side' || this.mode === 'approach') {
      const source = this.sourceCamera(m,t);
      this.cam.position.copy(source.position); this.cam.quaternion.copy(source.quaternion);
      this.lensFov = this.sourceFov(m,t);
      this.cameraSeeded = true;
    }
    const sourceAspect = c.aspect || 16/9;
    this.cam.fov = THREE.MathUtils.radToDeg(2*Math.atan(Math.tan((this.lensFov||c.fov_y)/2)*Math.max(1,sourceAspect/this.cam.aspect)));
    this.cam.updateProjectionMatrix();
    if(this.mode==='driver'){
      const lens=this.sourceIntrinsics(m,t);
      if(lens&&m.source_size){
        const [fx,fy,cx,cy]=lens,[w,h]=m.source_size,n=this.cam.near;
        this.cam.projectionMatrix.makePerspective(-cx*n/fx,(w-cx)*n/fx,cy*n/fy,-(h-cy)*n/fy,n,this.cam.far);
        this.cam.projectionMatrixInverse.copy(this.cam.projectionMatrix).invert();
      }
    }
    if (m.approach && this.mode === 'approach') {
      const ego = (m.vehicles || []).find(v => v.key === 'ego');
      const body = ego ? keyAt(ego.keys,t-m.t_frame,{clampBefore:true}) : null;
      const c0 = this.cam.position.clone(), fwd = this.cam.getWorldDirection(new THREE.Vector3());
      if (body) c0.set(body[0],c.pos[1],-body[1]);
      const flat = fwd.clone().setY(0).normalize();
      const right = new THREE.Vector3().crossVectors(flat,UP).normalize();
      this.cam.position.copy(c0).addScaledVector(flat,-10).addScaledVector(right,1.6);
      this.cam.position.y = c.pos[1];
      const target = c0.clone().addScaledVector(flat,4); target.y=.9;
      this.cam.lookAt(target);
    }
    if (this.mode === 'side') this.cam.position.x += 3;
    this.nav.update(now,this.mode==='free');
  }

  draw(now, video, state) {
    if (!this.cur) return;
    if (!this.canvas.width || this.canvas.width !== Math.floor(this.canvas.clientWidth * this.r.getPixelRatio())) this.resize();
    this.place(now, state);
    this.animate(now, state);
    this.renderTime = state?.t ?? this.cur.t_frame;
    this.renderedFrameRevision = state?.videoFrameRevision;
    const size=this.r.getSize(new THREE.Vector2());
    this.r.setScissorTest(false);this.r.setViewport(0,0,size.x,size.y);
    if(this.mode==='driver'){
      const aspect=this.cur.cam.aspect||16/9;
      const w=Math.min(size.x,size.y*aspect),h=w/aspect;
      this.r.setScissorTest(true);this.r.setScissor(0,0,size.x,size.y);this.r.clear();
      this.r.setViewport((size.x-w)/2,(size.y-h)/2,w,h);
      this.r.setScissor((size.x-w)/2,(size.y-h)/2,w,h);
    }
    this.r.render(this.scene, this.cam);
    const nativeSource=this.root.querySelector('video.scene-source-video');
    this.pip.hidden=!!nativeSource;
    if (video && !nativeSource && (video instanceof HTMLCanvasElement || video.readyState >= 2)) {
      const ctx = this.pip.getContext('2d');
      if (this.pip.width !== 480) { this.pip.width = 480; this.pip.height = 270; }
      ctx.drawImage(video, 0, 0, 480, 270);
    }
    this.comparison.hidden=!video;
    const view=this.mode==='driver'?'Dashcam view':this.mode==='approach'?'Approach view':'Free camera';
    this.comparison.textContent=`Video · T+${this.renderTime.toFixed(3)} s · ${view}`;
  }
}
