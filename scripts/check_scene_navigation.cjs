/* Real-browser regression checks: node scripts/check_scene_navigation.cjs (demo server on :8000). */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const distance=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
async function auditPersistentWorld(){
 const {keyAt,browserForward,actorVisible}=await import('/js/replay-pose.js');
 const THREE=await import('/vendor/three.module.min.js');
 const m=MERCY.moments,S=MERCY.S,c=MERCY.feed.case;
 S.playing=false;m.setMode('free');
 const original=m.cur.id,ids=new Map(m.layers.get(original).map(l=>[l.v.key,l.mesh.uuid]));
 let samples=0,maxSpeed=0,maxAcceleration=0;
 for(let t=0;t<=c.duration+.0001;t+=.1){
  S.t=t;m.animate(performance.now(),S);
  for(const l of m.layers.get(m.cur.id)){
   if(l.v.key==='ego')continue;
   if(l.mesh.visible!==actorVisible(l.v,t-m.cur.t_frame))throw Error(l.v.key+' source visibility mismatch at '+t);
   if(l.mesh.uuid!==ids.get(l.v.key))throw Error('actor identity replaced');
   const pose=keyAt(l.v.keys,t-m.cur.t_frame,{clampBefore:true});
   const actual=new THREE.Vector3(0,0,-1).applyQuaternion(l.mesh.quaternion);
   if(actual.distanceTo(new THREE.Vector3(...browserForward(pose[2])))>1e-8)throw Error('body nose mirrored');
  }
  const bodies=m.layers.get(m.cur.id).filter(l=>l.v.model).map(l=>{
   const [w,,length]=l.v.model.dimensions_m;
   const x=new THREE.Vector3(1,0,0).applyQuaternion(l.mesh.quaternion);
   const z=new THREE.Vector3(0,0,1).applyQuaternion(l.mesh.quaternion);
   return {key:l.v.key,p:[l.mesh.position.x,l.mesh.position.z],axes:[[x.x,x.z],[z.x,z.z]],r:[w*1.15/2,(length+.22)/2]};
  });
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
  for(let a=0;a<bodies.length;a++)for(let b=0;b<a;b++){
   const A=bodies[a],B=bodies[b],d=A.p.map((p,i)=>p-B.p[i]);
   const separated=[...A.axes,...B.axes].some(n=>Math.abs(dot(d,n))>=
    A.r[0]*Math.abs(dot(A.axes[0],n))+A.r[1]*Math.abs(dot(A.axes[1],n))+
    B.r[0]*Math.abs(dot(B.axes[0],n))+B.r[1]*Math.abs(dot(B.axes[1],n))-.0003);
   if(!separated)throw Error(`${A.key} intersects ${B.key} at ${t}`);
  }
  samples++;
 }
 S.t=m.cur.impact_t;m.animate(performance.now(),S);
 const positions=new Map(m.layers.get(m.cur.id).map(l=>[l.v.key,l.mesh.position.toArray()]));
 for(const moment of m.list){
  await m.show(moment.id);m.animate(performance.now(),S);
  for(const l of m.layers.get(m.cur.id)){
   if(l.mesh.uuid!==ids.get(l.v.key))throw Error('moment switch recreated '+l.v.key);
   if(distance3(l.mesh.position.toArray(),positions.get(l.v.key))>.002)throw Error('moment switch teleported '+l.v.key);
   if(l.v.key!=='ego'&&l.mesh.visible!==actorVisible(l.v,S.t-m.cur.t_frame))throw Error('moment switch changed source visibility '+l.v.key);
  }
 }
 await m.show(original);m.animate(performance.now(),S);
 for(const v of m.cur.vehicles){
  let last=null,prevVelocity=null,prevDt=null;
  for(const k of v.keys){
   if(last){
    const dt=k[0]-last[0],velocity=[(k[1]-last[1])/dt,(k[2]-last[2])/dt];
    maxSpeed=Math.max(maxSpeed,Math.hypot(...velocity));
    const impact=v.contacts?.impact_impulse_t;
    const straddlesImpact=Number.isFinite(impact)&&k[0]-dt-prevDt+m.cur.t_frame<impact&&k[0]+m.cur.t_frame>impact;
    if(prevVelocity&&v.key!=='ego'&&!straddlesImpact)maxAcceleration=Math.max(maxAcceleration,distance3(velocity,prevVelocity)/((dt+prevDt)*.5));
    prevVelocity=velocity;prevDt=dt;
   }
   last=k;
  }
 }
 if(maxSpeed>40.6)throw Error('unbounded actor speed '+maxSpeed);
 if(maxAcceleration>10.15)throw Error('unbounded actor acceleration '+maxAcceleration);
 const primary=m.cur.vehicles.find(v=>v.key===(c.id==='hero'||c.id==='snow_pileup'?'car:7':c.id==='night_rain'?'car:1':'car:4'));
 const at=t=>keyAt(primary.keys,t-m.cur.t_frame,{clampBefore:true})[2];
 if(c.id==='hero'){
  if(Math.abs(at(6.5)-180)>5||Math.abs(at(8)-90)>5)throw Error('crossing approach/turn heading wrong');
 }else if(Math.abs(at(m.cur.impact_t))>1)throw Error('side perspective mistaken for a turn');
 return {case:c.id,persistent_actors:ids.size-1,timeline_samples:samples,shared_meshes_across_moments:true,
  persistent_actor_visibility:true,no_despawns:true,stable_actor_identities:true,body_intersections:0,max_speed_mps:maxSpeed,max_acceleration_mps2:maxAcceleration};
 function distance3(a,b){return Math.hypot(...a.map((x,i)=>x-b[i]));}
}
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const report=[];
 try{
  const p=await browser.newPage({viewport:{width:1920,height:1080}});
  const errors=[];p.on('pageerror',e=>errors.push(e.message));
  await p.goto('http://localhost:8000/?case=cutin_rearend&view=scene&t=.95');
  await p.waitForFunction(()=>window.MERCY?.moments.cur?.url?.includes('/cases/cutin_rearend/')&&document.querySelector('#sceneLoad').hidden,{timeout:90000});
  const pose=()=>p.evaluate(()=>({p:MERCY.moments.cam.position.toArray(),q:MERCY.moments.cam.quaternion.toArray(),mode:MERCY.moments.mode}));
  const initial=await pose();assert.equal(initial.mode,'free');
  await p.waitForTimeout(1500);assert.deepEqual(await pose(),initial,'paused camera drifted');
  const rect=await p.locator('#gl').boundingBox(),x=rect.x+rect.width*.5,y=rect.y+rect.height*.55;
  await p.mouse.move(x,y);await p.mouse.down();await p.mouse.move(x+150,y-60,{steps:8});await p.mouse.up();
  const looked=await pose();assert(distance(looked.q,initial.q)>.1,'drag did not change angle');
  await p.waitForTimeout(800);assert.deepEqual(await pose(),looked,'manual angle did not stay put');
  await p.mouse.wheel(0,-160);await p.waitForTimeout(100);const dolly=await pose();
  assert(distance(dolly.p,looked.p)>2,'wheel did not move through the world');
  await p.locator('#gl').focus();await p.keyboard.down('w');await p.waitForTimeout(400);await p.keyboard.up('w');
  const walked=await pose();assert(distance(walked.p,dolly.p)>.3,'W did not move');
  await p.keyboard.down('e');await p.waitForTimeout(250);await p.keyboard.up('e');
  assert((await pose()).p[1]>walked.p[1]+.2,'height control failed');
  const manual=await pose();
  await p.evaluate(()=>{MERCY.S.t=4.02;MERCY.S.playing=false});await p.waitForTimeout(300);
  assert.deepEqual(await pose(),manual,'time change moved the free camera');
  await p.locator('.scene-reset').click();await p.waitForTimeout(200);
  const reset=await pose();assert(distance(reset.p,manual.p)>1,'reset failed');
  await p.locator('.scene-side').click();assert.equal((await pose()).mode,'driver');
  const follow=await pose();await p.evaluate(()=>MERCY.S.t=4.56);await p.waitForTimeout(200);
  assert(distance((await pose()).p,follow.p)>.2,'explicit video following failed');
  await p.locator('#gl').focus();await p.keyboard.press('a');assert.equal((await pose()).mode,'free','manual input did not leave follow mode');
  report.push({navigation:'pass',no_automatic_drift:true,drag:true,wheel:true,keyboard:true,height:true,reset:true,independent_of_timeline:true});
  for(const t of [.95,4.02,4.56]){
   await p.evaluate(t=>{MERCY.S.t=t;MERCY.S.playing=false;MERCY.moments.cameraSeeded=false;MERCY.moments.setMode('free')},t);
   await p.waitForTimeout(500);
   const check=await p.evaluate(async t=>{
    const m=MERCY.moments,source=await(await fetch('/cases/cutin_rearend/overlays.json')).json();
    const obs=Object.entries(source[String(Math.round(t*30))]).filter(([k,v])=>v.concept==='car'&&v.box[2]-v.box[0]>=7&&v.box[3]-v.box[1]>=6);
    const layers=m.layers.get(m.cur.id),missing=[];
    for(const [key] of obs){
     const layer=layers.find(l=>l.v.key===key||l.v.model?.source_keys?.includes(key));
     if(!layer||!layer.mesh.visible)missing.push(key);
    }
    return {t,source_vehicles:obs.length,missing,visible_tracks:layers.filter(l=>l.mesh.visible&&l.v.key!=='ego').map(l=>l.v.key)};
   },t);
   assert.deepEqual(check.missing,[],'a source vehicle is missing at '+t);
   report.push(check);
   await p.screenshot({path:'data/research/registration-fix/corrected-'+t+'.png'});
  }
  report.push(await p.evaluate(auditPersistentWorld));
  assert.deepEqual(errors,[]);await p.close();
  for(const id of ['hero','lanechange','night_rain','snow_pileup']){
   const page=await browser.newPage({viewport:{width:1600,height:1000}}),jsErrors=[];
   page.on('pageerror',e=>jsErrors.push(e.message));
   await page.goto('http://localhost:8000/?case='+id+'&view=scene&t=5.8');
   try{
    await page.waitForFunction(id=>window.MERCY?.moments.cur?.url?.includes('/cases/'+id+'/')&&document.querySelector('#sceneLoad').hidden,id,{timeout:90000});
   }catch(error){
    console.error('Case load diagnostic',id,jsErrors,await page.evaluate(()=>({url:location.href,
     cur:window.MERCY?.moments.cur?.url,mode:window.MERCY?.moments.mode,state:window.MERCY?.S,
     load:document.querySelector('#sceneLoad')?.textContent,hidden:document.querySelector('#sceneLoad')?.hidden})));
    await page.screenshot({path:'data/research/registration-fix/'+id+'-load-failure.png'});
    throw error;
   }
   const result=await page.evaluate(async id=>{
    const m=MERCY.moments,S=MERCY.S;S.playing=false;m.setMode('driver');
    const c=await(await fetch('/api/case/'+id)).json(),overlays=await(await fetch('/cases/'+id+'/overlays.json')).json();
    const counts={},concepts=new Set(['car','truck','bus','van','motorcycle','motorbike','pickup','vehicle']);
    for(const objects of Object.values(overlays))for(const [key,o] of Object.entries(objects)){
     if(concepts.has(o.concept.toLowerCase())&&o.box[2]-o.box[0]>=7&&o.box[3]-o.box[1]>=6)counts[key]=(counts[key]||0)+1;
    }
    const rows=[];
    let roadTracksChecked=0;
    for(const v of m.cur.vehicles){
     if(!v.road_motion?.straight_prior_applied)continue;
     const yaw=v.road_motion.heading_deg*Math.PI/180,forward=[Math.sin(yaw),Math.cos(yaw)],right=[Math.cos(yaw),-Math.sin(yaw)];
     for(let i=1;i<v.keys.length;i++){
      const a=v.keys[i-1],b=v.keys[i],dx=b[1]-a[1],dz=b[2]-a[2];
      if(Math.abs(dx*right[0]+dz*right[1])>1e-4)throw Error(v.key+' drifted sideways');
      if(dx*forward[0]+dz*forward[1]<-1e-4)throw Error(v.key+' moved backwards');
      if(Math.abs(a[3]-b[3])>1e-4)throw Error(v.key+' changed body heading on a straight lane');
     }
     roadTracksChecked++;
    }
    for(const t of [m.cur.replay_start+.5,m.cur.impact_t,m.cur.replay_end-.2]){
     S.t=t;m.draw(performance.now(),null,S);
     const objects=overlays[String(Math.round(t*c.meta.src_fps))]||{},missing=[];
     const ls=m.layers.get(m.cur.id);
     for(const [key,o] of Object.entries(objects)){
      if(!concepts.has(o.concept.toLowerCase())||counts[key]<3||o.box[2]-o.box[0]<7||o.box[3]-o.box[1]<6)continue;
      const l=ls.find(l=>l.v.key===key||l.v.model?.source_keys?.includes(key));
      if(!l||!l.mesh.visible)missing.push(key);
     }
     if(!m.cam.position.toArray().every(Number.isFinite))throw Error('invalid camera');
     for(const l of ls)if(l.v.model&&l.mesh.visible&&Math.abs(l.mesh.position.y)>1e-6)throw Error('vehicle floating off shared ground');
     rows.push({t,missing});
    }
    m.setMode('free');return {case:id,rows,vehicle_tracks:m.cur.vehicles.length-1,road_tracks_checked:roadTracksChecked};
   },id);
   assert.deepEqual(jsErrors,[]);
   for(const row of result.rows)assert.deepEqual(row.missing,[],id+' has missing vehicles at '+row.t);
   await page.screenshot({path:'data/research/registration-fix/'+id+'-final.png'});
   report.push(result);report.push(await page.evaluate(auditPersistentWorld));await page.close();console.log(id,'passed source coverage and continuous actor lifetime');
  }
  fs.writeFileSync('data/research/registration-fix/browser-check.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
