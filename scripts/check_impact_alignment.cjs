/* Verify the displayed bodies at the labelled impact, using the presented video clock. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const reports=[],directory='data/research/impact-alignment';
 fs.mkdirSync(directory,{recursive:true});
 try{
  for(const id of (process.argv.slice(2).length?process.argv.slice(2):['hero','cutin_rearend'])){
   const manifest=JSON.parse(fs.readFileSync(`data/cases/${id}/moments.json`));
   const event=manifest.contacts.event_contact;
   assert(event,'missing labelled contact');
   const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(`http://localhost:8000/?case=${id}&view=scene&t=${event.time}`);
   await page.waitForFunction(id=>MERCY?.feed.case?.id===id&&MERCY.moments.cur&&document.querySelector('#sceneLoad').hidden,id,{timeout:90000});
   const samples=[];
   for(const time of [event.time,event.time+.13]){
    await page.evaluate(t=>{MERCY.S.t=t;MERCY.S.playing=false;},time);
    await page.waitForFunction(t=>!MERCY.feed.video.seeking&&Math.abs(MERCY.feed.clock.time-t)<.04&&MERCY.moments.renderTime===MERCY.feed.clock.time,time);
    samples.push(await page.evaluate(async({actors,time})=>{
     const THREE=await import('/vendor/three.module.min.js'),m=MERCY.moments;
     const bodies=actors.map(key=>m.layers.get(m.cur.id).find(l=>l.v.key===key)).map(({v,mesh})=>{
      const x=new THREE.Vector3(1,0,0).applyQuaternion(mesh.quaternion),z=new THREE.Vector3(0,0,1).applyQuaternion(mesh.quaternion);
      return {key:v.key,p:[mesh.position.x,mesh.position.z],axes:[[x.x,x.z],[z.x,z.z]],r:[v.model.dimensions_m[0]*1.15/2,(v.model.dimensions_m[2]+.22)/2]};
     });
     const [a,b]=bodies,d=a.p.map((x,i)=>x-b.p[i]),dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
     const gap=Math.max(...[...a.axes,...b.axes].map(n=>Math.abs(dot(d,n))-
      a.r.reduce((s,r,i)=>s+r*Math.abs(dot(a.axes[i],n)),0)-b.r.reduce((s,r,i)=>s+r*Math.abs(dot(b.axes[i],n)),0)));
     m.setMode('free');m.cameraSeeded=true;
     const center=[(a.p[0]+b.p[0])/2,(a.p[1]+b.p[1])/2];
     m.cam.position.set(center[0]+8,11,center[1]+10);m.cam.lookAt(center[0],.7,center[1]);
     return {requested_time:time,presented_time:MERCY.feed.clock.time,scene_time:m.renderTime,surface_gap_m:gap,bodies};
    },{actors:event.actors,time}));
    await page.waitForTimeout(300);
    await page.screenshot({path:`${directory}/${id}-${time.toFixed(2)}.png`});
   }
   assert(samples[0].surface_gap_m>=-.001&&samples[0].surface_gap_m<.3,JSON.stringify(samples[0]));
   assert(samples[1].surface_gap_m>=-.001&&samples[1].surface_gap_m<1.8,JSON.stringify(samples[1]));
   assert.deepEqual(errors,[]);
   reports.push({case:id,samples,console_errors:errors});
   console.log(id,'rendered impact contact PASS',samples.map(v=>v.surface_gap_m.toFixed(3)));
   await page.close();
  }
  fs.writeFileSync(`${directory}/browser-impact.json`,JSON.stringify(reports,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
