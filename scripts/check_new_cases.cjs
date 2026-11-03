/* Verify the new examples in the real viewer, including decoded-frame sync and actor persistence. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const base=process.env.MERCY_TEST_URL||'http://localhost:8001';
const directory='data/research/new-scenarios';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const reports=[];
 try{
  for(const id of (process.argv.slice(2).length?process.argv.slice(2):['london_rearend','vegas_rearend'])){
   const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.route('**/new-case-projection/*',route=>route.fulfill({contentType:'application/json',
     body:fs.readFileSync(`${directory}/${id}-projection.json`,'utf8')}));
   await page.goto(`${base}/?case=${id}&view=scene&t=1`);
   await page.waitForFunction(id=>window.MERCY?.feed.case?.id===id&&MERCY.moments.cur&&document.querySelector('#sceneLoad').hidden,id,{timeout:120000});
   const caseRoot=process.env.MERCY_CASES_DIR||'data/cases';
   const manifest=JSON.parse(fs.readFileSync(`${caseRoot}/${id}/moments.json`));
   const plates=JSON.parse(fs.readFileSync(`${caseRoot}/${id}/plates.json`)).plates;
   const native=plates.find(p=>p.source_verified);
   for(const file of [native.images.raw,native.original_frame_image]){
    const response=await page.request.get(`${base}/cases/${id}/${file}`);
    assert(response.ok(),`original plate image unavailable: ${id}/${file}`);
   }
   const samples=[];
   const impact=manifest.moments[0].impact_t;
   await page.evaluate(()=>MERCY.moments.setMode('driver'));
   await page.waitForTimeout(150);
   for(const t of [native.t,impact-.3,impact,Math.min(impact+.8,manifest.moments[0].replay_end-.1)]){
    await page.evaluate(t=>{MERCY.S.t=t;MERCY.S.playing=false;},t);
    await page.waitForFunction(t=>!MERCY.feed.video.seeking&&Math.abs(MERCY.feed.clock.time-t)<.06&&MERCY.moments.renderTime===MERCY.feed.clock.time,t,{timeout:20000});
    samples.push(await page.evaluate(async()=>{
     const m=MERCY.moments,c=MERCY.feed.clock;
     const THREE=await import('/vendor/three.module.min.js');
     const [fx,fy,cx,cy]=m.sourceIntrinsics(m.cur,c.time),[w,h]=m.cur.source_size;
     const p=new THREE.Vector3(.4,-.2,-8).applyMatrix4(m.cam.projectionMatrix);
     const error=Math.hypot((p.x+1)*w/2-(cx+fx*.4/8),(1-p.y)*h/2-(cy+fy*.2/8));
     const fixtures=await(await fetch('/new-case-projection/'+MERCY.feed.case.id)).json();
     const fixture=fixtures.reduce((a,b)=>Math.abs(a.time-c.time)<Math.abs(b.time-c.time)?a:b);
     let poseError=0;
     for(let i=0;i<fixture.points_road.length;i++){
      const [x,y,z]=fixture.points_road[i],point=new THREE.Vector3(x,y,-z).project(m.cam);
      poseError=Math.max(poseError,Math.hypot((point.x+1)*w/2-fixture.pixels[i][0],(1-point.y)*h/2-fixture.pixels[i][1]));
     }
     const ctx=document.createElement('canvas');ctx.width=c.canvas.width;ctx.height=c.canvas.height;
     const cc=ctx.getContext('2d');cc.drawImage(MERCY.feed.video,0,0);
     const a=cc.getImageData(0,0,ctx.width,ctx.height).data,b=c.ctx.getImageData(0,0,ctx.width,ctx.height).data;
     let differences=0;for(let i=0;i<a.length;i++)if(a[i]!==b[i])differences++;
     const layers=m.layers.get(m.cur.id);
     return {time:c.time,scene_time:m.renderTime,intrinsics_error_px:error,world_projection_error_px:poseError,pixel_differences:differences,
       actors:layers.length,hidden_traffic:layers.filter(l=>l.v.key!=='ego'&&!l.mesh.visible).map(l=>l.v.key)};
    }));
   }
   await page.screenshot({path:`${directory}/${id}-browser-contact.png`});
   const moments=[];
   for(const moment of manifest.moments){
    await page.evaluate(async mid=>{await MERCY.moments.show(mid);},moment.id);
    await page.waitForFunction(()=>document.querySelector('#sceneLoad').hidden);
    moments.push(await page.evaluate(()=>({id:MERCY.moments.cur.id,actors:MERCY.moments.layers.get(MERCY.moments.cur.id).length})));
   }
   for(const sample of samples){assert.equal(sample.scene_time,sample.time);assert.equal(sample.pixel_differences,0);assert(sample.intrinsics_error_px<1e-5);assert(sample.world_projection_error_px<.01);assert.deepEqual(sample.hidden_traffic,[]);}
   assert(moments.every(m=>m.actors===manifest.moments[0].vehicles.length));
   const provenance=JSON.parse(fs.readFileSync(`${caseRoot}/${id}/source_provenance.json`));
   assert.equal(await page.locator('#tab-files .caveat a').first().getAttribute('href'),provenance.source_url);
   assert.equal(await page.locator('#tab-files .caveat a').nth(1).getAttribute('href'),provenance.license_url);
   assert((await page.request.get(`${base}/cases/${id}/source_provenance.json`)).ok());
   assert.deepEqual(errors,[]);reports.push({case:id,original_plate:native.text,samples,moments,errors});
   console.log(id,'native plate, decoded-frame sync, all moments and persistent actors PASS');
   await page.close();
  }
  fs.writeFileSync(`${directory}/browser-verification.json`,JSON.stringify(reports,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
