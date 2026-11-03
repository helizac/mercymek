/* Actual decoded-frame and calibrated-camera checks against the running app. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const reports=[];
 try{
  for(const id of (process.argv.slice(2).length?process.argv.slice(2):['cutin_rearend','hero','lanechange','night_rain','snow_pileup'])){
   const page=await browser.newPage({viewport:{width:1600,height:1000}});
   await page.route('**/contact-projection/*',route=>route.fulfill({contentType:'application/json',
    body:fs.readFileSync(`data/research/contacts/${id}-projection.json`,'utf8')}));
   const errors=[];page.on('pageerror',e=>errors.push(e.message));
   await page.goto(`http://localhost:8000/?case=${id}&view=scene&t=3.54`);
   try{await page.waitForFunction(id=>MERCY?.feed.case?.id===id&&MERCY.moments.cur&&document.querySelector('#sceneLoad').hidden&&Math.abs(MERCY.feed.clock.time-3.54)<.06,id,{timeout:90000});}
   catch(error){console.error('load',id,errors,await page.evaluate(()=>({case:window.MERCY?.feed.case?.id,state:window.MERCY?.S,clock:window.MERCY?.feed.clock.time,source:window.MERCY?.feed.video.currentSrc,cur:window.MERCY?.moments.cur?.id,load:document.querySelector('#sceneLoad').textContent,hidden:document.querySelector('#sceneLoad').hidden})));await page.screenshot({path:'data/research/contacts/load-failure.png'});throw error;}
   await page.locator('.scene-side').click();
   const samples=[];
   for(const t of [3.54,1.02,5.11,2.27]){
    await page.evaluate(t=>{MERCY.S.t=t;MERCY.S.playing=false;},t);
    try{await page.waitForFunction(t=>!MERCY.feed.video.seeking&&Math.abs(MERCY.feed.clock.time-t)<.06&&MERCY.moments.renderTime===MERCY.feed.clock.time&&MERCY.moments.renderedFrameRevision===MERCY.feed.clock.revision,t,{timeout:15000});}
    catch(error){console.error('seek',t,errors,await page.evaluate(()=>({state:MERCY.S,clock:MERCY.feed.clock.time,callbacks:MERCY.feed.clock.callbacks,scene:MERCY.moments.renderTime,video:MERCY.feed.video.currentTime,seeking:MERCY.feed.video.seeking,ready:MERCY.feed.video.readyState,parent:MERCY.feed.video.parentElement.id,rect:MERCY.feed.video.getBoundingClientRect().toJSON(),style:getComputedStyle(MERCY.feed.video).cssText})));await page.screenshot({path:'data/research/contacts/sync-failure.png'});throw error;}
    samples.push(await page.evaluate(async()=>{
     const m=MERCY.moments,clock=MERCY.feed.clock,t=clock.time;
     const THREE=await import('/vendor/three.module.min.js');
     const [fx,fy,cx,cy]=m.sourceIntrinsics(m.cur,t),[w,h]=m.cur.source_size;
     // Project an arbitrary camera-space point through the actual renderer matrix.
     const p=new THREE.Vector3(.4,-.2,-8).applyMatrix4(m.cam.projectionMatrix);
     const error=Math.hypot((p.x+1)*w/2-(cx+fx*.4/8),(1-p.y)*h/2-(cy+fy*.2/8));
     const original=document.createElement('canvas');original.width=clock.canvas.width;original.height=clock.canvas.height;
     const originalCtx=original.getContext('2d');originalCtx.drawImage(MERCY.feed.video,0,0);
     const actual=originalCtx.getImageData(0,0,original.width,original.height).data;
     const captured=clock.ctx.getImageData(0,0,original.width,original.height).data;
     let videoDifferent=0;for(let i=0;i<captured.length;i++)if(actual[i]!==captured[i])videoDifferent++;
     const fixtures=await(await fetch('/contact-projection/'+MERCY.feed.case.id)).json();
     const fixture=fixtures.reduce((a,b)=>Math.abs(a.time-t)<Math.abs(b.time-t)?a:b);
     let poseError=0;
     for(let i=0;i<fixture.points_road.length;i++){
      const [x,y,z]=fixture.points_road[i],projected=new THREE.Vector3(x,y,-z).project(m.cam);
      poseError=Math.max(poseError,Math.hypot((projected.x+1)*w/2-fixture.pixels[i][0],(1-projected.y)*h/2-fixture.pixels[i][1]));
     }
     return {presented_time:t,scene_time:m.renderTime,intrinsics_error_px:error,world_projection_error_px:poseError,
      clock_source:clock.source,native_video_in_comparison_window:MERCY.feed.video.parentElement===m.root,
      decoded_video_different_channels:videoDifferent};
    }));
   }
   await page.evaluate(()=>{MERCY.S.t=3.54;MERCY.S.playing=false;});
   await page.waitForFunction(()=>Math.abs(MERCY.feed.clock.time-3.54)<.06&&MERCY.moments.renderTime===MERCY.feed.clock.time);
   await page.screenshot({path:`data/research/contacts/${id}-dashcam.png`});
   await page.evaluate(()=>{MERCY.S.playing=true;});
   await page.waitForTimeout(700);
   const playing=await page.evaluate(()=>({clock:MERCY.feed.clock.time,scene:MERCY.moments.renderTime,control:MERCY.S.t}));
   assert(playing.clock>3.6,'video did not advance');assert.equal(playing.scene,playing.clock,'playing scene uses a different frame');
   for(const s of samples){assert.equal(s.scene_time,s.presented_time);assert(s.intrinsics_error_px<1e-6);assert(s.world_projection_error_px<.01);assert(s.native_video_in_comparison_window);assert.equal(s.decoded_video_different_channels,0,JSON.stringify(s));}
   assert.deepEqual(errors,[]);
   reports.push({case:id,seek_samples:samples,playing,frame_pixels_and_scene_share_timestamp:true});
   console.log(id,'frame sync and projection PASS');await page.close();
  }
  fs.writeFileSync('data/research/contacts/browser-sync.json',JSON.stringify(reports,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
