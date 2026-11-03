/* Export the real 4D viewer at deterministic 30fps; neural reconstruction remains on Kaggle.
   node scripts/export_approach.cjs cutin_rearend */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const path=require('node:path');
const fs=require('node:fs');
const caseId=process.argv[2]||'cutin_rearend';
if(!/^[a-z0-9_-]+$/.test(caseId))throw Error('Invalid case id');
const output=path.resolve('data/cases',caseId,'environment/approach/approach.mp4');
const temporary=output.replace(/\.mp4$/,'.rendering.mp4');
const html=`<!doctype html><meta charset="utf-8"><script type="importmap">{"imports":{"three":"/vendor/three.module.min.js","three/addons/":"/vendor/addons/"}}</script>
<style>html,body{margin:0;width:100%;height:100%;background:#101519}#v-scene{position:absolute;inset:0}#gl{width:100%;height:100%}
button,#pip,#hud3,#sceneLoad{display:none!important}footer{position:fixed;bottom:12px;left:14px;color:white;background:#14201fbb;padding:7px 10px;border-radius:6px;font:12px Arial}
#time{position:fixed;right:14px;bottom:12px;color:white;background:#14201fbb;padding:7px 10px;border-radius:6px;font:12px monospace}</style>
<div id="v-scene"><canvas id="gl"></canvas><canvas id="pip"></canvas><div id="hud3"></div><div id="sceneLoad"></div></div>
<footer>AI ile üretilmiş ortam · tahmini rekonstrüksiyon</footer><div id="time"></div>`;
let browser,ff;
(async()=>{
 browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1280,height:720}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/capture-approach',route=>route.fulfill({contentType:'text/html',body:html}));
 await page.goto('http://localhost:8000/capture-approach');
 const timing=await page.evaluate(async id=>{
   const c=await(await fetch('/api/case/'+id)).json();
   const m=c.moments.find(m=>m.id==='impact')||c.moments.at(-1);
   if(!m.approach)throw Error('No published approach extension');
   const {Moments}=await import('/js/moments.js');
   window.capture=new Moments();capture.setCase(c);await capture.show(m.id);capture.setMode('approach');
   capture.draw(0,null,{t:m.replay_start});
   return {start:m.replay_start,end:m.replay_end,height:m.cam.pos[1],distance:m.approach.distance_m,
     vehicle_registration:m.reconstruction,motion_version:m.motion_version,persistent_world_actors:true};
 },caseId);
 await page.waitForTimeout(1200);
 ff=spawn('ffmpeg',['-y','-loglevel','error','-f','image2pipe','-framerate','30','-i','pipe:0',
   '-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',temporary]);
 let stderr='';ff.stderr.on('data',data=>{stderr=(stderr+data).slice(-4000)});
 const completion=once(ff,'close');
 const frames=Math.ceil((timing.end-timing.start)*60);
 for(let i=0;i<=frames;i++){
   const t=Math.min(timing.end,timing.start+i/60);
   const height=await page.evaluate(t=>{capture.draw(t*1000,null,{t});document.querySelector('#time').textContent='T+'+t.toFixed(2)+'s';return capture.cam.position.y},t);
   if(Math.abs(height-timing.height)>1e-6)throw Error('Camera height changed during export');
   const png=await page.screenshot({type:'png'});
   if(i===Math.floor(frames/3))fs.writeFileSync(output.replace(/\.mp4$/,'-preview.png'),png);
   if(!ff.stdin.write(png))await once(ff.stdin,'drain');
   if(i%60===0)console.log('frame',i,'/',frames);
 }
 ff.stdin.end();const [code]=await completion;
 await browser.close();
 if(code!==0)throw Error(stderr||'ffmpeg failed');
 if(errors.length)throw Error(errors.join('\n'));
 fs.renameSync(temporary,output);
 fs.writeFileSync(output.replace(/\.mp4$/,'.json'),JSON.stringify({case:caseId,...timing,fps:30,speed:.5,frames:frames+1,
   generated_environment:true,poses_from_source_detections:true,unobserved_motion_estimated:true,camera_height_constant:true},null,2));
 console.log(output);
})().catch(error=>{console.error(error);process.exitCode=1}).finally(async()=>{
 if(ff&&ff.exitCode===null)ff.kill();
 if(browser)await browser.close();
 fs.rmSync(temporary,{force:true});
});
