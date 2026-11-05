/* Brief actual web-app 3D view, following the same-time comparison card.
 * Records the unmodified interface; no orbit, auto-flight or half-speed playback. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),media=path.join(root,'deliverables/production/revision6');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const python=path.join(root,'.venv','Scripts','python.exe');
function exec(file,args,buffer=false){const r=cp.spawnSync(file,args,{encoding:buffer?null:'utf8',windowsHide:true,maxBuffer:32*1024*1024});if(r.status!==0)throw Error(String(r.stderr||r.error?.message));return r.stdout;}
const ffmpeg=exec(python,['-c',"import shutil;print(shutil.which('ffmpeg'))"]).trim(),ffprobe=path.join(path.dirname(ffmpeg),'ffprobe.exe');
function syncPoint(file){const bytes=exec(ffmpeg,['-v','error','-i',file,'-vf','scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'],true);let last=-1;for(let i=0;i<bytes.length;i+=3)if(bytes[i]>240&&bytes[i+1]<20&&bytes[i+2]>240)last=i/3;assert(last>=0);const info=JSON.parse(exec(ffprobe,['-v','error','-show_streams','-of','json',file]));const [n,d]=info.streams[0].avg_frame_rate.split('/').map(Number);return (last+1)/(n/d);}
(async()=>{
 for(const dir of ['raw','screens'])fs.mkdirSync(path.join(media,dir),{recursive:true});
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist','--use-angle=d3d11']});
 const ctx=await browser.newContext({viewport:{width:1920,height:960},deviceScaleFactor:1,recordVideo:{dir:path.join(media,'raw'),size:{width:1920,height:960}}});
 const p=await ctx.newPage(),errors=[];p.on('pageerror',e=>errors.push(e.message));
 try{
  await p.goto('http://localhost:8001/?case=london_rearend&view=video&t=7.4');
  await p.waitForFunction(()=>window.MERCY?.app&&MERCY.feed.case?.id==='london_rearend'&&MERCY.feed.video.readyState>=2&&document.querySelector('#boot').classList.contains('out'),null,{timeout:120000});
  await p.locator('#viewNav [data-view="scene"]').click();
  await p.evaluate(()=>MERCY.app.local({view:'scene',camera:'driver',moment:'impact',t:7.4,playing:false,zoom:null,isolate:[],selected:null,layers:{masks:false,tracks:false,hud:false}}));
  await p.waitForFunction(()=>MERCY.moments.cur?.id==='impact'&&MERCY.moments.mode==='driver'&&document.querySelector('#sceneLoad').hidden&&!MERCY.feed.video.seeking&&Math.abs(MERCY.feed.clock.time-7.4)<.034&&Math.abs(MERCY.moments.renderTime-7.4)<.034,null,{timeout:120000});
  await wait(1300);await p.mouse.move(1900,945);
  const read=()=>p.evaluate(()=>({caseId:MERCY.feed.case.id,view:MERCY.S.view,camera:MERCY.moments.mode,sourceTime:MERCY.feed.clock.time,rendererTime:MERCY.moments.renderTime,videoTime:MERCY.feed.video.currentTime,playing:MERCY.S.playing,selectedNavigation:document.querySelector('#viewNav button.on')?.textContent,sourceVideoVisible:!!document.querySelector('video.scene-source-video')||!document.querySelector('#pip').hidden,estimateLabel:document.querySelector('#hud3 .ai')?.innerText}));
  const before=await read();assert.equal(before.view,'scene');assert.equal(before.camera,'driver');assert.equal(before.selectedNavigation,'3D moments');assert(before.sourceVideoVisible&&!before.playing&&before.estimateLabel);
  const comparison=JSON.parse(fs.readFileSync(path.join(media,'comparison-validation.json'),'utf8')).samples[0];assert(Math.abs(before.sourceTime-comparison.sourceTime)<.034,'Web-app view must use the same source frame as the comparison');
  await p.screenshot({path:path.join(media,'screens','website-3d-start.png')});
  await p.evaluate(()=>{const n=document.createElement('div');n.id='production-sync';n.style='position:fixed;inset:0;background:rgb(255,0,255);z-index:2147483647';document.documentElement.append(n);});await wait(220);await p.evaluate(()=>document.querySelector('#production-sync').remove());await p.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  const start=Date.now();await wait(4500);const after=await read();await p.screenshot({path:path.join(media,'screens','website-3d-end.png')});assert.deepEqual(errors,[]);assert(Math.abs(after.rendererTime-before.rendererTime)<.034&&Math.abs(after.sourceTime-before.sourceTime)<.034);
  const video=p.video();await ctx.close();const file=await video.path(),sync=syncPoint(file),relative=path.relative(media,file).replaceAll('\\','/');
  const records=JSON.parse(fs.readFileSync(path.join(media,'recordings.json'),'utf8'));records['07b_reconstruction']={clips:[{file:'comparison.png',image:true,full_frame:true,duration:2},{file:relative,in:sync,duration:3}]};fs.writeFileSync(path.join(media,'recordings.json'),JSON.stringify(records,null,2));
  const validation={errors,before,after,raw_file:relative,sync,wall_start:start,seconds_in_final:3,interface:'actual application, unchanged',camera_motion:'none',original_frame_visible:true};fs.writeFileSync(path.join(media,'website-3d-validation.json'),JSON.stringify(validation,null,2));
  const v=JSON.parse(fs.readFileSync(path.join(media,'capture-validation.json'),'utf8'));v.takes['07b_webapp']={...validation,real_asr:false};fs.writeFileSync(path.join(media,'capture-validation.json'),JSON.stringify(v,null,2));console.log('CAPTURED actual web-app 3D interface',JSON.stringify(validation));
 }catch(e){await p.screenshot({path:path.join(media,'screens','website-3d-failed.png')}).catch(()=>{});await ctx.close().catch(()=>{});throw e;}finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
