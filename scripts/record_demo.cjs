/* Revision 6. Every demonstrated command traverses the real microphone PCM,
 * Muse Voice ASR and command router. Only microphone INPUT is prerecorded.
 * Processing waits are cut; real results and source timings are preserved. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),media=path.join(root,'deliverables','production','revision6');
const plan=JSON.parse(fs.readFileSync(path.join(root,'docs','demo-storyboard.json'),'utf8'));
const base=process.env.MERCY_TEST_URL||'http://localhost:8001',only=process.argv.slice(2);
const wait=ms=>new Promise(r=>setTimeout(r,ms));
for(const dir of ['raw','screens'])fs.mkdirSync(path.join(media,dir),{recursive:true});
const python=path.join(root,'.venv','Scripts','python.exe');
function exec(file,args,buffer=false){const r=cp.spawnSync(file,args,{encoding:buffer?null:'utf8',windowsHide:true,maxBuffer:32*1024*1024});if(r.status!==0)throw Error(String(r.stderr||r.error?.message));return r.stdout;}
const ffmpeg=exec(python,['-c',"import shutil;print(shutil.which('ffmpeg'))"]).trim(),ffprobe=path.join(path.dirname(ffmpeg),'ffprobe.exe');
const json=(file,other={})=>fs.existsSync(file)?JSON.parse(fs.readFileSync(file,'utf8')):other;
const records=json(path.join(media,'recordings.json')),validation=json(path.join(media,'capture-validation.json'),{takes:{}});
function save(){fs.writeFileSync(path.join(media,'recordings.json'),JSON.stringify(records,null,2));fs.writeFileSync(path.join(media,'capture-validation.json'),JSON.stringify(validation,null,2));}
function alignment(id){const d=json(path.join(media,id+'.alignment.json'));return d.normalized_alignment||d.alignment;}
function sentenceEnd(id){const a=alignment(id),i=a.characters.findIndex(c=>'.?!'.includes(c));assert(i>=0);return a.character_end_times_seconds[i];}
function audio(id){const wav=path.join(media,id+'.command.wav'),end=sentenceEnd(id);exec(ffmpeg,['-y','-v','error','-i',path.join(media,id+'.mp3'),'-af',`atrim=duration=${end+.08},asetpts=PTS-STARTPTS,adelay=2250:all=1,apad=pad_dur=1`,'-ar','48000','-ac','1','-c:a','pcm_s16le',wav]);return {data:fs.readFileSync(wav).toString('base64'),seconds:end+3.33};}
async function marker(p){await p.evaluate(()=>{const n=document.createElement('div');n.id='production-sync';n.style='position:fixed;inset:0;background:rgb(255,0,255);z-index:2147483647';document.documentElement.append(n);});await wait(220);await p.evaluate(()=>document.querySelector('#production-sync').remove());await p.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));return Date.now();}
function syncPoint(file){const bytes=exec(ffmpeg,['-v','error','-i',file,'-vf','scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'],true);let last=-1;for(let i=0;i<bytes.length;i+=3)if(bytes[i]>240&&bytes[i+1]<20&&bytes[i+2]>240)last=i/3;assert(last>=0,'Missing production sync marker');const info=JSON.parse(exec(ffprobe,['-v','error','-show_streams','-of','json',file]));const [n,d]=info.streams[0].avg_frame_rate.split('/').map(Number);return (last+1)/(n/d);}
async function record(browser,shot){
 const ctx=await browser.newContext({viewport:{width:1920,height:960},deviceScaleFactor:1,recordVideo:{dir:path.join(media,'raw'),size:{width:1920,height:960}}});
 // A real MediaStream containing the spoken input. No transcripts, command
 // responses, network APIs, model results or UI result fields are mocked.
 await ctx.addInitScript(()=>{navigator.mediaDevices.getUserMedia=async()=>{const c=new AudioContext({sampleRate:48000});const bytes=Uint8Array.from(atob(window.__demoSpeech),v=>v.charCodeAt(0));const a=await c.decodeAudioData(bytes.buffer);const s=c.createBufferSource(),d=c.createMediaStreamDestination();s.buffer=a;s.connect(d);await c.resume();window.__demoInput={c,s};const begin=()=>{if(window.MERCY?.net.mic?.live)s.start();else setTimeout(begin,10);};setTimeout(begin,10);return d.stream;};});
 const p=await ctx.newPage(),events=[],errors=[],checks=[];
 p.on('pageerror',e=>errors.push(e.message));await p.exposeFunction('productionEvent',(type,data)=>(events.push({at:Date.now(),type,data}),fs.writeFileSync(path.join(media,'events-live.json'),JSON.stringify({shot:shot.id,events},null,2))));
 const state=patch=>p.evaluate(patch=>MERCY.app.local(patch),patch);
 async function seek(t){await state({t,playing:false});await p.waitForFunction(t=>!MERCY.feed.video.seeking&&Math.abs(MERCY.feed.video.currentTime-t)<.08,t);}
 async function guard(){const a=await p.evaluate(()=>MERCY.feed.video.currentTime);await wait(700);const b=await p.evaluate(()=>MERCY.feed.video.currentTime);checks.push({kind:'source_playback',from:a,to:b});assert(b>a+.15,'Main source video did not advance');}
 async function speech(){const a=audio(shot.id);await p.evaluate(data=>window.__demoSpeech=data,a.data);const box=await p.locator('#mic').boundingBox();await p.mouse.move(box.x+box.width/2,box.y+box.height/2);await p.mouse.down();await wait(a.seconds*1000);await p.mouse.up();await p.waitForFunction(()=>window.__productionFinal,null,{timeout:30000});const transcript=events.find(e=>e.type==='transcript'&&e.data.final);assert(transcript&&transcript.data.typed!==true,'Command did not originate in ASR');console.log('ASR',shot.id,JSON.stringify(transcript.data.text));const expected=alignment(shot.id).characters.join('').match(/^\w+/)[0].toLowerCase();assert(transcript.data.text.toLowerCase().startsWith(expected),'ASR missed the beginning of the command');}
 try{
  await p.goto(base+'/?case=london_rearend&view='+(shot.id==='05_map'?'map':'video')+'&t=7.4');
  await p.waitForFunction(()=>window.MERCY?.app&&MERCY.feed.case?.id==='london_rearend'&&document.querySelector('#boot').classList.contains('out')&&MERCY.feed.video.readyState>=2,null,{timeout:120000});
  if(shot.id==='05_map'){
   await p.waitForFunction(()=>MERCY.map.map?.getZoom()===15&&[...document.querySelectorAll('.leaflet-tile')].length>=12&&[...document.querySelectorAll('.leaflet-tile')].every(i=>i.complete&&i.naturalWidth===256),null,{timeout:60000});await wait(1000);
  }
  await state({view:'video',camera:'driver',playing:false,speed:1,zoom:null,isolate:[],selected:null,trajectory:[],layers:{masks:false,tracks:false,hud:false}});await seek(7.4);
  if(shot.id==='11_paper'){
   // Restore the already established report using a genuine spoken request
   // before this take, so paperwork continues the same investigation on screen.
   const warm=audio('10_report');await p.evaluate(data=>window.__demoSpeech=data,warm.data);
   const box=await p.locator('#mic').boundingBox();await p.mouse.move(box.x+box.width/2,box.y+box.height/2);await p.mouse.down();await wait(warm.seconds*1000);await p.mouse.up();
   await p.waitForFunction(()=>MERCY.panels.verdict?.fault?.length>0,null,{timeout:60000});
   await p.evaluate(async()=>{const m=MERCY.net.mic;if(m){m.stream.getTracks().forEach(t=>t.stop());await m.ctx.close();MERCY.net.mic=null;}if(window.__demoInput)await window.__demoInput.c.close();});
   await wait(700);
  }
  await p.evaluate(()=>{for(const type of ['transcript','intent','state','say','error','seal','enhance','whatif','verdict','replay','statement'])MERCY.net.on(type,data=>{productionEvent(type,data);if(type==='transcript'&&data.final)window.__productionFinal=true;if(type==='statement')window.__productionStatement=true;});});
  if(shot.id==='00_hook')await seek(0);
  if(shot.id==='06_isolate')await seek(8.4);
  console.log('loaded',shot.id);await wait(700);const markerWall=await marker(p),start=markerWall+(shot.id==='00_hook'?0:2000);let ready=start,cutAt=sentenceEnd(shot.id)+1.25;
  if(shot.id==='00_hook'){
   await state({playing:true});await guard();await wait(8500);await state({playing:false});cutAt=null;
  }else{
   await speech();console.log('waiting for result',shot.id);
   if(shot.id==='01_zoom')await p.waitForFunction(()=>!!MERCY.S.zoom,null,{timeout:60000});
   if(shot.id==='02_plate'){
    await p.waitForFunction(()=>document.querySelector('#enh .compare')&&!document.querySelector('#enh .spin')&&document.querySelector('#enh .plate-read .txt')?.innerText==='LM58 BLZ',null,{timeout:180000});cutAt=6.5;
   }
   if(['03_locate','05_map'].includes(shot.id)){
    await p.waitForFunction(()=>MERCY.S.view==='map'&&MERCY.map.map?.getZoom()===15,null,{timeout:60000});
    await p.waitForFunction(()=>{const im=[...document.querySelectorAll('.leaflet-tile')];return im.length>=12&&im.every(i=>i.complete&&i.naturalWidth===256);},null,{timeout:60000});await wait(500);
    // A loaded image is insufficient: check actual composited map pixels too.
    let coverage=0;
    for(let attempt=0;attempt<4;attempt++){
     const png=await p.screenshot();const rgb=cp.spawnSync(ffmpeg,['-v','error','-i','pipe:0','-vf','crop=1000:700:430:100,scale=25:18','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],{input:png,windowsHide:true});assert.equal(rgb.status,0,String(rgb.stderr));let lit=0;for(let i=0;i<rgb.stdout.length;i+=3)if((rgb.stdout[i]+rgb.stdout[i+1]+rgb.stdout[i+2])/3>45)lit++;coverage=lit/(rgb.stdout.length/3);if(coverage>.98)break;
     console.log('Redrawing actual map tiles; pixel coverage',coverage);
     await p.evaluate(()=>MERCY.map.map.eachLayer(l=>{if(l instanceof L.TileLayer)l.redraw();}));await wait(2200);
    }
    assert(coverage>.98,'Incomplete map pixels: '+coverage);
    checks.push({kind:'map_tiles',tiles:await p.locator('.leaflet-tile').count(),pixel_coverage:coverage});
   }
   if(shot.id==='06_isolate'){
    await p.waitForFunction(()=>MERCY.S.isolate.length>0&&Math.abs(MERCY.S.t-5.4)<.15,null,{timeout:60000});
   }
   if(shot.id==='07_slowmo'){cutAt=sentenceEnd(shot.id)+.3;
    await p.waitForFunction(()=>!document.querySelector('#replay').hidden&&document.querySelector('#replayVid').readyState>=2,null,{timeout:60000});
    const a=await p.locator('#replayVid').evaluate(v=>v.currentTime);await wait(600);const b=await p.locator('#replayVid').evaluate(v=>v.currentTime);assert(b>a+.3);checks.push({kind:'replay_playback',from:a,to:b});
    await p.waitForFunction(()=>document.querySelector('#replayVid').currentTime>=2,null,{timeout:10000});
    const clock=await p.locator('#replayVid').evaluate(v=>v.currentTime),info=events.find(e=>e.type==='replay').data;const resultDuration=shot.duration-cutAt;
    assert(info.t0+(clock+resultDuration)/info.factor>=info.impact_t+.1,'The shorter replay must still show the impact');
    checks.push({kind:'impact_in_short_replay',replay_start:clock,result_duration:resultDuration,source_start:info.t0+clock/info.factor,source_end:info.t0+(clock+resultDuration)/info.factor,impact:info.impact_t});
   }
   if(shot.id==='08_evidence'){
    await p.waitForFunction(()=>MERCY.S.view==='evidence'&&document.querySelector('#dossier1 .photo')&&document.querySelector('#platesCard .plate-item')&&document.querySelector('#tab-files .file'),null,{timeout:60000});
    await p.waitForFunction(()=>[...document.querySelectorAll('#v-evidence img')].every(i=>i.complete&&i.naturalWidth>0),null,{timeout:30000});
    checks.push({kind:'evidence_page',dossiers:await p.locator('.dossier:not([hidden])').count(),plate:await p.locator('#platesCard .txt').innerText(),files:await p.locator('#tab-files h3').innerText()});
   }
   if(shot.id==='09_whatif'){
    await p.waitForFunction(()=>document.querySelector('#whatif').innerText.includes('11 to 13'),null,{timeout:60000});await guard();
   }
   if(shot.id==='10_report')await p.waitForFunction(()=>MERCY.S.view==='report'&&MERCY.panels.verdict?.fault?.length>0,null,{timeout:60000});
   if(shot.id==='11_paper'){
    await p.waitForFunction(()=>window.__productionStatement&&document.querySelector('#stmtCard .stmt')&&MERCY.panels.sealState?.intact===true,null,{timeout:60000});await p.locator('#stmtCard').scrollIntoViewIfNeeded();
   }
   await p.waitForFunction(()=>getComputedStyle(document.querySelector('.view.on')).filter==='none',null,{timeout:5000});await wait(100);ready=Date.now();await p.screenshot({path:path.join(media,'screens',shot.id+'-result.png')});
   if(shot.id==='06_isolate'){await state({playing:true,speed:.5});await guard();}
   if(shot.id==='02_plate'){
    const box=await p.locator('#enh .compare').boundingBox();await p.mouse.move(box.x+box.width*.2,box.y+box.height*.5);await wait(800);for(let i=0;i<=30;i++){await p.mouse.move(box.x+box.width*(.2+.6*i/30),box.y+box.height*.5);await wait(40);}await p.mouse.move(1900,940);
   }
  }
  const resultSeconds=Math.max(0,shot.duration-(cutAt||0));await wait(Math.max(0,resultSeconds*1000-(Date.now()-ready))+1300);
  await p.screenshot({path:path.join(media,'screens',shot.id+'-end.png')});
  const end=Date.now(),video=p.video();await ctx.close();const file=await video.path(),relative=path.relative(media,file).replaceAll('\\','/'),sync=syncPoint(file)+(shot.id==='00_hook'?0:2);
  let clips;if(cutAt!==null&&(ready-start)/1000>cutAt+.15){clips=[{file:relative,in:sync,duration:cutAt},{file:relative,in:sync+(ready-start)/1000,duration:shot.duration-cutAt}];}else clips=[{file:relative,in:sync,duration:shot.duration}];
  const bad=events.filter(e=>e.type==='error');assert.deepEqual(errors,[]);assert.deepEqual(bad,[],JSON.stringify(bad));
  records[shot.id]={clips};validation.takes[shot.id]={errors,checks,events,raw_file:relative,sync,wall_start:start,result_at:ready,wall_end:end,processing_seconds_omitted:clips.length>1?(ready-start)/1000-cutAt:0,real_asr:shot.id!=='00_hook'};save();console.log('CAPTURED',shot.id,JSON.stringify(clips));
 }catch(e){validation.takes[shot.id]={failed:true,message:e.message,events,errors};save();await p.screenshot({path:path.join(media,'screens',shot.id+'-failed.png')}).catch(()=>{});await ctx.close().catch(()=>{});throw e;}
}
async function cards(browser){
 const p=await browser.newPage({viewport:{width:1920,height:1080}});
 const style=`*{box-sizing:border-box}body{margin:0;background:#070b0c;color:#eff5f2;font-family:'Segoe UI',sans-serif;width:1920px;height:1080px}.kicker{position:absolute;left:86px;top:48px;color:#85cfc3;font-size:20px;letter-spacing:4px}.headline{position:absolute;left:86px;top:87px;font-size:42px;font-weight:400}.photo{position:absolute;left:86px;top:180px;width:1230px;height:705px;overflow:hidden;border:1px solid #26332f}.photo img{position:absolute}.notes{position:absolute;left:1380px;top:260px;width:450px;font-size:32px;line-height:1.5}.notes small{display:block;font-size:20px;color:#9aada6;margin-top:30px;line-height:1.6}.tag{color:#85cfc3;font-size:20px;text-transform:uppercase;letter-spacing:2px;margin-bottom:20px}.credit{position:absolute;left:86px;top:910px;color:#6d827a;font-size:15px}`;
 const source=name=>'data:image/jpeg;base64,'+fs.readFileSync(path.join(root,'data/cases/london_rearend/geo',name)).toString('base64');
 const items=[
  {file:'clue-sign.png',name:'k_08.40.jpg',title:'The signs identify the corridor.',note:'A10 · M25<br>London · Enfield',small:'Route numbers and destination names, read from the original frame.',css:'width:7680px;left:-5050px;top:-1500px;',duration:8},
  {file:'clue-business.png',name:'k_05.40.jpg',title:'Local details narrow the search.',note:'A local business<br>A UK rear plate',small:'Visible business text and the yellow plate provide additional geographic clues.',css:'width:1920px;left:-300px;top:-160px;',duration:5},
  {file:'clue-road.png',name:'k_08.40.jpg',title:'The whole environment contributes.',note:'Road layout<br>Traffic direction<br>Street furniture',small:'The automatic analysis combines visible clues and checks the named places.',css:'width:1253px;left:-12px;top:0;',duration:5}
 ];
 for(const item of items){await p.setContent(`<html><head><style>${style}</style></head><body><div class="kicker">LOCATION · FROM THE VIDEO</div><div class="headline">${item.title}</div><div class="photo">${item.file==='clue-business.png'?`<div style="position:absolute;left:0;top:0;width:580px;height:705px;overflow:hidden"><img src="${source(item.name)}" style="width:4800px;left:10px;top:-900px"></div><div style="position:absolute;left:610px;top:0;width:620px;height:705px;overflow:hidden"><img src="${source(item.name)}" style="width:1920px;left:-950px;top:-280px"></div>`:`<img src="${source(item.name)}" style="${item.css}">`}</div><div class="notes"><div class="tag">Visual evidence</div>${item.note}<small>${item.small}</small></div><div class="credit">Original dashcam frame · Dash Cam Owners World · CC BY 3.0 · Crop used to highlight visible clues</div></body></html>`);await p.screenshot({path:path.join(media,item.file)});}
 records['04_clues']={clips:items.map(i=>({file:i.file,image:true,full_frame:true,duration:i.duration}))};
 await p.setContent(`<html><head><style>${style}body{padding:160px 140px}b{font-size:115px;font-weight:500;letter-spacing:3px}h1{font-size:49px;font-weight:400;color:#85cfc3;margin-top:36px}p{font-size:26px;color:#9aada6}.features{margin-top:68px;font-size:26px;line-height:1.7;color:#d4dfda}.credits{position:absolute;left:140px;bottom:150px;font-size:15px;line-height:1.8;color:#7b9187}</style></head><body><b>mercymek</b><h1>Your voice runs the investigation.</h1><p>One conversation, from footage to findings.</p><div class="features">Zoom · Read plates · Locate the scene<br>Replay · Reconstruct · Test scenarios · Prepare the report</div><div class="credits">Muse Voice · Muse Spark · Muse Wonder · Muse Image · SAM 3.1 · VGGT · RAFT · ElevenLabs<br>Footage: Dash Cam Owners World, CC BY 3.0 · Full attribution accompanies the demo.</div></body></html>`);await p.screenshot({path:path.join(media,'close.png')});records['12_close']={clips:[{file:'close.png',image:true,full_frame:true,duration:8}]};save();await p.close();
}
(async()=>{
 for(let i=0;i<180;i++){try{if((await(await fetch(base+'/api/status')).json()).laya)break;}catch{}await wait(500);}
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist',process.env.MERCY_RECORD_SOFTWARE?'--use-angle=swiftshader':'--use-angle=d3d11','--enable-unsafe-swiftshader','--autoplay-policy=no-user-gesture-required']});
 try{for(const shot of plan.shots){if(['04_clues','12_close','07b_reconstruction'].includes(shot.id))continue;if(only.length&&!only.includes(shot.id))continue;await record(browser,shot);}await cards(browser);}finally{await browser.close();}
 console.log('Recording complete. Actual microphone input and ASR validation saved.');
})().catch(e=>{console.error(e);process.exitCode=1});
