/* A short, frozen comparison of the real video frame and the actual app renderer.
 * No generated replacement footage, camera flight or animated model claims. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),media=path.join(root,'deliverables/production/revision5');
const probe=process.argv.includes('--probe'),selected=+(process.env.MERCY_COMPARE_TIME||'7.4');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
function writeImage(file,url){fs.writeFileSync(path.join(media,file),Buffer.from(url.split(',')[1],'base64'));}
(async()=>{
 fs.mkdirSync(media,{recursive:true});
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist','--use-angle=d3d11']});
 const p=await browser.newPage({viewport:{width:1920,height:960},deviceScaleFactor:1}),errors=[];p.on('pageerror',e=>errors.push(e.message));
 try{
  await p.goto('http://localhost:8001/?case=london_rearend&view=scene&t=7.8');
  await p.waitForFunction(()=>window.MERCY?.app&&MERCY.feed.case?.id==='london_rearend'&&MERCY.moments.cur&&document.querySelector('#sceneLoad').hidden&&MERCY.feed.video.readyState>=2,null,{timeout:120000});
  const samples=[];
  for(const t of probe?[6.8,7.4,7.8,8.4]:[selected]){
   await p.evaluate(t=>MERCY.app.local({view:'scene',camera:'driver',moment:'impact',t,playing:false,zoom:null,isolate:[],layers:{masks:false,tracks:false,hud:false}}),t);
   await p.waitForFunction(t=>!MERCY.feed.video.seeking&&Math.abs(MERCY.feed.video.currentTime-t)<.034&&MERCY.moments.mode==='driver'&&document.querySelector('#sceneLoad').hidden,t,{timeout:60000});await wait(1500);
   const shot=await p.evaluate(()=>{
    const f=MERCY.feed.video,cv=document.createElement('canvas');cv.width=f.videoWidth;cv.height=f.videoHeight;cv.getContext('2d').drawImage(f,0,0);
    const m=MERCY.moments;m.r.render(m.scene,m.cam);
    return {original:cv.toDataURL('image/png'),reconstruction:m.canvas.toDataURL('image/png'),sourceTime:f.currentTime,stateTime:MERCY.S.t,camera:m.mode,moment:m.cur.id,paused:f.paused,caseId:MERCY.feed.case.id,renderer:'actual application WebGL scene',estimate:m.cur.reconstruction};
   });
   assert(shot.paused&&Math.abs(shot.sourceTime-t)<.034&&Math.abs(shot.stateTime-t)<.034);assert.equal(shot.camera,'driver');
   const tag=t.toFixed(2);writeImage(`compare-${tag}-original.png`,shot.original);writeImage(`compare-${tag}-3d.png`,shot.reconstruction);
   const {original,reconstruction,...meta}=shot;samples.push(meta);console.log('Captured paired frame',tag);
  }
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(media,probe?'comparison-probes.json':'comparison-validation.json'),JSON.stringify({errors,samples},null,2));
  if(!probe){
   const q=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});const tag=selected.toFixed(2);
   const url=f=>'data:image/png;base64,'+fs.readFileSync(path.join(media,f)).toString('base64');
   await q.setContent(`<html><head><style>*{box-sizing:border-box}body{margin:0;width:1920px;height:1080px;background:#070b0c;color:#eff5f2;font-family:'Segoe UI',sans-serif}.kicker{position:absolute;left:80px;top:55px;color:#85cfc3;font-size:19px;letter-spacing:4px}.title{position:absolute;left:80px;top:100px;font-size:44px}.grid{position:absolute;left:80px;top:250px;display:flex;gap:40px}.panel{width:860px}.label{font-size:23px;margin-bottom:20px;color:#d9e5df}.label span{float:right;color:#85cfc3;font-size:20px}.image{width:860px;height:484px;border:1px solid #314239;border-radius:10px;overflow:hidden}.image img{width:100%;height:100%;object-fit:contain}.foot{position:absolute;left:80px;top:820px;color:#98ada3;font-size:23px}.credit{position:absolute;left:80px;top:910px;font-size:15px;color:#6f867a}</style></head><body><div class="kicker">RECONSTRUCTION · SAME MOMENT</div><div class="title">Original footage beside estimated 3D positions.</div><div class="grid"><div class="panel"><div class="label">Original frame <span>T+${tag} s</span></div><div class="image"><img src="${url('compare-'+tag+'-original.png')}"></div></div><div class="panel"><div class="label">Estimated 3D · dashcam angle <span>T+${tag} s</span></div><div class="image"><img src="${url('compare-'+tag+'-3d.png')}"></div></div></div><div class="foot">A spatial view of the approach · Geometry and vehicle bodies are estimates.</div><div class="credit">Actual mercymek renderer · Estimated geometry and vehicle bodies · Dash Cam Owners World, CC BY 3.0</div></body></html>`);
   await q.screenshot({path:path.join(media,'comparison.png')});await q.close();
   const records=JSON.parse(fs.readFileSync(path.join(media,'recordings.json'),'utf8'));records['07b_reconstruction']={clips:[{file:'comparison.png',image:true,full_frame:true,duration:5}]};fs.writeFileSync(path.join(media,'recordings.json'),JSON.stringify(records,null,2));
  }
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
