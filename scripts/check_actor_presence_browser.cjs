/* Departed traffic stays in the world and follows its observed camera exit. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const base=process.env.MERCY_TEST_URL||'http://localhost:8000';
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const directory='data/research/actor-presence';fs.mkdirSync(directory,{recursive:true});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`${base}/?case=hero&view=scene&t=10.95`);
  await page.waitForFunction(()=>window.MERCY?.feed.case?.id==='hero'&&MERCY.moments.cur&&document.querySelector('#sceneLoad').hidden,{timeout:90000});
  async function seek(t){
   await page.evaluate(t=>{MERCY.S.t=t+.002;MERCY.S.playing=false;MERCY.feed.video.currentTime=MERCY.S.t;MERCY.moments.setMode('driver');},t);
   const expected=Math.floor((t+.002)*30.5+1e-6)/30.5;
   await page.waitForFunction(expected=>!MERCY.feed.video.seeking&&Math.abs(MERCY.feed.clock.time-expected)<.001&&MERCY.moments.renderTime===MERCY.feed.clock.time,expected);
   await page.waitForTimeout(100);
  }
  async function sample(){return page.evaluate(async()=>{
   const THREE=await import('/vendor/three.module.min.js'),m=MERCY.moments;
   m.cam.updateMatrixWorld(true);
   const frustum=new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(m.cam.projectionMatrix,m.cam.matrixWorldInverse));
   const actors=m.layers.get(m.cur.id).map(({v,mesh})=>{
    mesh.updateMatrixWorld(true);const max=frustum.planes.map(()=>-Infinity),p=new THREE.Vector3();
    mesh.traverse(o=>{
     const a=o.geometry?.attributes.position;if(!a||o.material?.transparent)return;
     for(let i=0;i<a.count;i++){
      p.fromBufferAttribute(a,i).applyMatrix4(o.matrixWorld);
      frustum.planes.forEach((plane,j)=>{max[j]=Math.max(max[j],plane.distanceToPoint(p));});
     }
    });
    return {key:v.key,id:mesh.uuid,visible:!!mesh.visible,inside_camera:max.every(x=>x>=0),position:mesh.position.toArray()};
   });
   return {time:MERCY.feed.clock.time,scene_time:m.renderTime,actors};
  });}
  const samples=[];
  for(const t of [10.95,5.5,8.13,10.95]){
   await seek(t);const s=await sample();assert.equal(s.scene_time,s.time);
   for(const v of s.actors)if(v.key!=='ego')assert(v.visible,v.key+' disappeared after a source observation ended');
   if(t===10.95){
    assert(!s.actors.find(v=>v.key==='car:3').inside_camera,'departed body re-entered the source camera');
    assert(s.actors.find(v=>v.key==='car:16').inside_camera,'actual white source car moved outside the image');
    await page.screenshot({path:`${directory}/hero-persistent-source-10.95.png`});
   }
   samples.push(s);
  }
  await page.evaluate(()=>{
   const m=MERCY.moments,p=m.layers.get(m.cur.id).find(l=>l.v.key==='car:3').mesh.position;
   m.setMode('free');m.cameraSeeded=true;m.cam.position.set(p.x+7,4,p.z+6);m.cam.lookAt(p.x,.7,p.z);
  });
  await page.waitForTimeout(200);const followed=await sample(),departed=followed.actors.find(v=>v.key==='car:3');
  assert(departed.visible&&departed.inside_camera,'departed actor cannot be inspected from a free camera');
  await page.screenshot({path:`${directory}/hero-follow-departed.png`});
  const ids=new Map(samples[0].actors.map(v=>[v.key,v.id]));
  for(const s of [...samples,followed])for(const v of s.actors)assert.equal(v.id,ids.get(v.key));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(`${directory}/browser.json`,JSON.stringify({case:'hero',samples,followed,console_errors:errors},null,2));
  console.log('Persistent actors PASS; departed body outside source camera and visible from free camera; actual source traffic retained');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
