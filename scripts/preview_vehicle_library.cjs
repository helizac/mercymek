/* Render the real body meshes, including domed hatch and off-road SUV. */
const {chromium}=require('../data/browser-tests/node_modules/playwright');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const directory='data/research/ui-vehicles';fs.mkdirSync(directory,{recursive:true});
 const base=process.env.MERCY_TEST_URL||'http://localhost:8001';
 try{
  const page=await browser.newPage({viewport:{width:1800,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/model-gallery.html',route=>route.fulfill({contentType:'text/html',body:`<!doctype html><html><head><meta charset="utf-8">
  <script type="importmap">{"imports":{"three":"/vendor/three.module.min.js"}}</script>
  <style>body{margin:0;padding:36px;background:#0c1215;color:#e8eeee;font:18px system-ui}h1{font-size:28px;font-weight:500;margin:0 0 8px}p{color:#98abae;margin:0 0 30px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}.card{background:#182228;border:1px solid #304048;border-radius:18px;overflow:hidden}.card b{display:block;padding:12px 22px 20px;font-weight:500}canvas{display:block;width:100%;height:380px}</style></head>
  <body><h1>Vehicle body library</h1><p>Distinct roof, rear-body and off-road geometry · 18 supported types</p><div class="grid"></div>
  <script type="module">
  import * as THREE from 'three';import {createVehicle} from '/js/vehicle-models.js';
  const types=[['sedan','Sedan',[1.82,1.47,4.7],'#aab6be'],['hatchback','Hatchback',[1.78,1.48,4.15],'#287bc0'],['rounded_hatchback','Rounded hatchback',[1.75,1.52,3.9],'#e6d7ad'],['crossover','Crossover',[1.86,1.64,4.55],'#71877f'],['suv','SUV',[1.92,1.75,4.8],'#718ca2'],['offroad_suv','Off-road SUV / Jeep-like',[1.98,1.87,4.65],'#8b9a64']];
  for(const [type,label,dims,paint] of types){
   const card=document.createElement('div');card.className='card';document.querySelector('.grid').append(card);
   const r=new THREE.WebGLRenderer({antialias:true});r.setSize(560,380);r.setPixelRatio(1);card.append(r.domElement);
   const caption=document.createElement('b');caption.textContent=label;card.append(caption);
   const scene=new THREE.Scene();scene.background=new THREE.Color('#182228');scene.add(new THREE.HemisphereLight(0xe5f1ff,0x625f4d,3));
   const light=new THREE.DirectionalLight(0xfff3dc,3);light.position.set(-4,7,5);scene.add(light);
   const model=createVehicle({body_type:type,dimensions_m:dims,paint_hex:paint},{plateTexture:false});scene.add(model);
   const cam=new THREE.PerspectiveCamera(34,560/380,.1,100);cam.position.set(5,2.9,7);cam.lookAt(0,.8,0);r.render(scene,cam);
  }window.galleryReady=true;
  </script></body></html>`}));
  await page.goto(`${base}/model-gallery.html`);await page.waitForFunction(()=>window.galleryReady);
  if(errors.length)throw Error(errors.join('\n'));
  await page.screenshot({path:`${directory}/vehicle-library.png`});console.log('Vehicle geometry preview saved');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
