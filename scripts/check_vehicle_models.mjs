import {registerHooks} from 'node:module';
import assert from 'node:assert/strict';
registerHooks({resolve(s,c,next){if(s==='three')return {url:new URL('../frontend/vendor/three.module.min.js',import.meta.url).href,shortCircuit:true};return next(s,c)}});
const THREE=await import('three');
const {createVehicle,VEHICLE_TYPES}=await import('../frontend/js/vehicle-models.js');
const {keyAt,browserForward,actorVisible}=await import('../frontend/js/replay-pose.js');
assert.equal(keyAt([[0,0,0,179],[1,0,0,-179]],.5)[2],180,'heading took the long way around');
assert.equal(keyAt([[1,2,3,0]],0),null);
assert.deepEqual(keyAt([[1,2,3,0]],0,{clampBefore:true}),[2,3,0]);
for(const yaw of [-90,-30,0,30,90,180]){
 const nose=new THREE.Vector3(0,0,-1),direction=new THREE.Vector3(...browserForward(yaw));
 const q=new THREE.Quaternion().setFromUnitVectors(nose,direction);
 const actual=nose.clone().applyQuaternion(q);
 assert(actual.distanceTo(direction)<1e-10,'body and road direction disagree');
 if(Math.abs(yaw)<180 && yaw!==0)assert.equal(Math.sign(actual.x),Math.sign(yaw),'left/right nose mirrored');
}
const persistent={model:{},persistent:true,visible_range:[1,2],observed_range:[1,2]};
for(const t of [-10,0,1,1.5,2,3,20])assert(actorVisible(persistent,t),'a detection boundary removed a physical actor');
assert(!actorVisible({...persistent,existence_range:[1.2,1.8]},1),'explicit physical lifetime ignored');
assert(!actorVisible({visible_range:[1,2]},3),'legacy visibility changed');
function rearHeight(type){
 const m=createVehicle({body_type:type,dimensions_m:[1.82,1.6,4.5],paint_hex:'#7799bb'},{plateTexture:false});
 let height=0;
 m.updateMatrixWorld(true);
 m.traverse(o=>{if(!o.geometry)return;const a=o.geometry.attributes.position;
  for(let i=0;i<a.count;i++){
   const p=new THREE.Vector3().fromBufferAttribute(a,i).applyMatrix4(o.matrixWorld);
   assert(p.toArray().every(Number.isFinite),type+' has invalid geometry');
   assert(Math.abs(p.x)<=1.82*1.15/2+1e-6,type+' extends outside its collision width');
   assert(Math.abs(p.z)<=(4.5+.22)/2+1e-6,type+' extends outside its collision length');
   if(p.z>4.5*.35)height=Math.max(height,p.y);
  }});
 assert.deepEqual(m.userData.forward,[0,0,-1]);m.dispose();return height;
}
const heights=Object.fromEntries(VEHICLE_TYPES.map(t=>[t,rearHeight(t)]));
for(const t of ['hatchback','rounded_hatchback','suv','offroad_suv','crossover','wagon','mpv'])assert(heights[t]>heights.sedan+.15,t+' still has a sedan boot');
const rounded=createVehicle({body_type:'rounded_hatchback',dimensions_m:[1.75,1.52,3.9]},{plateTexture:false});
const shell=rounded.getObjectByName('rounded-body-shell');assert(shell,'domed body missing');assert.equal(shell.geometry.groups.length,2,'body should batch paint/glass into two draws');
const p=shell.geometry.attributes.position,n=shell.geometry.attributes.normal;
let right=0,top=0;for(let i=1;i<p.count;i++){if(p.getX(i)>p.getX(right))right=i;if(p.getY(i)>p.getY(top))top=i;}
assert(n.getX(right)>0&&n.getY(top)>0,'rounded body normals face inward');rounded.dispose();
console.log('All '+VEHICLE_TYPES.length+' models: finite geometry, forward -z; hatch/MPV/SUV rear silhouettes differ from sedan.');
