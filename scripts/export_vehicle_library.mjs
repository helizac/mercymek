/* node scripts/export_vehicle_library.mjs
   Exports the exact browser mesh templates as reusable GLBs; no GPU or image model required. */
import { registerHooks } from 'node:module';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
registerHooks({ resolve(specifier, context, nextResolve) {
  if (specifier === 'three') return { url: new URL('../frontend/vendor/three.module.min.js', import.meta.url).href, shortCircuit: true };
  return nextResolve(specifier, context);
}});
// GLTFExporter uses the browser FileReader API for a binary buffer. No DOM is required without textures.
globalThis.FileReader = class {
  readAsArrayBuffer(blob) { blob.arrayBuffer().then(data => { this.result = data; this.onloadend?.(); }); }
  readAsDataURL(blob) { blob.arrayBuffer().then(data => { this.result = `data:${blob.type};base64,${Buffer.from(data).toString('base64')}`; this.onloadend?.(); }); }
};
const { createVehicle, VEHICLE_TYPES } = await import('../frontend/js/vehicle-models.js');
const { Vector3 } = await import('../frontend/vendor/three.module.min.js');
const { GLTFExporter } = await import('../frontend/vendor/exporters/GLTFExporter.js');
const dimensions = { sedan:[1.82,1.47,4.7], hatchback:[1.78,1.48,4.15], suv:[1.92,1.75,4.8], coupe:[1.85,1.35,4.5],
  rounded_hatchback:[1.75,1.52,3.9], offroad_suv:[1.98,1.87,4.65],
  crossover:[1.86,1.64,4.55], wagon:[1.84,1.50,4.75], mpv:[1.92,1.86,4.90], fastback:[1.86,1.47,4.75],
  pickup:[2,1.85,5.5], van:[2,2.3,5.4], box_truck:[2.5,3.5,8], semi_truck:[2.55,3.9,16], tanker:[2.55,3.6,12],
  dump_truck:[2.5,3.3,8], bus:[2.55,3.2,12], motorcycle:[.8,1.2,2.1] };
const dst = path.join(root, 'data', 'models', 'vehicles'); await mkdir(dst, { recursive: true });
const index = [];
const fitPoints = {};
for (const type of VEHICLE_TYPES) {
  const profile = { body_type:type, paint_hex:'#a7afb8', dimensions_m:dimensions[type], plate:{text:''} };
  const mesh = createVehicle(profile, {plateTexture:false}); mesh.name = type;
  // Use the displayed body silhouette for image fitting. A full-height cuboid puts
  // roof corners over the bonnet and biases the depth of close, partly visible cars.
  mesh.updateMatrixWorld(true);
  const points=[];
  mesh.traverse(object=>{
    const positions=object.geometry?.getAttribute('position');
    if(!positions)return;
    for(let i=0;i<positions.count;i++){
      const p=new Vector3().fromBufferAttribute(positions,i).applyMatrix4(object.matrixWorld);
      points.push([p.x,p.y,-p.z]);
    }
  });
  const extreme=new Set();
  for(let i=0;i<512;i++){
    const y=1-2*(i+.5)/512,r=Math.sqrt(1-y*y),angle=i*Math.PI*(3-Math.sqrt(5));
    const n=[r*Math.cos(angle),y,r*Math.sin(angle)];let best=0,score=-Infinity;
    for(let j=0;j<points.length;j++){
      const dot=points[j].reduce((sum,x,k)=>sum+x*n[k],0);
      if(dot>score){score=dot;best=j;}
    }
    extreme.add(best);
  }
  fitPoints[type]={dimensions_m:dimensions[type],points:[...extreme].map(i=>points[i])};
  // Material instances in runtime-only userData are not glTF metadata.
  mesh.userData = { type, dimensions_m:dimensions[type], geometry:'parametric complete body' };
  const buffer = await new GLTFExporter().parseAsync(mesh, {binary:true});
  const file = `${type}.glb`; await writeFile(path.join(dst, file), Buffer.from(buffer));
  index.push({ type, dimensions_m:dimensions[type], file, bytes:buffer.byteLength });
  mesh.dispose();
}
await writeFile(path.join(dst, 'index.json'), JSON.stringify({ units:'metres', axes:'x right, y up, nose -z', models:index }, null, 2));
await writeFile(path.join(dst, 'fit-points.json'), JSON.stringify({axes:'x right, y up, nose +z',models:fitPoints}));
console.log(`Saved ${index.length} reusable vehicle GLBs to ${dst}`);
