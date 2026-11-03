/* Complete reusable vehicle meshes. Units: metres, bottom on y=0, nose toward -z.
   Paint/type/plate come from observed footage; hidden surfaces are a parametric approximation. */
import * as THREE from 'three';

export const VEHICLE_TYPES = ['sedan', 'hatchback', 'rounded_hatchback', 'suv', 'offroad_suv', 'crossover', 'wagon', 'mpv', 'fastback', 'coupe', 'pickup', 'van', 'box_truck', 'semi_truck', 'tanker', 'dump_truck', 'bus', 'motorcycle'];

export function createVehicle(profile, { plateTexture = true } = {}) {
  const root = new THREE.Group();
  const [w, h, l] = profile.dimensions_m || [1.82, 1.47, 4.7];
  const type = profile.body_type || 'sedan';
  const paint = new THREE.MeshStandardMaterial({ color: profile.paint_hex || '#737a80', metalness: .28, roughness: .33 });
  const glass = new THREE.MeshStandardMaterial({ color: '#263b48', metalness: .45, roughness: .16 });
  const rubber = new THREE.MeshStandardMaterial({ color: '#15181b', roughness: .9 });
  const trim = new THREE.MeshStandardMaterial({ color: '#333b40', metalness: .55, roughness: .4 });
  const alloy = new THREE.MeshStandardMaterial({ color: '#bbc2c5', metalness: .7, roughness: .3 });
  const white = new THREE.MeshStandardMaterial({ color: '#faf3da', emissive: '#b5a477', emissiveIntensity: .28 });
  const red = new THREE.MeshStandardMaterial({ color: '#a71922', emissive: '#8a0711', emissiveIntensity: .3 });
  if (plateTexture) {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d'), gradient = ctx.createRadialGradient(64,64,12,64,64,64);
    gradient.addColorStop(0,'rgba(0,0,0,0.45)'); gradient.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=gradient; ctx.fillRect(0,0,128,128);
    const map=new THREE.CanvasTexture(canvas);
    const shadow=new THREE.Mesh(new THREE.PlaneGeometry(w*1.45,l*1.10),
      new THREE.MeshBasicMaterial({map,transparent:true,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-2}));
    shadow.rotation.x=-Math.PI/2;shadow.position.y=.025;root.add(shadow);
  }
  function box(x, y, z, sx, sy, sz, mat = paint) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
    mesh.position.set(x, y, z); root.add(mesh); return mesh;
  }
  function cylinder(x, y, z, radius, length, mat, axis = 'x') {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 24), mat);
    if (axis === 'x') mesh.rotation.z = Math.PI / 2;
    if (axis === 'z') mesh.rotation.x = Math.PI / 2;
    mesh.position.set(x, y, z); root.add(mesh); return mesh;
  }
  const heavy = ['box_truck', 'semi_truck', 'tanker', 'dump_truck', 'bus'].includes(type);
  const offroad = type === 'offroad_suv';
  const rounded = type === 'rounded_hatchback';
  const radius = type === 'motorcycle' ? .31 : heavy ? .49 : offroad ? .39 : ['suv','crossover','pickup','mpv'].includes(type) ? .36 : .31;
  const chassis = heavy ? .76 : type === 'motorcycle' ? .48 : offroad ? .66 : rounded ? .40 : .54;
  box(0, chassis, 0, w * (rounded?.68:.91), chassis * .53, l * (rounded?.82:.93), trim);
  if (type === 'motorcycle') {
    box(0, .68, -.12, .4, .38, .85);
    box(0, .83, .37, .4, .16, .7, rubber);
    box(0, 1.08, -.65, .78, .06, .08, alloy);
    box(0, 1.0, -.68, .16, .45, .08, trim);
    box(0, .96, -.85, .23, .18, .12, white);
    box(0, .78, .85, .2, .12, .06, red);
    for (const z of [-.74, .74]) cylinder(0, radius, z, radius, .15, rubber);
  } else {
    let frontAxle = -l * .31;
    let axles = [frontAxle, l * .31];
    if (heavy && type !== 'bus') axles = [frontAxle, l * .21, l * .34];
    if (type === 'semi_truck') axles = [-l * .41, -l * .20, -l * .13, l * .29, l * .37];
    for (const z of axles) for (const side of [-1, 1]) {
      cylinder(side * w * .46, radius, z, radius, w * .13, rubber);
      cylinder(side * w * .53, radius, z, radius * .60, .025, alloy);
      cylinder(side * w * .545, radius, z, radius * .22, .028, trim);
    }
    if (heavy && type !== 'bus') {
      const cabL = Math.min(2.7, l * .3), cabZ = -l / 2 + cabL / 2;
      box(0, h * .46, cabZ, w * .96, h * .65, cabL * .96);
      box(0, h * .65, -l / 2 - .012, w * .81, h * .22, .025, glass);
      for (const side of [-1, 1]) box(side * w * .485, h * .65, cabZ, .025, h * .22, cabL * .64, glass);
      const cargoL = l - cabL - .3, cargoZ = l / 2 - cargoL / 2;
      if (type === 'tanker') {
        cylinder(0, h * .57, cargoZ, w * .46, cargoL * .96, alloy, 'z');
        for (const dz of [-.35, 0, .35]) {
          const ring = new THREE.Mesh(new THREE.TorusGeometry(w * .465, .035, 6, 32), trim);
          ring.position.set(0, h * .57, cargoZ + cargoL * dz); root.add(ring);
        }
      } else if (type === 'dump_truck') {
        box(0, h * .47, cargoZ, w * .95, .2, cargoL);
        for (const side of [-1, 1]) box(side * w * .46, h * .65, cargoZ, .12, h * .37, cargoL);
        box(0, h * .65, l / 2 - .1, w * .95, h * .37, .12);
      } else {
        box(0, h * .59, cargoZ, w * .99, h * .78, cargoL, type === 'semi_truck' ? alloy : paint);
        for (const side of [-1, 1]) box(side * w * .501, h * .26, cargoZ, .025, .08, cargoL * .96, white);
        box(0, h * .58, l / 2 + .012, .045, h * .72, .02, trim);
      }
    } else if (type === 'bus' || type === 'van') {
      box(0, h * .53, 0, w * .96, h * .84, l * .97);
      box(0, h * .69, -l / 2 - .01, w * .85, h * .3, .025, glass);
      box(0, h * .69, l / 2 + .01, w * .75, h * .26, .025, glass);
      const count = type === 'bus' ? 8 : 3;
      for (const side of [-1, 1]) for (let i = 0; i < count; i++) {
        box(side * w * .485, h * .69, -l * .41 + (i + .5) * l * .82 / count, .025, h * .29, l * .72 / count, glass);
      }
    } else if (type === 'rounded_hatchback') {
      // A continuous domed body: no sedan boot, flat tail plate or rectangular extrusion.
      const roof=new THREE.SplineCurve([
        new THREE.Vector2(-.5,.47),new THREE.Vector2(-.34,.67),new THREE.Vector2(-.16,.94),
        new THREE.Vector2(.07,.995),new THREE.Vector2(.27,.94),new THREE.Vector2(.43,.75),new THREE.Vector2(.505,.42)
      ]).getPoints(128).sort((a,b)=>a.x-b.x);
      const section=z=>{
        let j=1;while(j<roof.length-1&&roof[j].x<z)j++;
        const a=roof[j-1],b=roof[j],f=THREE.MathUtils.clamp((z-a.x)/Math.max(b.x-a.x,1e-6),0,1);
        return THREE.MathUtils.clamp(a.y+(b.y-a.y)*f,.4,1)*h;
      };
      const vertices=[],surfaces=[[],[]],rings=81,sides=48;
      for(let i=0;i<rings;i++){
        const z=-.5+1.005*i/(rings-1),tip=Math.max(0,(Math.abs(z)-.29)/.22);
        const half=w*(.48-.29*Math.pow(Math.min(1,tip),2)),bottom=h*.27,top=section(z);
        for(let j=0;j<sides;j++){
          const angle=2*Math.PI*j/sides,c=Math.cos(angle),s=Math.sin(angle);
          vertices.push(half*Math.sign(c)*Math.pow(Math.abs(c),.7),bottom+(top-bottom)*(1+Math.sign(s)*Math.pow(Math.abs(s),.72))/2,z*l);
        }
        if(i)for(let j=0;j<sides;j++){
          const a=(i-1)*sides+j,b=(i-1)*sides+(j+1)%sides,c=i*sides+j,d=i*sides+(j+1)%sides;
          const angle=2*Math.PI*(j+.5)/sides,s=Math.sin(angle);
          const sideWindow=z>-.13&&z<.30&&s>.12&&s<.89&&Math.abs(z-.04)>.016;
          const frontWindow=z>-.29&&z<-.15&&s>.75;
          const rearWindow=z>.32&&z<.435&&s>.75;
          surfaces[sideWindow||frontWindow||rearWindow?1:0].push(a,b,c,b,d,c);
        }
      }
      for(let j=1;j<sides-1;j++){
        surfaces[0].push(0,j+1,j);
        const base=(rings-1)*sides;surfaces[0].push(base,base+j,base+j+1);
      }
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.setIndex(surfaces.flat());
      geometry.addGroup(0,surfaces[0].length,0);geometry.addGroup(surfaces[0].length,surfaces[1].length,1);
      geometry.computeVertexNormals();
      const shell=new THREE.Mesh(geometry,[paint,glass]);shell.name='rounded-body-shell';root.add(shell);
      for(const side of [-1,1]){
        box(side*w*.505,h*.63,-l*.17,w*.075,.08,.15);
        box(side*w*.47,h*.55,l*.08,.025,.025,.13,alloy);
      }
    } else {
      // A loft has a genuinely curved roof/tail and narrows towards its roof. Hatchbacks/MPVs/SUVs no longer
      // reuse a sedan's separate boot. The same local axes are used by the pose fitter and the GLB export.
      const pickup = type === 'pickup';
      const hatch = ['hatchback','suv','offroad_suv','crossover','wagon','mpv'].includes(type);
      const tall = ['suv','offroad_suv','crossover','mpv'].includes(type);
      const waist = tall ? .57 : .60;
      const hoodEnd = type === 'mpv' ? -.34 : -.25;
      const roofStart = offroad ? -.17 : type === 'mpv' ? -.14 : -.08;
      const roofEnd = offroad ? .40 : type === 'suv' ? .35 : pickup ? .035 : type === 'fastback' || type === 'coupe' ? .10 : hatch ? .28 : .15;
      const cabinEnd = pickup ? .11 : hatch ? .49 : type === 'fastback' ? .45 : .32;
      const roofScale = type === 'coupe' ? .97 : 1;
      const curve = new THREE.SplineCurve([
        new THREE.Vector2(hoodEnd,waist+.015),new THREE.Vector2((hoodEnd+roofStart)*.5,.84*roofScale),
        new THREE.Vector2(roofStart,.97*roofScale),new THREE.Vector2((roofStart+roofEnd)*.5,.99*roofScale),
        new THREE.Vector2(roofEnd,.96*roofScale),new THREE.Vector2((roofEnd+cabinEnd)*.5,offroad?.95:type==='suv'?.91:hatch?.86:.79),
        new THREE.Vector2(cabinEnd,waist+.015)
      ]);
      const roofPoints = curve.getPoints(96).sort((a,b)=>a.x-b.x);
      function section(z) {
        let i=1;while(i<roofPoints.length-1&&roofPoints[i].x<z)i++;
        const a=roofPoints[i-1],b=roofPoints[i],f=THREE.MathUtils.clamp((z-a.x)/Math.max(b.x-a.x,1e-5),0,1);
        const y=THREE.MathUtils.clamp(a.y+(b.y-a.y)*f,waist+.01,1)*h;
        return {y, half:w*(offroad?.43:tall?.385:.365), base:w*.455};
      }
      function surface(vertices, indices, material) {
        const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
        g.setIndex(indices);g.computeVertexNormals();const mesh=new THREE.Mesh(g,material);root.add(mesh);return mesh;
      }
      // Lower hull: rounded bumper corners, hood and rear shoulders.
      const lower=new THREE.Shape();
      lower.moveTo(-l*.49,h*.31);lower.quadraticCurveTo(-l*.515,h*.43,-l*.48,h*.51);
      lower.bezierCurveTo(-l*.42,h*.59,-l*.32,h*waist,-l*.21,h*waist);
      lower.lineTo(l*.35,h*waist);lower.quadraticCurveTo(l*.49,h*.60,l*.495,h*.46);
      lower.quadraticCurveTo(l*.51,h*.34,l*.47,h*.30);lower.lineTo(-l*.49,h*.31);
      const hull=new THREE.ExtrudeGeometry(lower,{depth:w*.84,bevelEnabled:true,bevelSegments:5,steps:1,
        bevelSize:Math.min(.075,h*.035),bevelThickness:w*.065,curveSegments:18});
      hull.rotateY(-Math.PI/2);hull.translate(w*.42,0,0);root.add(new THREE.Mesh(hull,paint));
      // Cabin cross sections. Shared vertex rings keep the roof and C-pillar smoothly rounded.
      const vertices=[],indices=[],rings=65;
      for(let i=0;i<rings;i++){
        const z=hoodEnd+(cabinEnd-hoodEnd)*i/(rings-1),s=section(z),base=h*waist;
        const ring=[[-s.base,base],[-s.half,s.y-h*.025],[-s.half*.78,s.y],
          [s.half*.78,s.y],[s.half,s.y-h*.025],[s.base,base]];
        for(const [x,y] of ring)vertices.push(x,Math.max(base,y),z*l);
        if(i)for(let j=0;j<6;j++){
          const a=(i-1)*6+j,b=(i-1)*6+(j+1)%6,c=i*6+j,d=i*6+(j+1)%6;
          indices.push(a,c,b,b,c,d);
        }
      }
      for(let j=1;j<5;j++){indices.push(0,j,j+1);const b=(rings-1)*6;indices.push(b,b+j+1,b+j);}
      surface(vertices,indices,paint);
      // Windows lie on the curved cabin, including a sloping/recurved rear hatch window.
      function roofGlass(z0,z1){
        const v=[],ind=[],n=15;
        for(let i=0;i<=n;i++){
          const z=z0+(z1-z0)*i/n,s=section(z);
          v.push(-s.half*.91,s.y+.012,z*l,s.half*.91,s.y+.012,z*l);
          if(i){const a=(i-1)*2;ind.push(a,a+1,a+2,a+1,a+3,a+2);}
        }
        const mat=glass.clone();mat.side=THREE.DoubleSide;surface(v,ind,mat);
      }
      roofGlass(hoodEnd+.025,roofStart-.015);
      roofGlass(roofEnd+.018,cabinEnd-.022);
      for(const side of [-1,1]){
        for(const [start,end] of [[hoodEnd+.045,.015],[.035,cabinEnd-.065]]){
          if(end<=start)continue;
          const v=[],ind=[],n=16;
          for(let i=0;i<=n;i++){
            const z=start+(end-start)*i/n,s=section(z),base=waist*h;
            for(const f of [.14,.90]){
              const x=s.base+(s.half-s.base)*f;
              v.push(side*(x+.012),base+(s.y-base)*f,z*l);
            }
            if(i){const a=(i-1)*2;ind.push(a,a+1,a+2,a+1,a+3,a+2);}
          }
          const mat=glass.clone();mat.side=THREE.DoubleSide;surface(v,ind,mat);
        }
        box(side*w*.525,h*.66,-l*.18,w*.09,.085,.16);
        for(const z of [-.08,.18])box(side*w*.49,h*.54,l*z,.025,.025,.14,alloy);
        // Body-coloured wheel arch moulding and a small sill distinguish tall crossovers from low hatchbacks.
        box(side*w*.465,h*.28,0,.045,.055,l*.52,trim);
      }
      if(hatch||type==='fastback'){
        box(0,h*.54,l*.496,w*.69,.035,.025,red);
        box(0,h*.73,l*(cabinEnd-.018),w*.84,.035,.10,paint);
        box(0,h*.40,l*.504,w*.50,.03,.025,trim);
      }
      if(pickup){
        box(0,h*.55,l*.30,w*.72,.05,l*.30,rubber);
        for(const side of [-1,1])box(side*w*.44,h*.65,l*.30,w*.11,h*.23,l*.32);
      }
      if(offroad){
        cylinder(0,h*.57,l/2+.025,radius*.93,.14,rubber,'z');
        cylinder(0,h*.57,l/2+.029,radius*.52,.145,alloy,'z');
        for(const side of [-1,1])box(side*w*.39,h*1.015,l*.05,.045,.045,l*.64,trim);
        for(const z of [-.25,.28])box(0,h*1.015,l*z,w*.8,.045,.045,trim);
      }
    }
    // Bumpers, grille, headlights and brake lights give front/back an unambiguous orientation.
    box(0, chassis * .8, rounded?-l*.49:-l / 2 - .025, w * (rounded?.5:.9), .15, .11, trim);
    box(0, chassis * .8, rounded?l*.49:l / 2 + .025, w * (rounded?.5:.9), .15, .11, trim);
    box(0, h * (heavy ? .35 : .42), rounded?-l*.495:-l / 2 - .05, w * (rounded?.3:.4), heavy ? .35 : .13, .025, trim);
    for (const side of [-1, 1]) {
      box(side * w * (rounded?.21:.33), h * (heavy ? .3 : rounded?.43:.46), rounded?-l*.475:-l / 2 - .065, w * (rounded?.13:.19), .12, .035, white);
      box(side * w * (rounded?.21:.35), h * (heavy ? .25 : rounded?.43:.46), rounded?l*.475:l / 2 + .065, w * (rounded?.13:.15), .13, .035, red);
    }
  }
  const text = profile.plate?.text;
  if (text && plateTexture) {
    const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ecece1'; ctx.fillRect(0, 0, 512, 128);
    ctx.strokeStyle = '#333'; ctx.lineWidth = 5; ctx.strokeRect(5, 5, 502, 118);
    ctx.fillStyle = '#111'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = 'bold 76px monospace'; ctx.fillText(text, 256, 68, 475);
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.MeshStandardMaterial({ map: texture, roughness: .65 });
    for (const sign of [-1, 1]) {
      const plate = new THREE.Mesh(new THREE.PlaneGeometry(Math.min(.52, w * .6), .13), mat);
      plate.position.set(0, heavy ? .68 : .36, sign * (rounded?l*.502+.01:l / 2 + .09));
      if (sign < 0) plate.rotation.y = Math.PI;
      root.add(plate);
    }
  }
  root.userData.profile = profile;
  root.userData.forward = [0,0,-1];
  root.userData.brakeLights = red;
  root.dispose = () => {
    const materials = new Set(), geometries = new Set();
    root.traverse(obj => { if (obj.geometry) geometries.add(obj.geometry); if (obj.material) for(const m of Array.isArray(obj.material)?obj.material:[obj.material])materials.add(m); });
    for (const g of geometries) g.dispose();
    for (const m of materials) { m.map?.dispose(); m.dispose(); }
  };
  return root;
}
