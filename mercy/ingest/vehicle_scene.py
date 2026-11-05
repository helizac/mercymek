"""Source-box constrained vehicle poses in the SAME road frame as the background and video cameras.

All observed vehicles participate, including those discarded by neural depth. A source-image heading prior
resolves the front/back ambiguity of cuboids. Missing detections use an explicitly estimated continuation,
never a new vehicle spawned when a depth track happens to become available. This is CPU geometric fitting.
"""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicHermiteSpline

from mercy.generation.vehicles import vehicle_observations


class Cameras:
    def __init__(self, times, matrices, intrinsics, position_velocity=None):
        self.times=np.asarray(times,float)
        self.matrices=np.asarray(matrices,float)
        self.intrinsics=np.asarray(intrinsics,float)
        self.position_curve=CubicHermiteSpline(self.times,self.matrices[:,:3,3],position_velocity) if position_velocity is not None else None
        # OpenCV cameras -> road axes contain a reflection. Remove it before quaternion interpolation.
        self.flip=np.diag([1.,-1.,1.])
        self.rotations=Slerp(self.times,Rotation.from_matrix(self.matrices[:,:3,:3]@self.flip))

    def at(self,t):
        requested=float(t)
        t=float(np.clip(t,self.times[0],self.times[-1]))
        C=self.position_curve(t) if self.position_curve is not None else np.array([np.interp(t,self.times,self.matrices[:,j,3]) for j in range(3)])
        if self.position_curve is not None:C=C+(requested-t)*self.position_curve(t,1)
        R=self.rotations(t).as_matrix()@self.flip
        K=np.array([[np.interp(t,self.times,self.intrinsics[:,i,j]) for j in range(3)] for i in range(3)])
        return C,R,K

    def browser_poses(self):
        A=np.diag([1.,1.,-1.]);cv_to_gl=np.diag([1.,-1.,-1.])
        q=Rotation.from_matrix(A@self.matrices[:,:3,:3]@cv_to_gl).as_quat()
        return [[round(float(t),6),(C*[1,1,-1]).tolist(),r.tolist(),
                 *([(self.position_curve(t,1)*[1,1,-1]).tolist()] if self.position_curve is not None else [])]
                for t,C,r in zip(self.times,self.matrices[:,:3,3],q)]


def continuous_cameras(times,matrices,start,end):
    """PnP was calibrated in the replay window. Do not splice its poses to an unaligned outside trajectory."""
    matrices=np.array(matrices,copy=True)
    core=np.flatnonzero((times>=start)&(times<=end))
    if len(core)<3:return matrices
    first,last=core[0],core[-1]
    for boundary,indices,outside in [(first,core[:6],np.flatnonzero(times<times[first])),
                                     (last,core[-6:],np.flatnonzero(times>times[last]))]:
        t=np.asarray(times)[indices];P=matrices[indices,:3,3]
        velocity=np.linalg.lstsq(np.column_stack([t-t.mean(),np.ones(len(t))]),P,rcond=None)[0][0]
        velocity[1]=0
        velocity*=min(1.,40/max(np.linalg.norm(velocity),1e-6))
        matrices[outside,:3,:3]=matrices[boundary,:3,:3]
        matrices[outside,:3,3]=matrices[boundary,:3,3]+(times[outside]-times[boundary])[:,None]*velocity
    return matrices


def cuboid(dimensions, yaw_deg):
    w,h,l=dimensions
    points=np.array([[x,y,z] for x in [-w/2,w/2] for y in [0,h] for z in [-l/2,l/2]])
    yaw=np.radians(yaw_deg)
    R=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    return points@R.T


def project_box(xz, yaw_deg, dimensions, C, R, K):
    P=(cuboid(dimensions,yaw_deg)+[xz[0],0,xz[1]]-C)@R
    z=np.maximum(P[:,2],.10)
    u=K[0,0]*P[:,0]/z+K[0,2];v=K[1,1]*P[:,1]/z+K[1,2]
    return np.array([u.min(),v.min(),u.max(),v.max()])


@lru_cache(maxsize=1)
def fitting_library():
    path=Path(__file__).resolve().parents[1]/'assets/vehicle-fit-points.json'
    return json.loads(path.read_text(encoding='utf-8'))['models']


def fitting_points(model, yaw_deg):
    """Sampled silhouette of the renderer's vehicle body; cuboid for unknown templates."""
    template=fitting_library().get(model.get('body_type'))
    if template is None:return cuboid(model['dimensions_m'],yaw_deg)
    points=np.asarray(template['points'])*np.asarray(model['dimensions_m'])/template['dimensions_m']
    a=np.radians(yaw_deg)
    return points@np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]).T


def project_fitting_points(points, xz, C, R, K, clip_x=None):
    pc=(points+[xz[0],0,xz[1]]-C)@R
    uv=pc[:,:2]/np.maximum(pc[:,2,None],.1)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    box=np.r_[uv.min(axis=0),uv.max(axis=0)]
    if clip_x is not None:
        # Clip the projected convex silhouette to the visible horizontal frustum. Its
        # offscreen roof/front cannot define the top/bottom edge of a truncated mask.
        polygon=cv2.convexHull(uv.astype(np.float32))[:,0].astype(float)
        for boundary,sign in [(clip_x[0],1),(clip_x[1],-1)]:
            clipped=[]
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                inside_a=sign*(a[0]-boundary)>=0;inside_b=sign*(b[0]-boundary)>=0
                if inside_a:clipped.append(a)
                if inside_a != inside_b:
                    clipped.append(a+(b-a)*(boundary-a[0])/(b[0]-a[0]))
            polygon=np.asarray(clipped).reshape(-1,2)
            if not len(polygon):break
        if len(polygon):box[[1,3]]=[polygon[:,1].min(),polygon[:,1].max()]
    return box


def heading(profile, frame, fps, camera_rotation):
    samples=sorted(profile.get('headings',[]),key=lambda x:x['source_frame'])
    if samples:
        yaw=np.unwrap(np.radians([s['relative_yaw_deg'] for s in samples]))
        angle=float(np.degrees(np.interp(frame,[s['source_frame'] for s in samples],yaw)))
        confidence=float(np.interp(frame,[s['source_frame'] for s in samples],[s['confidence'] for s in samples]))
    else:
        angle,confidence=0.,0.1
    forward=camera_rotation[:,2]
    return float(np.degrees(np.arctan2(forward[0],forward[2]))+angle),confidence


def contours(value):
    if not isinstance(value,(list,tuple)) or not value:return []
    if isinstance(value[0],(list,tuple)) and len(value[0])==2 and all(isinstance(x,(int,float)) for x in value[0]):
        return [np.asarray(value,np.float32)] if len(value)>=3 else []
    return [part for child in value for part in contours(child)]


def observed_edges(key,obj,objects,image_size):
    """A visible fragment's occlusion boundary is not the full vehicle's outer edge."""
    W,H=image_size;box=np.asarray(obj['box'],float)
    valid=np.array([box[0]>2,box[1]>2,box[2]<W-3,box[3]<H-3])
    if box[3]>H*.79 and box[2]-box[0]>W*.25:valid[3]=False
    parts=contours(obj.get('poly',[]))
    if not parts:return valid
    poly=np.concatenate(parts)
    area=(box[2]-box[0])*(box[3]-box[1])
    for j,(axis,sign) in enumerate([(0,-1),(1,-1),(0,1),(1,1)]):
        if not valid[j]:continue
        tolerance=max(2.,.02*(box[j+2]-box[j] if j<2 else box[j]-box[j-2]))
        tangent=1-axis;span=box[tangent+2]-box[tangent]
        points=[]
        # Sample the LENGTH of the boundary, not just polygon corners. A bottom corner touching a
        # foreground mask must not incorrectly mark the entire left and right edges as occluded.
        for part in parts:
            for a,b in zip(part,np.roll(part,-1,axis=0)):
                if max(abs(a[axis]-box[j]),abs(b[axis]-box[j]))<tolerance:
                    points.extend(a+(b-a)*f for f in np.linspace(.1,.9,9))
        if not points:
            points=[p for p in poly if abs(p[axis]-box[j])<tolerance and
                    box[tangent]+span*.06<p[tangent]<box[tangent+2]-span*.06]
        if not points:continue
        border=np.array(points)
        shifted=border.copy();shifted[:,axis]+=sign*4
        occluded=np.zeros(len(border),bool)
        for other_key,other in objects.items():
            if other_key==key or other.get('concept','').lower() not in {'car','truck','bus','van','motorcycle','vehicle'}:continue
            b=other['box'];other_area=(b[2]-b[0])*(b[3]-b[1])
            op=contours(other.get('poly',[]))
            if other_area<area*.85 or not op:continue
            for i,p in enumerate(shifted):
                if b[0]-2<=p[0]<=b[2]+2 and b[1]-2<=p[1]<=b[3]+2 and any(cv2.pointPolygonTest(part,(float(p[0]),float(p[1])),False)>=0 for part in op):
                    occluded[i]=True
        if occluded.mean()>=.5:valid[j]=False
    return valid


def fit_box(box, dimensions, C, R, K, yaw_prior, image_size, confidence=.5, edge_valid=None, position_prior=None, neural_anchor=None):
    target=np.asarray(box,float);W,H=image_size
    bw,bh=max(5.,target[2]-target[0]),max(5.,target[3]-target[1])
    valid=np.array([target[0]>2,target[1]>2,target[2]<W-3,target[3]<H-3])
    if target[3]>H*.79 and bw>W*.25:valid[3]=False  # dashboard hides contact, not the road plane
    if edge_valid is not None:valid &= edge_valid
    forward=R[:,2].copy();forward[1]=0;forward/=max(1e-8,np.linalg.norm(forward))
    right=np.array([forward[2],0,-forward[0]])
    # A complete class-height prior seeds depth; each observed image refines it on one shared ground plane.
    extent=.5*np.hypot(dimensions[0],dimensions[2])
    depth=np.clip(K[1,1]*dimensions[1]/bh+extent,extent+.15,400.)
    lateral=((target[0]+target[2])*.5-K[0,2])/K[0,0]*depth
    initial=np.array([np.clip(lateral,-250,250),depth,0.])
    if position_prior is not None:
        delta=np.array([position_prior[0]-C[0],0,position_prior[1]-C[2]])
        initial[:2]=[np.clip(delta@right,-299,299),np.clip(delta@forward,extent+.13,449)]
    if neural_anchor is not None:
        delta=np.array([neural_anchor['position'][0]-C[0],0,neural_anchor['position'][1]-C[2]])
        initial[:2]=[np.clip(delta@right,-299,299),np.clip(delta@forward,extent+.13,449)]
    norm=np.array([bw,bh,bw,bh])

    def position(params):
        P=C+right*params[0]+forward*params[1]
        return P[[0,2]]

    def residual(params):
        pred=project_box(position(params),yaw_prior+params[2],dimensions,C,R,K)
        residual=(pred-target)/norm
        # Occluded or image-truncated edges are inequalities: the complete model must contain its visible part.
        contain=residual*np.array([1,1,-1,-1])
        extra=np.maximum(0,contain[~valid])*.3
        prior=(position(params)-position_prior)/max(5.,depth)*.2 if position_prior is not None else []
        neural=(position(params)-neural_anchor['position'])/max(2.,depth*.15)*.3 if neural_anchor is not None else []
        return np.r_[residual[valid],extra,params[2]/(12 if confidence>.65 else 25)*.10,
                     (params[1]-depth)/max(depth,10)*.015,prior,neural]

    fitted=least_squares(residual,initial,loss='soft_l1',f_scale=.08,max_nfev=40,
                         bounds=([-300,extent+.12,-24],[300,450,24]))
    xz=position(fitted.x);yaw=yaw_prior+fitted.x[2]
    predicted=project_box(xz,yaw,dimensions,C,R,K)
    error=float(np.median(np.abs((predicted-target)[valid]))) if valid.any() else 0.
    return xz,float(yaw),error


def smooth(values, strength=2.):
    """Light temporal regularization; keeps every source observation and never splits a vehicle at depth spikes."""
    values=np.asarray(values,float)
    if len(values)<4:return values
    D=diags([np.ones(len(values)-2),-2*np.ones(len(values)-2),np.ones(len(values)-2)],[0,1,2],
            shape=(len(values)-2,len(values)),format='csc')
    A=diags(np.ones(len(values)),format='csc')+strength*(D.T@D)
    return spsolve(A,values)


def endpoint_velocity(times, positions, first):
    take=times<=times[0]+.65 if first else times>=times[-1]-.65
    t=times[take];p=positions[take]
    if len(t)<3 or np.ptp(t)<.15:return np.zeros(2)
    v=np.linalg.lstsq(np.column_stack([t-t.mean(),np.ones(len(t))]),p,rcond=None)[0][0]
    # Unobserved continuation is a prior; a bad/truncated box cannot launch a car across the scene.
    v[0]=np.clip(v[0],-5,5)
    speed=np.linalg.norm(v)
    return v*min(1.,40/max(speed,1e-6))


def continue_track(times, poses, sample_times):
    positions=poses[:,:2]
    result=np.column_stack([np.interp(sample_times,times,poses[:,j]) for j in range(3)])
    before,after=sample_times<times[0],sample_times>times[-1]
    result[before,:2]=positions[0]+(sample_times[before]-times[0])[:,None]*endpoint_velocity(times,positions,True)
    result[after,:2]=positions[-1]+(sample_times[after]-times[-1])[:,None]*endpoint_velocity(times,positions,False)
    return result


def merge_observations(observations, appearance, image_size, links=(), primary=()):
    """Merge duplicate masks and reviewed ID switches, never neighbouring cars with disjoint masks."""
    parent={k:k for k in observations}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    maps={k:dict(rows) for k,rows in observations.items()};keys=list(maps)
    W,H=image_size
    def mask_iou(a,b):
        A,B=np.array(a['box']),np.array(b['box'])
        wh=np.maximum(0,np.minimum(A[2:],B[2:])-np.maximum(A[:2],B[:2]));inter=wh.prod()
        box_iou=inter/max(1,np.prod(A[2:]-A[:2])+np.prod(B[2:]-B[:2])-inter)
        if box_iou<.55:return 0.
        if not a.get('poly') or not b.get('poly'):return float(box_iou) if box_iou>.9 else 0.
        ma=np.zeros((H//4+1,W//4+1),np.uint8);mb=np.zeros_like(ma)
        cv2.fillPoly(ma,[(part/4).astype(np.int32) for part in contours(a['poly'])],1)
        cv2.fillPoly(mb,[(part/4).astype(np.int32) for part in contours(b['poly'])],1)
        return float(np.logical_and(ma,mb).sum()/max(1,np.logical_or(ma,mb).sum()))
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            common=sorted(maps[a].keys()&maps[b].keys())
            if len(common)<3:continue
            take=common[::max(1,len(common)//15)]
            overlap=[mask_iou(maps[a][f],maps[b][f]) for f in take]
            if np.median(overlap)>.72 and np.mean(np.array(overlap)>.65)>.75:union(a,b)
    for link in links:
        a,b=link['earlier'],link['later']
        if a not in parent or b not in parent:continue
        # Simultaneously observed cars cannot become the same car through transitive identity links.
        ga=[k for k in keys if find(k)==find(a)];gb=[k for k in keys if find(k)==find(b)]
        if any(len(maps[x].keys()&maps[y].keys())>=2 for x in ga for y in gb):continue
        union(a,b)
    groups={}
    for key in keys:groups.setdefault(find(key),[]).append(key)
    merged={};profiles={};members={}
    for group in groups.values():
        canonical=min(group,key=lambda k:(k not in primary,observations[k][0][0],-len(observations[k])))
        members[canonical]=group
        frames={}
        for key in group:
            for f,obj in observations[key]:
                if f not in frames or obj.get('confidence',0)>frames[f].get('confidence',0):frames[f]=dict(obj,_source_key=key)
        merged[canonical]=sorted(frames.items())
        best=max(group,key=lambda k:appearance[k].get('confidence',0)*np.log1p(max(
            (r[1]['box'][2]-r[1]['box'][0])*(r[1]['box'][3]-r[1]['box'][1]) for r in observations[k])))
        model=dict(appearance[best]);model['headings']=sorted([h for k in group for h in appearance[k].get('headings',[])],key=lambda h:h['source_frame'])
        model['source_keys']=group;profiles[canonical]=model
    return merged,profiles,members


def reconstruct(case, cameras, meta, appearance, identity_links=(), primary=()):
    from mercy.ingest.wilddet import load, anchors_for, local_prior, rebase
    neural_report=rebase(load(case),cameras)
    overlays=json.loads((case/'overlays.json').read_text())
    observations=vehicle_observations(overlays)
    observations,appearance,members=merge_observations(observations,appearance,(meta['width'],meta['height']),identity_links,primary)
    fps=float(meta['src_fps']);duration=float(meta['duration'])
    times=np.arange(0,duration+1e-6,1/fps)
    template=[];report={}
    for key,rows in observations.items():
        model=dict(appearance.get(key) or {})
        if not model:raise ValueError(f'Missing appearance for observed vehicle {key}')
        neural_anchors=anchors_for(neural_report,members[key])
        dimensions=np.array([a['dimensions'] for a in neural_anchors])
        if len(dimensions)>=3 and np.max(np.std(dimensions,axis=0)/np.mean(dimensions,axis=0))<.25:
            model['class_dimensions_m']=model['dimensions_m']
            model['dimensions_m']=np.median(dimensions,axis=0).round(4).tolist()
            model['dimensions_source']='WildDet3D robust multi-frame RGB estimate; not measured dimensions'
        neural_used=0
        fitted=[];errors=[];edge_masks=[];trusted=[]
        for frame,obj in rows:
            t=frame/fps;C,R,K=cameras.at(t)
            prior,conf=heading(model,frame,fps,R)
            anchor=local_prior(neural_anchors,t)
            if anchor is not None:
                # A geometric cuboid does not independently establish front versus rear. Resolve its 180-degree
                # ambiguity against source appearance; ordinary traffic is further constrained jointly below.
                angle=prior+(anchor['yaw']-prior+90)%180-90
                if abs(angle-prior)<=50:prior=angle;conf=max(conf,.65)
                neural_used+=1
            valid=observed_edges(obj.get('_source_key',key),obj,overlays[str(frame)],(meta['width'],meta['height']))
            pos,yaw,error=fit_box(obj['box'],model['dimensions_m'],C,R,K,prior,(meta['width'],meta['height']),conf,valid,neural_anchor=anchor)
            fitted.append([t,*pos,yaw]);errors.append(error)
            edge_masks.append(valid)
            prediction=project_box(pos,yaw,model['dimensions_m'],C,R,K)
            max_error=np.max(abs((prediction-obj['box'])[valid])) if valid.any() else np.inf
            trusted.append(valid.sum()>=3 and max_error<max(8.,(obj['box'][3]-obj['box'][1])*.16))
        data=np.array(fitted)
        # Fit partial early/late fragments against the trajectory supported by complete views. A roof appearing
        # from behind a nearer car must not be assigned an enormous depth because its wheels are hidden.
        good=np.array(trusted)
        if good.sum()>=3:
            clean=data[good].copy();clean[:,1:3]=smooth(clean[:,1:3],strength=5)
            priors=continue_track(clean[:,0],clean[:,1:],data[:,0])
            for i,((frame,obj),valid) in enumerate(zip(rows,edge_masks)):
                if good[i] or valid.sum()>=3:continue
                if not clean[0,0]-1.0<=data[i,0]<=clean[-1,0]+1.0:continue
                C,R,K=cameras.at(frame/fps);yaw,conf=heading(model,frame,fps,R)
                pos,angle,error=fit_box(obj['box'],model['dimensions_m'],C,R,K,yaw,(meta['width'],meta['height']),conf,
                                        valid,priors[i,:2],neural_anchor=local_prior(neural_anchors,frame/fps))
                data[i,1:]=[*pos,angle]
        data[:,1:3]=smooth(data[:,1:3])
        data[:,3]=np.degrees(smooth(np.unwrap(np.radians(data[:,3])),strength=5.))
        poses=continue_track(data[:,0],data[:,1:],times)
        keys=np.column_stack([times,poses]).round(5).tolist()
        # Recompute post-regularization reprojection, including every detector box rather than accepted fits only.
        final_errors=[]
        for (frame,obj),entry,valid in zip(rows,data,edge_masks):
            C,R,K=cameras.at(frame/fps)
            pred=project_box(entry[1:3],entry[3],model['dimensions_m'],C,R,K)
            if valid.any():final_errors.append(float(np.median(abs((pred-obj['box'])[valid]))))
        info={'method':'all source boxes + shared road plane + class-size cuboid + source-image body heading',
              'source_keys':members[key],
              'observations':len(rows),'source_first_t':float(data[0,0]),'source_last_t':float(data[-1,0]),
              'median_box_error_px':round(float(np.median(final_errors)),2),'depth_track_required':False,
              'missing_observations':'continuous interpolated/extrapolated motion, estimated; identity is retained',
              'heading_source':'visible vehicle front/rear, not noisy position velocity',
              'dimensions':'class prior; not measured vehicle dimensions'}
        if neural_report:
            info['neural_backend']='WildDet3D' if neural_used else 'source-box fallback; no qualified WildDet3D anchor'
            info['neural_anchor_count']=len(neural_anchors);info['neural_supported_observations']=neural_used
            info['dimensions']=model.get('dimensions_source','class prior; not measured vehicle dimensions')
        template.append({'key':key,'label':key,'model':model,'pivot':[0,0,0],'keys_absolute':keys,
                         'persistent':True,'registration':info,'visible_absolute':[max(0,float(data[0,0])-1/fps),
                           min(duration,float(data[-1,0])+1/fps)]})
        report[key]=info
        print(case.name,key,len(rows),'source observations; median box error',info['median_box_error_px'],'px',flush=True)
    # Ego remains behind the source camera, with its nose from the camera orientation, even when stopped.
    ego=[]
    for t in times:
        C,R,_=cameras.at(t);forward=R[:,2].copy();forward[1]=0;forward/=np.linalg.norm(forward)
        P=C-forward*.35;yaw=np.degrees(np.arctan2(forward[0],forward[2]))
        ego.append([float(t),float(P[0]),float(P[2]),float(yaw)])
    model={'body_type':'sedan','paint_hex':'#737a80','paint_name':'unobserved camera vehicle',
           'dimensions_m':[1.82,1.47,4.7],'plate':{'text':'','observed':False},
           'geometry':'generic camera-vehicle prior; type and paint not observed'}
    template.append({'key':'ego','label':'Camera vehicle','model':model,'pivot':[0,0,0],'keys_absolute':ego,
                     'persistent':True,'registration':{'method':'camera trajectory and a .35m body-centre offset'}})
    return template,report


def publish(case: Path, appearance=None):
    """Refine vehicles/camera metadata without touching the reviewed static environment or its extension."""
    from mercy.generation.vehicles import profiles
    if appearance is None:appearance=profiles(case)
    from mercy.generation.heading_review import review as review_headings
    review_headings(case.name)
    manifest=json.loads((case/'moments.json').read_text())
    meta=json.loads((case/'meta.json').read_text())
    with np.load(case/'recon.npz') as rec:
        times=rec['times'];K=rec['intrinsic'].copy()
        rh,rw=rec['depth'].shape[1:]
    K*=np.array([[meta['width']/rw],[meta['height']/rh],[1.]])
    with np.load(case/'environment/cameras.npz') as cache:matrices=cache['matrices']
    first=manifest['moments'][0]
    matrices=continuous_cameras(times,matrices,first['replay_start'],first['replay_end'])
    from mercy.ingest.playback_motion import camera_path, VERSION
    anchor_t=json.loads((case/'environment/input.json').read_text())['anchor_t']
    matrices,velocity=camera_path(case,times,matrices,anchor_t,return_velocity=True)
    cameras=Cameras(times,matrices,K,position_velocity=velocity)
    from mercy.generation.vehicle_identity import review
    identity=review(case,appearance,vehicle_observations(json.loads((case/'overlays.json').read_text())))
    physics=json.loads((case/'physics.json').read_text())
    primary={o['id'] for o in physics['objects'] if o.get('vid')}
    templates,report=reconstruct(case,cameras,meta,appearance,identity['accepted'],primary)
    from mercy.generation.traffic_motion import review as review_motion
    from mercy.ingest.road_motion import apply as apply_motion, save_baseline
    save_baseline(case,templates)
    motion_review=review_motion(case)
    road_motion=apply_motion(case,templates,cameras,meta,primary,motion_review)
    from mercy.ingest.traffic_constraints import solve_world, VERSION as CONTACT_VERSION
    contacts=solve_world(case,templates,cameras,meta,first)
    browser=cameras.browser_poses()
    camera_keys=[[float(t),float(M[0,3]),float(M[2,3]),float(np.degrees(np.arctan2(M[0,2],M[2,2])))] for t,M in zip(times,matrices)]
    aliases=json.loads((case/'tracks.json').read_text()).get('aliases',{})
    for m in manifest['moments']:
        m.pop('wilddet3d',None)
        m['cam']['aspect']=meta['width']/meta['height']
        # Source intrinsics are used by BOTH fitting and rendering, rather than mixing SHARP and VGGT lenses.
        _,_,intrinsic=cameras.at(m['t_frame'])
        m['cam']['fov_y']=float(2*np.arctan(meta['height']/(2*intrinsic[1,1])))
        m['camera_poses']=browser;m['camera_keys']=camera_keys
        m['camera_lenses']=[[float(t),float(2*np.arctan(meta['height']/(2*k[1,1])))] for t,k in zip(times,K)]
        m['camera_intrinsics']=[[float(t),float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])] for t,k in zip(times,K)]
        m['source_size']=[meta['width'],meta['height']]
        m['contacts']={'version':CONTACT_VERSION,'continuous_nonpenetration':True,'event_contact':contacts['event_contact']}
        m['vehicles']=[{**{k:v for k,v in item.items() if k not in {'keys_absolute','visible_absolute'}},'label':aliases.get(item['key'],item['label']),
             'visibility':'persistent-world',
             **({'observed_range':[x-m['t_frame'] for x in item['visible_absolute']]} if item.get('visible_absolute') else {}),
             'keys':[[round(k[0]-m['t_frame'],6),*k[1:]] for k in item['keys_absolute']]} for item in templates]
        m['reconstruction']='lane-constrained-vehicles-v3'
        m['road_motion']={'version':'lane-traffic-v1','default':'straight forward; fixed lane; continuous nonnegative speed'}
        m['motion_version']=VERSION
    info={'method':'source-observation registration v2','vehicle_count':len(templates)-1,'vehicles':report,
          'motion_version':VERSION,'visibility':'source detection intervals are observations, not physical existence',
          'identity_review':identity,
          'camera':'source-camera orientation + continuous video-derived translation prior; free navigation is independent',
          'outside_replay_camera':'continuous boundary-pose/velocity prior; unaligned legacy camera splice removed',
          'limitations':['single-camera metric scale and class sizes are estimates',
                         'camera translation is a bounded source-video speed prior, not independently measured odometry',
                         'unobserved motion/occlusion intervals use estimated continuations',
                         'rounded vehicle meshes are class approximations; crash deformation is not recovered'],
          'gaussiancar':'reviewed camera/radar BEV implementation; pretrained fusion not executed without radar'}
    from mercy.ingest.wilddet import load as load_wilddet
    neural_report=load_wilddet(case)
    if neural_report:
        info['method']='WildDet3D RGB anchors + source-box registration + joint road-motion constraints'
        info['wilddet3d']={k:neural_report[k] for k in ['version','backend','sensors','frames','predictions']}
        info['wilddet3d'].update(accepted=len(neural_report['accepted']),rejected=len(neural_report['rejected']),
                               report='vehicles/wilddet3d.json',fallback='source-box geometry for unqualified/missing neural predictions')
        for m in manifest['moments']:
            m['reconstruction']='wilddet-rgb-road-vehicles-v4'
            m['wilddet3d']=info['wilddet3d']
    manifest['road_motion']={'version':'lane-traffic-v1','source_review':motion_review,'tracks':road_motion,'constraint_is_a_prior':True}
    manifest['contacts']=contacts
    manifest['camera_motion']=json.loads((case/'vehicles/playback-camera.json').read_text())
    (case/'vehicles/road-motion.json').write_text(json.dumps(manifest['road_motion'],indent=2))
    (case/'vehicles/registration.json').write_text(json.dumps(info,indent=2))
    manifest['vehicle_registration']=info
    path=case/'moments.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=1));temp.replace(path)
    return info
