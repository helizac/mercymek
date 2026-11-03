"""Joint, physically constrained traffic motion. Box noise must not turn straight cars into sideways skaters.

Normal tracks share a fixed road heading/lateral position. Positive continuous speed controls allow slowing
and stopping, but not reversing. Incident vehicles and video-supported maneuvers retain flexible paths.
These are explicit road-motion priors, not measured ground truth or predictions from a new neural detector.
"""
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from mercy.ingest.vehicle_scene import cuboid, observed_edges


def integrated_speed_basis(times,knots):
    times=np.asarray(times,float);knots=np.asarray(knots,float)
    B=np.zeros((len(times),len(knots)))
    for j,dt in enumerate(np.diff(knots)):
        u=np.clip(times-knots[j],0,dt)
        B[:,j]+=u-u*u/(2*dt);B[:,j+1]+=u*u/(2*dt)
    return B


def project_sequence(positions,corners,C,R,K):
    points=(corners[None]+np.column_stack([positions[:,0],np.zeros(len(positions)),positions[:,1]])[:,None]-C[:,None])
    pc=np.einsum('nki,nij->nkj',points,R)
    z=np.maximum(pc[:,:,2],.15)
    u=K[:,0,0,None]*pc[:,:,0]/z+K[:,0,2,None]
    v=K[:,1,1,None]*pc[:,:,1]/z+K[:,1,2,None]
    return np.column_stack([u.min(1),v.min(1),u.max(1),v.max(1)])


def fit_lane(times,boxes,valid,camera_values,dimensions,old_positions,sample_times,heading_deg=0.,stationary=False,neural_anchors=()):
    times=np.asarray(times,float);boxes=np.asarray(boxes,float);valid=np.asarray(valid,bool)
    old_positions=np.asarray(old_positions,float);C,R,K=map(np.asarray,zip(*camera_values))
    angle=np.radians(heading_deg)
    forward=np.array([np.sin(angle),np.cos(angle)]);right=np.array([np.cos(angle),-np.sin(angle)])
    along=old_positions@forward;lateral=old_positions@right
    span=max(.1,float(times[-1]-times[0]))
    n=min(10,max(2,int(np.ceil(span/.8))+1))
    knots=np.linspace(times[0],times[-1],n)
    B=integrated_speed_basis(times,knots)
    speed=0. if stationary else float(np.clip(np.polyfit(times-times[0],along,1)[0],0,35))
    start=float(np.median(along-speed*(times-times[0])))
    lane=float(np.clip(np.median(lateral),-59,59))
    initial=np.array([lane,np.clip(start,-399,499)]+([] if stationary else [max(.02,speed)]*n))
    norms=np.column_stack([boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]]*2).clip(6)
    # Distant and heavily occluded observations cannot dominate a complete nearby vehicle.
    weights=np.sqrt(np.clip((boxes[:,3]-boxes[:,1])/35,.12,1))[:,None]
    corners=cuboid(dimensions,heading_deg)
    anchor_basis=integrated_speed_basis([a['t'] for a in neural_anchors],knots)
    anchor_positions=np.array([a['position'] for a in neural_anchors]).reshape(-1,2)
    anchor_sigma=np.maximum(2.,np.linalg.norm(anchor_positions-np.median(C[:,[0,2]],axis=0),axis=1)*.15)

    def positions(params,basis):
        progress=np.full(len(basis),params[1]) if stationary else params[1]+basis@params[2:]
        return params[0]*right+progress[:,None]*forward

    def residual(params):
        P=positions(params,B)
        delta=(project_sequence(P,corners,C,R,K)-boxes)/norms
        # Preserve full edges; partial/image-cut edges only constrain containment.
        contain=np.maximum(0,delta*np.array([1,1,-1,-1]))*.3
        errors=np.where(valid,delta,contain)*weights
        priors=np.r_[(P-old_positions).ravel()/1000,(params[0]-lane)/100]
        if neural_anchors:
            priors=np.r_[priors,((positions(params,anchor_basis)-anchor_positions)/anchor_sigma[:,None]*.3).ravel()]
        if stationary:return np.r_[errors.ravel(),priors]
        accel=np.diff(params[2:])/np.diff(knots)
        return np.r_[errors.ravel(),priors,accel*.04,np.maximum(0,abs(accel)-6)*.4]

    lower=[-60,-400]+([] if stationary else [0.]*n)
    upper=[60,500]+([] if stationary else [40.]*n)
    fitted=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.10,max_nfev=160)
    if not stationary:
        from mercy.ingest.playback_motion import bounded_velocity
        fixed=bounded_velocity(fitted.x[2:,None],knots,max_speed=40.,max_acceleration=6.)[:,0]
        params=fitted.x.copy();params[2:]=fixed
        offset=least_squares(lambda p:residual(np.r_[p,fixed]),params[:2],bounds=(lower[:2],upper[:2]),
                             loss='soft_l1',f_scale=.10,max_nfev=40)
        params[:2]=offset.x;fitted.x=params
    sample_times=np.asarray(sample_times,float)
    # Continue the boundary speed outside detections; source visibility still gates actual scene entry/exit.
    E=integrated_speed_basis(np.clip(sample_times,times[0],times[-1]),knots)
    result=positions(fitted.x,E)
    if not stationary:
        result+=(np.minimum(0,sample_times-times[0])*fitted.x[2])[:,None]*forward
        result+=(np.maximum(0,sample_times-times[-1])*fitted.x[-1])[:,None]*forward
    prediction=project_sequence(positions(fitted.x,B),corners,C,R,K)
    edge_errors=abs(prediction-boxes)[valid]
    speeds=np.array([0.]) if stationary else fitted.x[2:]
    metrics={'model':'stationary road prior' if stationary else 'fixed lane + fixed body heading + nonnegative continuous-speed spline',
             'assumption':True,'heading_deg':float(heading_deg),'lane_offset_m':float(fitted.x[0]),
             'speed_range_mps':[float(speeds.min()),float(speeds.max())],
             'median_valid_edge_error_px':float(np.median(edge_errors)) if len(edge_errors) else None,
             'lateral_span_before_m':float(np.ptp(lateral)),'lateral_span_after_m':float(np.ptp(result@right)),
             'max_step_m':float(np.max(np.linalg.norm(np.diff(result,axis=0),axis=1))) if len(result)>1 else 0.,
             'optimizer_success':bool(fitted.success)}
    metrics['max_acceleration_mps2']=float(np.max(abs(np.diff(fitted.x[2:])/np.diff(knots)))) if not stationary else 0.
    if neural_anchors:
        metrics['neural_anchor_count']=len(neural_anchors)
        metrics['neural_backend']='WildDet3D RGB; soft position anchors, fixed road heading retained'
    return np.column_stack([result,np.full(len(result),heading_deg)]),metrics


def direction(profile,behavior):
    if behavior=='opposite_straight':return 180.
    # A small/side-on crop is an unreliable front/back signal. Opposing traffic needs video evidence;
    # ordinary traffic uses the user's straight-forward prior, even if a crop classifier was confused.
    return 0.


def apply(case,templates,cameras,meta,primary,review):
    from mercy.ingest.wilddet import load, anchors_for, rebase
    from mercy.ingest.playback_motion import body_headings, fit_incident, curved_road
    neural_report=rebase(load(case),cameras)
    overlays=json.loads((case/'overlays.json').read_text())
    manifest=json.loads((case/'moments.json').read_text())
    window=manifest['moments'][0]
    explicit={x['track']:x for x in review.get('exceptions',[]) if x['confidence']>=.85}
    report={}
    for item in templates:
        key=item['key'];members=item['model'].get('source_keys',[key])
        evidence=next((explicit[k] for k in members if k in explicit),None)
        behavior=evidence['behavior'] if evidence else 'straight_forward'
        if key=='ego':
            old=np.asarray(item['keys_absolute'],float)
            for i,t in enumerate(old[:,0]):
                C,R,_=cameras.at(t);f=R[:,2].copy();f[1]=0;f/=max(1e-9,np.linalg.norm(f))
                P=C-f*.35;old[i,1:]=[P[0],P[2],np.degrees(np.arctan2(f[0],f[2]))]
            item['keys_absolute']=old.tolist()
            info={'model':'continuous source-video ego camera path','straight_prior_applied':False}
            item['road_motion']=info;report[key]=info;continue
        curved=curved_road(case)
        flexible=bool(curved or set(members)&set(primary) or behavior in {'lane_change','turn','crossing','skidding'})
        if behavior=='stationary':flexible=False
        rows=[]
        for f,objects in overlays.items():
            obj=next((objects[k] for k in members if k in objects and objects[k].get('concept','').lower() in {'car','truck','bus','van','motorcycle','motorbike','pickup','vehicle'}),None)
            if not obj:continue
            x1,y1,x2,y2=obj['box']
            if x2-x1<7 or y2-y1<6:continue
            rows.append((int(f),obj))
        rows.sort(key=lambda r:r[0])
        if len(rows)<3:raise ValueError(f'No source support for lane-motion track {key}')
        core=[row for row in rows if window['replay_start']<=row[0]/meta['src_fps']<=window['replay_end']]
        if not curved and len(core)>=3 and (core[-1][0]-core[0][0])/meta['src_fps']>=.15:rows=core
        source_t=np.array([f/meta['src_fps'] for f,_ in rows])
        old=np.asarray(item['keys_absolute'],float)
        source_pos=np.column_stack([np.interp(source_t,old[:,0],old[:,j]) for j in [1,2]])
        valid=[observed_edges(obj.get('object_id',key),obj,overlays[str(f)],(meta['width'],meta['height'])) for f,obj in rows]
        anchors=[a for a in anchors_for(neural_report,members) if source_t[0]<=a['t']<=source_t[-1]]
        if flexible:
            headings,heading_source=body_headings(case,item,source_t,cameras,meta['src_fps'])
            fitted,metrics=fit_incident(source_t,[o['box'] for _,o in rows],valid,[cameras.at(t) for t in source_t],
                item['model']['dimensions_m'],source_pos,old[:,0],headings,anchors)
            metrics.update(reason='curved roadway' if curved else 'incident participant or video-supported maneuver',heading_source=heading_source,
                           source_keys=members,fitting_window_s=[float(source_t[0]),float(source_t[-1])],
                           forward_only=bool(curved and behavior=='straight_forward'))
            item['keys_absolute']=np.column_stack([old[:,0],fitted]).round(6).tolist()
            item['road_motion']=metrics;report[key]=metrics
            registration=item.setdefault('registration',{})
            registration.update(method='joint bounded incident motion + source heading + qualified WildDet3D anchors',
                                median_box_error_px=metrics['median_valid_edge_error_px'],heading_source=heading_source,
                                road_motion=metrics)
            print(case.name,key,'joint incident motion',round(metrics['max_speed_mps'],2),'m/s;',heading_source,flush=True)
            continue
        stationary=behavior=='stationary'
        body_yaw=float(np.median(body_headings(case,item,source_t,cameras,meta['src_fps'])[0])) if stationary else direction(item['model'],behavior)
        fitted,metrics=fit_lane(source_t,[o['box'] for _,o in rows],valid,[cameras.at(t) for t in source_t],
             item['model']['dimensions_m'],source_pos,old[:,0],body_yaw,stationary,anchors)
        item['keys_absolute']=np.column_stack([old[:,0],fitted]).round(6).tolist()
        metrics.update(straight_prior_applied=not stationary,stationary_prior_applied=stationary,behavior=behavior,evidence=evidence,
                       source_keys=members,reprojection_is_not_physical_validation=True,
                       fitting_window_s=[float(source_t[0]),float(source_t[-1])],
                       outside_fit_window='boundary-speed continuation; not used as camera calibration evidence')
        item['road_motion']=metrics;report[key]=metrics
        registration=item.setdefault('registration',{})
        registration.setdefault('per_frame_baseline_median_box_error_px',registration.get('median_box_error_px'))
        registration.update(method='joint source-box fit with explicit straight-lane motion prior',
                            median_box_error_px=metrics['median_valid_edge_error_px'],road_motion=metrics,
                            heading_source='straight-road prior; opposing traffic requires full-video evidence')
        print(case.name,key,'road prior',behavior,'lateral span',round(metrics['lateral_span_before_m'],2),'->',round(metrics['lateral_span_after_m'],5),flush=True)
    return report


def baseline_signature(case):
    h=hashlib.sha256(b'unconstrained-traffic-baseline-v3-curved-camera')
    for file in ['overlays.json','vehicles/profiles.json','environment/cameras.npz','physics.json']:
        h.update((case/file).read_bytes())
    if (case/'vehicles/wilddet3d.json').exists():h.update((case/'vehicles/wilddet3d.json').read_bytes())
    if (case/'vehicles/heading-review.json').exists():h.update((case/'vehicles/heading-review.json').read_bytes())
    return h.hexdigest()


def save_baseline(case,templates):
    (case/'vehicles/unconstrained-motion.json').write_text(json.dumps({'signature':baseline_signature(case),'templates':templates},indent=1))


def regularize_existing(case):
    """Apply the joint motion stage to saved source fits without rerunning neural models or per-frame fitting."""
    from mercy.ingest.vehicle_scene import Cameras,continuous_cameras
    from mercy.generation.traffic_motion import review
    meta=json.loads((case/'meta.json').read_text());manifest=json.loads((case/'moments.json').read_text())
    first=manifest['moments'][0];path=case/'vehicles/unconstrained-motion.json'
    cached=json.loads(path.read_text()) if path.exists() else {}
    if cached.get('signature')==baseline_signature(case):templates=cached['templates']
    else:
        if first.get('road_motion'):raise ValueError('Source inputs changed: rebuild unconstrained source fits before regularizing motion')
        templates=[]
        for v in first['vehicles']:
            item={k:value for k,value in v.items() if k not in {'keys','visible_range','road_motion'}}
            item['keys_absolute']=[[k[0]+first['t_frame'],*k[1:]] for k in v['keys']]
            if v.get('visible_range'):item['visible_absolute']=[t+first['t_frame'] for t in v['visible_range']]
            templates.append(item)
        save_baseline(case,templates)
    with np.load(case/'recon.npz') as rec:
        times=rec['times'];K=rec['intrinsic'].copy();rh,rw=rec['depth'].shape[1:]
    K*=np.array([[meta['width']/rw],[meta['height']/rh],[1.]])
    with np.load(case/'environment/cameras.npz') as cache:matrices=cache['matrices']
    matrices=continuous_cameras(times,matrices,first['replay_start'],first['replay_end'])
    from mercy.ingest.playback_motion import camera_path, VERSION
    matrices,velocity=camera_path(case,times,matrices,json.loads((case/'environment/input.json').read_text())['anchor_t'],return_velocity=True)
    cameras=Cameras(times,matrices,K,position_velocity=velocity)
    physics=json.loads((case/'physics.json').read_text());primary={o['id'] for o in physics['objects'] if o.get('vid')}
    source_review=review(case)
    report=apply(case,templates,cameras,meta,primary,source_review)
    from mercy.ingest.traffic_constraints import solve_world, VERSION as CONTACT_VERSION
    contacts=solve_world(case,templates,cameras,meta,first)
    summary={'version':'lane-traffic-v1','default':'straight forward; fixed lane; continuous nonnegative speed',
             'source_review':source_review,'tracks':report,'constraint_is_a_prior':True}
    for m in manifest['moments']:
        by_key={v['key']:v for v in m['vehicles']}
        for item in templates:
            v=by_key[item['key']]
            v['keys']=[[round(k[0]-m['t_frame'],6),*k[1:]] for k in item['keys_absolute']]
            v['road_motion']=item['road_motion'];v['registration']=item['registration']
            v['contacts']=item['contacts']
            v['visibility']='persistent-world'
            if 'visible_range' in v:v['observed_range']=v.pop('visible_range')
        m['road_motion']={'version':summary['version'],'default':summary['default']}
        m['reconstruction']='wilddet-rgb-road-vehicles-v4' if (case/'vehicles/wilddet3d.json').exists() else 'lane-constrained-vehicles-v3'
        m['motion_version']=VERSION
        m['contacts']={'version':CONTACT_VERSION,'continuous_nonpenetration':True,'event_contact':contacts['event_contact']}
        m['camera_intrinsics']=[[float(t),float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])] for t,k in zip(times,K)]
        m['source_size']=[meta['width'],meta['height']]
    manifest['road_motion']=summary
    manifest['contacts']=contacts
    manifest['camera_motion']=json.loads((case/'vehicles/playback-camera.json').read_text())
    if manifest.get('vehicle_registration'):
        manifest['vehicle_registration']['camera']='source-camera orientation + continuous video-derived translation prior; free navigation is independent'
    for key,info in manifest.get('vehicle_registration',{}).get('vehicles',{}).items():
        item=next((v for v in templates if v['key']==key),None)
        if item:info.update(item['registration'])
    (case/'vehicles/road-motion.json').write_text(json.dumps(summary,indent=2))
    if manifest.get('vehicle_registration'):
        (case/'vehicles/registration.json').write_text(json.dumps(manifest['vehicle_registration'],indent=2))
    tmp=(case/'moments.json').with_suffix('.tmp');tmp.write_text(json.dumps(manifest,indent=1));tmp.replace(case/'moments.json')
    return summary
