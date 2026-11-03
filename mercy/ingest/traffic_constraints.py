"""Joint source-image fitting and swept, oriented vehicle nonpenetration constraints.

Positions are solved together over the whole clip. Contacts constrain body surfaces, not just centres.
Continuous SAT checks cover interpolation between published keys, including rotating vehicles. Source
evidence is primary; corrupt detector masks and unobserved continuation have explicitly weaker weight.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import clarabel
import numpy as np
import osqp
from scipy.sparse import block_diag, coo_matrix, eye, hstack, triu, vstack

from mercy.ingest.vehicle_scene import cuboid, fitting_points, observed_edges, project_fitting_points

VERSION = 'joint-source-contact-v3'


def body_dimensions(model):
    w, h, length = model['dimensions_m']
    # Include mirrors, wheels, bumper trim and plates extending beyond the nominal fitting cuboid.
    return np.array([w*1.15, length+.22])


def axes(yaw):
    a = np.radians(yaw)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def radius(dim, yaw, axis):
    return .5*(dim[0]*abs(axes(yaw)[0]@axis)+dim[1]*abs(axes(yaw)[1]@axis))


def interval_radius(dim, yaw0, yaw1, axis):
    delta = (yaw1-yaw0+180)%360-180
    # Support is Lipschitz in angle. This envelope contains every interpolated orientation.
    return max(radius(dim, yaw0, axis), radius(dim, yaw1, axis)) + np.linalg.norm(dim)*.25*abs(np.radians(delta))


def swept_conflicts(P, yaw, dims, tolerance=1e-4):
    """Find overlapping swept intervals, including objects swapping places between clear endpoints."""
    result = []
    N, F, _ = P.shape
    for a in range(N):
        for b in range(a):
            d0 = P[a, :-1]-P[b, :-1]; d1 = P[a, 1:]-P[b, 1:]
            motion = d1-d0
            u = np.clip(-np.sum(d0*motion, axis=1)/np.maximum(1e-12, np.sum(motion*motion, axis=1)), 0, 1)
            near = np.linalg.norm(d0+u[:, None]*motion, axis=1) < (np.linalg.norm(dims[a])+np.linalg.norm(dims[b]))*.5+.05
            for k in np.flatnonzero(near):
                candidates = np.vstack([axes((yaw[a, k]+yaw[a, k+1])*.5), axes((yaw[b, k]+yaw[b, k+1])*.5)])
                enter, leave = 0., 1.
                supports = []
                for n in candidates:
                    r = interval_radius(dims[a], yaw[a,k], yaw[a,k+1], n)+interval_radius(dims[b], yaw[b,k], yaw[b,k+1], n)
                    supports.append(r)
                    p, v = d0[k]@n, motion[k]@n
                    if abs(v)<1e-12:
                        if abs(p)>=r-tolerance:enter=2.;break
                    else:
                        lo, hi = sorted(((-r+tolerance-p)/v, (r-tolerance-p)/v))
                        enter=max(enter,lo);leave=min(leave,hi)
                if enter < leave and enter < 1 and leave > 0:
                    fraction=np.clip((enter+leave)*.5,0,1)
                    d=d0[k]+fraction*motion[k]
                    gaps=np.abs(candidates@d)-supports
                    j=int(np.argmax(gaps));n=candidates[j]
                    sign=1. if n@d>=0 else -1.
                    result.append((a,b,int(k),n*sign,float(supports[j]),float(-gaps[j])))
    return result


def source_quality(obj, size):
    W,H=size;x1,y1,x2,y2=obj['box'];area=max(1.,(x2-x1)*(y2-y1))
    if y2<H*.35:return False
    if obj.get('area',area)/area<.12:return False
    if x1<=1 and y1<=1 and (x2-x1)>W*.4 and y2<H*.8:return False
    return x2-x1>=7 and y2-y1>=6


def collision_event(case, templates, window):
    """Only a labelled collision with an identified pair can supply a contact constraint."""
    path = Path(case)/'physics.json'
    if not path.exists() or window.get('event_kind') != 'collision':
        return None
    physics = json.loads(path.read_text())
    if physics.get('impact', {}).get('kind') != 'collision':
        return None
    participants = {o['id'] for o in physics.get('objects', []) if o.get('vid')}
    actors = [i for i, v in enumerate(templates) if v['key'] != 'ego' and
              participants.intersection(v['model'].get('source_keys', [v['key']]))]
    ego = next((i for i, v in enumerate(templates) if v['key'] == 'ego'), None)
    if 'ego' in participants and ego is not None and len(actors) == 1:
        return actors[0], ego, float(window['impact_t'])
    if 'ego' not in participants and len(actors) == 2:
        return actors[0], actors[1], float(window['impact_t'])
    return None


class Rows:
    def __init__(self, indices, fixed):
        self.indices=indices;self.fixed=fixed;self.rr=[];self.cc=[];self.vv=[];self.lo=[];self.hi=[]

    def add(self, terms, lo, hi=None, scale=1.):
        if hi is None:hi=lo
        values={};offset=0.
        for actor,frame,axis,coefficient in terms:
            index=self.indices[actor,frame,axis]
            if index<0:offset+=coefficient*self.fixed[actor,frame,axis]
            else:values[index]=values.get(index,0.)+coefficient
        if not values:return
        row=len(self.lo)
        for index,coefficient in values.items():
            if abs(coefficient)>1e-15:self.rr.append(row);self.cc.append(index);self.vv.append(coefficient*scale)
        self.lo.append((lo-offset)*scale);self.hi.append((hi-offset)*scale)

    def matrix(self, count):
        return coo_matrix((self.vv,(self.rr,self.cc)),shape=(len(self.lo),count)).tocsc()


def separation_plane(planes, a, b, k, proposed):
    """Adjacent swept intervals share an endpoint and cannot demand opposite sides.

    Keep one separating face across each connected contact run. Disconnected runs
    may choose another face, so a crossing can pass from in front to beside ego.
    """
    left=right=k
    while (a,b,left-1) in planes:left-=1
    while (a,b,right+1) in planes:right+=1
    existing=next((planes[a,b,j] for j in range(left,right+1) if (a,b,j) in planes),None)
    chosen=np.asarray(proposed if existing is None else existing)
    for j in range(left,right+1):planes[a,b,j]=chosen
    return chosen


def quadratic_solution(Q, q, A, lower, upper, warm):
    """Interior-point solve avoids slow ADMM convergence for dense curved traffic.

    Check feasibility in the original units. OSQP remains the fallback, including
    its infeasibility certificate for diagnosing incompatible source constraints.
    """
    lo=np.isfinite(lower);hi=np.isfinite(upper)
    matrix=vstack([-A[lo],A[hi]],format='csc');bound=np.r_[-lower[lo],upper[hi]]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=150;settings.tol_feas=1e-9
    interior=clarabel.DefaultSolver(triu(Q,format='csc'),q,matrix,bound,
                                   [clarabel.NonnegativeConeT(len(bound))],settings).solve()
    x=np.asarray(interior.x);values=A@x
    violation=float(max(np.max(lower-values,initial=0),np.max(values-upper,initial=0)))
    if str(interior.status) in {'Solved','AlmostSolved'} and np.isfinite(x).all() and violation<=5e-5:
        return SimpleNamespace(x=x,info=SimpleNamespace(status=str(interior.status),status_val=1,iter=interior.iterations),
                               primal_violation=violation,backend='clarabel')
    solver=osqp.OSQP();solver.setup(P=Q,q=q,A=A,l=lower,u=upper,verbose=False,
        eps_abs=1e-5,eps_rel=1e-6,max_iter=200000,polishing=True,adaptive_rho=True,adaptive_rho_interval=50)
    if np.isfinite(x).all() and violation<.1 and np.isfinite(interior.z).all():
        # Preserve the near-optimal primal AND dual solution when the interior
        # solver misses the strict feasibility cutoff by a small amount.
        dual=np.zeros(len(lower));z=np.asarray(interior.z)
        dual[lo]-=z[:int(lo.sum())];dual[hi]+=z[int(lo.sum()):]
        solver.warm_start(x=x,y=dual)
    else:solver.warm_start(x=warm)
    return solver.solve(raise_error=False)


def solve_world(case, templates, cameras, meta, window, max_iterations=40, write_report=True):
    case=Path(case);overlays=json.loads((case/'overlays.json').read_text())
    keys=[item['key'] for item in templates];N=len(keys)
    source=np.asarray([item['keys_absolute'] for item in templates],float)
    stride=max(1,round(1/np.median(np.diff(source[0,:,0]))/10))
    selected=np.unique(np.r_[np.arange(0,source.shape[1]-1,stride),source.shape[1]-1])
    source=source[:,selected]
    event=collision_event(case, templates, window)
    if event and source[0,0,0] <= event[2] <= source[0,-1,0]:
        # Keep native observations around a fast impact and the exact event time, rather
        # than dropping two of every three frames before imposing an instantaneous contact.
        raw=np.asarray([item['keys_absolute'] for item in templates],float)
        near=raw[0,:,0][abs(raw[0,:,0]-event[2])<=.4]
        grid=np.unique(np.round(np.r_[source[0,:,0],near,event[2]],7))
        grid[np.abs(grid-event[2])<1e-4]=event[2]
        grid=np.unique(grid)
        source=np.array([np.column_stack([grid,*(np.interp(grid,row[:,0],row[:,j]) for j in [1,2]),
                         np.degrees(np.interp(grid,row[:,0],np.unwrap(np.radians(row[:,3]))))]) for row in raw])
    else:
        event=None
    times=source[0,:,0];F=len(times);steps=np.diff(times)
    original=source[:,:,1:3].copy();yaw=source[:,:,3]
    dims=np.array([body_dimensions(item['model']) for item in templates])
    ego=keys.index('ego') if 'ego' in keys else -1
    straight=[bool(v.get('road_motion',{}).get('straight_prior_applied')) for v in templates]
    stationary=[bool(v.get('road_motion',{}).get('stationary_prior_applied')) for v in templates]
    indices=np.full((N,F,2),-1,int);count=0
    for a in range(N):
        if a==ego:continue
        if stationary[a]:
            indices[a,:,0]=count;indices[a,:,1]=count+1;count+=2;continue
        if straight[a]:indices[a,:,0]=count;count+=1
        else:indices[a,:,0]=np.arange(count,count+F);count+=F
        indices[a,:,1]=np.arange(count,count+F);count+=F
    camera_values=[cameras.at(t) for t in times]
    observations=[[] for _ in templates];valid_at=np.zeros((N,F),bool);rejected=0
    baseline_path=case/'vehicles/unconstrained-motion.json'
    baseline={v['key']:np.array(v['keys_absolute']) for v in json.loads(baseline_path.read_text()).get('templates',[])} if baseline_path.exists() else {}
    reference=original.copy()
    for a,item in enumerate(templates):
        if a==ego:continue
        members=item['model'].get('source_keys',[item['key']])
        for k,t in enumerate(times):
            objects=overlays.get(str(round(t*meta['src_fps'])),{})
            choices=[(key,objects[key]) for key in members if key in objects]
            if not choices:continue
            key,obj=max(choices,key=lambda x:x[1].get('confidence',0))
            if not source_quality(obj,(meta['width'],meta['height'])):rejected+=1;continue
            edge=observed_edges(key,obj,objects,(meta['width'],meta['height']))
            if not edge.any():continue
            observations[a].append((k,obj,edge));valid_at[a,k]=True
        rows=observations[a]
        if rows and item['key'] in baseline and not item.get('contacts',{}).get('continuous_nonpenetration'):
            raw=baseline[item['key']];at=np.array([times[k] for k,_,_ in rows])
            p=np.column_stack([np.interp(at,raw[:,0],raw[:,j]) for j in [1,2]])
            for j in range(2):reference[a,:,j]=np.interp(times,at,p[:,j])
            for first in [True,False]:
                take=at<=at[0]+.7 if first else at>=at[-1]-.7
                velocity=np.linalg.lstsq(np.column_stack([at[take]-at[take].mean(),np.ones(take.sum())]),p[take],rcond=None)[0][0] if take.sum()>=3 else np.zeros(2)
                velocity=np.clip(velocity,-30,30)
                if straight[a]:velocity[0]=0;velocity[1]=np.clip(velocity[1],0,40)*(1 if item['road_motion'].get('heading_deg',0)==0 else -1)
                boundary=at[0] if first else at[-1];point=p[0] if first else p[-1]
                outside=times<boundary if first else times>boundary
                reference[a,outside]=point+(times[outside]-boundary)[:,None]*velocity
            if straight[a]:
                reference[a,:,0]=np.median(p[:,0])
    if ego>=0:reference[ego]=original[ego]
    fixed=original.copy()
    constraints=Rows(indices,fixed);outside_evidence=[];boundary_rows=[];soft_boundaries=[]
    from mercy.ingest.playback_motion import curved_road
    clear_curved_exits=curved_road(case)
    for a,item in enumerate(templates):
        if a==ego:continue
        for k,obj,edge in observations[a]:
            C,R,_=camera_values[k];n=R[:,2]
            # An observed actor cannot escape its image residual by moving behind
            # the camera. Complete horizontal views also keep the whole body in front.
            depths=fitting_points(item['model'],yaw[a,k])@n
            minimum=float(np.min(depths) if edge[0] and edge[2] else np.max(depths))
            constraints.add([(a,k,j,float(n[[0,2]][j])) for j in range(2)],float(C@n)-minimum+.3,np.inf)
        if item.get('road_motion',{}).get('forward_only'):
            # Following a curved road does not imply reversing. Keep progress in
            # the source-supported nose direction while allowing lateral turns.
            for k in range(F-1):
                n=axes((yaw[a,k]+yaw[a,k+1])*.5)[1]
                constraints.add([(a,f,j,sign*n[j]) for f,sign in [(k,-1),(k+1,1)] for j in range(2)],0,np.inf)
        dynamic_axes=[np.array([0.,1.])] if straight[a] else axes(0).tolist()+axes(45).tolist()
        for n in dynamic_axes:
            n=np.asarray(n);speed=40. if straight[a] else 40*np.cos(np.pi/8)
            accel=6. if straight[a] else 10*np.cos(np.pi/8)
            for k in range(F-1):
                terms=[(a,f,j,sign*n[j]) for f,sign in [(k,-1),(k+1,1)] for j in range(2) if n[j]]
                if straight[a]:
                    direction=1 if item['road_motion'].get('heading_deg',0)==0 else -1
                    constraints.add(terms,0 if direction>0 else -speed*steps[k],speed*steps[k] if direction>0 else 0)
                else:constraints.add(terms,-speed*steps[k],speed*steps[k])
            for k in range(F-2):
                if event and a in event[:2] and times[k]<event[2]<times[k+2]:
                    # An impact can change velocity abruptly. Applying the ordinary
                    # acceleration bound THROUGH it moves the approach away from video.
                    # Both sides of the event retain their continuous-motion bounds.
                    continue
                h0,h1=steps[k:k+2];limit=accel*(h0+h1)*.5*h0*h1
                constraints.add([(a,f,j,sign*n[j]) for f,sign in [(k,h1),(k+1,-h0-h1),(k+2,h0)] for j in range(2) if n[j]],-limit,limit,
                                scale=1/(h0*h1*(h0+h1)*.5))
        # A car that exits a camera edge remains in the world, on the outside of that camera frustum.
        rows=observations[a]
        if len(rows)>=3:
            for first in [True,False]:
                frame,obj,_=rows[0] if first else rows[-1];box=obj['box'];W,H=meta['width'],meta['height']
                side='right' if box[2]>=W-3 else 'left' if box[0]<=2 else None
                if side is None:continue
                # A broad truncated detection may end because of an occluder or ID
                # loss. Only a narrow boundary fragment supports a full-frame exit.
                if box[2]-box[0]>W*.15:continue
                mask=times<times[frame]-.65 if first else times>times[frame]+.65
                for k in np.flatnonzero(mask):
                    C,R,K=camera_values[k];edge=W if side=='right' else 0
                    n3=R[:,0]-(edge-K[0,2])/K[0,0]*R[:,2]
                    if side=='left':n3=-n3
                    w,length=dims[a]
                    corner=cuboid([w,item['model']['dimensions_m'][1],length],yaw[a,k])
                    boundary=float(C@n3-np.min(corner@n3))
                    outside_evidence.append((a,k,n3[[0,2]],boundary))
                    # Continue beyond the observed exit edge while retaining the actor.
                    if not clear_curved_exits:
                        boundary_rows.append(len(constraints.lo));soft_boundaries.append((a,k,n3[[0,2]],boundary))
                    constraints.add([(a,k,j,float(n3[[0,2]][j])) for j in range(2)],boundary+.1,np.inf)
    contact=None
    event_normal=None
    if event:
        a,b,t=event;k=int(np.argmin(abs(times-t)))
        if not any(abs(times[f]-t)<=.3 for f,_,_ in observations[a]):
            raise ValueError('Labelled collision has no source observation near its event time')
        if keys[b] == 'ego':
            # An involved dashcam meets the other vehicle at its front.
            n=axes(yaw[b,k])[1]
        else:
            # A witness camera is not one of the colliding bodies. Choose the
            # separating body face from the two participants' source geometry.
            normals=np.vstack([axes(yaw[a,k]),axes(yaw[b,k])])
            delta=original[a,k]-original[b,k]
            gaps=[abs(delta@axis)-radius(dims[a],yaw[a,k],axis)-radius(dims[b],yaw[b,k],axis)
                  for axis in normals]
            n=normals[int(np.argmax(gaps))]
            n=n*(1 if delta@n>=0 else -1)
        event_normal=n.copy()
        r=radius(dims[a],yaw[a,k],n)+radius(dims[b],yaw[b,k],n)
        constraints.add([(a,k,j,float(n[j])) for j in range(2)]+[(b,k,j,float(-n[j])) for j in range(2)],r+.012,r+.15)
        # Prevent satisfying the front plane while remaining metres away to its side.
        for tangent in np.vstack([axes(yaw[a,k]),axes(yaw[b,k])]):
            alignment=abs(tangent@n)
            if alignment>1-1e-10:continue
            extent=radius(dims[a],yaw[a,k],tangent)+radius(dims[b],yaw[b,k],tangent)
            margin=.1 if alignment<.5 else -.15
            constraints.add([(a,k,j,float(tangent[j])) for j in range(2)]+[(b,k,j,float(-tangent[j])) for j in range(2)],-extent+margin,extent-margin)
        contact={'actors':[keys[a],keys[b]],'time':t,'max_surface_gap_m':.15,
                 'source':'labelled video collision and identified incident pair; kinematic contact prior'}
    collisions=[];planes={};pair_choice={};initial_conflicts=len(swept_conflicts(original,yaw,dims));P=reference.copy()
    base_rows=len(constraints.lo);base_values=len(constraints.vv)
    anchor_index=int(np.argmin(abs(times-window['impact_t'])))
    rank_position=reference[:,anchor_index].copy();rank_position+=np.arange(N)[:,None]*1e-5
    for iteration in range(max_iterations):
        objective=Rows(indices,fixed)
        for a,item in enumerate(templates):
            if a==ego:continue
            for k in range(F):
                published=item.get('contacts',{}).get('continuous_nonpenetration')
                weight=.5 if valid_at[a,k] and published else .02 if valid_at[a,k] or published else .0003
                for j in range(2):objective.add([(a,k,j,1)],reference[a,k,j],scale=np.sqrt(weight))
            for k in range(1,F-1):
                if event and a in event[:2] and times[k-1]<event[2]<times[k+1]:continue
                h0,h1=steps[k-1:k+1]
                for j in range(2):
                    objective.add([(a,k-1,j,1/h0),(a,k,j,-1/h0-1/h1),(a,k+1,j,1/h1)],0,scale=.004/(h0+h1))
            for k,obj,edge in observations[a]:
                C,R,K=camera_values[k];corners=fitting_points(item['model'],yaw[a,k])
                pc=(corners+[P[a,k,0],0,P[a,k,1]]-C)@R
                if pc[:,2].min()<.29:continue
                u=K[0,0]*pc[:,0]/pc[:,2]+K[0,2];v=K[1,1]*pc[:,1]/pc[:,2]+K[1,2]
                box=np.asarray(obj['box']);norm=np.array([box[2]-box[0],box[3]-box[1]]*2).clip(6)
                clip_x=(0,meta['width']) if obj['box'][0]<=2 or obj['box'][2]>=meta['width']-3 else None
                predictions=project_fitting_points(corners,P[a,k],C,R,K,clip_x)
                corners_idx=[u.argmin(),v.argmin(),u.argmax(),v.argmax()]
                for e in range(4):
                    if not edge[e]:continue
                    xyz=pc[corners_idx[e]];axis=e%2;focal=K[axis,axis]
                    J=focal*(R[[0,2],axis]*xyz[2]-xyz[axis]*R[[0,2],2])/(xyz[2]**2)
                    if clip_x is not None and axis==1:
                        J=np.array([(project_fitting_points(corners,P[a,k]+np.eye(2)[j]*.001,C,R,K,clip_x)[e]-
                                     project_fitting_points(corners,P[a,k]-np.eye(2)[j]*.001,C,R,K,clip_x)[e])/.002 for j in range(2)])
                    target=box[e]-predictions[e]+J@P[a,k]
                    objective.add([(a,k,j,float(J[j])) for j in range(2)],float(target),scale=np.sqrt(30)/norm[e])
        M=objective.matrix(count);rhs=np.asarray(objective.lo)
        A=constraints.matrix(count)
        Q=(2*(M.T@M)+eye(count,format='csc')*1e-8).tocsc();q=-2*(M.T@rhs)
        lower=np.asarray(constraints.lo);upper=np.asarray(constraints.hi);B=len(boundary_rows)
        if B:
            # Exit timing is a monocular continuation prior, not measured dynamics.
            # Explicit slack lets source observations and body nonpenetration win if
            # a guessed exit conflicts with them, instead of warping observed traffic.
            slack=coo_matrix((np.ones(B),(boundary_rows,np.arange(B))),shape=(A.shape[0],B)).tocsc()
            A=vstack([hstack([A,slack]),hstack([coo_matrix((B,count)),eye(B)])],format='csc')
            Q=block_diag([Q,eye(B)*8],format='csc');q=np.r_[q,np.zeros(B)]
            lower=np.r_[lower,np.zeros(B)];upper=np.r_[upper,np.full(B,np.inf)]
        warm=np.zeros(count+B)
        for a in range(N):
            for j in range(2):
                good=indices[a,:,j]>=0;warm[indices[a,good,j]]=P[a,good,j]
        for i,(a,k,n,boundary) in enumerate(soft_boundaries):warm[count+i]=max(0.,boundary+.1-n@P[a,k])
        result=quadratic_solution(Q,q,A,lower,upper,warm)
        if result.info.status_val not in [1,2]:
            print('QP diagnostics',result.info,flush=True)
            certificate=getattr(result,'prim_inf_cert',None)
            if certificate is not None and write_report:
                descriptions={int(indices[a,k,j]):[keys[a],float(times[k]),j]
                              for a in range(N) for k in range(F) for j in range(2) if indices[a,k,j]>=0}
                diagnostic=[]
                for row in np.argsort(abs(certificate))[-25:][::-1]:
                    values=A.getrow(int(row))
                    diagnostic.append({'row':int(row),'certificate':float(certificate[row]),
                        'lower':float(lower[row]),'upper':float(upper[row]),
                        'terms':[[descriptions.get(int(i),['exit_slack',int(i)]),float(v)] for i,v in zip(values.indices,values.data)]})
                (case/'vehicles/constraint-failure.json').write_text(json.dumps(diagnostic,indent=2))
            raise RuntimeError(f'Joint vehicle constraints failed: {result.info.status}')
        for a in range(N):
            for j in range(2):
                good=indices[a,:,j]>=0;P[a,good,j]=result.x[indices[a,good,j]]
        conflicts=swept_conflicts(P,yaw,dims,tolerance=2e-4)
        print(case.name,'joint contacts',iteration,'conflicting intervals',len(conflicts),'QP',result.info.iter,flush=True)
        new=0
        for a,b,k,n,r,depth in conflicts:
            # The SAT plane belongs to this interval. A crossing vehicle can be ahead
            # before impact and beside ego afterwards; a clip-wide side cannot express it.
            fixed_lane_pair=(straight[a] or a==ego) and (straight[b] or b==ego)
            aligned_contact=bool(event and {a,b}=={event[0],event[1]} and abs(np.cos(np.radians(yaw[a,anchor_index]-yaw[b,anchor_index])))>.9)
            if fixed_lane_pair or aligned_contact:
                if (a,b) not in pair_choice:
                    d=rank_position[a]-rank_position[b]
                    if aligned_contact:
                        n=event_normal*(1 if a==event[0] else -1)
                    else:
                        axis=0 if abs(d[0])>(dims[a,0]+dims[b,0])*.5*.65 else 1
                        n=np.zeros(2);n[axis]=1 if d[axis]>=0 else -1
                    pair_choice[a,b]=n
                n=pair_choice[a,b]
            r=interval_radius(dims[a],yaw[a,k],yaw[a,k+1],n)+interval_radius(dims[b],yaw[b,k],yaw[b,k+1],n)
            token=(a,b,k)
            if token in planes:continue
            separation_plane(planes,a,b,k,n);new+=1
        # A newly connected run can merge two old face choices. Rebuild only SAT
        # cuts so an obsolete opposite face cannot survive at their shared endpoint.
        constraints.lo=constraints.lo[:base_rows];constraints.hi=constraints.hi[:base_rows]
        constraints.rr=constraints.rr[:base_values];constraints.cc=constraints.cc[:base_values];constraints.vv=constraints.vv[:base_values]
        for (a,b,k),n in planes.items():
            r=interval_radius(dims[a],yaw[a,k],yaw[a,k+1],n)+interval_radius(dims[b],yaw[b,k],yaw[b,k+1],n)
            for frame in [k,k+1]:
                constraints.add([(a,frame,j,float(n[j])) for j in range(2)]+[(b,frame,j,float(-n[j])) for j in range(2)],r+.012,np.inf)
        collisions.append(len(conflicts))
        if not conflicts and iteration>=3:break
        if conflicts and not new:raise RuntimeError('Contact solver has unresolved nonpenetration constraints')
    else:raise RuntimeError('Contact solver failed continuous nonpenetration verification')
    for a,item in enumerate(templates):
        k=source[a].copy();k[:,1:3]=P[a]
        item['keys_absolute']=k.round(7).tolist();item['contacts']={'version':VERSION,'continuous_nonpenetration':True}
        if event and a in event[:2]:item['contacts']['impact_impulse_t']=event[2]
        item.setdefault('registration',{})['joint_source_contacts']=True
    reprojection={}
    for a,item in enumerate(templates):
        before=[];after=[]
        for k,obj,edge in observations[a]:
            for positions,errors in [(original,before),(P,after)]:
                clip_x=(0,meta['width']) if obj['box'][0]<=2 or obj['box'][2]>=meta['width']-3 else None
                predicted=project_fitting_points(fitting_points(item['model'],yaw[a,k]),positions[a,k],*camera_values[k],clip_x)
                errors.extend(abs(predicted-np.asarray(obj['box']))[edge])
        if after:
            reprojection[item['key']]={'edges':len(after),'before_median_px':float(np.median(before)),
                                      'after_median_px':float(np.median(after)),'after_p90_px':float(np.quantile(after,.9))}
            item['registration']['post_contact_median_edge_error_px']=reprojection[item['key']]['after_median_px']
    report={'version':VERSION,'initial_conflicting_intervals':initial_conflicts,'final_conflicting_intervals':len(conflicts),
        'continuous_interval_validation':True,'iterations':collisions,'discarded_corrupt_source_rows':rejected,
        'actors':len(templates),'body_extents':'nominal width * 1.15, length + .22m includes mirrors/bumper/plates',
        'source_pixels_used':sum(len(rows) for rows in observations),
        'max_position_adjustment_m':float(np.linalg.norm(P-original,axis=2).max()),
        'event_contact':contact,
        'source_reprojection':reprojection,
        'source_fit_geometry':'renderer body silhouette; horizontal frustum clipping for truncated masks',
        'source_boundary_constraints':len(outside_evidence),
        'source_boundary_model':('narrow observed curved-road exits retained outside source frustum with 0.1m margin'
                                 if clear_curved_exits else 'soft exit continuation with 0.1m frustum margin; source fitting and physical constraints retained'),
        'max_source_boundary_violation_m':max([max(0.,float(boundary-n@P[a,k])) for a,k,n,boundary in outside_evidence],default=0.),
        'limitation':'joint monocular reconstruction; missing observations remain estimated, not measured'}
    if contact:
        a,b,t=event;k=int(np.argmin(abs(times-t)))
        normals=np.vstack([axes(yaw[a,k]),axes(yaw[b,k])])
        contact['surface_gap_before_m']=float(max(abs(n@(original[a,k]-original[b,k]))-radius(dims[a],yaw[a,k],n)-radius(dims[b],yaw[b,k],n) for n in normals))
        contact['surface_gap_after_m']=float(max(abs(n@(P[a,k]-P[b,k]))-radius(dims[a],yaw[a,k],n)-radius(dims[b],yaw[b,k],n) for n in normals))
        if 0<k<F-1:
            before=(P[a,k]-P[a,k-1])/steps[k-1];after=(P[a,k+1]-P[a,k])/steps[k]
            contact['velocity_jump_mps']=(after-before).tolist()
            contact['impact_dynamics']='piecewise bounded kinematic motion; velocity impulse at the labelled event, forces not recovered'
        if not -.0002 <= contact['surface_gap_after_m'] <= contact['max_surface_gap_m']+.0002:
            raise RuntimeError(f'Collision contact does not match the labelled event: {contact}')
    if write_report:
        (case/'vehicles/contacts.json').write_text(json.dumps(report,indent=2))
        (case/'vehicles/constraint-failure.json').unlink(missing_ok=True)
    return report


def publish(case, dry_run=False):
    from mercy.ingest.vehicle_scene import Cameras, continuous_cameras
    from mercy.ingest.playback_motion import camera_path
    case=Path(case);meta=json.loads((case/'meta.json').read_text());manifest=json.loads((case/'moments.json').read_text())
    first=manifest['moments'][0]
    with np.load(case/'recon.npz') as rec:
        times=rec['times'];K=rec['intrinsic'].copy();rh,rw=rec['depth'].shape[1:]
    K*=np.array([[meta['width']/rw],[meta['height']/rh],[1.]])
    with np.load(case/'environment/cameras.npz') as rec:matrices=rec['matrices']
    matrices=continuous_cameras(times,matrices,first['replay_start'],first['replay_end'])
    matrices,velocity=camera_path(case,times,matrices,json.loads((case/'environment/input.json').read_text())['anchor_t'],return_velocity=True,write_report=not dry_run)
    cameras=Cameras(times,matrices,K,position_velocity=velocity)
    templates=[]
    for v in first['vehicles']:
        item={k:value for k,value in v.items() if k!='keys'}
        item['keys_absolute']=[[k[0]+first['t_frame'],*k[1:]] for k in v['keys']];templates.append(item)
    # Regenerate ego from the same calibrated camera used for source fitting/rendering.
    for item in templates:
        if item['key'] != 'ego':continue
        for k in item['keys_absolute']:
            C,R,_=cameras.at(k[0]);n=R[:,2].copy();n[1]=0;n/=np.linalg.norm(n)
            p=C-n*.35;k[1:]=[float(p[0]),float(p[2]),float(np.degrees(np.arctan2(n[0],n[2])))]
    report=solve_world(case,templates,cameras,meta,first,write_report=not dry_run)
    if dry_run:return report,templates
    for m in manifest['moments']:
        by_key={v['key']:v for v in m['vehicles']}
        for item in templates:
            v=by_key[item['key']]
            v['keys']=[[round(k[0]-m['t_frame'],7),*k[1:]] for k in item['keys_absolute']]
            v['contacts']=item['contacts'];v['registration']=item['registration']
        m['contacts']={'version':VERSION,'continuous_nonpenetration':True,'event_contact':report['event_contact']}
        m['camera_intrinsics']=[[float(t),float(lens[0,0]),float(lens[1,1]),float(lens[0,2]),float(lens[1,2])] for t,lens in zip(times,K)]
        m['source_size']=[meta['width'],meta['height']]
        m['camera_poses']=cameras.browser_poses()
    manifest['camera_motion']=json.loads((case/'vehicles/playback-camera.json').read_text())
    manifest['contacts']=report
    temporary=(case/'moments.json').with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=1));temporary.replace(case/'moments.json')
    return report,templates
