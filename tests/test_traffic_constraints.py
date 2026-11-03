import json

import numpy as np
import pytest

from mercy.ingest.traffic_constraints import body_dimensions, collision_event, interval_radius, radius, solve_world, source_quality, swept_conflicts
from mercy.ingest.vehicle_scene import Cameras, project_box


def test_adjacent_swept_cuts_merge_faces_without_contradicting_shared_endpoints():
    from mercy.ingest.traffic_constraints import separation_plane
    planes={}
    separation_plane(planes,1,0,3,[1.,0.])
    separation_plane(planes,1,0,5,[-1.,0.])
    separation_plane(planes,1,0,4,[-1.,0.])
    assert all(np.array_equal(planes[1,0,k],[1.,0.]) for k in [3,4,5])
    separation_plane(planes,1,0,9,[-1.,0.])
    assert np.array_equal(planes[1,0,9],[-1.,0.])


def test_swept_collision_detects_tunnelling_between_clear_endpoints():
    positions=np.array([[[-6.,0.],[6.,0.]],[[0.,0.],[0.,0.]]])
    assert swept_conflicts(positions,np.zeros((2,2)),np.array([[2.,4.],[2.,4.]]))
    positions[0,:,1]=5
    assert not swept_conflicts(positions,np.zeros((2,2)),np.array([[2.,4.],[2.,4.]]))


def test_touching_bodies_are_allowed_but_interpenetration_is_detected():
    positions=np.array([[[2.,0.],[2.,0.]],[[0.,0.],[0.,0.]]])
    dims=np.array([[2.,4.],[2.,4.]])
    assert not swept_conflicts(positions,np.zeros((2,2)),dims)
    positions[0,:,0]-=.01
    assert swept_conflicts(positions,np.zeros((2,2)),dims)


def test_rotation_support_envelope_contains_intermediate_body_orientations():
    dim=np.array([2.,5.]);axis=np.array([1.,0.])
    bound=interval_radius(dim,-30,65,axis)
    assert all(radius(dim,yaw,axis)<=bound for yaw in np.linspace(-30,65,501))
    assert body_dimensions({'dimensions_m':[2.,1.5,5.]})==pytest.approx([2.3,5.22])


def test_corrupt_tracking_mask_is_not_geometric_evidence():
    assert not source_quality({'box':[0,0,902,444],'area':2904},(1280,720))
    assert not source_quality({'box':[200,10,230,40]},(1280,720))
    assert source_quality({'box':[510,398,537,439],'area':900},(1280,720))


def test_forward_curve_prior_does_not_turn_depth_noise_into_reversing(tmp_path):
    (tmp_path/'vehicles').mkdir()
    times=np.linspace(0,2,21);M=np.repeat(np.eye(4)[None],len(times),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.4
    lens=np.repeat(np.array([[[900.,0,640],[0,900,360],[0,0,1]]]),len(times),axis=0)
    cameras=Cameras(times,M,lens);dimensions=[1.8,1.5,4.5]
    # Monocular depth noise suggests reverse motion, although source review says forward flow.
    positions=np.column_stack([times,20-times])
    overlays={str(i):{'car:1':{'box':project_box(p,0,dimensions,*cameras.at(t)).tolist(),'concept':'car'}}
              for i,(t,p) in enumerate(zip(times,positions))}
    (tmp_path/'overlays.json').write_text(json.dumps(overlays))
    templates=[{'key':'car:1','model':{'dimensions_m':dimensions},'road_motion':{'forward_only':True},
                'keys_absolute':np.column_stack([times,positions,np.zeros(len(times))]).tolist()}]
    solve_world(tmp_path,templates,cameras,{'src_fps':10,'width':1280,'height':720},{'impact_t':1.})
    result=np.array(templates[0]['keys_absolute'])
    assert np.diff(result[:,2]).min()>=-1e-6
    assert np.ptp(result[:,1])>1.5  # the flexible path can still move laterally


def test_waiting_vehicle_stays_in_world_after_camera_passes_curved_road_exit(tmp_path):
    (tmp_path/'vehicles').mkdir()
    (tmp_path/'scene.json').write_text(json.dumps({'road_type':'roundabout'}))
    times=np.linspace(0,2,41);M=np.repeat(np.eye(4)[None],len(times),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.4;M[:,2,3]=times*8
    lens=np.repeat(np.array([[[900.,0,640],[0,900,360],[0,0,1]]]),len(times),axis=0)
    cameras=Cameras(times,M,lens);dimensions=[1.8,1.5,4.5];overlays={}
    for i,t in enumerate(times):
        if t>.9:continue
        box=project_box([-4.5,10.],0,dimensions,*cameras.at(t))
        box=box.clip([0,0,0,0],[1280,720,1280,720])
        if box[2]-box[0]>=7:overlays[str(i)]={'car:waiting':{'box':box.tolist(),'concept':'car'}}
    assert 0<list(overlays.values())[-1]['car:waiting']['box'][2]<1280*.15
    (tmp_path/'overlays.json').write_text(json.dumps(overlays))
    seed=np.column_stack([times,np.full(len(times),-4.5),np.full(len(times),10.),np.zeros(len(times))])
    templates=[{'key':'car:waiting','model':{'dimensions_m':dimensions},
                'road_motion':{'stationary_prior_applied':True},'keys_absolute':seed.tolist()}]
    report=solve_world(tmp_path,templates,cameras,{'src_fps':20,'width':1280,'height':720},{'impact_t':1.})
    result=np.array(templates[0]['keys_absolute'])
    assert len(result)>10 and np.ptp(result[:,1:3],axis=0).max()<1e-7
    assert report['source_boundary_constraints']>0
    assert report['max_source_boundary_violation_m']<1e-5


def test_joint_solver_preserves_source_alignment_and_all_physical_actors(tmp_path):
    (tmp_path/'vehicles').mkdir()
    times=np.linspace(0,2,21);M=np.repeat(np.eye(4)[None],len(times),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.4;M[:,2,3]=times*4
    K=np.repeat(np.array([[[900.,0,640],[0,900,360],[0,0,1]]]),len(times),axis=0)
    cameras=Cameras(times,M,K);dims=[1.8,1.5,4.5]
    true=[np.column_stack([np.full(len(times),x),20+times*4]) for x in [-1.6,1.6]]
    templates=[];overlays={}
    for a in range(2):
        key=f'car:{a}'
        templates.append({'key':key,'model':{'dimensions_m':dims},'road_motion':{'straight_prior_applied':True,'heading_deg':0},
                          'keys_absolute':np.column_stack([times,np.zeros(len(times)),20+times*4,np.zeros(len(times))]).tolist()})
        for k,t in enumerate(times):
            box=project_box(true[a][k],0,dims,*cameras.at(t)).tolist()
            overlays.setdefault(str(k),{})[key]={'box':box,'concept':'car'}
    (tmp_path/'overlays.json').write_text(json.dumps(overlays))
    report=solve_world(tmp_path,templates,cameras,{'src_fps':10,'width':1280,'height':720},{'impact_t':1.})
    assert report['initial_conflicting_intervals']>0
    assert report['final_conflicting_intervals']==0
    assert len(templates)==2
    for item,p in zip(templates,true):
        assert np.array(item['keys_absolute'])[:,1:3]==pytest.approx(p,abs=.03)


@pytest.mark.parametrize('impulse',[False,True])
def test_labelled_collision_reaches_contact_at_exact_event_time(tmp_path,impulse):
    (tmp_path/'vehicles').mkdir()
    times=np.linspace(0,2,61);M=np.repeat(np.eye(4)[None],len(times),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.35;M[:,2,3]=times*4
    K=np.repeat(np.array([[[700.,0,640],[0,700,360],[0,0,1]]]),len(times),axis=0)
    cameras=Cameras(times,M,K);dims=[1.8,1.5,4.5]
    extent=body_dimensions({'dimensions_m':dims})
    contact_z=(extent[0]+extent[1])/2+.04
    ego=np.column_stack([times,np.zeros(len(times)),times*4-.35,np.zeros(len(times))])
    incident=ego.copy();incident[:,1]=-.5+times*.5;incident[:,2]+=contact_z;incident[:,3]=90
    if impulse:incident[:,2]+=8*np.abs(times-1.03)
    overlays={str(i):{'car:target':{'box':project_box(k[1:3],k[3],dims,*cameras.at(k[0])).tolist(),'concept':'car'}} for i,k in enumerate(incident)}
    (tmp_path/'overlays.json').write_text(json.dumps(overlays))
    (tmp_path/'physics.json').write_text(json.dumps({'impact':{'kind':'collision'},'objects':[{'id':'ego','vid':'v1'},{'id':'car:target','vid':'v2'}]}))
    seed=incident.copy();seed[:,2]+=5
    templates=[{'key':'car:target','model':{'dimensions_m':dims},'keys_absolute':seed.tolist()},
               {'key':'ego','model':{'dimensions_m':dims},'keys_absolute':ego.tolist()}]
    report=solve_world(tmp_path,templates,cameras,{'src_fps':30,'width':1280,'height':720},
                       {'impact_t':1.03,'event_kind':'collision'},write_report=False)
    contact=report['event_contact']
    assert contact['surface_gap_before_m']>4
    assert 0<=contact['surface_gap_after_m']<=.1502
    assert report['final_conflicting_intervals']==0
    assert any(abs(k[0]-1.03)<1e-7 for k in templates[0]['keys_absolute'])
    assert not (tmp_path/'vehicles/contacts.json').exists()
    if impulse:
        assert np.linalg.norm(contact['velocity_jump_mps'])>10
        fitted=np.array(templates[0]['keys_absolute'])
        for t in [.5,1.5]:
            assert np.interp(t,fitted[:,0],fitted[:,2])==pytest.approx(np.interp(t,incident[:,0],incident[:,2]),abs=.3)


def test_near_miss_and_ambiguous_participants_do_not_create_contact(tmp_path):
    physics={'impact':{'kind':'near-miss'},'objects':[{'id':'ego','vid':'v1'},{'id':'car:1','vid':'v2'}]}
    path=tmp_path/'physics.json';path.write_text(json.dumps(physics))
    templates=[{'key':k,'model':{}} for k in ['ego','car:1','car:2']]
    window={'impact_t':1.,'event_kind':'near-miss'}
    assert collision_event(tmp_path,templates,window) is None
    window['event_kind']='collision'
    assert collision_event(tmp_path,templates,window) is None
    physics['impact']['kind']='collision';physics['objects'].append({'id':'car:2','vid':'v3'})
    path.write_text(json.dumps(physics))
    assert collision_event(tmp_path,templates,window) is None


def test_witness_camera_is_excluded_from_two_other_vehicles_contact(tmp_path):
    (tmp_path/'physics.json').write_text(json.dumps({'impact':{'kind':'collision'},'objects':[
        {'id':'car:rear','vid':'v1'}, {'id':'truck:front','vid':'v2'}, {'id':'ego','vid':None}]}))
    templates=[{'key':key,'model':{}} for key in ['ego','car:rear','truck:front']]
    assert collision_event(tmp_path,templates,{'impact_t':3.2,'event_kind':'collision'}) == (1,2,3.2)


def test_two_other_vehicles_reach_contact_without_a_camera_body(tmp_path):
    (tmp_path/'vehicles').mkdir()
    times=np.linspace(0,2,61);M=np.repeat(np.eye(4)[None],len(times),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.35;M[:,2,3]=times*4
    K=np.repeat(np.array([[[700.,0,640],[0,700,360],[0,0,1]]]),len(times),axis=0)
    cameras=Cameras(times,M,K);dims=[1.8,1.5,4.5]
    overlays={};templates=[]
    for key,offset in [('car:rear',15.),('car:front',19.77)]:
        source=np.column_stack([times,np.zeros(len(times)),offset+times*4,np.zeros(len(times))])
        for i,k in enumerate(source):
            overlays.setdefault(str(i),{})[key]={'box':project_box(k[1:3],0,dims,*cameras.at(k[0])).tolist(),'concept':'car'}
        seed=source.copy()
        if key=='car:front':seed[:,2]+=4
        templates.append({'key':key,'model':{'dimensions_m':dims},'keys_absolute':seed.tolist()})
    (tmp_path/'overlays.json').write_text(json.dumps(overlays))
    (tmp_path/'physics.json').write_text(json.dumps({'impact':{'kind':'collision'},'objects':[
        {'id':'car:rear','vid':'v1'},{'id':'car:front','vid':'v2'}]}))
    report=solve_world(tmp_path,templates,cameras,{'src_fps':30,'width':1280,'height':720},
                       {'impact_t':1.03,'event_kind':'collision'},write_report=False)
    assert report['event_contact']['actors']==['car:rear','car:front']
    assert -.0002<=report['event_contact']['surface_gap_after_m']<=.1502
    assert report['final_conflicting_intervals']==0


def test_exit_fragment_continues_outside_camera_without_removing_physical_actor(tmp_path):
    times=np.linspace(0,3,31);M=np.repeat(np.eye(4)[None],len(times),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.35;M[:,2,3]=times*5
    K=np.repeat(np.array([[[700.,0,640],[0,700,360],[0,0,1]]]),len(times),axis=0)
    cameras=Cameras(times,M,K);dims=[1.8,1.5,4.5];overlays={}
    for i,t in enumerate(times[:11]):
        box=project_box([-5.,8.],0,dims,*cameras.at(t))
        box=np.clip(box,[0,0,0,0],[1280,720,1280,720])
        overlays[str(i)]={'car:edge':{'concept':'car','box':box.tolist()}}
    (tmp_path/'overlays.json').write_text(json.dumps(overlays));(tmp_path/'vehicles').mkdir()
    templates=[{'key':'car:edge','model':{'dimensions_m':dims},'persistent':True,
        'road_motion':{'straight_prior_applied':True,'heading_deg':0},
        'keys_absolute':np.column_stack([times,np.full(len(times),-5.),8+times*5,np.zeros(len(times))]).tolist()}]
    report=solve_world(tmp_path,templates,cameras,{'src_fps':10,'width':1280,'height':720},
                       {'impact_t':1.,'event_kind':'near-miss'},write_report=False)
    assert report['source_boundary_constraints']>0
    assert report['max_source_boundary_violation_m']<.001
    assert len(templates)==1 and templates[0]['persistent']
    assert templates[0]['keys_absolute'][-1][0]==pytest.approx(3.)
    assert report['final_conflicting_intervals']==0
