import json

import numpy as np
import pytest

from mercy.ingest.playback_motion import bounded_velocity, body_headings, camera_path, fit_incident
from mercy.ingest.vehicle_scene import Cameras, project_box


def test_velocity_projection_removes_teleports_and_obeys_acceleration_bound():
    t=np.linspace(0,4,17)
    velocity=np.column_stack([np.sin(t)*20,np.full(len(t),8.)]);velocity[8]=[500,-800]
    result=bounded_velocity(velocity,t,max_speed=30,max_acceleration=8)
    assert np.max(np.linalg.norm(result,axis=1))<=30+1e-8
    assert np.max(np.linalg.norm(np.diff(result,axis=0)/np.diff(t)[:,None],axis=1))<=8+1e-8


def test_camera_does_not_alternate_between_pnp_and_zero_fallback(tmp_path):
    t=np.linspace(0,4,33);M=np.repeat(np.eye(4)[None],len(t),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.35;M[::7,2,3]=np.arange(len(M[::7]))*8
    (tmp_path/'vehicles').mkdir();(tmp_path/'environment').mkdir()
    (tmp_path/'environment/cameras.npz').write_bytes(b'raw evidence untouched')
    (tmp_path/'physics.json').write_text(json.dumps({'series':{'objects':{'ego':{'t':[0,4],'speed_kmh':[36,36]}}}}))
    fixed=camera_path(tmp_path,t,M,2.)
    assert np.diff(fixed[:,2,3])==pytest.approx(np.full(32,1.25))
    assert fixed[16,2,3]==pytest.approx(M[16,2,3])
    assert fixed[:,:3,:3]==pytest.approx(M[:,:3,:3])


def test_full_video_heading_controls_left_right_not_a_box_fit(tmp_path):
    (tmp_path/'vehicles').mkdir()
    review={'headings':[{'track':'car:1','confidence':.9,'source_frame':0,'road_yaw_deg':-25},
                        {'track':'car:1','confidence':.9,'source_frame':30,'road_yaw_deg':-10}]}
    (tmp_path/'vehicles/heading-review.json').write_text(json.dumps(review))
    yaw,source=body_headings(tmp_path,{'key':'car:1','model':{}},np.linspace(0,1,11),None,30)
    assert np.all(yaw<0) and yaw[[0,-1]]==pytest.approx([-25,-10])
    assert 'full-video' in source


def test_playback_camera_uses_published_road_height_instead_of_stale_pnp(tmp_path):
    t=np.linspace(0,2,17);M=np.repeat(np.eye(4)[None],len(t),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.8;M[:,2,3]=t*4
    (tmp_path/'vehicles').mkdir();(tmp_path/'environment').mkdir()
    (tmp_path/'environment/cameras.npz').write_bytes(b'preserved PnP evidence')
    (tmp_path/'environment/reconstruction.json').write_text(json.dumps({'road':{'camera_height_m':1.35}}))
    (tmp_path/'physics.json').write_text(json.dumps({'series':{'objects':{'ego':{'t':[0,2],'speed_kmh':[14.4,14.4]}}}}))
    fixed,velocity=camera_path(tmp_path,t,M,1.,return_velocity=True)
    assert fixed[:,1,3]==pytest.approx(np.full(len(t),1.35))
    assert fixed[:,2,3]==pytest.approx(M[:,2,3])
    assert velocity[:,1]==pytest.approx(np.zeros(len(t)))
    assert (tmp_path/'environment/cameras.npz').read_bytes()==b'preserved PnP evidence'


def test_roundabout_camera_velocity_follows_view_heading(tmp_path):
    from scipy.spatial.transform import Rotation
    t=np.linspace(0,4,41);angle=np.linspace(0,40,len(t))
    M=np.repeat(np.eye(4)[None],len(t),axis=0)
    M[:,:3,:3]=Rotation.from_euler('y',angle[:,None],degrees=True).as_matrix()@np.diag([1.,-1.,1.])
    M[:,1,3]=1.35
    (tmp_path/'vehicles').mkdir();(tmp_path/'environment').mkdir()
    (tmp_path/'environment/cameras.npz').write_bytes(b'original')
    (tmp_path/'scene.json').write_text(json.dumps({'road_type':'multi-lane roundabout'}))
    (tmp_path/'physics.json').write_text(json.dumps({'series':{'objects':{'ego':{'t':[0,4],'speed_kmh':[36,36]}}}}))
    fixed,velocity=camera_path(tmp_path,t,M,2.,return_velocity=True)
    assert np.linalg.norm(velocity[:,[0,2]],axis=1)==pytest.approx(np.full(len(t),10.))
    assert velocity[-1,0]>6 and velocity[-1,2]<8
    assert fixed[-1,0,3]-fixed[0,0,3]>10


def test_roundabout_body_heading_uses_relative_view_instead_of_fixed_global_yaw(tmp_path):
    from scipy.spatial.transform import Rotation
    (tmp_path/'vehicles').mkdir()
    (tmp_path/'scene.json').write_text(json.dumps({'road_type':'roundabout'}))
    (tmp_path/'vehicles/heading-review.json').write_text(json.dumps({'headings':[
        {'track':'car:1','confidence':.99,'source_frame':0,'road_yaw_deg':0}]}))
    times=np.linspace(0,2,21);M=np.repeat(np.eye(4)[None],len(times),axis=0)
    M[:,:3,:3]=Rotation.from_euler('y',np.linspace(0,40,len(times))[:,None],degrees=True).as_matrix()@np.diag([1.,-1.,1.])
    K=np.repeat(np.eye(3)[None],len(times),axis=0);cameras=Cameras(times,M,K)
    profile={'headings':[{'source_frame':0,'relative_yaw_deg':0,'confidence':.9}]}
    yaw,_=body_headings(tmp_path,{'key':'car:1','model':profile},times,cameras,30)
    assert yaw==pytest.approx(np.linspace(0,40,len(times)))


def test_playback_camera_continues_at_boundary_instead_of_freezing_last_video_frames():
    t=np.array([0.,1.,2.]);M=np.repeat(np.eye(4)[None],3,axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.4;M[:,2,3]=t*10
    K=np.repeat(np.eye(3)[None],3,axis=0);velocity=np.tile([0.,0.,10.],(3,1))
    cameras=Cameras(t,M,K,position_velocity=velocity)
    assert cameras.at(2.1)[0]==pytest.approx([0,1.4,21])
    assert cameras.at(-.1)[0]==pytest.approx([0,1.4,-1])
    poses=cameras.browser_poses()
    assert poses[-1][3]==pytest.approx([0,0,-10])


def test_incident_fit_keeps_a_turn_continuous_despite_bad_per_frame_depth():
    t=np.linspace(0,4,41);M=np.repeat(np.eye(4)[None],len(t),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.4
    K=np.repeat(np.array([[[900.,0,640],[0,900,360],[0,0,1]]]),len(t),axis=0)
    cameras=Cameras(t,M,K);dims=[1.85,1.6,4.5]
    heading=np.linspace(-20,0,len(t));true=np.column_stack([3-t*.5,30+t*4])
    boxes=np.array([project_box(p,y,dims,*cameras.at(tt)) for p,y,tt in zip(true,heading,t)])
    noisy=true.copy();noisy[12:15,1]+=70
    anchors=[dict(t=float(t[i]),position=true[i].tolist()) for i in [0,10,20,30,40]]
    result,info=fit_incident(t,boxes,np.ones_like(boxes,bool),[cameras.at(tt) for tt in t],dims,noisy,t,heading,anchors)
    speed=np.linalg.norm(np.diff(result[:,:2],axis=0),axis=1)/np.diff(t)
    assert speed.max()<=40.0001
    assert info['max_acceleration_mps2']<=10.0001
    assert result[:,2]==pytest.approx(heading)
    assert np.max(abs(result[:,:2]-true))<.5
