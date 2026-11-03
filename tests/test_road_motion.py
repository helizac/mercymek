import numpy as np
import pytest

from mercy.ingest.road_motion import direction, fit_lane, integrated_speed_basis
from mercy.ingest.vehicle_scene import Cameras,project_box


def fixture_scene(speed=8.,heading=0.):
    times=np.linspace(0,4,41)
    matrices=np.repeat(np.eye(4)[None],len(times),axis=0)
    matrices[:,:3,:3]=np.diag([1.,-1.,1.]);matrices[:,1,3]=1.4;matrices[:,2,3]=times*5
    K=np.repeat(np.array([[[900.,0,640],[0,900,360],[0,0,1]]]),len(times),axis=0)
    cameras=Cameras(times,matrices,K);dims=[1.85,1.65,4.6]
    positions=np.column_stack([np.full(len(times),3.),50+speed*times])
    boxes=np.array([project_box(p,heading,dims,*cameras.at(t)) for p,t in zip(positions,times)])
    return times,cameras,dims,positions,boxes


def test_integrating_positive_continuous_speeds_cannot_reverse_or_teleport():
    knots=np.array([0.,1.,3.]);t=np.linspace(0,3,301);B=integrated_speed_basis(t,knots)
    z=B@np.array([0.,2.,1.])
    assert z[0]==0 and z[-1]==pytest.approx(4.)
    assert np.all(np.diff(z)>=0)
    v=np.gradient(z,t)
    assert abs(v[99]-v[101])<.05


def test_joint_fit_removes_sideways_skating_and_heading_wobble():
    t,c,dims,true,boxes=fixture_scene()
    rng=np.random.default_rng(7);boxes+=rng.normal(0,.5,boxes.shape)
    bad=true+np.column_stack([np.sin(t*8)*8,np.sin(t*5)*12])
    result,info=fit_lane(t,boxes,np.ones_like(boxes,dtype=bool),[c.at(x) for x in t],dims,bad,t)
    assert np.ptp(result[:,0])<1e-8
    assert np.all(np.diff(result[:,1])>=-1e-8)
    assert np.max(abs(result[:,2]))==0
    assert np.max(abs(result[:,:2]-true))<.25
    assert info['median_valid_edge_error_px']<1


def test_stationary_vehicle_is_not_forced_to_drive_forward():
    t,c,dims,true,boxes=fixture_scene(speed=0)
    result,info=fit_lane(t,boxes,np.ones_like(boxes,dtype=bool),[c.at(x) for x in t],dims,true,t,stationary=True)
    assert np.ptp(result[:,0])<1e-8 and np.ptp(result[:,1])<1e-8
    assert result[0,:2]==pytest.approx(true[0],abs=.1)
    assert info['speed_range_mps']==[0.,0.]


def test_opposing_traffic_moves_towards_the_camera_without_moonwalking():
    t,c,dims,true,boxes=fixture_scene(speed=-3.,heading=180)
    result,info=fit_lane(t,boxes,np.ones_like(boxes,dtype=bool),[c.at(x) for x in t],dims,true,t,heading_deg=180)
    assert np.all(np.diff(result[:,1])<=1e-8)
    assert np.max(abs(result[:,:2]-true))<.25
    assert np.all(result[:,2]==180)


def test_ambiguous_crop_cannot_turn_normal_traffic_backwards():
    profile={'headings':[{'relative_yaw_deg':180,'confidence':.99,'view':'front'}]}
    assert direction(profile,'straight_forward')==0
    assert direction(profile,'opposite_straight')==180
