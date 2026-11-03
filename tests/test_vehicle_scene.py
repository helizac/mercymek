import json

import numpy as np
import pytest

from mercy.generation.vehicles import vehicle_observations
from mercy.ingest.vehicle_scene import Cameras, continue_track, continuous_cameras, fit_box, fitting_points, heading, merge_observations, observed_edges, project_box, project_fitting_points, reconstruct


def cameras():
    t=np.linspace(0,1,9)
    M=np.repeat(np.eye(4)[None],len(t),axis=0)
    M[:,:3,:3]=np.diag([1.,-1.,1.]);M[:,1,3]=1.5;M[:,2,3]=t*4
    K=np.repeat(np.array([[[800.,0,640],[0,800,360],[0,0,1]]]),len(t),axis=0)
    return Cameras(t,M,K)


def test_source_boxes_are_used_without_neural_depth_tracks():
    obs={str(f):{'car:missing-depth':{'concept':'car','box':[700,350,760,400]},
                 'plate:0':{'concept':'license plate','box':[710,380,730,390]}} for f in range(4)}
    assert list(vehicle_observations(obs))==['car:missing-depth']


def test_camera_quaternion_matches_the_source_projection():
    c=cameras();t,p,q=c.browser_poses()[4]
    from scipy.spatial.transform import Rotation
    browser_R=Rotation.from_quat(q).as_matrix()
    C,R,K=c.at(t);world=np.array([2.,.7,16.])
    pc=(world-C)@R
    gl=(world*[1,1,-1]-p)@browser_R
    assert gl==pytest.approx(pc*[1,-1,-1],abs=1e-8)


def test_cuboid_fit_recovers_ground_position_and_keeps_a_rear_facing_car_forward():
    c=cameras();C,R,K=c.at(.4);dims=[1.85,1.65,4.6]
    box=project_box([3.,22.],0,dims,C,R,K)
    xz,yaw,error=fit_box(box,dims,C,R,K,0,(1280,720),.95)
    assert xz==pytest.approx([3,22],abs=.05)
    assert abs(yaw)<2 and error<.5
    profile={'headings':[{'source_frame':0,'relative_yaw_deg':0,'confidence':.9}]}
    assert heading(profile,12,30,R)[0]==pytest.approx(0)


def test_offscreen_roof_does_not_define_truncated_source_top():
    C=np.array([0.,1.35,0.]);R=np.diag([1.,-1.,1.])
    K=np.array([[700.,0,640],[0,700.,360],[0,0,1.]])
    # The tall end is entirely outside the right edge; only the lower end is visible.
    points=np.array([[-1.,0.,0.],[-1.,1.4,0.],[3.,0.,0.],[3.,2.5,0.]])
    whole=project_fitting_points(points,[0.,2.],C,R,K)
    visible=project_fitting_points(points,[0.,2.],C,R,K,(0,1280))
    assert visible[1]>whole[1]+100
    assert visible[[0,2]]==pytest.approx(whole[[0,2]])
    assert np.isfinite(visible).all()


def test_fitting_geometry_uses_the_displayed_hatch_roof_instead_of_a_tall_bonnet():
    model={'body_type':'hatchback','dimensions_m':[1.78,1.48,4.15]}
    points=fitting_points(model,0)
    assert len(points)>8
    front=points[points[:,2]>4.15*.4]
    assert front[:,1].max()<1.48*.8
    assert points[:,1].max()>1.48*.95


def test_reconstruction_keeps_all_observed_vehicles_and_identity_across_gaps(tmp_path):
    c=cameras();dims=[1.85,1.65,4.6]
    overlays={}
    for frame in range(9):
        C,R,K=c.at(frame/8)
        overlays[str(frame)]={}
        for i in range(3):
            if i==2 and frame in [3,4,5]:continue
            box=project_box([i*2+1,22+i*8],0,dims,C,R,K)
            overlays[str(frame)][f'car:{i}']={'concept':'car','box':box.tolist()}
    (tmp_path/'overlays.json').write_text(json.dumps(overlays))
    appearance={f'car:{i}':{'dimensions_m':dims,'headings':[{'source_frame':0,'relative_yaw_deg':0,'confidence':1}]} for i in range(3)}
    templates,report=reconstruct(tmp_path,c,{'src_fps':8,'duration':1,'width':1280,'height':720},appearance)
    assert len(report)==3 and len(templates)==4
    assert all(t['persistent'] and len(t['keys_absolute'])==9 for t in templates)
    last=next(t for t in templates if t['key']=='car:2')
    assert max(abs(k[3]) for k in last['keys_absolute'])<3
    assert max(abs(k[2]-38) for k in last['keys_absolute'])<.2


def test_unobserved_continuation_does_not_spawn_or_reverse_a_stopped_car():
    times=np.array([1.,1.2,1.4,1.6]);poses=np.tile([3.,24.,0.],(4,1))
    continued=continue_track(times,poses,np.linspace(0,3,31))
    assert continued==pytest.approx(np.tile([3.,24.,0.],(31,1)))


def test_partial_roof_is_not_fitted_as_a_complete_far_away_car():
    C,R,K=cameras().at(.4);dims=[1.85,1.65,4.6]
    box=project_box([3,22],0,dims,C,R,K)
    box[3]=box[1]+(box[3]-box[1])*.35
    xz,yaw,error=fit_box(box,dims,C,R,K,0,(1280,720),.9,
                        edge_valid=np.array([True,True,True,False]),position_prior=np.array([3.,22.]))
    assert xz==pytest.approx([3.,22.],abs=.3)


def test_nested_masks_mark_the_occlusion_edge_without_crashing():
    behind={'box':[100,100,160,140],'poly':[[[100,100],[160,100],[160,140],[100,140]],
            [[110,110],[120,110],[120,115],[110,115]]],'concept':'car'}
    front={'box':[80,140,200,240],'poly':[[80,140],[200,140],[200,240],[80,240]],'concept':'car'}
    edges=observed_edges('a',behind,{'a':behind,'b':front},(1280,720))
    assert edges[:3].all() and not edges[3]


def test_duplicate_masks_merge_but_neighbouring_cars_stay_separate():
    obj={'box':[100,100,180,160],'poly':[[100,100],[180,100],[180,160],[100,160]],'concept':'car'}
    other={'box':[140,100,220,160],'poly':[[181,100],[220,100],[220,160],[181,160]],'concept':'car'}
    obs={key:[(f,dict(v)) for f in range(4)] for key,v in [('car:1',obj),('truck:1',obj),('car:2',other)]}
    profiles={k:{'confidence':.8,'headings':[]} for k in obs}
    merged,_,groups=merge_observations(obs,profiles,(1280,720))
    assert len(merged)==2 and sorted(len(v) for v in groups.values())==[1,2]


def test_calibrated_camera_window_is_not_spliced_to_an_unaligned_legacy_pose():
    t=np.arange(0,2,.125);M=np.repeat(np.eye(4)[None],len(t),axis=0)
    M[:,2,3]=100+t*3;core=(t>=.5)&(t<=1.5);M[core,2,3]=t[core]*3
    out=continuous_cameras(t,M,.5,1.5)
    assert out[core]==pytest.approx(M[core])
    assert np.max(abs(np.diff(out[:,2,3])))<.5
