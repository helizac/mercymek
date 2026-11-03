import json

import numpy as np
import pytest

from mercy.ingest.incident_tracks import ensure
from mercy.ingest.vehicle_scene import project_box


def test_missed_depth_party_uses_real_boxes_and_keeps_other_vehicles(tmp_path):
    times=np.array([0., .5, 1.])
    matrices=np.repeat(np.eye(4)[None],3,axis=0)
    matrices[:,:3,:3]=np.diag([1.,-1.,1.]);matrices[:,1,3]=1.4
    lens=np.array([[600.,0,640],[0,600,360],[0,0,1.]])
    dimensions=[1.9,1.8,5.3]
    # The pickup is visibly observed at 25m but has no neural depth rows.
    box=project_box([2.,25.],0,dimensions,matrices[0,:3,3],matrices[0,:3,:3],lens).tolist()
    values={
        'scene.json':{'impact_time_s':.5,'vehicles':[
            {'label':'pickup','body_type':'pickup','role':'struck'},
            {'label':'dark car','body_type':'sedan','role':'striking'}]},
        'meta.json':{'src_fps':2,'time_of_event':.5,'width':1280,'height':720},
        'frames4d.json':{'frames':[{'M':m.tolist()} for m in matrices]},
        'tracks.json':{'tracks':{'car:1':[{'t':float(t),'center':[0,1,20]} for t in times],
                                'car:2':[{'t':float(t),'center':[5,1,20]} for t in times]},
                       'aliases':{'truck:3':'pickup','car:1':'dark car','car:2':'dark car'}},
        'overlays.json':{str(i):{'truck:3':{'box':box,'concept':'truck'},
                                'car:1':{'box':[450,360,550,420],'concept':'car'},
                                'car:2':{'box':[400,360,450,400],'concept':'car'}} for i in range(3)}}
    for name,value in values.items():(tmp_path/name).write_text(json.dumps(value))
    np.savez(tmp_path/'recon.npz',times=times,intrinsic=np.repeat(lens[None],3,axis=0),depth=np.ones((3,720,1280)))
    original=(tmp_path/'tracks.json').read_bytes()
    result=ensure(tmp_path)
    assert result['incident_bindings']=={'pickup':'truck:3','dark car':'car:1'}
    assert 'car:2' in result['tracks'] and 'car:2' not in result['aliases']
    assert result['unselected_incident_aliases']['car:2']=='dark car'
    assert not result['track_seeds']['truck:3']['metric_depth_measured']
    assert len(result['tracks']['truck:3'])==3
    assert all(row['n']==0 and row['source_frame']==i for i,row in enumerate(result['tracks']['truck:3']))
    # Class dimensions are a prior: they recover the source depth to class-size accuracy.
    assert result['tracks']['truck:3'][1]['center'][2]==pytest.approx(25.,rel=.12)
    assert (tmp_path/'tracks.neural-original.json').read_bytes()==original
    assert ensure(tmp_path)['tracks']==result['tracks']
