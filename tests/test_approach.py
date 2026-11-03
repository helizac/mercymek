import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from mercy.ingest import approach
from mercy.ingest.environment import write_splat


def make_case(tmp_path):
    case=tmp_path/'case';root=case/'environment'/'approach';root.mkdir(parents=True)
    base=case/'environment'/'world.splat'
    write_splat(base,np.array([[0.,0.,-5.]]),np.full((1,3),.1),np.array([[1.,0,0,0]]),np.ones((1,3))*.5,np.ones(1))
    image=root/'extended.png';cv2.imwrite(str(image),np.zeros((48,80,3),np.uint8))
    sha=hashlib.sha256(image.read_bytes()).hexdigest()
    (root/'review.json').write_text(json.dumps({'accepted':True,'image_sha256':sha,
        'road_polygon':[[0,500],[1000,500],[1000,1000],[0,1000]]}))
    np.savez_compressed(root/'guide-depth.npz',depth=np.zeros((48,80)),alpha=np.zeros((48,80)))
    keys=[[-1.,0,0,0],[0.,0,10,0]]
    (case/'moments.json').write_text(json.dumps({'moments':[{'id':'impact','file':'environment/world.splat',
        'vehicles':[{'key':'ego','keys':keys}]}]}))
    plan={'case':'case','plan_id':'known-plan','camera_to_road':np.diag([1.,-1.,1.]).tolist(),
          'camera_height_m':1.5,'distance_m':20.,'width':80,'height':48,'focal_px':40.,
          'base_world_sha256':approach.digest(base)}
    output=tmp_path/'output';output.mkdir()
    (output/'result.json').write_text(json.dumps({'plan_id':'known-plan','image_sha256':sha}))
    rng=np.random.default_rng(3)
    points=np.column_stack([rng.uniform(-3,3,10000),np.full(10000,1.5),rng.uniform(5,20,10000)])
    g={'pos':points,'scales':np.full((len(points),3),.05),'quat':np.tile([1.,0,0,0],(len(points),1)),
       'rgb':np.full((len(points),3),.4),'op':np.full(len(points),.9)}
    return case,plan,output,g,keys


def test_extension_preserves_original_world_and_observed_trajectory(tmp_path,monkeypatch):
    case,plan,output,g,keys=make_case(tmp_path)
    original=(case/'environment'/'world.splat').read_bytes()
    monkeypatch.setattr(approach,'read_gaussians',lambda path:copy.deepcopy(g))
    result=approach.fuse(case,plan,output)
    combined=(case/'environment'/'extended-world.splat').read_bytes()
    assert combined.startswith(original)
    assert (case/'environment'/'world.splat').read_bytes()==original
    assert result['extension_splats']>5000
    data=np.frombuffer((case/'environment'/'approach'/'extension.splat').read_bytes(),dtype=np.dtype([
        ('pos','<f4',3),('scale','<f4',3),('rgba','u1',4),('quat','u1',4)]))
    assert np.max(abs(data['pos'][:,1]))<1e-4
    assert data['pos'][:,2].min()>-3 and data['pos'][:,2].max()<20
    manifest=json.loads((case/'moments.json').read_text())
    assert manifest['moments'][0]['vehicles'][0]['keys']==keys
    assert manifest['moments'][0]['approach']['camera_height_m']==1.5


def test_extension_rejects_stale_base_geometry(tmp_path,monkeypatch):
    case,plan,output,g,_=make_case(tmp_path)
    (case/'environment'/'world.splat').write_bytes(b'changed')
    monkeypatch.setattr(approach,'read_gaussians',lambda path:copy.deepcopy(g))
    with pytest.raises(ValueError,match='original world changed'):
        approach.fuse(case,plan,output)


def test_extension_rejects_unreviewed_image(tmp_path):
    case,plan,output,_,_=make_case(tmp_path)
    (case/'environment'/'approach'/'extended.png').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='valid accepted review'):
        approach.fuse(case,plan,output)
