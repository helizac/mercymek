import hashlib
import json

import cv2
import numpy as np
import pytest

from mercy import config
from mercy.analysis.enhance import verified_source_plate
from mercy.generation.vehicles import plate_for


def case_with_native_plate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    data=tmp_path/'data';data.mkdir()
    image=np.arange(32*48*3,dtype=np.uint8).reshape(32,48,3)
    original=data/'original.png';cv2.imwrite(str(original),image)
    case=tmp_path/'case';case.mkdir()
    audit={'text':'AB123CD','confidence':.92,'frame':90,'box':[20,12,35,22],
           'enhanced':False,'evidence_file':'data/original.png',
           'evidence_sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
    (case/'source_provenance.json').write_text(json.dumps({'original_plate_audit':audit}))
    meta={'width':48,'height':32,'src_fps':30,'trim_start':2.,'duration':12.}
    overlays={'30':{'car:1':{'box':[10,8,40,30]},'car:2':{'box':[0,0,12,12]}}}
    return case,meta,overlays,image,original


def test_original_plate_keeps_source_pixels_and_correct_trimmed_frame(tmp_path, monkeypatch):
    case,meta,overlays,image,original=case_with_native_plate(tmp_path,monkeypatch)
    reading=verified_source_plate(case,meta,overlays,['car:1','car:2'])
    assert reading['vehicle']=='car:1' and reading['t']==1.
    assert reading['original_frame']==90 and reading['source_verified']
    assert np.array_equal(cv2.imread(str(case/reading['images']['raw'])),image[12:22,20:35])
    assert (case/reading['original_frame_image']).read_bytes()==original.read_bytes()
    assert 'enhanced' not in reading['images'] and reading['frames_used']==1


def test_original_plate_rejects_changed_evidence_and_omitted_frame(tmp_path, monkeypatch):
    case,meta,overlays,_,original=case_with_native_plate(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='omits'):
        verified_source_plate(case,dict(meta,trim_start=4),overlays,['car:1'])
    original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash changed'):
        verified_source_plate(case,meta,overlays,['car:1'])


def test_generated_plate_detail_cannot_override_verified_original_text():
    plates={'plates':[
        {'key':'native','vehicle':'car:1','legible':True,'confidence':.92,'text':'AB123CD','source_verified':True},
        {'key':'sr','vehicle':'car:1','legible':True,'confidence':.99,'text':'AB128CD'}]}
    assert plate_for('car:1',plates)['text']=='AB123CD'
    assert not plate_for('car:2',plates)['observed']
