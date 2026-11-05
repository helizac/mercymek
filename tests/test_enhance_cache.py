"""Cached derived images must match both the original crops and saved output hashes."""
import cv2
import numpy as np

from mercy.analysis import enhance as E


def test_reuses_real_outputs_but_invalidates_changed_input_and_tampering(tmp_path, monkeypatch):
    frames = {i:np.full((24,48,3),80+i,dtype=np.uint8) for i in range(3)}
    calls = []
    monkeypatch.setattr(E,"_read_frames",lambda *_:frames)
    monkeypatch.setattr(E,"fuse",lambda crops:(crops[1],len(crops)))

    def upscale(image):
        calls.append(True)
        return cv2.resize(image,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)

    monkeypatch.setattr(E,"super_resolve",upscale)
    args = (tmp_path/"clip.mp4",30,1,lambda _: [0,0,48,24],tmp_path)
    first = E.enhance(*args,"first",n=3,pad=0)
    second = E.enhance(*args,"second",n=3,pad=0)
    assert len(calls)==1 and second["cache_hit"]
    assert second["paths"]["enhanced"].read_bytes()==first["paths"]["enhanced"].read_bytes()
    frames[1][0,0]=255
    changed = E.enhance(*args,"changed",n=3,pad=0)
    assert len(calls)==2 and not changed.get("cache_hit")
    changed["paths"]["enhanced"].write_bytes(b"tampered")
    repaired = E.enhance(*args,"repaired",n=3,pad=0)
    assert len(calls)==3 and not repaired.get("cache_hit")
    assert cv2.imread(str(repaired["paths"]["enhanced"])) is not None
