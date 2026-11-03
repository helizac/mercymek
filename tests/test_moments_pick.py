import json

from mercy.ingest.moments import pick


def test_witness_collision_without_braking_has_a_real_aftermath_view(tmp_path):
    (tmp_path/'physics.json').write_text(json.dumps({'impact':{'kind':'collision','t':12.9},'measurements':[],'objects':[]}))
    (tmp_path/'meta.json').write_text(json.dumps({'duration':17.,'time_of_event':12.9}))
    moments=pick(tmp_path)
    assert [(m['id'],m['t']) for m in moments]==[('before',12.3),('impact',12.9),('after',13.7)]
    assert all(m['id'] not in {'brake','hazard'} for m in moments)
