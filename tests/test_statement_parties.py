import json

from mercy.analysis import statement


def test_witness_and_readable_bystander_plate_are_excluded_from_form(tmp_path,monkeypatch):
    artifacts={
        'physics.json':{'objects':[{'id':'ego','label':'witness','role':'other'},
            {'id':'car:4','label':'dark hatchback','vid':'v1'},
            {'id':'truck:10','label':'silver pickup','vid':'v2'}]},
        'plates.json':{'plates':[{'vehicle':'car:1','text':'9296C','legible':True}]}}
    for name,value in artifacts.items():(tmp_path/name).write_text(json.dumps(value))
    def generate(model,messages,*args,**kwargs):
        facts=json.loads(messages[1]['content'])
        assert facts['form_parties']['A']['id']=='car:4'
        assert facts['form_parties']['B']['id']=='truck:10'
        assert '9296C' not in json.dumps(facts)
        # A model cannot override ownership with an unrelated readable plate.
        return {'A':{'label':'camera car','plate':'9296C','circumstances':[8]},
                'B':{'label':'other','plate':'9296C','circumstances':[]},'remarks':''}
    monkeypatch.setattr(statement,'structured',generate)
    result=statement.build(tmp_path)
    assert result['A']['label']=='dark hatchback' and result['A']['plate']==''
    assert result['B']['label']=='silver pickup' and result['B']['plate']==''
