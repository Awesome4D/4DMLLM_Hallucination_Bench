from copy import deepcopy
from lidar_hallu.priors import predict, context, feature

def binary(scene,answer):
    return dict(category='object_existence',question_type='binary',scene_id=scene,
                question='Does car exist in frame 004?',answer=answer)

def mcq(scene,answer):
    return dict(category='temporal_grounding',question_type='mcq',scene_id=scene,
                question='When does the event occur?',answer=answer,correct_option=answer,
                options={'A':'from frame 000 to frame 002','B':'from frame 010 to frame 015',
                         'C':'from frame 020 to frame 030','D':'from frame 002 to frame 018'})

def test_entire_scene_excluded_binary():
    items=[binary('held','Yes.'),binary('held','No.'),binary('other','Yes.')]
    before=predict(items)[:2]
    changed=deepcopy(items)
    for x in changed[:2]: x['answer']='No.'
    assert predict(changed)[:2]==before==['Yes.','Yes.']

def test_entire_scene_excluded_mcq():
    items=[mcq('held','A'),mcq('held','B'),mcq('other','C')]
    before=predict(items)[:2]
    changed=deepcopy(items)
    for x in changed[:2]: x['answer']=x['correct_option']='D'
    assert predict(changed)[:2]==before==['C','C']

def test_ground_truth_metadata_not_used():
    items=[binary('held','Yes.'),binary('other','Yes.')]
    before=predict(items)
    for x in items:
        x.update(object_class='forbidden oracle data',distance_meters=999,position='back')
    assert predict(items)==before

def test_temporal_feature_ignores_absolute_endpoints():
    x=mcq('s','A')
    assert feature(x,'from frame 001 to frame 009')==feature(x,'from frame 011 to frame 019')==8

def test_spatial_query_context():
    x=dict(category='ego_relative_spatial',question_type='binary',question='In frame 010, is the nearest car at the back position relative to the ego vehicle?')
    assert context(x)==('car','back')
    x.update(question_type='mcq',question='In frame 010, at which position is the nearest car relative to the ego vehicle?')
    assert context(x)==('car',)

def test_unseen_context_tie():
    assert predict([binary('one','Yes.')])==['No.']
    assert predict([mcq('one','D')])==['A']
