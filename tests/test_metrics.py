import numpy as np
import pytest
from lidar_hallu.metrics import INVALID, parse_answer, summarize, paired_diagnostics, clustered_interval

@pytest.mark.parametrize('text,want', [('A','A'), ('(B)','B'), ('C. because','C'), ('The answer is D','D'), ('Answer: B','B'), ('car',INVALID), ('Definitely A',INVALID), ('',INVALID)])
def test_explicit_mcq_contract(text,want):
    assert parse_answer(text,'mcq') == want

@pytest.mark.parametrize('text,want', [('Yes.','Yes.'), ('no','No.'), ('Yes, it is.','Yes.'), ('Nobody',INVALID), ('Possibly yes',INVALID)])
def test_binary_contract(text,want):
    assert parse_answer(text,'binary') == want

def test_unique_full_option_and_error():
    assert parse_answer('car','mcq',{'A':'car','B':'bus'}) == 'A'
    assert parse_answer('car','mcq',{'A':'car','B':'car'}) == INVALID
    assert parse_answer('A','mcq',error='inference failed') == INVALID
    with pytest.raises(ValueError): parse_answer('A','free')

def test_invalid_denominator():
    s=summarize(['A','B'],['A',INVALID],['A','B','C','D'])
    assert s['accuracy']==.5 and s['valid_rate']==.5

def test_exact_prior_identity():
    rng=np.random.default_rng(1)
    for _ in range(30):
        gold=rng.choice(['A','B','C','D'],100).tolist()
        pred=rng.choice(['A','B','C','D',INVALID],100).tolist()
        s=summarize(gold,pred,['A','B','C','D'])
        assert s['accuracy']-s['always_a'] == pytest.approx(s['excess'])
        assert abs(s['excess']) <= 1-s['a_rate']+1e-12

def test_paired_repair_identity():
    s=paired_diagnostics(['A','B','C','D'],['A','A','A','B'],['B','B','D','B'])
    assert (s['repaired'],s['regressed'],s['stable_wrong'])==(1,1,1)
    assert s['delta']==0 and s['flip_rate']==.75

def test_cluster_resampling_reproducible():
    lo,hi=clustered_interval([1,1,1,1],['s1','s1','s2','s2'],draws=200)
    assert (lo,hi)==(1.,1.)
    x=clustered_interval([1,0,0],['s1','s1','s2'],draws=300)
    assert x==clustered_interval([1,0,0],['s1','s1','s2'],draws=300)
    for vals,scenes in [([],[]),([1],[])]:
        with pytest.raises(ValueError): clustered_interval(vals,scenes)

def test_length_validation():
    with pytest.raises(ValueError): summarize([],[],['A'])
    with pytest.raises(ValueError): paired_diagnostics(['A'],[],['A'])
