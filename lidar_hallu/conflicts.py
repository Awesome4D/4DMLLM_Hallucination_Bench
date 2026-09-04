"""Diagnose model errors on scene-held-out sensor-free prior conflict strata."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .metrics import parse_answer, clustered_interval

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    priors=[json.loads(s) for s in (a.output/'sensor_free_predictions.jsonl').read_text().splitlines()]
    lookup={(x['category'],x['scene_id'],x['question']):x for x in priors}
    result=[]
    for model,ext in [('NoMeta',''),('Meta','_metatoken')]:
        merged=[]
        for path in sorted((a.predictions/('hallucination_v2'+ext)).glob('*_predictions.jsonl')):
            for line in path.read_text().splitlines():
                x=json.loads(line);v=lookup[(x['category'],x['scene_id'],x['question'])]
                pred=parse_answer(x['prediction'],x['question_type'],x.get('options'),x.get('error',''))
                merged.append(dict(category=x['category'],scene=x['scene_id'],gold=x['reference_answer'],pred=pred,prior=v['pred'],kind=x['question_type']))
        for cat in sorted({x['category'] for x in merged})+['Overall','All MCQ']:
            rows=[x for x in merged if cat=='Overall' or x['category']==cat or(cat=='All MCQ' and x['kind']=='mcq')]
            for group in ['all','prior-correct','prior-wrong']:
                sub=[x for x in rows if group=='all' or (x['prior']==x['gold'])==(group=='prior-correct')]
                corr=[x['pred']==x['gold'] for x in sub]
                delta=[float(x['pred']==x['gold'])-float(x['prior']==x['gold']) for x in sub]
                lo,hi=clustered_interval(delta,[x['scene'] for x in sub])
                result.append(dict(model=model,category=cat,group=group,n=len(sub),accuracy=float(np.mean(corr)),prior_accuracy=float(np.mean([x['prior']==x['gold'] for x in sub])),excess=float(np.mean(delta)),ci_low=lo,ci_high=hi))
    (a.output/'prior_conflicts.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result:
        if r['category'] in ['Overall','All MCQ','temporal_grounding']:print(r)
if __name__=='__main__':main()
