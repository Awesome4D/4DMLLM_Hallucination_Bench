"""Leave-one-scene-out, sensor-free categorical controls.

Every scored example's entire scene is excluded from fitting. Only fields visible
in the question/option strings are used. No ground-truth coordinates or motion
values are inputs. This is cross-fitted benchmark diagnosis, not an independently
trained language-model baseline or a replacement for question-only model inference.
"""
from __future__ import annotations
import argparse,csv,json,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .metrics import summarize,clustered_interval

CATEGORIES=['object_existence','temporal_grounding','ego_relative_spatial','motion_action','distance_depth']

def context(item):
    q=item['question']
    cat=item['category']
    if cat=='object_existence' and item['question_type']=='binary':
        m=re.search(r'Does (.*?) exist in frame',q)
        if not m:raise ValueError(q)
        return (m[1],)
    if cat=='ego_relative_spatial':
        if item['question_type']=='binary':
            m=re.search(r'nearest (.*?) at the (.*?) position relative',q)
            if not m:raise ValueError(q)
            return(m[1],m[2])
        m=re.search(r'nearest (.*?) relative to',q)
        if not m:raise ValueError(q)
        return(m[1],)
    if cat=='motion_action':
        m=re.search(r'does the nearest (.*?)\?',q)
        if not m:raise ValueError(q)
        # Extract the entire proposition, excluding interval indices.
        return(m[1],)
    return ()

def feature(item, option):
    if item['category']=='temporal_grounding':
        a,b=map(int,re.findall(r'\d+',option))
        return b-a
    return option.lower().strip()

def predict(items):
    """Laplace-smoothed label/option prior with full-scene subtraction."""
    kind=items[0]['question_type'];counts=defaultdict(Counter);scounts=defaultdict(Counter)
    for x in items:
        ctx=context(x);s=x['scene_id']
        if kind=='binary':
            counts[ctx][x['answer']]+=1;scounts[(s,ctx)][x['answer']]+=1
        else:
            for k,txt in x['options'].items():
                token=(feature(x,txt),k==x['correct_option'])
                counts[ctx][token]+=1;scounts[(s,ctx)][token]+=1
    out=[]
    for x in items:
        ctx=context(x);c=counts[ctx]-scounts[(x['scene_id'],ctx)]
        if kind=='binary':
            pred='Yes.' if c['Yes.']>c['No.'] else 'No.'
        else:
            # Ratio of Laplace-smoothed correct and distractor counts. Constants
            # shared by candidate features cancel in argmax within each question.
            score={k:(c[(feature(x,v),True)]+1)/(c[(feature(x,v),False)]+1) for k,v in x['options'].items()}
            pred=max(score,key=score.get)
        out.append(pred)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    results=[];per_category=[];preds=[]
    for cat in CATEGORIES:
        items=json.loads((a.dataset/'hallucination_v2'/f'{cat}.json').read_text())
        catcorrect=[]
        for kind in sorted({x['question_type'] for x in items}):
            sub=[dict(x,category=cat) for x in items if x['question_type']==kind]
            pr=predict(sub);s=summarize([x['answer'] for x in sub],pr,['A','B','C','D'] if kind=='mcq' else ['Yes.','No.']);s.update(category=cat,kind=kind)
            s['ci_low'],s['ci_high']=clustered_interval([float(x['answer']==y) for x,y in zip(sub,pr)],[x['scene_id'] for x in sub])
            results.append(s);catcorrect.extend(x['answer']==y for x,y in zip(sub,pr))
            preds.extend(dict(category=cat,kind=kind,scene_id=x['scene_id'],question=x['question'],gold=x['answer'],pred=y) for x,y in zip(sub,pr))
        per_category.append(dict(category=cat,n=len(items),accuracy=float(np.mean(catcorrect))))
    (a.output/'sensor_free_priors.json').write_text(json.dumps(results,indent=2)+'\n')
    (a.output/'sensor_free_category.json').write_text(json.dumps(per_category,indent=2)+'\n')
    with (a.output/'sensor_free_predictions.jsonl').open('w') as f:
        for r in preds:f.write(json.dumps(r)+'\n')
    for r in results:print(r['category'],r['kind'],round(r['accuracy']*100,2),'CI',round(r['ci_low']*100,2),round(r['ci_high']*100,2))
    print(per_category)
if __name__=='__main__':main()
