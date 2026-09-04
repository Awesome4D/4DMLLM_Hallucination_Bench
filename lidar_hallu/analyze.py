"""Validate all archived rows and reproduce submission statistics.

Usage: python -m lidar_hallu.analyze --dataset PATH/b4dl_dataset --predictions PATH/b4dl_eval --output results
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter
from pathlib import Path
import numpy as np
from .metrics import INVALID, parse_answer, summarize, paired_diagnostics, clustered_interval

CATEGORIES = ['object_existence','temporal_grounding','ego_relative_spatial','motion_action','distance_depth']
CONDITIONS = [('Base',''),('Prompt','_no_position_bias'),('TSCD .5','_temporal_shuffle_cd_alpha05'),('TSCD 1','_temporal_shuffle_cd'),('TSCD 1.5','_temporal_shuffle_cd_alpha15')]

def read_json(path: Path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)

def write_csv(path: Path, rows: list[dict]):
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out, fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results'));p.add_argument('--bootstrap-draws',type=int,default=10000)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    base={c:read_json(args.dataset/'hallucination_v2'/f'{c}.json') for c in CATEGORIES}
    audit={'records':0,'prediction_errors':0,'parser_changes':[],'files':{},'variant_checks':{},'dataset':{}}
    for path in sorted(args.dataset.glob('*/*.json')):
        audit['files'][str(path.relative_to(args.dataset))]=hashlib.sha256(path.read_bytes()).hexdigest()
    all_scenes=set(); all_keys=set(); feature_lengths={}
    for cat,items in base.items():
        all_scenes.update(x['scene_id'] for x in items)
        keys=[(cat,x['scene_id'],x['question']) for x in items]
        assert len(set(keys))==len(keys),f'Duplicate query in {cat}'
        all_keys.update(keys)
        audit['dataset'][cat]={'n':len(items),'scenes':len({x['scene_id'] for x in items}),
            'question_types':dict(Counter(x['question_type'] for x in items)), 'split':dict(Counter(x['split'] for x in items))}
    audit['unique_queries']=len(all_keys);audit['unique_scenes']=len(all_scenes)
    # Validate all benchmark variants semantically before pairing model outputs.
    for folder in sorted(args.dataset.glob('hallucination_v2*')):
        total=0
        for cat in CATEGORIES:
            items=read_json(folder/f'{cat}.json');assert len(items)==len(base[cat])
            for a,b in zip(base[cat],items):
                for key in ['scene_id','answer','options','question_type']:
                    assert a.get(key)==b.get(key),(folder.name,cat,key)
                qa=a['question'].split('Do not favor any option')[0].strip()
                qb=b['question'].split('Do not favor any option')[0].strip()
                assert qa==qb,(folder.name,cat,'question')
                total+=1
        audit['variant_checks'][folder.name]=total
    metrics=[];paired=[];binary=[];mcq=[];per_answer=[];scenes_rows=[];conditions={}
    for meta in [False,True]:
        model='Meta' if meta else 'NoMeta'
        for name,suffix in CONDITIONS:
            folder='hallucination_v2'+suffix+('_metatoken' if meta else '')
            expected_dataset='hallucination_v2'+('_no_position_bias' if name=='Prompt' else '')+('_metatoken' if meta else '')
            merged=[]
            for cat in CATEGORIES:
                path=args.predictions/folder/f'{cat}_predictions.jsonl'
                rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                items=read_json(args.dataset/expected_dataset/f'{cat}.json')
                assert len(rows)==len(items)==2000,(folder,cat,len(rows))
                audit['files'][str(path.relative_to(args.predictions))]=hashlib.sha256(path.read_bytes()).hexdigest()
                for i,(r,d) in enumerate(zip(rows,items)):
                    assert r['index']==i,(folder,cat,i)
                    for key in ['scene_id','question','options','prompt','question_type']:
                        assert r.get(key)==d.get(key),(folder,cat,i,key)
                    assert r['reference_answer']==d['answer'],(folder,cat,i,'reference')
                    if r.get('shuffle_permutation'):
                        perm=list(map(int,r['shuffle_permutation'].split()))
                        assert sorted(perm)==list(range(len(perm)))
                        sid=r['scene_id']
                        assert sid not in feature_lengths or feature_lengths[sid]==len(perm)
                        feature_lengths[sid]=len(perm)
                    r['pred']=parse_answer(r['prediction'],r['question_type'],r.get('options'),r.get('error',''))
                    r['gold']=r['reference_answer']; r['strict_correct']=r['pred']==r['gold']
                    audit['records']+=1;audit['prediction_errors']+=bool(r.get('error'))
                    old=r.get('parsed_prediction') or INVALID
                    if old !=r['pred']:
                        audit['parser_changes'].append(dict(condition=folder,category=cat,index=i,before=old,after=r['pred'],raw=r['prediction'],gold=r['gold']))
                merged.extend(rows)
                for kind in sorted({r['question_type'] for r in rows}):
                    sub=[r for r in rows if r['question_type']==kind]
                    s=summarize([r['gold'] for r in sub],[r['pred'] for r in sub],['A','B','C','D'] if kind=='mcq' else ['Yes.','No.'])
                    s.update(model=model,condition=name,category=cat,kind=kind)
                    if name=='Base':
                        (mcq if kind=='mcq' else binary).append(s)
                        for label,recall in s['recall'].items():per_answer.append(dict(model=model,category=cat,kind=kind,label=label,recall=recall))
                result=dict(model=model,condition=name,category=cat,n=len(rows),
                    accuracy=np.mean([r['strict_correct'] for r in rows]),
                    legacy_accuracy=np.mean([r['correct'] for r in rows]),
                    valid_rate=np.mean([r['pred']!=INVALID for r in rows]))
                metrics.append(result)
            conditions[(model,name)]=merged
            y=[r['gold'] for r in merged];pr=[r['pred'] for r in merged]
            metrics.append(dict(model=model,condition=name,category='Overall',n=len(merged),accuracy=np.mean(np.array(y)==np.array(pr)),
                legacy_accuracy=np.mean([r['correct'] for r in merged]),valid_rate=np.mean(np.array(pr)!=INVALID)))
            if name!='Base':
                before=conditions[(model,'Base')]
                for cat in CATEGORIES+['Overall','All MCQ','All binary']:
                    idx=[i for i,r in enumerate(merged) if cat=='Overall' or r['category']==cat or (cat=='All MCQ' and r['question_type']=='mcq') or (cat=='All binary' and r['question_type']=='binary')]
                    gold=[y[i] for i in idx];a=[before[i]['pred'] for i in idx];b=[pr[i] for i in idx]
                    scene=[merged[i]['scene_id'] for i in idx]
                    ss=paired_diagnostics(gold,a,b)
                    delta=[float(g==bb)-float(g==aa) for g,aa,bb in zip(gold,a,b)]
                    lo,hi=clustered_interval(delta,scene,draws=args.bootstrap_draws)
                    ss.update(model=model,condition=name,category=cat,ci_low=lo,ci_high=hi)
                    paired.append(ss)
            for scene in sorted(all_scenes):
                sub=[r for r in merged if r['scene_id']==scene]
                scenes_rows.append(dict(model=model,condition=name,scene_id=scene,n=len(sub),correct=sum(r['strict_correct'] for r in sub)))
    # Relative-prior intervals and pooled baseline evidence.
    for row in mcq:
        sub=[r for r in conditions[(row['model'],'Base')] if r['category']==row['category'] and r['question_type']=='mcq']
        delta=[float(r['pred']==r['gold'])-float(r['gold']=='A') for r in sub]
        row['ci_low'],row['ci_high']=clustered_interval(delta,[r['scene_id'] for r in sub],draws=args.bootstrap_draws)
    for model in ['NoMeta','Meta']:
        sub=[r for r in conditions[(model,'Base')] if r['question_type']=='mcq']
        s=summarize([r['gold'] for r in sub],[r['pred'] for r in sub],['A','B','C','D']);s.update(model=model,condition='Base',category='All MCQ',kind='mcq')
        delta=[float(r['pred']==r['gold'])-float(r['gold']=='A') for r in sub]
        s['ci_low'],s['ci_high']=clustered_interval(delta,[r['scene_id'] for r in sub],draws=args.bootstrap_draws)
        mcq.append(s)
    # Validate generated motion labels and temporal distractors from retained metadata.
    for d in base['motion_action']:
        dd,dy=d['distance_delta'],d['lateral_delta']
        truth={'becomes closer to the ego vehicle':dd<=-2.,'moves farther away from the ego vehicle':dd>=2.,'moves left relative to the ego vehicle':dy>=1.,'moves right relative to the ego vehicle':dy<=-1.,'maintains a similar relative position':abs(dd)<1. and abs(dy)<.5}[d['action']]
        assert truth==(d['answer']=='Yes.')
        assert d['end_frame']-d['start_frame']==8
    exact_temporal=0; lengths=[]
    for d in base['temporal_grounding']:
        gt=d['gt_interval']; ok=True; candidates=[]
        for k,txt in d['options'].items():
            nums=list(map(int,re.findall(r'\d+',txt))); assert len(nums)==2
            a,b=nums; assert 0<=a<=b<feature_lengths[d['scene_id']]; candidates.append((a,b))
            if k==d['correct_option']: assert nums==list(gt)
            else:
                union=max(gt[1],b)-min(gt[0],a)
                iou=max(0,min(gt[1],b)-max(gt[0],a))/union if union>0 else 0.
                assert iou<=.20000001
        assert len(set(candidates))==4
        exact_temporal+=1;lengths.append(gt[1]-gt[0])
    audit['validated_motion_labels']=2000;audit['validated_temporal_options']=exact_temporal
    audit['temporal_gt_duration_outside_distractor_range']=sum(not 5<=x<=19 for x in lengths)
    audit['feature_length_scenes']=dict(Counter(feature_lengths.values()))
    audit['feature_lengths']=feature_lengths
    for cat,items in base.items():
        for d in items:
            for key in ['frame','start_frame','end_frame']:
                if d.get(key) is not None: assert 0<=d[key]<feature_lengths[d['scene_id']],(cat,key,d)
    audit['bootstrap_draws']=args.bootstrap_draws;audit['bootstrap_seed']=20260904
    for name,rows in [('accuracy',metrics),('paired',paired),('binary',binary),('mcq',mcq),('per_answer',per_answer),('scenes',scenes_rows)]:
        write_csv(args.output/f'{name}.csv',rows)
        (args.output/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['records','unique_queries','unique_scenes','prediction_errors','validated_motion_labels','validated_temporal_options','temporal_gt_duration_outside_distractor_range']},indent=2))
    print('Parser differences:',len(audit['parser_changes']))
    for r in mcq: print(r['model'],r['category'],'acc',round(r['accuracy'],4),'A',round(r['a_rate'],4),'excess',round(r['excess'],4),'rescue/damage',r['rescue'],r['damage'])
    for r in paired:
        if r['category']=='Overall':print(r)
if __name__=='__main__':main()
