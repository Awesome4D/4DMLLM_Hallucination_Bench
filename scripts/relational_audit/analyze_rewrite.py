"""CPU-only reanalysis of immutable thesis archives for the relational rewrite.
Uses the committed explicit parser contract, scene bootstrap, and leave-scene-out
candidate controls. No new neural inference, logits, or model attention is produced.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, string
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
CATS=['object_existence','temporal_grounding','ego_relative_spatial','motion_action','distance_depth']
INVALID='<invalid>'
def parse_answer(text,kind,options=None,error=''):
    if error:return INVALID
    raw=str(text or '').strip()
    if kind=='binary':
        m=re.match(r'^(yes|no)(?=$|[\s.!?,:;])',raw,re.I)
        return m[1].capitalize()+'.' if m else INVALID
    if kind!='mcq':raise ValueError(kind)
    m=re.match(r'^\s*[\(\[]?\s*([abcd])(?=$|[\s\)\].:\-])',raw,re.I)
    if not m:m=re.match(r'^(?:the\s+)?(?:option|answer|choice)\s*(?:is|:|-)\s*([abcd])\b',raw,re.I)
    if m:return m[1].upper()
    def norm(s):return re.sub(r'\s+',' ',str(s).lower().strip().strip(string.punctuation+' '))
    matches=[k for k,v in (options or {}).items() if norm(v)==norm(raw)]
    return matches[0] if len(matches)==1 else INVALID

def ci(values,scenes,draws=10000):
    _,ind=np.unique(scenes,return_inverse=True)
    n=ind.max()+1;sums=np.bincount(ind,weights=values);counts=np.bincount(ind)
    ix=np.random.default_rng(20260904).integers(0,n,size=(draws,n))
    vals=sums[ix].sum(1)/counts[ix].sum(1)
    return [float(v) for v in np.quantile(vals,[.025,.975])]

def context(x):
    q=x['question'];cat=x['category']
    if cat=='object_existence' and x['question_type']=='binary':return (re.search(r'Does (.*?) exist in frame',q)[1],)
    if cat=='ego_relative_spatial':
        if x['question_type']=='binary':return tuple(re.search(r'nearest (.*?) at the (.*?) position relative',q).groups())
        return (re.search(r'nearest (.*?) relative to',q)[1],)
    if cat=='motion_action':return (re.search(r'does the nearest (.*?)\?',q)[1],)
    return ()
def feature(x,v):
    if x['category']=='temporal_grounding':
        a,b=map(int,re.findall(r'\d+',v));return b-a
    return v.lower().strip()
def prior_predict(items):
    kind=items[0]['question_type'];cnt=defaultdict(Counter);sc=defaultdict(Counter)
    for x in items:
        c=context(x);s=x['scene_id']
        if kind=='binary':cnt[c][x['answer']]+=1;sc[s,c][x['answer']]+=1
        else:
            for k,v in x['options'].items():
                t=(feature(x,v),k==x['correct_option']);cnt[c][t]+=1;sc[s,c][t]+=1
    pred=[]
    for x in items:
        c=context(x);h=cnt[c]-sc[x['scene_id'],c]
        if kind=='binary':p='Yes.' if h['Yes.']>h['No.'] else 'No.'
        else:p=max(x['options'],key=lambda k:(h[feature(x,x['options'][k]),True]+1)/(h[feature(x,x['options'][k]),False]+1))
        pred.append(p)
    return pred

def stats(rows):
    y=np.array([r['gold'] for r in rows]);p=np.array([r['pred'] for r in rows]);s=[r['scene_id'] for r in rows]
    out={'n':len(rows),'correct':int((y==p).sum()),'accuracy':float((y==p).mean()),'valid_rate':float((p!=INVALID).mean())}
    if all(r['question_type']=='mcq' for r in rows):
        out.update(a_count=int((p=='A').sum()),a_rate=float((p=='A').mean()),always_a=float((y=='A').mean()),rescue=int(((y!='A')&(p==y)).sum()),damage=int(((y=='A')&(p!='A')).sum()))
        out['excess_ci']=ci((p==y).astype(float)-(y=='A'),s)
    if all(r['question_type']=='binary' for r in rows):
        out.update(n_yes=int((y=='Yes.').sum()),n_no=int((y=='No.').sum()),tpr=float((p[y=='Yes.']=='Yes.').mean()) if np.any(y=='Yes.') else None,fpr=float((p[y=='No.']=='Yes.').mean()) if np.any(y=='No.') else None)
    return out

def dump_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    datasets={d.name:{cat:json.loads((d/f'{cat}.json').read_text()) for cat in CATS} for d in (a.data/'b4dl_dataset').iterdir() if d.is_dir()}
    base=datasets['hallucination_v2']; byrun={};summaries=[];pairs=[];binary=[];motions=[]
    for run in sorted((a.data/'b4dl_eval').iterdir()):
        if not run.is_dir():continue
        meta=run.name.endswith('_metatoken');model='Meta' if meta else 'NoMeta'
        cond=('Prompt' if 'no_position_bias' in run.name else 'TSCD 0.5' if 'alpha05' in run.name else 'TSCD 1.5' if 'alpha15' in run.name else 'TSCD 1' if 'temporal_shuffle_cd' in run.name else 'Base')
        variant='hallucination_v2'+('_no_position_bias' if cond=='Prompt' else '')+('_metatoken' if meta else '')
        allrows=[]
        for cat in CATS:
            rows=[json.loads(l) for l in (run/f'{cat}_predictions.jsonl').read_text().splitlines() if l.strip()]
            if len(rows)!=2000:raise ValueError((run,cat,len(rows)))
            for i,(r,q) in enumerate(zip(rows,datasets[variant][cat])):
                for key in ['question','prompt','options','scene_id']:
                    if r.get(key)!=q.get(key):raise ValueError((run,cat,i,key))
                if r['reference_answer']!=q['answer']:raise ValueError('reference mismatch')
                r['gold']=q['answer'];r['pred']=parse_answer(r['prediction'],r['question_type'],r.get('options'),r.get('error',''));r['source_index']=i
            allrows.extend(rows)
            summaries.append(dict(model=model,condition=cond,category=cat,**stats(rows)))
            if cond=='Base':
                for kind in ['binary','mcq']:
                    sub=[r for r in rows if r['question_type']==kind]
                    if sub:binary.append(dict(model=model,category=cat,kind=kind,**stats(sub)))
        byrun[model,cond]=allrows
        summaries.append(dict(model=model,condition=cond,category='Overall',**stats(allrows)))
        for kind in ['binary','mcq']:summaries.append(dict(model=model,condition=cond,category='All '+kind,**stats([r for r in allrows if r['question_type']==kind])))
    for (model,cond),rows in byrun.items():
        if cond=='Base':continue
        orig=byrun[model,'Base']; gold=np.array([r['gold'] for r in rows]);p=np.array([r['pred'] for r in rows]);p0=np.array([r['pred'] for r in orig]);s=[r['scene_id'] for r in rows];c=p==gold;c0=p0==gold
        pairs.append(dict(model=model,condition=cond,n=len(rows),repairs=int((c&~c0).sum()),regressions=int((~c&c0).sum()),wrong_to_wrong=int(((p!=p0)&~c&~c0).sum()),flip_rate=float((p!=p0).mean()),delta=float((c.astype(float)-c0).mean()),ci=ci(c.astype(float)-c0,s)))
    priors=[];priorrows=[]
    for cat in CATS:
        for kind in sorted({r['question_type'] for r in base[cat]}):
            sub=[r for r in base[cat] if r['question_type']==kind];p=prior_predict(sub)
            rows=[dict(r,gold=r['answer'],pred=v) for r,v in zip(sub,p)];stat=stats(rows)
            stat['ci']=ci([r['gold']==r['pred'] for r in rows],[r['scene_id'] for r in rows]);priors.append(dict(category=cat,kind=kind,**stat));priorrows.extend(rows)
    # Predicate-level analysis: all five predicates, both configurations, all ten runs.
    for (model,cond),allrows in byrun.items():
        rows=[r for r in allrows if r['category']=='motion_action']
        for action in sorted({r['action'] for r in base['motion_action']}):
            sub=[r for r,q in zip(rows,base['motion_action']) if q['action']==action]
            motions.append(dict(model=model,condition=cond,action=action,**stats(sub)))
    summary={'scope':'Rescoring archived outputs only; no new inference or logit measurements.','questions':sum(map(len,base.values())),'scenes':len({r['scene_id'] for rows in base.values() for r in rows}),'outputs':sum(map(len,byrun.values())),'temporal_all_A':all(r['pred']=='A' for rows in byrun.values() for r in rows if r['category']=='temporal_grounding'),'summary':summaries,'paired':pairs,'binary':binary,'priors':priors,'motion_predicates':motions}
    (a.output/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,rows in [('accuracy',summaries),('paired',pairs),('binary',binary),('priors',priors),('motion_predicates',motions)]:dump_csv(a.output/f'{name}.csv',rows)
    for r in pairs:print(r)
    print('\nMOTION BASELINES')
    for r in motions:
        if r['condition']=='Base':print(r)
    print('\nPRIORS')
    for r in priors:print(r['category'],r['kind'],r['accuracy'],r['ci'])
if __name__=='__main__':main()
