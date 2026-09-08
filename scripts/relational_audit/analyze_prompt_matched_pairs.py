"""Compile disjoint cross-scene binary pairs from archived questions, before reading predictions.
Pairing is exact in category, full input prompt, and observed feature sequence length.
References must differ. Matching is deterministic and does not select on model outcomes.
These are natural cross-scene contrasts, not surgically edited counterfactuals.
"""
from __future__ import annotations
import argparse, hashlib, json, csv, sys
from collections import defaultdict
from pathlib import Path
from analyze_rewrite import parse_answer, CATS

def build_pairs(data:Path,meta:bool,lengths:dict):
    variant='hallucination_v2'+('_metatoken' if meta else '')
    groups=defaultdict(lambda: {'Yes.':[], 'No.':[]})
    for cat in CATS:
        for index,q in enumerate(json.loads((data/'b4dl_dataset'/variant/f'{cat}.json').read_text())):
            if q['question_type']!='binary':continue
            key=(cat,q['prompt'],lengths[q['scene_id']])
            groups[key][q['answer']].append({'index':index,'scene_id':q['scene_id'],'gold':q['answer'],'question':q['question']})
    pairs=[]
    for (cat,prompt,length),g in sorted(groups.items()):
        ys=sorted(g['Yes.'],key=lambda x:(x['scene_id'],x['index']))
        ns=sorted(g['No.'],key=lambda x:(x['scene_id'],x['index']))
        # Deterministic maximum bipartite matching. Same-scene pairs are forbidden.
        match={}
        def augment(yi,seen):
            for ni,n in enumerate(ns):
                if ni in seen or n['scene_id']==ys[yi]['scene_id']:continue
                seen.add(ni)
                if ni not in match or augment(match[ni],seen):
                    match[ni]=yi;return True
            return False
        for yi in range(len(ys)):augment(yi,set())
        for ni,yi in sorted(match.items()):
            pairs.append({'category':cat,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'sequence_length':length,'positive':ys[yi],'negative':ns[ni]})
    used=set()
    for i,p in enumerate(pairs):
        p['pair_id']=i
        for key in ['positive','negative']:
            k=(p['category'],p[key]['index'])
            if k in used:raise ValueError('reused question')
            used.add(k)
        if p['positive']['scene_id']==p['negative']['scene_id']:raise ValueError('same scene')
    return pairs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lengths={}
    for c in CATS:
        for line in (a.data/'b4dl_eval/hallucination_v2_temporal_shuffle_cd'/f'{c}_predictions.jsonl').read_text().splitlines():
            r=json.loads(line);perm=r['shuffle_permutation'];T=len(perm.split()) if isinstance(perm,str) else len(perm)
            if r['scene_id'] in lengths and lengths[r['scene_id']]!=T:raise ValueError('inconsistent length')
            lengths[r['scene_id']]=T
    allresults=[];summary={}
    for model,meta in [('NoMeta',False),('Meta',True)]:
        pairs=build_pairs(a.data,meta,lengths)
        (a.output/f'prompt_matched_pairs_{model}.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
        summary[model]={'pairs':len(pairs),'questions':2*len(pairs),'scenes':len({p[k]['scene_id'] for p in pairs for k in ['positive','negative']}),'full_prompts':len({p['prompt'] for p in pairs}),'per_category':{c:sum(p['category']==c for p in pairs) for c in CATS}}
        for cond,suffix in [('Base',''),('Prompt','_no_position_bias'),('TSCD 0.5','_temporal_shuffle_cd_alpha05'),('TSCD 1','_temporal_shuffle_cd'),('TSCD 1.5','_temporal_shuffle_cd_alpha15')]:
            folder='hallucination_v2'+suffix+('_metatoken' if meta else '')
            pred={c:[json.loads(l) for l in (a.data/'b4dl_eval'/folder/f'{c}_predictions.jsonl').read_text().splitlines()] for c in CATS}
            for cat in ['All','object_existence','ego_relative_spatial','motion_action']:
                ps=pairs if cat=='All' else [p for p in pairs if p['category']==cat]
                both=one=neither=invalid=same=0
                for p in ps:
                    c=p['category'];r0=pred[c][p['positive']['index']];r1=pred[c][p['negative']['index']]
                    if r0['prompt']!=r1['prompt'] or r0['prompt']!=p['prompt']:raise ValueError('prompt mismatch')
                    if (r0['reference_answer'],r1['reference_answer'])!=('Yes.','No.'):raise ValueError('gold mismatch')
                    p0=parse_answer(r0['prediction'],'binary',error=r0.get('error',''));p1=parse_answer(r1['prediction'],'binary',error=r1.get('error',''))
                    cs=(p0=='Yes.')+(p1=='No.');both+=cs==2;one+=cs==1;neither+=cs==0;invalid+=p0=='<invalid>' or p1=='<invalid>';same+=p0==p1
                allresults.append({'model':model,'condition':cond,'category':cat,'n_pairs':len(ps),'both_correct':both,'one_correct':one,'neither_correct':neither,'pair_correct':both/len(ps) if ps else None,'item_accuracy':(2*both+one)/(2*len(ps)) if ps else None,'same_answer':same,'invalid_pairs':invalid})
    (a.output/'prompt_matched_summary.json').write_text(json.dumps({'construction':{'key':'category + exact full prompt + feature sequence length','matching':'maximum disjoint opposite-label bipartite matching, sorted by scene and index','prediction_independent':True,'cross_scene':True,'scope':'Natural cross-scene contrasts from existing benchmark. Not a new independently sampled test set or isolated causal intervention.'},'subsets':summary,'results':allresults},indent=2)+'\n')
    with (a.output/'prompt_matched_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allresults[0]));w.writeheader();w.writerows(allresults)
    print(json.dumps(summary,indent=2))
    for d in allresults:
        if d['category']=='All' or d['condition']=='Base':print(d)
if __name__=='__main__':main()
