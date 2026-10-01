"""Measure fixed-shortlist oracle bounds on held-out identities; no inference."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from track6.data import sha256,strict_json,write_json
from track6.diagnostics import shortlist_bounds
from track6.protocol import load_split


def diagnose(source,split_path,run_dir,output,cutoffs=(10,50,100)):
    _,records,split=load_split(source,split_path)
    by_id={r.image_id:r for r in records}
    run=Path(run_dir)
    original=strict_json((run/'report.json').read_text(encoding='utf-8'))
    rankings=strict_json((run/'validation-rankings.json').read_text(encoding='utf-8'))
    cache_sha=sha256(run/'features.npz')
    if (original['split_sha256']!=sha256(split_path) or original['source_sha256']!=split['source_sha256']
        or original['feature_cache_sha256']!=cache_sha):
        raise ValueError('Feature/report/data/split fingerprint mismatch')
    if set(rankings)!=set(split['tasks']):
        raise ValueError('Stored direction coverage differs')
    with np.load(run/'features.npz',allow_pickle=False) as cache:
        ids=cache['ids'].tolist(); features=np.asarray(cache['embeddings'],dtype=np.float32)
    if features.ndim!=2 or features.shape[0]!=len(ids) or len(ids)!=len(set(ids)) or not np.isfinite(features).all():
        raise ValueError('Unique IDs and finite matching feature rows required')
    norms=np.linalg.norm(features,axis=1,keepdims=True)
    if not np.isfinite(norms).all() or (norms==0).any():
        raise ValueError('Finite nonzero feature norms required')
    features=features/norms; lookup={i:j for j,i in enumerate(ids)}
    result={mode:{} for mode in ['min_relevant_k','all_relevant']}
    for direction,task in split['tasks'].items():
        q,g=task['queries'],sorted(task['gallery'])
        if set(rankings[direction])!=set(q):
            raise ValueError('Stored query coverage differs')
        similarities=features[[lookup[i] for i in q]]@features[[lookup[i] for i in g]].T
        order=np.argsort(-similarities,axis=1,kind='stable')
        for row,query in zip(order,q):
            if [g[i] for i in row[:10]]!=rankings[direction][query]:
                raise ValueError('Full-gallery recomputation does not reproduce stored top ten')
        identities=np.asarray([by_id[i].ship_id for i in g])
        relevance=identities[order]==np.asarray([by_id[i].ship_id for i in q])[:,None]
        for mode in result:
            row=shortlist_bounds(relevance,cutoffs,mode)
            recorded=original['metrics'][mode]['directions'][direction]
            if any(not math.isclose(row['actual'][k],recorded[k],abs_tol=1e-12,rel_tol=0) for k in ['R@1','mAP@10','score']):
                raise ValueError('Independent actual metric recomputation differs')
            result[mode][direction]=row
    weights={'O2S':.45,'S2O':.45,'O2O':.1}
    report={'method':'fixed-shortlist-label-oracle-v1','diagnostic_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'feature_model_code_commit':original['code_commit'],'seed':split['seed'],
            'source_sha256':split['source_sha256'],'split_sha256':sha256(split_path),
            'model_report_sha256':sha256(run/'report.json'),'feature_cache_sha256':cache_sha,
            'diagnostic_tool_sha256':sha256(__file__),'bound_function_sha256':sha256(Path(__file__).resolve().parents[1]/'track6/diagnostics.py'),
            'test_used':False,'validation_rankings_reproduced':True,'actual_metrics_recomputed':True,
            'oracle_uses_heldout_labels':True,'oracle_is_model_result':False,'official_scorer_confirmed':False,
            'scope':'Upper bound for ordering inside a fixed shortlist only; not an information limit or evidence an image-only reranker can realize it.',
            'directions':result,
            'weighted_actual':{mode:math.fsum(result[mode][d]['actual']['score']*w for d,w in weights.items()) for mode in result},
            'weighted_oracle':{mode:{str(k):math.fsum(result[mode][d]['shortlists'][str(k)]['oracle_direction_score']*w for d,w in weights.items()) for k in sorted(cutoffs)} for mode in result}}
    write_json(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cutoffs',type=int,nargs='+',default=[10,50,100])
    args=p.parse_args()
    result=diagnose(args.source,args.split,args.run_dir,args.output,args.cutoffs)
    print(json.dumps({'weighted_actual':result['weighted_actual']['min_relevant_k'],
        'label_oracle_is_not_a_model_result':True,'weighted_oracle':result['weighted_oracle']['min_relevant_k'],
        'candidate_recall':{d:{k:v['candidate_recall'] for k,v in row['shortlists'].items()} for d,row in result['directions']['min_relevant_k'].items()}},indent=2))
