"""Explain held-out retrieval errors using public labels and observed pixels only.

All per-image details and optional original-pixel panels stay in the output
directory (normally ignored local/). Only the aggregate report is publishable.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from track6.data import decode_image, sha256, strict_json, write_json
from track6.protocol import load_split


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {'count': len(values), 'mean': float(values.mean()),
            'p10': float(np.quantile(values, .1)), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9))} if len(values) else {'count': 0}


def image_properties(pixels):
    maximum = pixels.max(axis=2)
    mask = maximum > 8
    ys, xs = np.where(mask)
    width = int(xs.max()-xs.min()+1) if len(xs) else 0
    height = int(ys.max()-ys.min()+1) if len(ys) else 0
    return {'nonzero_fraction': float((maximum>0).mean()),
            'threshold8_fraction': float(mask.mean()),
            'threshold8_width': width, 'threshold8_height': height,
            'threshold8_aspect': height/max(width,1),
            'mean_nonzero_intensity': float(maximum[maximum>0].mean()) if (maximum>0).any() else 0.}


def panel(store, by_id, cases, output):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (828, 80+310*len(cases)), '#e8eef3')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('arial.ttf', 18)
    small = ImageFont.truetype('arial.ttf', 15)
    draw.text((20,15), 'Real held-out images: query / true match / wrong top-1', fill='#172a3a', font=font)
    draw.text((20,43), 'Original 256x256 pixels. Identity details remain in the private manifest.', fill='#172a3a', font=small)
    for row, case in enumerate(cases):
        y = 80+row*310
        draw.text((20,y), f"Case {row+1}: {case['direction']}, true match rank {case['best_positive_rank']}",
                  fill='#172a3a',font=font)
        for col, (label,key,color) in enumerate([('Query','query','#244b70'),
                                                ('True match','positive','#167742'),
                                                ('Wrong top-1','top1','#a02424')]):
            x = 20+col*270
            draw.text((x,y+24),label,fill=color,font=small)
            r = by_id[case[key]]
            canvas.paste(Image.fromarray(decode_image(store.read(r.image_path))), (x,y+45))
            draw.rectangle((x-1,y+44,x+256,y+301),outline=color,width=2)
    canvas.save(output)


def analyze(source, split_path, run_dir, output_dir, make_panel=False):
    store, records, split = load_split(source, split_path)
    by_id = {r.image_id:r for r in records}
    original = strict_json((run_dir/'report.json').read_text(encoding='utf-8'))
    rankings = strict_json((run_dir/'validation-rankings.json').read_text(encoding='utf-8'))
    digest = sha256(run_dir/'features.npz')
    if original['split_sha256']!=sha256(split_path) or original['feature_cache_sha256']!=digest:
        raise ValueError('Features/report/split fingerprint mismatch')
    with np.load(run_dir/'features.npz', allow_pickle=False) as cache:
        ids, features = cache['ids'].tolist(), np.asarray(cache['embeddings'],dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if len(ids)!=len(set(ids)) or not np.isfinite(features).all() or (norms==0).any():
        raise ValueError('Invalid feature cache')
    features = features/norms
    lookup = {i:j for j,i in enumerate(ids)}
    # Never use test identities, filenames, guessed associations or test distributions.
    observed = {i:image_properties(decode_image(store.read(by_id[i].image_path)))
                for i in split['validation_images']}
    directions, private_rows, selected_cases = {}, {}, []
    for direction, task in split['tasks'].items():
        q, g = task['queries'], sorted(task['gallery'])
        similarities = features[[lookup[i] for i in q]] @ features[[lookup[i] for i in g]].T
        order = np.argsort(-similarities,axis=1,kind='stable')
        rows = []
        for index, query in enumerate(q):
            ranked = [g[j] for j in order[index]]
            if ranked[:10] != rankings[direction][query]:
                raise ValueError('CPU rankings differ from the stored experiment')
            positives = [rank+1 for rank,image in enumerate(ranked)
                         if by_id[image].ship_id==by_id[query].ship_id]
            best_rank = min(positives)
            positive = ranked[best_rank-1]
            negatives = [rank for rank,image in enumerate(ranked)
                         if by_id[image].ship_id!=by_id[query].ship_id]
            neg_rank = negatives[0]
            pscore = float(similarities[index,order[index,best_rank-1]])
            nscore = float(similarities[index,order[index,neg_rank]])
            qs, ps = observed[query], observed[positive]
            rows.append({'query':query,'positive':positive,'top1':ranked[0],
                         'direction':direction,'best_positive_rank':best_rank,
                         'positive_count':len(positives),'positive_similarity':pscore,
                         'hardest_negative_similarity':nscore,'identity_margin':pscore-nscore,
                         'query_properties':qs,'positive_properties':ps,
                         'foreground_area_log_ratio':float(abs(np.log((qs['threshold8_fraction']+1e-6)/
                                                                      (ps['threshold8_fraction']+1e-6))))})
        ranks = np.array([r['best_positive_rank'] for r in rows])
        counts = Counter(r['top1'] for r in rows)
        bins = []
        for low,high in [(0,.01),(.01,.03),(.03,.06),(.06,.12),(.12,.3),(.3,.9),(.9,1.01)]:
            subset = [r for r in rows if low<=r['query_properties']['threshold8_fraction']<high]
            bins.append({'threshold8_fraction_interval':[low,min(high,1.)], 'queries':len(subset),
                         'R@1':float(np.mean([r['best_positive_rank']==1 for r in subset])) if subset else None,
                         'R@10':float(np.mean([r['best_positive_rank']<=10 for r in subset])) if subset else None})
        for category, predicate in [('top1_correct',lambda r:r['best_positive_rank']==1),
                                    ('rank2_to10',lambda r:1<r['best_positive_rank']<=10),
                                    ('beyond10',lambda r:r['best_positive_rank']>10)]:
            subset = [r for r in rows if predicate(r)]
            # Physical ship length cannot be recovered from these threshold masks.
            directions.setdefault(direction,{})[category] = {
                'queries':len(subset),
                'query_threshold8_fraction':describe([r['query_properties']['threshold8_fraction'] for r in subset]),
                'foreground_area_log_ratio':describe([r['foreground_area_log_ratio'] for r in subset]),
                'positive_similarity':describe([r['positive_similarity'] for r in subset]),
                'identity_margin':describe([r['identity_margin'] for r in subset])}
        directions[direction].update(queries=len(q),gallery=len(g),
            recall={f'R@{k}':float((ranks<=k).mean()) for k in [1,5,10,50]},
            best_positive_rank=describe(ranks),query_foreground_bins=bins,
            unique_top1_gallery=len(counts),max_queries_selecting_same_top1=max(counts.values()),
            top10_hub_query_fraction=sum(sorted(counts.values(),reverse=True)[:10])/len(q),
            gallery_foreground_fraction=describe([observed[i]['threshold8_fraction'] for i in g]),
            gallery_threshold8_width=describe([observed[i]['threshold8_width'] for i in g]),
            gallery_threshold8_height=describe([observed[i]['threshold8_height'] for i in g]))
        private_rows[direction] = rows
        if direction!='O2O':
            for condition in [lambda r:1<r['best_positive_rank']<=10,lambda r:r['best_positive_rank']>50]:
                candidates = [r for r in rows if condition(r)]
                if candidates:
                    # Illustrative examples, deliberately selected; not a random sample.
                    selected_cases.append(min(candidates,key=lambda r:(r['query_properties']['threshold8_fraction'],r['query'])))
    output_dir.mkdir(parents=True,exist_ok=True)
    report = {'method':'heldout-full-gallery-error-analysis-v1','seed':split['seed'],
              'source_sha256':split['source_sha256'],'split_sha256':sha256(split_path),
              'model_report_sha256':sha256(run_dir/'report.json'),'feature_cache_sha256':digest,
              'analysis_tool_sha256':sha256(__file__),
              'validation_rankings_reproduced':True,'test_used':False,
              'threshold8_mask_is_ship_segmentation':False,
              'pixel_size_is_physical_size':False,'observational_only':True,
              'directions':directions}
    write_json(output_dir/'report.json',report)
    write_json(output_dir/'private-errors.json',private_rows)
    write_json(output_dir/'private-panel-cases.json',selected_cases)
    if make_panel:
        panel(store,by_id,selected_cases,output_dir/'real-errors.png')
    return report


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--panel',action='store_true')
    args=p.parse_args()
    report=analyze(args.source,args.split,args.run_dir,args.output_dir,args.panel)
    print(json.dumps({k:{'recall':v['recall'],'rank_groups':{n:v[n]['queries'] for n in
                       ['top1_correct','rank2_to10','beyond10']},'foreground_bins':v['query_foreground_bins']}
                      for k,v in report['directions'].items()},indent=2))
