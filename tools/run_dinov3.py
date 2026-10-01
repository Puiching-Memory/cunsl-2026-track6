"""Compare reviewed DINOv3 token poolings in a single frozen encoder pass."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track6.backbones import POOLING, encoder_metadata, load_encoder, pool_tokens, preprocess
from track6.data import decode_image, load_task, sha256
from track6.experiments import save_retrieval
from track6.protocol import load_split

def run(args):
    import numpy as np
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; wrap invocation with shared GPU lock')
    store,records,split = load_split(args.source,args.split)
    by_id = {r.image_id:r for r in records}
    selected = [by_id[i] for i in split['validation_images']]
    if args.cache_train:
        selected += [by_id[i] for i in split['train_images']]
    queries = gallery = None
    if args.predict_test:
        queries,gallery,_ = load_task(store)
        selected += queries+gallery
    model,resource = load_encoder(args.resource,args.weights,args.image_size)
    model = model.eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    all_features = {method:[] for method in args.poolings}
    feature_start = time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(selected),args.batch_size):
            pixels = torch.from_numpy(np.stack([decode_image(store.read(r.image_path))
                    for r in selected[start:start+args.batch_size]]))
            x,mask = preprocess(pixels,model,args.image_size)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens = model.forward_features(x)
            for method in args.poolings:
                all_features[method].append(pool_tokens(tokens,model,mask,method).cpu().numpy())
            if start % (args.batch_size*20)==0:
                print(f'Embedded {min(start+args.batch_size,len(selected))}/{len(selected)}',flush=True)
    torch.cuda.synchronize()
    feature_seconds = time.monotonic()-feature_start
    reports = {}
    for method in args.poolings:
        metadata = encoder_metadata(model,resource,args.image_size,method)
        metadata.update(device='cuda',torch=torch.__version__,cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(),precision='BF16 forward; FP32 pooling and cosine',
                        batch_size=args.batch_size,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report = {'model':'dinov3', 'metadata':metadata,'trained':False,
                  'training_embeddings_cached':args.cache_train,'feature_seconds':feature_seconds,
                  'feature_seconds_scope':'shared encoder pass for all reported poolings; includes decoding',
                  'resource_manifest_sha256':sha256(args.resource),
                  'elapsed_seconds':time.monotonic()-started}
        reports[method] = save_retrieval(args.source,args.split,records,split,selected,
                      np.concatenate(all_features[method]),args.output_root/method,report,
                      queries=queries,gallery=gallery)
        print(f'{method}: weighted_score={reports[method]["metrics"]["min_relevant_k"]["weighted_score"]:.6f}',flush=True)
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--resource',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--poolings',nargs='+',choices=POOLING,default=['cls','avg','cls_foreground'])
    p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--cache-train',action='store_true')
    p.add_argument('--predict-test',action='store_true')
    args=p.parse_args()
    if args.batch_size<1 or len(args.poolings)!=len(set(args.poolings)):
        p.error('Positive batch size and distinct pooling settings required')
    print(json.dumps(run(args),indent=2))
