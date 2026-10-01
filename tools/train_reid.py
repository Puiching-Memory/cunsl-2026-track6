"""Fine-tune reviewed DINOv3 layers on training identities only, then rank images."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track6.backbones import POOLING, encoder_metadata, load_encoder
from track6.data import decode_image, load_task, sha256
from track6.experiments import save_retrieval
from track6.protocol import load_split

def setup(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; wrap invocation with shared GPU lock')

def extract(model,store,selected,batch_size):
    import numpy as np
    import torch
    model.eval()
    features = []
    with torch.inference_mode():
        for start in range(0,len(selected),batch_size):
            batch = selected[start:start+batch_size]
            pixels = torch.from_numpy(np.stack([decode_image(store.read(r.image_path)) for r in batch]))
            modalities = torch.tensor([r.modality=='sar' for r in batch],device='cuda',dtype=torch.long)
            features.append(model.encode(pixels,modalities).cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(features)

def selection(store,records,split,predict_test):
    by_id = {r.image_id:r for r in records}
    selected = [by_id[i] for i in split['validation_images']]
    queries = gallery = None
    if predict_test:
        queries,gallery,_ = load_task(store)
        selected += queries+gallery
    return selected,queries,gallery

def train(args):
    setup(args.seed)
    import numpy as np
    import torch
    from track6.learning import SensorReID, retrieval_losses, structural_pair_loss, trainable_state
    started = time.monotonic()
    store,records,split = load_split(args.source,args.split)
    by_id = {r.image_id:r for r in records}
    grouped = defaultdict(lambda:defaultdict(list))
    for i in split['train_images']:
        r = by_id[i]
        grouped[r.ship_id][r.modality].append(r)
    paired = sorted(i for i,g in grouped.items() if g['optical'] and g['sar'])
    encoder,resource = load_encoder(args.resource,args.weights,args.image_size)
    model = SensorReID(encoder,len(paired),args.image_size,args.pooling,args.train_last_blocks,
                       args.stem_mode,args.structure_layer).cuda()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(args.seed)
    # Cache only allowed training pixels. Validation/test pixels are decoded after fitting.
    preload_start = time.monotonic()
    train_pixels = {r.image_id:decode_image(store.read(r.image_path))
                    for identity in paired for entries in grouped[identity].values() for r in entries}
    preload_seconds = time.monotonic()-preload_start
    backbone_parameters = [p for name,p in model.named_parameters()
                           if p.requires_grad and not name.startswith(('heads.','classifier.'))]
    head_parameters = list(model.heads.parameters())+list(model.classifier.parameters())
    groups = [{'params':head_parameters,'lr':args.head_lr,'base_lr':args.head_lr}]
    if backbone_parameters:
        groups.append({'params':backbone_parameters,'lr':args.backbone_lr,'base_lr':args.backbone_lr})
    optimizer = torch.optim.AdamW(groups,weight_decay=.01)
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    history = []
    training_start = time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train()
        factor = min(epoch/2,1.) * (.1+.9*(1+math.cos(math.pi*(epoch-1)/max(1,args.epochs-1)))/2)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*factor
        losses = []
        components = defaultdict(list)
        epoch_start = time.monotonic()
        order = rng.permutation(len(paired))
        for start in range(0,len(order),args.identities_per_batch):
            indices = order[start:start+args.identities_per_batch]
            if len(indices)<2:
                continue
            batch_ids = [paired[int(i)] for i in indices]
            optical = [rng.choice(grouped[i]['optical']) for i in batch_ids]
            sar = [rng.choice(grouped[i]['sar']) for i in batch_ids]
            pixels = torch.from_numpy(np.stack([train_pixels[r.image_id] for r in optical+sar]))
            labels = torch.tensor(np.concatenate([indices,indices]),device='cuda',dtype=torch.long)
            modalities = torch.cat([torch.zeros(len(indices),device='cuda',dtype=torch.long),
                                    torch.ones(len(indices),device='cuda',dtype=torch.long)])
            structure = None
            if args.structure_weight:
                embeddings,structure = model.encode(pixels,modalities,augment=args.augment,return_structure=True)
            else:
                embeddings = model.encode(pixels,modalities,augment=args.augment)
            loss,parts = retrieval_losses(model,embeddings,labels,modalities,args.temperature,
                                          args.classification_weight,args.triplet_weight)
            if args.structure_weight:
                if not structure.requires_grad:
                    raise ValueError('Structural constraint has no trainable path at this layer')
                parts['structure'] = structural_pair_loss(structure,labels,modalities)
                loss = loss+args.structure_weight*parts['structure']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite image-level ReID loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained_parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            for key,value in parts.items():
                components[key].append(float(value.detach()))
        torch.cuda.synchronize()
        entry = {'epoch':epoch,'mean_loss':float(np.mean(losses)),'steps':len(losses),
                 'components':{k:float(np.mean(v)) for k,v in components.items()},
                 'seconds':time.monotonic()-epoch_start,
                 'learning_rates':[g['lr'] for g in optimizer.param_groups]}
        history.append(entry)
        print(f'Epoch {epoch}/{args.epochs}: loss={entry["mean_loss"]:.5f}, seconds={entry["seconds"]:.1f}',flush=True)
    training_seconds = time.monotonic()-training_start
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path = args.output_dir/'reid.pt'
    config = {'image_size':args.image_size,'pooling':args.pooling,'train_last_blocks':args.train_last_blocks,
              'identities':len(paired),'inference_batch_size':args.inference_batch_size,
              'stem_mode':args.stem_mode,'structure_layer':args.structure_layer}
    torch.save({'trainable_state':trainable_state(model),'config':config,'seed':args.seed,
                'split_sha256':sha256(args.split),'source_sha256':split['source_sha256'],
                'weights_sha256':resource['weights']['sha256'],'variant':resource['variant']},checkpoint_path)
    selected,queries,gallery = selection(store,records,split,args.predict_test)
    inference_start = time.monotonic()
    embeddings = extract(model,store,selected,args.inference_batch_size)
    inference_seconds = time.monotonic()-inference_start
    metadata = encoder_metadata(encoder,resource,args.image_size,args.pooling)
    metadata.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    precision='BF16 encoder; FP32 embeddings, heads, losses and cosine')
    report = {'model':'image-level-dinov3-cross-modal-reid','seed':args.seed,'metadata':metadata,
              'trained':True,'checkpoint_sha256':sha256(checkpoint_path),'resource_manifest_sha256':sha256(args.resource),
              'training':{'epochs':args.epochs,'identities_per_batch':args.identities_per_batch,
                  'paired_training_identities':len(paired),'paired_training_images_available':len(train_pixels),
                  'training_images_available':len(split['train_images']),'train_last_blocks':args.train_last_blocks,
                  'stem_mode':args.stem_mode,'structure_weight':args.structure_weight,
                  'structure_layer':args.structure_layer,
                  'structure_loss':'mean squared difference of channel-standardized adjacent patch gradient energies' if args.structure_weight else 'disabled',
                  'optimizer':'AdamW','weight_decay':.01,'backbone_lr':args.backbone_lr,'head_lr':args.head_lr,
                  'schedule':'2 epoch warmup times cosine decay, floor .1','temperature':args.temperature,
                  'classification_weight':args.classification_weight,'triplet_weight':args.triplet_weight,
                  'loss':'symmetric paired InfoNCE + cosine identity CE (smoothing .1) + soft batch-hard triplet (margin .2)',
                  'augmentation':'independent H/V flips without cropping' if args.augment else 'none',
                  'trainable_parameters':sum(p.numel() for p in trained_parameters),
                  'selection':'fixed final epoch; no validation checkpoint selection',
                  'validation_used_for_training':False,'test_used_for_training':False,'history':history},
              'data_preload_seconds':preload_seconds,'training_seconds':training_seconds,
              'inference_seconds':inference_seconds,'elapsed_seconds':time.monotonic()-started}
    report = save_retrieval(args.source,args.split,records,split,selected,embeddings,args.output_dir,
                            report,queries=queries,gallery=gallery)
    print(f'weighted_score={report["metrics"]["min_relevant_k"]["weighted_score"]:.6f}',flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--resource',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261001)
    p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--pooling',choices=POOLING,default='cls_foreground')
    p.add_argument('--train-last-blocks',type=int,default=4)
    p.add_argument('--stem-mode',choices=['frozen','shared','separate'],default='frozen')
    p.add_argument('--structure-weight',type=float,default=0.)
    p.add_argument('--structure-layer',type=int,default=6)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--identities-per-batch',type=int,default=32)
    p.add_argument('--inference-batch-size',type=int,default=32)
    p.add_argument('--backbone-lr',type=float,default=2e-5)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--temperature',type=float,default=.07)
    p.add_argument('--classification-weight',type=float,default=.5)
    p.add_argument('--triplet-weight',type=float,default=.2)
    p.add_argument('--augment',action='store_true')
    p.add_argument('--predict-test',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.identities_per_batch-1,args.inference_batch_size,args.backbone_lr,args.head_lr,args.temperature)<=0:
        p.error('Positive training settings and at least two identities per batch required')
    if min(args.classification_weight,args.triplet_weight,args.structure_weight)<0:
        p.error('Loss weights cannot be negative')
    if args.structure_weight and args.stem_mode=='frozen':
        p.error('A trainable patch stem is required for the structural ablation')
    print(json.dumps(train(args),indent=2))
