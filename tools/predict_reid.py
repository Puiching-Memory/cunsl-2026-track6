"""Replay a trained ReID model with its exact public initialization and split."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track6.backbones import encoder_metadata, load_encoder
from track6.data import sha256
from track6.experiments import save_retrieval
from track6.protocol import load_split
from tools.train_reid import setup,extract,selection

def predict(args):
    import torch
    from track6.learning import SensorReID, restore_trainable_state
    started = time.monotonic()
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    setup(checkpoint['seed'])
    store,records,split = load_split(args.source,args.split)
    if checkpoint['source_sha256']!=split['source_sha256'] or checkpoint['split_sha256']!=sha256(args.split):
        raise ValueError('Checkpoint data and identity split must match exactly')
    encoder,resource = load_encoder(args.resource,args.weights,checkpoint['config']['image_size'])
    if checkpoint['weights_sha256']!=resource['weights']['sha256'] or checkpoint['variant']!=resource['variant']:
        raise ValueError('Checkpoint public backbone initialization differs')
    cfg = checkpoint['config']
    model = SensorReID(encoder,cfg['identities'],cfg['image_size'],cfg['pooling'],cfg['train_last_blocks'],
                       cfg.get('stem_mode','frozen'),cfg.get('structure_layer',6))
    restore_trainable_state(model,checkpoint['trainable_state'])
    model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    selected,queries,gallery = selection(store,records,split,args.predict_test)
    embeddings = extract(model,store,selected,cfg['inference_batch_size'])
    metadata = encoder_metadata(encoder,resource,cfg['image_size'],cfg['pooling'])
    metadata.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return save_retrieval(args.source,args.split,records,split,selected,embeddings,args.output_dir,
             {'model':'saved-dinov3-reid-replay','seed':checkpoint['seed'],'metadata':metadata,
              'trained':False,'checkpoint_sha256':sha256(args.checkpoint),'elapsed_seconds':time.monotonic()-started},
             queries=queries,gallery=gallery)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--resource',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--predict-test',action='store_true')
    print(json.dumps(predict(p.parse_args()),indent=2))
