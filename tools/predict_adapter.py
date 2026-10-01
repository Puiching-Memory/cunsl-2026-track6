"""Replay a saved adapter checkpoint using its exact audited feature cache."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track6.data import DIRECTIONS, load_task, sha256, write_json
from track6.metrics import AP_MODES, cosine_top10, evaluate
from track6.protocol import load_split
from track6.submission import validate_submission

def predict(args):
    import numpy as np
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.nn.functional as F
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    store, records, split = load_split(args.source,args.split)
    checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    cache_path = args.features_dir / "features.npz"
    report = json.loads((args.features_dir / "report.json").read_text(encoding="utf-8"))
    if checkpoint["source_sha256"] != split["source_sha256"] or checkpoint["split_sha256"] != sha256(args.split):
        raise ValueError("Checkpoint belongs to different data or validation split")
    if checkpoint["feature_cache_sha256"] != sha256(cache_path) or report["feature_cache_sha256"] != sha256(cache_path):
        raise ValueError("Checkpoint requires its exact original feature cache")
    if checkpoint["feature_weights_sha256"] != report["metadata"]["weights_sha256"]:
        raise ValueError("Checkpoint and backbone weights differ")
    with np.load(cache_path,allow_pickle=False) as cache:
        ids,raw = cache["ids"].tolist(),cache["embeddings"].astype(np.float32)
    if len(set(ids)) != len(ids) or not np.isfinite(raw).all() or raw.shape != (len(ids),checkpoint["feature_dimension"]):
        raise ValueError("Malformed feature cache")
    index = {i:j for j,i in enumerate(ids)}
    by_id = {r.image_id:r for r in records}
    queries,gallery,_ = load_task(store)
    selected = [by_id[i] for i in split["validation_images"]] + queries + gallery
    if not {r.image_id for r in selected} <= set(index):
        raise ValueError("Cache is missing validation or test embeddings")
    dim = checkpoint["feature_dimension"]
    heads = torch.nn.ModuleDict({m:torch.nn.Linear(dim,dim,bias=False) for m in ["optical","sar"]})
    heads.load_state_dict(checkpoint["state_dict"],strict=True)
    heads.eval().cuda()
    features = F.normalize(torch.from_numpy(raw).cuda(),dim=1)
    adapted = {}
    with torch.inference_mode():
        for modality in ["optical","sar"]:
            batch_ids = [r.image_id for r in selected if r.modality == modality]
            output = F.normalize(heads[modality](features[[index[i] for i in batch_ids]]),dim=1).cpu().numpy()
            adapted.update(zip(batch_ids,output))
    rankings = {}
    for direction,task in split["tasks"].items():
        q,g = task["queries"],task["gallery"]
        rankings[direction] = cosine_top10([adapted[i] for i in q],[adapted[i] for i in g],q,g)
    prediction = {}
    for direction,(_,modality) in DIRECTIONS.items():
        q = [r.image_id for r in queries if r.query_type == direction]
        g = [r.image_id for r in gallery if r.modality == modality]
        if q:
            prediction.update(cosine_top10([adapted[i] for i in q],[adapted[i] for i in g],q,g))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir / "prediction.json",prediction)
    write_json(args.output_dir / "validation-rankings.json",rankings)
    result = {"state":"completed","checkpoint_sha256":sha256(args.checkpoint),
              "feature_cache_sha256":sha256(cache_path),"split_sha256":sha256(args.split),
              "metrics":{mode:evaluate(rankings,split,records,denominator=mode) for mode in AP_MODES},
              "submission_validation":validate_submission(args.source,args.output_dir / "prediction.json"),
              "prediction_sha256":sha256(args.output_dir / "prediction.json"),
              "validation_rankings_sha256":sha256(args.output_dir / "validation-rankings.json"),
              "elapsed_seconds":time.monotonic()-started,"trained":False,"leaderboard_submitted":False}
    write_json(args.output_dir / "report.json",result)
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--split",type=Path,required=True)
    p.add_argument("--features-dir",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    print(json.dumps(predict(p.parse_args()),indent=2))
