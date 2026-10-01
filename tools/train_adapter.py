"""Train two-modality contrastive adapters on frozen, audited DINOv2/v3 features."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track6.data import DIRECTIONS, load_task, sha256, write_json
from track6.metrics import AP_MODES, cosine_top10, evaluate
from track6.protocol import load_split
from track6.submission import validate_submission

def train(args):
    import numpy as np
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.nn.functional as F
    started = time.monotonic()
    store, records, split = load_split(args.source, args.split)
    base_report = json.loads((args.features_dir / "report.json").read_text(encoding="utf-8"))
    if base_report["model"] not in {"dinov2_vits14","dinov3"} or base_report["split_sha256"] != sha256(args.split):
        raise ValueError("Features must come from the reviewed DINO baseline on this exact split")
    if base_report["source_sha256"] != split["source_sha256"]:
        raise ValueError("Feature archive fingerprint mismatch")
    cache_path = args.features_dir / "features.npz"
    if base_report["feature_cache_sha256"] != sha256(cache_path):
        raise ValueError("Feature cache checksum differs from the baseline report")
    with np.load(cache_path, allow_pickle=False) as cache:
        ids, raw = cache["ids"].tolist(), cache["embeddings"].astype(np.float32)
    if len(set(ids)) != len(ids) or not np.isfinite(raw).all():
        raise ValueError("Duplicate feature IDs or nonfinite features")
    index = {i:j for j,i in enumerate(ids)}
    required = set(split["train_images"]) | set(split["validation_images"])
    if not required <= set(index):
        raise ValueError("Cache must contain all usable training and validation images")
    by_id = {r.image_id:r for r in records}
    grouped = defaultdict(lambda:defaultdict(list))
    for image_id in split["train_images"]:
        r = by_id[image_id]
        grouped[r.ship_id][r.modality].append(index[image_id])
    paired = sorted(s for s,v in grouped.items() if v["optical"] and v["sar"])
    if len(paired) < 2:
        raise ValueError("Training requires at least two paired identities")
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    features = torch.from_numpy(raw).cuda()
    features = F.normalize(features, dim=1)
    dim = features.shape[1]
    heads = torch.nn.ModuleDict({m:torch.nn.Linear(dim,dim,bias=False) for m in ["optical","sar"]}).cuda()
    identity = torch.eye(dim,device="cuda")
    with torch.no_grad():
        for head in heads.values():
            head.weight.copy_(identity)
    optimizer = torch.optim.AdamW(heads.parameters(),lr=args.lr,weight_decay=1e-4)
    history = []
    train_start = time.monotonic()
    for epoch in range(1,args.epochs+1):
        heads.train()
        losses = []
        order = rng.permutation(len(paired))
        for start in range(0,len(order),args.identities_per_batch):
            batch_ids = [paired[i] for i in order[start:start+args.identities_per_batch]]
            if len(batch_ids) < 2:
                continue
            optical = [int(rng.choice(grouped[s]["optical"])) for s in batch_ids]
            sar = [int(rng.choice(grouped[s]["sar"])) for s in batch_ids]
            a = F.normalize(heads["optical"](features[optical]),dim=1)
            b = F.normalize(heads["sar"](features[sar]),dim=1)
            logits = a @ b.T / args.temperature
            target = torch.arange(len(batch_ids),device="cuda")
            contrastive = (F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
            regularization = sum((head.weight-identity).square().mean() for head in heads.values())
            loss = contrastive + args.identity_regularization * regularization
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in heads.parameters()):
                raise RuntimeError("Nonfinite adapter gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch":epoch,"mean_loss":mean_loss,"steps":len(losses)})
        print(f"Epoch {epoch}/{args.epochs}: loss={mean_loss:.6f}",flush=True)
    torch.cuda.synchronize()
    training_seconds = time.monotonic()-train_start
    heads.eval()
    selected = [by_id[i] for i in split["validation_images"]]
    if args.predict_test:
        queries,gallery,_ = load_task(store)
        selected += queries+gallery
        if not {r.image_id for r in selected} <= set(index):
            raise ValueError("Test embeddings missing from cache")
    adapted = {}
    inference_start = time.monotonic()
    with torch.inference_mode():
        for modality in ["optical","sar"]:
            batch_ids = [r.image_id for r in selected if r.modality==modality]
            output = F.normalize(heads[modality](features[[index[i] for i in batch_ids]]),dim=1).cpu().numpy()
            adapted.update(zip(batch_ids,output))
    rankings = {}
    for direction, task in split["tasks"].items():
        q,g = task["queries"],task["gallery"]
        rankings[direction] = cosine_top10([adapted[i] for i in q],[adapted[i] for i in g],q,g)
    results = {mode:evaluate(rankings,split,records,denominator=mode) for mode in AP_MODES}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint = args.output_dir / "adapter.pt"
    torch.save({"state_dict":heads.cpu().state_dict(),"feature_dimension":dim,
                "split_sha256":sha256(args.split),"seed":args.seed,"epochs":args.epochs,
                "feature_weights_sha256":base_report["metadata"]["weights_sha256"],
                "feature_cache_sha256":sha256(cache_path),"source_sha256":split["source_sha256"]},checkpoint)
    write_json(args.output_dir / "validation-rankings.json",rankings)
    test_validation = None
    if args.predict_test:
        prediction = {}
        for direction,(_,modality) in DIRECTIONS.items():
            q = [r.image_id for r in queries if r.query_type==direction]
            g = [r.image_id for r in gallery if r.modality==modality]
            if q:
                prediction.update(cosine_top10([adapted[i] for i in q],[adapted[i] for i in g],q,g))
        write_json(args.output_dir / "prediction.json",prediction)
        test_validation = validate_submission(args.source,args.output_dir / "prediction.json")
    np.savez_compressed(args.output_dir / "features.npz",ids=np.asarray([r.image_id for r in selected]),
                        embeddings=np.asarray([adapted[r.image_id] for r in selected],dtype=np.float32))
    report = {"model":f"frozen-{base_report['model']}-two-linear-contrastive-adapters", "state":"completed",
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "seed":args.seed,"split_sha256":sha256(args.split),"source_sha256":split["source_sha256"],
        "input_feature_cache_sha256":sha256(cache_path),"feature_report_sha256":sha256(args.features_dir / "report.json"),
        "feature_cache_sha256":sha256(args.output_dir / "features.npz"),
        "backbone_metadata":base_report["metadata"], "feature_dimension":dim,
        "feature_weights_sha256":base_report["metadata"]["weights_sha256"],
        "training":{"epochs":args.epochs,"identities_per_batch":args.identities_per_batch,
                    "paired_training_identities":len(paired),"training_images_available":len(split["train_images"]),
                    "optimizer":"AdamW","lr":args.lr,"weight_decay":1e-4,"temperature":args.temperature,
                    "identity_regularization":args.identity_regularization,"precision":"float32",
                    "initialization":"identity matrices, no bias","loss":"symmetric optical/SAR paired InfoNCE",
                    "selection":"fixed final epoch; no validation-based checkpoint selection",
                    "backbone_frozen":True,"test_used_for_training":False,"history":history},
        "metrics":results,"adapter_sha256":sha256(checkpoint),"adapter_parameters":2*dim*dim,
        "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "training_seconds":training_seconds,"inference_and_scoring_seconds":time.monotonic()-inference_start,
        "feature_extraction_seconds":base_report["feature_seconds"],
        "elapsed_seconds":time.monotonic()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"submission_validation":test_validation,
        "leaderboard_submitted":False}
    write_json(args.output_dir / "report.json",report)
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--split",type=Path,required=True)
    p.add_argument("--features-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261001)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--identities-per-batch",type=int,default=128)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--temperature",type=float,default=0.07)
    p.add_argument("--identity-regularization",type=float,default=0.01)
    p.add_argument("--predict-test",action="store_true")
    a=p.parse_args()
    if a.epochs<1 or a.identities_per_batch<2 or a.lr<=0 or a.temperature<=0 or a.identity_regularization<0:
        p.error("Training settings must be positive (regularization can be zero)")
    print(json.dumps(train(a),indent=2))
