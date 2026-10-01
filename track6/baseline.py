"""Fixed-feature retrieval baselines with one shared validation protocol."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .data import DIRECTIONS, Store, decode_image, load_task, sha256, write_json
from .metrics import AP_MODES, cosine_top10, evaluate
from .protocol import load_split
from .submission import validate_submission

def gray_features(store, records):
    features = []
    for record in records:
        pixels = decode_image(store.read(record.image_path)).astype(np.float32)
        gray = pixels @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        small = np.asarray(Image.fromarray(gray).resize((32, 32), Image.Resampling.BILINEAR)).ravel().copy()
        small = (small - small.mean()) / max(float(small.std()), 1e-6)
        # Bias keeps constant-image embeddings defined; no test image is dropped.
        features.append(np.concatenate([small, np.ones(1, dtype=np.float32)]))
    return np.asarray(features, dtype=np.float32), {"feature":"32x32 float grayscale standardized per image plus bias",
                                                  "device":"cpu", "peak_allocated_bytes":0}

def dino_features(store, records, vendor, weights, resource, batch_size=32):
    import os
    import subprocess
    import sys
    import torch
    import torch.nn.functional as F
    resource_data = json.loads(Path(resource).read_text(encoding="utf-8"))
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if revision != resource_data["code"]["commit"]:
        raise ValueError("DINOv2 source commit differs from reviewed version")
    if not resource_data["weights"]["sha256"] or sha256(weights) != resource_data["weights"]["sha256"]:
        raise ValueError("A reviewed, fingerprinted exact checkpoint is required")
    if not torch.cuda.is_available():
        raise RuntimeError("This GPU baseline requires CUDA")
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(Path(vendor).resolve()))
    from dinov2.hub.backbones import dinov2_vits14
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1,3,1,1)
    features = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = np.stack([decode_image(store.read(r.image_path)) for r in records[start:start+batch_size]])
            x = torch.from_numpy(batch).permute(0,3,1,2).to("cuda", dtype=torch.float32) / 255
            x = F.interpolate(x, size=(224,224), mode="bicubic", align_corners=False, antialias=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model((x-mean)/std)
            features.append(y.float().cpu().numpy())
            if start % (batch_size * 10) == 0:
                print(f"Embedded {min(start+batch_size,len(records))}/{len(records)} images", flush=True)
    torch.cuda.synchronize()
    return np.concatenate(features), {"feature":"DINOv2-S/14 CLS, L2 normalization at cosine retrieval", "device":"cuda",
             "torch":torch.__version__, "cuda":torch.version.cuda, "gpu":torch.cuda.get_device_name(),
             "precision":"bf16 autocast; float32 embeddings and cosine", "batch_size":batch_size,
             "input":"256x256 uint8 RGB -> whole-image bicubic 224x224 -> ImageNet mean/std",
             "weights_sha256":sha256(weights), "vendor_commit":revision,
             "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
             "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
             "parameters":sum(p.numel() for p in model.parameters()),
             "embedding_seconds":time.monotonic()-started}

def run_baseline(source, split_path, output_dir, model="gray32", vendor=None, weights=None,
                 resource=None, batch_size=32, predict_test=False, cache_train=False):
    started = time.monotonic()
    store, records, split = load_split(source, split_path)
    by_id = {r.image_id:r for r in records}
    selected = [by_id[i] for i in split["validation_images"]]
    if cache_train:
        selected += [by_id[i] for i in split["train_images"]]
    if predict_test:
        queries, gallery, _ = load_task(store)
        selected += queries + gallery
    feature_start = time.monotonic()
    if model == "gray32":
        features, metadata = gray_features(store, selected)
    elif model == "dinov2_vits14":
        features, metadata = dino_features(store, selected, vendor, weights, resource, batch_size)
    else:
        raise ValueError("Unknown baseline")
    feature_seconds = time.monotonic() - feature_start
    features_by_id = {r.image_id:f for r,f in zip(selected, features)}
    rankings = {}
    for direction, task in split["tasks"].items():
        q, g = task["queries"], task["gallery"]
        rankings[direction] = cosine_top10([features_by_id[i] for i in q], [features_by_id[i] for i in g], q, g)
    results = {mode:evaluate(rankings, split, records, denominator=mode) for mode in AP_MODES}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "validation-rankings.json", rankings)
    np.savez_compressed(out / "features.npz", ids=np.asarray([r.image_id for r in selected]), embeddings=features)
    test_validation = None
    if predict_test:
        prediction = {}
        for direction, (_, target) in DIRECTIONS.items():
            q = [r.image_id for r in queries if r.query_type == direction]
            g = [r.image_id for r in gallery if r.modality == target]
            if q:
                prediction.update(cosine_top10([features_by_id[i] for i in q], [features_by_id[i] for i in g], q, g))
        write_json(out / "prediction.json", prediction)
        test_validation = validate_submission(source, out / "prediction.json")
    import subprocess
    code = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    report = {"model":model, "seed":split["seed"], "split_sha256":sha256(split_path),
              "labels_sha256":split["labels_sha256"], "source_sha256":split["source_sha256"],
              "code_commit":code.stdout.strip() if code.returncode == 0 else None,
              "feature_cache_sha256":sha256(out / "features.npz"),
              "metrics":results, "metadata":metadata, "feature_seconds":feature_seconds,
              "elapsed_seconds":time.monotonic() - started, "submission_validation":test_validation,
              "trained":False, "training_embeddings_cached":cache_train,
              "leaderboard_submitted":False, "state":"completed"}
    write_json(out / "report.json", report)
    return report
