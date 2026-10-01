"""Full TIFF decoding and private per-image audit with a shareable summary."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import datetime
import hashlib
import time

import numpy as np
from PIL import Image

from .data import Store, decode_image, load_labels, load_task, sha256, write_json

def image_stats(store, record):
    row = asdict(record)
    try:
        raw = store.read(record.image_path)
        pixels = decode_image(raw)
        replicated = bool(np.array_equal(pixels[..., 0], pixels[..., 1]) and
                          np.array_equal(pixels[..., 0], pixels[..., 2]))
        # Floating-point grayscale avoids rounding faint SAR thumbnails to zero.
        full_gray = pixels.astype(np.float32) @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        gray_image = Image.fromarray(full_gray)
        gray = np.asarray(gray_image.resize((9, 8), Image.Resampling.BILINEAR))
        bits = (gray[:, 1:] > gray[:, :-1]).ravel()
        dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
        thumbnail = np.asarray(gray_image.resize((16, 16), Image.Resampling.BILINEAR)).ravel().copy()
        thumbnail -= thumbnail.mean()
        norm = float(np.linalg.norm(thumbnail))
        thumbnail /= max(norm, 1e-12)
        row.update(bytes=len(raw), file_sha256=hashlib.sha256(raw).hexdigest(),
                   pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                   shape=list(pixels.shape), dtype=str(pixels.dtype), replicated_channels=replicated,
                   minimum=int(pixels.min()), maximum=int(pixels.max()),
                   std=float(pixels.std()), mean=float(pixels.mean()), dhash=f"{dhash:016x}",
                   constant=bool(pixels.min() == pixels.max()), passed=True)
        row.update(_thumbnail=thumbnail, gray_thumbnail_degenerate=norm < 1e-6)
        if record.modality == "sar" and not replicated:
            row.update(passed=False, error="SAR channels are not replicated")
    except Exception as exc:
        row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    return row

def duplicate_groups(rows, field):
    groups = defaultdict(list)
    for r in rows:
        if r.get(field):
            groups[r[field]].append(r["image_id"])
    return [sorted(v) for v in groups.values() if len(v) > 1]

def near_duplicates(rows, distance=2):
    """Candidate flags requiring dHash <=2 and thumbnail correlation >=0.9998.

    Three disjoint bands guarantee a shared band for Hamming distance <= 2.
    Constant images and exact pixel duplicates are handled separately.
    """
    buckets = defaultdict(list)
    pairs = []
    eligible = [r for r in rows if r.get("passed") and not r["constant"] and not r["gray_thumbnail_degenerate"]]
    thumbnails = np.asarray([r["_thumbnail"] for r in eligible], dtype=np.float32)
    for i, r in enumerate(eligible):
        h = int(r["dhash"], 16)
        keys = [(0, h & ((1 << 22) - 1)), (1, (h >> 22) & ((1 << 21) - 1)), (2, h >> 43)]
        candidates = {j for key in keys for j in buckets[key]}
        close = [j for j in sorted(candidates) if (h ^ int(eligible[j]["dhash"], 16)).bit_count() <= distance]
        if close:
            correlations = thumbnails[close] @ thumbnails[i]
            close = [j for j,c in zip(close, correlations) if c >= 0.9998]
        for j in close:
            other = eligible[j]
            if r["pixel_sha256"] == other["pixel_sha256"]:
                continue
            d = (h ^ int(other["dhash"], 16)).bit_count()
            if d <= distance:
                pairs.append({"left":other["image_id"], "right":r["image_id"], "hamming":d})
        for key in keys:
            buckets[key].append(i)
    return pairs

def audit(source, output_dir, workers=4):
    started = time.monotonic()
    store = Store(source)
    train, labels_member = load_labels(store)
    queries, gallery, task_member = load_task(store)
    records = train + queries + gallery
    if len({r.image_id for r in records}) != len(records):
        raise ValueError("Image IDs collide across train and test")
    referenced = {r.image_path for r in records}
    image_members = {n for n in store.names if n.lower().endswith((".tif", ".tiff"))}
    if image_members != referenced:
        raise ValueError("Unreferenced or missing TIFF members")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda r: image_stats(store, r), records))
    by_id = {r["image_id"]:r for r in rows}
    pixel_groups = duplicate_groups(rows, "pixel_sha256")
    file_groups = duplicate_groups(rows, "file_sha256")
    near_pairs = near_duplicates(rows)
    identities = defaultdict(Counter)
    for r in train:
        identities[r.ship_id][r.modality] += 1
    conflicts = [g for g in pixel_groups if len({by_id[i]["ship_id"] for i in g
                                                if by_id[i]["role"] == "train"}) > 1]
    roles = lambda g: sorted({by_id[i]["role"] for i in g})
    summaries = {}
    for role in ["train", "query", "gallery"]:
        selected = [r for r in rows if r["role"] == role]
        summaries[role] = {
            "images":len(selected), "modalities":dict(Counter(r["modality"] for r in selected)),
            "decode_failures":sum(not r["passed"] for r in selected),
            "constant_images":sum(r.get("constant", False) for r in selected),
            "low_std_lt_1":sum(r.get("std", 999) < 1 for r in selected),
            "replicated_channels_by_modality":dict(Counter(r["modality"] for r in selected
                                                              if r.get("replicated_channels"))),
            "shape_counts":dict(Counter(str(r.get("shape")) for r in selected)),
            "dtype_counts":dict(Counter(r.get("dtype", "decode_failed") for r in selected))}
    result = {
        "checked_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256":sha256(source) if store.zipped else None,
        "labels_sha256":hashlib.sha256(store.read(labels_member)).hexdigest(),
        "task_sha256":hashlib.sha256(store.read(task_member)).hexdigest(),
        "official_schema":{"labels":["image_id", "ship_id", "modality", "image_path"],
                           "task":["queries", "gallery"],
                           "query":["query_id", "image_path", "query_type"],
                           "gallery":["image_id", "image_path", "modality"]},
        "roles":summaries, "query_directions":dict(Counter(q.query_type for q in queries)),
        "train_identities":len(identities),
        "train_identity_modality_patterns":dict(Counter(f"optical={v['optical']},sar={v['sar']}"
                                                       for v in identities.values())),
        "train_both_modalities":sum(v["optical"] > 0 and v["sar"] > 0 for v in identities.values()),
        "train_at_least_two_optical":sum(v["optical"] >= 2 for v in identities.values()),
        "file_duplicate_groups":len(file_groups), "pixel_duplicate_groups":len(pixel_groups),
        "pixel_duplicate_images":sum(len(g) for g in pixel_groups),
        "pixel_duplicate_role_patterns":dict(Counter("+".join(roles(g)) for g in pixel_groups)),
        "train_duplicate_identity_conflicts":len(conflicts),
        "near_duplicate_pairs":len(near_pairs),
        "near_duplicate_criteria":"float grayscale dHash<=2 AND 16x16 normalized correlation>=0.9998",
        "near_duplicate_scope":"heuristic flags only; no hidden test identity inference",
        "test_identities_known":False,
        "passed":all(r["passed"] for r in rows),
        "elapsed_seconds":time.monotonic() - started,
        "warnings":["Constant and duplicate images require protocol handling; raw data is preserved.",
                    "Integrity and decoding do not establish label truth or benchmark accuracy."]}
    from pathlib import Path
    out = Path(output_dir)
    write_json(out / "summary.json", result)
    write_json(out / "images.json", [{k:v for k,v in r.items() if k != "_thumbnail"} for r in rows])
    write_json(out / "duplicates.json", {"file_groups":file_groups, "pixel_groups":pixel_groups,
                                          "train_identity_conflicts":conflicts, "near_pairs":near_pairs})
    return result
