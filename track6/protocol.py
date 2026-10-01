"""Identity-disjoint holdout; exact duplicates removed, near duplicates grouped."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .data import DIRECTIONS, Store, load_labels, sha256, strict_json, write_json

def make_split(source, audit_dir, output, seed=20261001, val_fraction=0.2):
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    store = Store(source)
    records, label_member = load_labels(store)
    summary = strict_json((Path(audit_dir) / "summary.json").read_text(encoding="utf-8"))
    rows = strict_json((Path(audit_dir) / "images.json").read_text(encoding="utf-8"))
    duplicates = strict_json((Path(audit_dir) / "duplicates.json").read_text(encoding="utf-8"))
    if not summary["passed"] or summary["labels_sha256"] != hashlib.sha256(store.read(label_member)).hexdigest():
        raise ValueError("A passing audit of these exact labels is required")
    if store.zipped and summary["source_sha256"] != sha256(source):
        raise ValueError("Audit belongs to another archive")
    audited = {r["image_id"]:r for r in rows if r["role"] == "train"}
    by_id = {r.image_id:r for r in records}
    if set(audited) != set(by_id) or any(audited[k]["image_path"] != by_id[k].image_path for k in by_id):
        raise ValueError("Per-image audit does not cover training metadata")
    excluded = {}
    for r in records:
        if not audited[r.image_id]["passed"] or audited[r.image_id]["constant"]:
            excluded[r.image_id] = "constant_or_invalid"
    groups = defaultdict(list)
    for r in records:
        if r.image_id not in excluded:
            groups[audited[r.image_id]["pixel_sha256"]].append(r)
    for group in groups.values():
        if len({r.ship_id for r in group}) > 1:
            for r in group:
                excluded[r.image_id] = "identical_pixels_conflicting_identity"
        else:
            # Keep one copy, including across modalities, to avoid trivial same-pixel positives.
            for r in sorted(group, key=lambda r:r.image_id)[1:]:
                excluded[r.image_id] = "identical_pixel_copy"
    usable = [r for r in records if r.image_id not in excluded]
    identities = sorted({r.ship_id for r in usable})
    parent = {s:s for s in identities}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for pair in duplicates["near_pairs"]:
        a, b = pair["left"], pair["right"]
        if a in by_id and b in by_id and a not in excluded and b not in excluded:
            ra, rb = find(by_id[a].ship_id), find(by_id[b].ship_id)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    components = defaultdict(list)
    for s in identities:
        components[find(s)].append(s)
    order = sorted(components.values(), key=lambda c:hashlib.sha256(f"{seed}:{min(c)}".encode()).hexdigest())
    target = max(1, round(len(identities) * val_fraction))
    validation_ids = set()
    for component in order:
        if len(validation_ids) < target:
            validation_ids.update(component)
    training_ids = set(identities) - validation_ids
    train = [r for r in usable if r.ship_id in training_ids]
    val = [r for r in usable if r.ship_id in validation_ids]
    if not train or not val:
        raise ValueError("Duplicate grouping left no nonempty identity holdout")
    train_hashes = {audited[r.image_id]["pixel_sha256"] for r in train}
    val_hashes = {audited[r.image_id]["pixel_sha256"] for r in val}
    if train_hashes & val_hashes or training_ids & validation_ids:
        raise ValueError("Identity or pixel leakage")
    tasks = {}
    by_identity = defaultdict(list)
    for r in val:
        by_identity[r.ship_id].append(r)
    for direction, (qm, gm) in DIRECTIONS.items():
        gallery = sorted((r.image_id for r in val if r.modality == gm))
        if direction != "O2O":
            positive_ids = {r.ship_id for r in val if r.modality == gm}
            queries = sorted(r.image_id for r in val if r.modality == qm and r.ship_id in positive_ids)
        else:
            queries = []
            for s, identity_rows in sorted(by_identity.items()):
                optical = [r for r in identity_rows if r.modality == "optical"]
                if len(optical) >= 2:
                    q = min(optical, key=lambda r:hashlib.sha256(f"{seed}:{r.image_id}".encode()).hexdigest())
                    queries.append(q.image_id)
            gallery = sorted(set(gallery) - set(queries))
        if not queries or len(gallery) < 10:
            raise ValueError(f"Holdout cannot evaluate {direction}; change split seed or fraction")
        tasks[direction] = {"queries":sorted(queries), "gallery":gallery}
    # Fingerprint only the training evidence used in this decision. Audit timestamps,
    # timings, unrelated test duplicates and floating descriptive statistics cannot
    # change an otherwise identical split file.
    decision_evidence = {
        "images":[{k:audited[i][k] for k in ["image_id","image_path","pixel_sha256","passed","constant"]}
                  for i in sorted(by_id)],
        "near_pairs":sorted(sorted([p["left"],p["right"]]) for p in duplicates["near_pairs"]
                            if p["left"] in by_id and p["right"] in by_id)}
    evidence_digest = hashlib.sha256(json.dumps(decision_evidence,sort_keys=True,separators=(",",":"),
                                               ensure_ascii=False).encode("utf-8")).hexdigest()
    protocol = {
        "version":"identity-holdout-v1", "seed":seed, "requested_val_fraction":val_fraction,
        "labels_sha256":summary["labels_sha256"], "source_sha256":summary["source_sha256"],
        "audit_training_decisions_sha256":evidence_digest,
        "train_identities":sorted(training_ids), "validation_identities":sorted(validation_ids),
        "train_images":sorted(r.image_id for r in train), "validation_images":sorted(r.image_id for r in val),
        "excluded_images":dict(sorted(excluded.items())), "tasks":tasks,
        "duplicate_policy":"constant/conflicting/exact copies excluded; dHash<=2 AND thumbnail correlation>=0.9998 training identity components kept together",
        "identity_components":len(components), "largest_identity_component":max(map(len, components.values())),
        "identity_overlap":0, "pixel_overlap":0, "near_duplicate_edges_across_split":0,
        "test_used_for_split":False}
    write_json(output, protocol)
    return split_summary(protocol, sha256(output))

def split_summary(protocol, digest):
    return {"protocol":protocol["version"], "seed":protocol["seed"], "split_sha256":digest,
            "train_identities":len(protocol["train_identities"]),
            "validation_identities":len(protocol["validation_identities"]),
            "train_images":len(protocol["train_images"]), "validation_images":len(protocol["validation_images"]),
            "exclusions":dict(Counter(protocol["excluded_images"].values())),
            "identity_components":protocol["identity_components"],
            "largest_identity_component":protocol["largest_identity_component"],
            "tasks":{k:{f:len(v) for f,v in task.items()} for k,task in protocol["tasks"].items()},
            "identity_overlap":protocol["identity_overlap"], "pixel_overlap":protocol["pixel_overlap"],
            "near_duplicate_edges_across_split":protocol["near_duplicate_edges_across_split"],
            "test_used_for_split":False}

def load_split(source, split_path):
    store = Store(source)
    records, labels_member = load_labels(store)
    split = strict_json(Path(split_path).read_text(encoding="utf-8"))
    if split["version"] != "identity-holdout-v1":
        raise ValueError("Unknown split version")
    if split["labels_sha256"] != hashlib.sha256(store.read(labels_member)).hexdigest():
        raise ValueError("Split labels do not match data")
    if store.zipped and split["source_sha256"] != sha256(source):
        raise ValueError("Split archive does not match data")
    by_id = {r.image_id:r for r in records}
    partitions = [set(split[k]) for k in ["train_images", "validation_images", "excluded_images"]]
    if set.union(*partitions) != set(by_id) or any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise ValueError("Split coverage or image overlap error")
    actual_train = {by_id[i].ship_id for i in split["train_images"]}
    actual_val = {by_id[i].ship_id for i in split["validation_images"]}
    if actual_train & actual_val or actual_train != set(split["train_identities"]) or actual_val != set(split["validation_identities"]):
        raise ValueError("Identity leakage or inconsistent identity list")
    if set(split["tasks"]) != set(DIRECTIONS):
        raise ValueError("Missing validation direction")
    for direction, task in split["tasks"].items():
        q, g = task["queries"], task["gallery"]
        if not q or len(g) < 10 or len(set(q)) != len(q) or len(set(g)) != len(g) or set(q) & set(g):
            raise ValueError("Invalid validation task or query self-match")
        if not (set(q) | set(g)) <= actual_image_ids(split):
            raise ValueError("Task uses a non-validation image")
        qm, gm = DIRECTIONS[direction]
        if any(by_id[i].modality != qm for i in q) or any(by_id[i].modality != gm for i in g):
            raise ValueError("Task modality mismatch")
        positive = {by_id[i].ship_id for i in g}
        if any(by_id[i].ship_id not in positive for i in q):
            raise ValueError("Query has no positive")
    return store, records, split

def actual_image_ids(split):
    return set(split["validation_images"])
