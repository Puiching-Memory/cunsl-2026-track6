"""Explicit AP@10 alternatives pending an official scoring implementation."""
import numpy as np

WEIGHTS = {"O2S":0.45, "S2O":0.45, "O2O":0.10}
AP_MODES = ("min_relevant_k", "all_relevant")

def average_precision_at_k(relevance, total_relevant, *, denominator, k=10):
    if denominator not in AP_MODES or total_relevant <= 0 or k <= 0:
        raise ValueError("Choose a known denominator, positive relevance count and positive k")
    rel = np.asarray(relevance, dtype=bool)[:k]
    if int(rel.sum()) > total_relevant:
        raise ValueError("More returned positives than total positives")
    precision = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    divisor = min(total_relevant, k) if denominator == "min_relevant_k" else total_relevant
    return float(np.sum(precision * rel) / divisor)

def direction_metrics(rankings, positives, *, denominator):
    if set(rankings) != set(positives) or not rankings:
        raise ValueError("Query coverage mismatch or empty evaluation")
    r1, aps = [], []
    for query in rankings:
        ranking = rankings[query]
        if not isinstance(ranking, list) or len(ranking) != 10 or any(not isinstance(g,str) for g in ranking) or len(set(ranking)) != 10:
            raise ValueError("Expected ten unique retrieval candidates")
        relevance = [g in positives[query] for g in ranking]
        r1.append(float(relevance[0]))
        aps.append(average_precision_at_k(relevance, len(positives[query]), denominator=denominator))
    recall, mean_ap = float(np.mean(r1)), float(np.mean(aps))
    return {"queries":len(rankings), "R@1":recall, "mAP@10":mean_ap, "score":(recall + mean_ap)/2,
            "max_relevant_in_gallery":max(map(len, positives.values()))}

def evaluate(rankings, split, records, *, denominator):
    by_id = {r.image_id:r for r in records}
    if set(rankings) != set(WEIGHTS):
        raise ValueError("Three directions are required")
    results = {}
    for direction, task in split["tasks"].items():
        allowed = set(task["gallery"])
        if set(rankings[direction]) != set(task["queries"]):
            raise ValueError("Direction query coverage mismatch")
        if any(g not in allowed for ranking in rankings[direction].values() for g in ranking):
            raise ValueError("Gallery ID outside validation protocol")
        positives = {q:{g for g in allowed if by_id[q].ship_id == by_id[g].ship_id} for q in task["queries"]}
        results[direction] = direction_metrics(rankings[direction], positives, denominator=denominator)
    return {"directions":results, "weighted_score":sum(WEIGHTS[d]*r["score"] for d,r in results.items()),
            "ap_denominator":denominator, "official_scorer_confirmed":False,
            "scope":"local identity-disjoint holdout; not a leaderboard result"}

def cosine_top10(query_features, gallery_features, query_ids, gallery_ids):
    q, g = np.asarray(query_features, dtype=np.float32), np.asarray(gallery_features, dtype=np.float32)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or len(q) != len(query_ids) or len(g) != len(gallery_ids):
        raise ValueError("Feature shape and ID mismatch")
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids) or len(gallery_ids) < 10:
        raise ValueError("Duplicate IDs or too few gallery images")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Nonfinite embedding")
    qnorm, gnorm = np.linalg.norm(q, axis=1, keepdims=True), np.linalg.norm(g, axis=1, keepdims=True)
    if (qnorm == 0).any() or (gnorm == 0).any():
        raise ValueError("Zero embedding cannot be cosine-normalized")
    similarities = (q/qnorm) @ (g/gnorm).T
    # Sorting gallery IDs first makes exact score ties reproducible.
    gallery_order = np.argsort(np.asarray(gallery_ids), kind="stable")
    order = gallery_order[np.argsort(-similarities[:,gallery_order], axis=1, kind="stable")[:,:10]]
    return {query_ids[i]:[gallery_ids[j] for j in row] for i,row in enumerate(order)}
