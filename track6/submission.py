"""Strict validation against the actual official task.json metadata."""
from collections import Counter
from pathlib import Path

from .data import DIRECTIONS, Store, load_task, strict_json

def validate_submission(source, prediction_path):
    store = Store(source)
    queries, gallery, _ = load_task(store)
    prediction = strict_json(Path(prediction_path).read_text(encoding="utf-8"))
    if not isinstance(prediction, dict):
        raise ValueError("Prediction top level must be an object")
    query_ids = {q.image_id for q in queries}
    if set(prediction) != query_ids:
        raise ValueError(f"Query coverage error: missing={len(query_ids-set(prediction))}, extra={len(set(prediction)-query_ids)}")
    by_id = {g.image_id:g for g in gallery}
    for q in queries:
        ranking = prediction[q.image_id]
        if not isinstance(ranking, list) or len(ranking) != 10 or any(not isinstance(g, str) for g in ranking):
            raise ValueError(f"Query {q.image_id}: expected ten string IDs")
        if len(set(ranking)) != 10:
            raise ValueError(f"Query {q.image_id}: duplicate candidates")
        target = DIRECTIONS[q.query_type][1]
        if any(g not in by_id for g in ranking):
            raise ValueError(f"Query {q.image_id}: invalid Gallery ID")
        if any(by_id[g].modality != target for g in ranking):
            raise ValueError(f"Query {q.image_id}: wrong target modality")
    return {"passed":True, "queries":len(queries), "gallery":len(gallery),
            "directions":dict(Counter(q.query_type for q in queries)), "candidates_per_query":10,
            "metrics_validated":False, "scope":"structure only; hidden identities are unavailable"}
