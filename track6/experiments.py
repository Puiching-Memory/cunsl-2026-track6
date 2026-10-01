"""Save private features/rankings and anonymous, independently checkable reports."""
import subprocess
from pathlib import Path

import numpy as np

from .data import DIRECTIONS, sha256, write_json
from .metrics import AP_MODES, cosine_top10, evaluate
from .submission import validate_submission

def save_retrieval(source, split_path, records, split, selected, embeddings,
                   output_dir, report, *, queries=None, gallery=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = [r.image_id for r in selected]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(ids) != len(set(ids)) or embeddings.shape[0] != len(ids):
        raise ValueError('Unique feature IDs and matching rows required')
    features = dict(zip(ids,embeddings))
    rankings = {}
    for direction, task in split['tasks'].items():
        q,g = task['queries'],task['gallery']
        rankings[direction] = cosine_top10([features[i] for i in q],[features[i] for i in g],q,g)
    metrics = {mode:evaluate(rankings,split,records,denominator=mode) for mode in AP_MODES}
    write_json(out/'validation-rankings.json',rankings)
    np.savez_compressed(out/'features.npz',ids=np.asarray(ids),embeddings=embeddings)
    submission_validation = None
    if queries is not None:
        prediction = {}
        for direction,(_,modality) in DIRECTIONS.items():
            q = [r.image_id for r in queries if r.query_type==direction]
            g = [r.image_id for r in gallery if r.modality==modality]
            if q:
                prediction.update(cosine_top10([features[i] for i in q],[features[i] for i in g],q,g))
        write_json(out/'prediction.json',prediction)
        submission_validation = validate_submission(source,out/'prediction.json')
    report = dict(report)
    report.update(state='completed',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  seed=report.get('seed',split['seed']),split_sha256=sha256(split_path),
                  source_sha256=split['source_sha256'],labels_sha256=split['labels_sha256'],
                  feature_cache_sha256=sha256(out/'features.npz'),metrics=metrics,
                  submission_validation=submission_validation,leaderboard_submitted=False)
    write_json(out/'report.json',report)
    return report
