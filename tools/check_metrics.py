"""Independently recompute saved rankings with scalar AP arithmetic, then check artifacts."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track6.data import sha256,strict_json,write_json
from track6.protocol import load_split
from track6.submission import validate_submission

def reference_metrics(rankings,split,records,mode):
    if mode not in {"min_relevant_k","all_relevant"}:
        raise ValueError("Unknown AP denominator")
    by_id = {r.image_id:r for r in records}
    directions = {}
    if set(rankings) != set(split["tasks"]):
        raise ValueError("Direction coverage mismatch")
    for direction,task in split["tasks"].items():
        if set(rankings[direction]) != set(task["queries"]):
            raise ValueError("Query coverage mismatch")
        recalls,aps,totals = [],[],[]
        allowed = set(task["gallery"])
        for query in task["queries"]:
            ranked = rankings[direction][query]
            if not isinstance(ranked,list) or len(ranked)!=10 or any(not isinstance(i,str) for i in ranked):
                raise ValueError("Ranking must contain ten string IDs")
            if len(set(ranked))!=10 or not set(ranked)<=allowed:
                raise ValueError("Duplicate or invalid gallery ID")
            identity = by_id[query].ship_id
            total = sum(by_id[g].ship_id==identity for g in task["gallery"])
            if total==0:
                raise ValueError("Query has no positive")
            hits,numerator = 0,0.0
            for rank,candidate in enumerate(ranked,1):
                if by_id[candidate].ship_id==identity:
                    hits += 1
                    numerator += hits/rank
            recalls.append(float(by_id[ranked[0]].ship_id==identity))
            aps.append(numerator/(min(total,10) if mode=="min_relevant_k" else total))
            totals.append(total)
        r1,ap = math.fsum(recalls)/len(recalls),math.fsum(aps)/len(aps)
        directions[direction] = {"queries":len(recalls),"R@1":r1,"mAP@10":ap,
                                 "score":(r1+ap)/2,"max_relevant_in_gallery":max(totals)}
    score = math.fsum(directions[d]["score"]*w for d,w in [("O2S",0.45),("S2O",0.45),("O2O",0.1)])
    return {"directions":directions,"weighted_score":score,"ap_denominator":mode,
            "official_scorer_confirmed":False,"scope":"local identity-disjoint holdout; not a leaderboard result"}

def check(source,split_path,run_dir,output=None):
    _,records,split = load_split(source,split_path)
    run_dir = Path(run_dir)
    report = strict_json((run_dir / "report.json").read_text(encoding="utf-8"))
    rankings = strict_json((run_dir / "validation-rankings.json").read_text(encoding="utf-8"))
    if report["split_sha256"] != sha256(split_path) or report["source_sha256"] != split["source_sha256"]:
        raise ValueError("Report data or split checksum mismatch")
    verified = {}
    for mode in ["min_relevant_k","all_relevant"]:
        actual = reference_metrics(rankings,split,records,mode)
        claimed = report["metrics"][mode]
        for direction,row in actual["directions"].items():
            for field,value in row.items():
                if not math.isclose(value,claimed["directions"][direction][field],abs_tol=1e-12,rel_tol=0):
                    raise ValueError(f"Recorded metric differs: {mode}/{direction}/{field}")
        if not math.isclose(actual["weighted_score"],claimed["weighted_score"],abs_tol=1e-12,rel_tol=0):
            raise ValueError("Recorded weighted score differs")
        verified[mode] = actual
    artifacts = {}
    for name,key in [("features.npz","feature_cache_sha256"),("adapter.pt","adapter_sha256")]:
        path = run_dir / name
        if path.exists():
            digest = sha256(path)
            if report[key] != digest:
                raise ValueError(f"Artifact checksum differs: {name}")
            artifacts[name] = digest
    submission = None
    if (run_dir / "prediction.json").exists():
        submission = validate_submission(source,run_dir / "prediction.json")
        artifacts["prediction.json"] = sha256(run_dir / "prediction.json")
    result = {"passed":True,"model":report["model"],"code_commit":report["code_commit"],
              "report_sha256":sha256(run_dir / "report.json"),"split_sha256":sha256(split_path),
              "source_sha256":split["source_sha256"],"validation_rankings_sha256":sha256(run_dir / "validation-rankings.json"),
              "independent_method":"scalar rank loop with math.fsum; does not call track6.metrics",
              "metric_tolerance":1e-12,"metrics":verified,"artifacts":artifacts,
              "submission_validation":submission,"official_scorer_confirmed":False,
              "leaderboard_submitted":False}
    if output:
        write_json(output,result)
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--split",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    print(json.dumps(check(a.source,a.split,a.run_dir,a.output),indent=2))
