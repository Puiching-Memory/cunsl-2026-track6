"""Validate prediction.json without requiring hidden ground truth."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track6.submission import validate_submission

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("prediction", type=Path)
    a = p.parse_args()
    try:
        print(json.dumps(validate_submission(a.source, a.prediction), indent=2))
    except (ValueError, OSError) as exc:
        print(json.dumps({"passed":False, "error":str(exc)}))
        raise SystemExit(1)
