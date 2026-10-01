"""Audit every official TIFF and metadata record; retain private details outside Git."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track6.audit import audit

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    result = audit(a.source, a.output_dir, a.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
