"""Freeze an audited identity-disjoint validation protocol."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track6.protocol import make_split

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--val-fraction", type=float, default=0.2)
    a = p.parse_args()
    print(json.dumps(make_split(a.source, a.audit_dir, a.output, a.seed, a.val_fraction), indent=2))
