"""Run a fixed-feature baseline; GPU runs must be wrapped by run_job --gpu-lock."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track6.baseline import run_baseline

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=["gray32", "dinov2_vits14"], default="gray32")
    p.add_argument("--vendor", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--resource", type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--predict-test", action="store_true")
    p.add_argument("--cache-train", action="store_true")
    a = p.parse_args()
    print(json.dumps(run_baseline(a.source, a.split, a.output_dir, a.model, a.vendor, a.weights,
                                 a.resource, a.batch_size, a.predict_test, a.cache_train), indent=2))
