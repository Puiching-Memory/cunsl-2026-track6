#!/usr/bin/env python3
"""Record a real CUDA smoke test without loading datasets or pretrained weights."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

def probe():
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    result = {
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "compiled_arches": torch.cuda.get_arch_list(),
        "packages": {},
        "scope": "synthetic CUDA correctness and backward checks; not a task benchmark"
    }
    for name in ("torchvision", "timm", "numpy", "Pillow", "safetensors"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if not result["cuda_available"]:
        result["passed"] = False
        return result
    try:
        device = torch.cuda.get_device_properties(0)
        result.update(device=device.name, vram_bytes=device.total_memory,
                      capability=list(torch.cuda.get_device_capability(0)))
        torch.manual_seed(20261001)
        torch.cuda.reset_peak_memory_stats()
        x = torch.arange(16, device="cuda", dtype=torch.float32)
        result["sum_of_squares"] = float((x * x).sum().cpu())
        layer = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
        inputs = torch.randn(2, 3, 64, 64, device="cuda")
        loss = layer(inputs).square().mean()
        loss.backward()
        result["fp32_backward_finite"] = bool(torch.isfinite(layer.weight.grad).all().cpu())
        layer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            amp_loss = layer(inputs).float().square().mean()
        amp_loss.backward()
        result["bf16_backward_finite"] = bool(torch.isfinite(layer.weight.grad).all().cpu())
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["passed"] = (result["sum_of_squares"] == 1240.0
                            and result["fp32_backward_finite"]
                            and result["bf16_backward_finite"])
    except Exception as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
