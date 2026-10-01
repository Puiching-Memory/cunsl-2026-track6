#!/usr/bin/env python3
"""Run a user-selected command and retain its log and reproducibility manifest."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout.strip() if result.returncode == 0 else None

def dirty_status():
    result = subprocess.run(["git", "status", "--porcelain"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout.splitlines() if result.returncode == 0 else None

def acquire_gpu_lock(path, timeout):
    """Cooperate with track5, then reject any pre-existing GPU compute process."""
    if sys.platform != "linux":
        raise RuntimeError("Shared GPU locking requires Linux flock")
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise RuntimeError("Shared GPU is locked by another task; no process was stopped")
            print("Waiting for shared GPU lock", flush=True)
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    check = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check.returncode != 0 or check.stdout.strip():
        handle.close()
        raise RuntimeError("GPU process check failed or GPU is occupied; no process was stopped")
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid":__import__("os").getpid(), "name":"track6", "acquired_at_utc":utc()}) + "\n")
    handle.flush()
    return handle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("local/runs"))
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--gpu-wait-seconds", type=float, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.name):
        parser.error("--name must use lowercase letters, digits, '_' or '-'")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.output_root / f"{stamp}-{args.name}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "name": args.name, "started_at_utc": utc(), "state": "running",
        "command": command, "working_directory": str(Path.cwd()),
        "git_commit": revision(), "git_dirty_files":dirty_status(), "metrics_validated": False,
        "gpu_lock":str(args.gpu_lock) if args.gpu_lock else None,
        "log": "stdout.log"
    }
    manifest_path = directory / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    save()
    started = time.monotonic()
    process = None
    lock = None
    try:
        if args.gpu_lock:
            lock = acquire_gpu_lock(args.gpu_lock, args.gpu_wait_seconds)
            manifest["gpu_lock_acquired_at_utc"] = utc()
            save()
        with (directory / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            manifest["pid"] = process.pid
            save()
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            manifest["return_code"] = process.wait()
        manifest["state"] = "completed" if manifest["return_code"] == 0 else "failed"
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        manifest.update(state="interrupted", return_code=130)
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        manifest.update(state="failed", return_code=1,
                        error=f"{type(exc).__name__}: {exc}")
    finally:
        if lock is not None:
            lock.close()
        manifest.update(finished_at_utc=utc(), elapsed_seconds=time.monotonic() - started)
        save()
        print(f"Experiment record: {manifest_path}")
    return manifest["return_code"]

if __name__ == "__main__":
    raise SystemExit(main())
