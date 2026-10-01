"""Download an individually reviewed public checkpoint and retain its fingerprint."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track6.data import sha256, write_json

def fetch(resource_path, destination):
    resource = json.loads(Path(resource_path).read_text(encoding="utf-8"))
    if not (resource["public_before_cutoff_reviewed"] and resource["code"]["license_reviewed"]
            and resource["weights"]["license_reviewed"]):
        raise ValueError("Resource date, code and specific weight licenses must be reviewed first")
    if resource["weights"]["public_url_present_by"] >= resource["cutoff"]:
        raise ValueError("Resource does not predate competition cutoff")
    url = resource["weights"]["url"]
    if not url.startswith("https://") or "?" in url or "@" in url:
        raise ValueError("Expected a public HTTPS checkpoint URL without credentials")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = resource["weights"].get("sha256")
    started = time.monotonic()
    if not destination.exists():
        part = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(url, timeout=30) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            copied, next_report = 0, 8 * 1024 * 1024
            for block in iter(lambda:response.read(1024 * 1024), b""):
                out.write(block)
                copied += len(block)
                if copied >= next_report:
                    print(f"Downloaded {copied}/{total or 'unknown'} bytes", flush=True)
                    next_report += 8 * 1024 * 1024
        if total and copied != total:
            raise ValueError("Download size mismatch; partial file retained")
        digest = sha256(part)
        if expected and digest != expected:
            raise ValueError("Weight SHA-256 mismatch; partial file retained")
        part.rename(destination)
    digest = sha256(destination)
    if expected and digest != expected:
        raise ValueError("Existing weight SHA-256 mismatch")
    result = {"resource":resource["variant"], "url":url, "sha256":digest,
              "bytes":destination.stat().st_size, "download_verified_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "elapsed_seconds":time.monotonic() - started, "resource_manifest_sha256":sha256(resource_path)}
    write_json(destination.with_suffix(destination.suffix + ".source.json"), result)
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("resource", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    print(json.dumps(fetch(a.resource, a.destination), indent=2))
