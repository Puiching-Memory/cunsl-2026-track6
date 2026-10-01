#!/usr/bin/env python3
"""Read a ZIP inventory and optional CRCs; never extract or execute its contents."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def unsafe_name(name):
    normal = name.replace("\\", "/")
    parts = PurePosixPath(normal).parts
    return (normal.startswith("/") or ".." in parts
            or (bool(parts) and ":" in parts[0]))

def inspect(path, verify_crc=False):
    result = {
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "archive_name": path.name,
        "archive_bytes": path.stat().st_size,
        "sha256": digest(path),
        "extracted": False,
        "image_decode_checked": False
    }
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        files = [item for item in entries if not item.is_dir()]
        names = [item.filename for item in files]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        unsafe = [name for name in names if unsafe_name(name)]
        encrypted = [item.filename for item in files if item.flag_bits & 1]
        result.update(
            entries=len(entries), files=len(files),
            directories=len(entries) - len(files),
            uncompressed_bytes=sum(item.file_size for item in files),
            compressed_bytes=sum(item.compress_size for item in files),
            extensions=dict(sorted(Counter(PurePosixPath(n).suffix.lower() or "(none)" for n in names).items())),
            top_level=dict(sorted(Counter(n.replace("\\", "/").split("/")[0] for n in names).items())),
            duplicate_names=duplicates, unsafe_paths=unsafe, encrypted_entries=encrypted,
            metadata_files=[n for n in names if PurePosixPath(n).suffix.lower() in {".json", ".csv", ".txt", ".md"}],
            zero_byte_files=[item.filename for item in files if item.file_size == 0],
            smallest_files=[{"name":item.filename, "bytes":item.file_size}
                            for item in sorted(files, key=lambda x:x.file_size)[:10]],
            crc_checked=False
        )
        if verify_crc and not encrypted:
            try:
                bad = archive.testzip()
                result.update(crc_checked=True, crc_failed_file=bad)
            except Exception as exc:
                result.update(crc_checked=True, crc_error=f"{type(exc).__name__}: {exc}")
        result["passed"] = not (duplicates or unsafe or encrypted or result["zero_byte_files"]
                                or result.get("crc_failed_file") or result.get("crc_error"))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--verify-crc", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.archive, args.verify_crc)
    except Exception as exc:
        result = {"passed":False, "extracted":False, "error":f"{type(exc).__name__}: {exc}"}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
