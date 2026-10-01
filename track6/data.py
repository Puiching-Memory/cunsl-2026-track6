"""Read the observed official schema without guessing modality from pixels."""
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import threading
import zipfile

import numpy as np
from PIL import Image

DIRECTIONS = {"O2S": ("optical", "sar"), "S2O": ("sar", "optical"),
              "O2O": ("optical", "optical")}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a nonempty POSIX relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(c in {"", ".", ".."} or ":" in c for c in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return p

def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {v}")))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")

class Store:
    """A ZIP or extracted directory with the same relative member names."""
    def __init__(self, source):
        self.source = Path(source)
        self.zipped = self.source.is_file()
        self._archive = None
        self._pid = None
        self._read_lock = threading.RLock()
        if self.zipped:
            with zipfile.ZipFile(self.source) as z:
                entries = z.infolist()
                self.names = sorted(e.filename for e in entries if not e.is_dir())
                if len(self.names) != len(set(self.names)):
                    raise ValueError("Duplicate ZIP members")
                for e in entries:
                    safe_relative(e.filename.rstrip("/"))
                    if e.flag_bits & 1 or stat.S_ISLNK(e.external_attr >> 16):
                        raise ValueError("Encrypted or symlink ZIP member")
        else:
            if not self.source.is_dir():
                raise ValueError("Data source must be a ZIP or directory")
            self.names = sorted(p.relative_to(self.source).as_posix()
                                for p in self.source.rglob("*") if p.is_file())
    def read(self, name):
        safe_relative(name)
        if self.zipped:
            # Serialize access to the shared seek/decompression stream. The
            # target Python 3.12.3 runtime produced intermittent corrupted reads
            # under simultaneous ZipFile.read calls, despite valid entry CRCs.
            with self._read_lock:
                if self._archive is None or self._pid != os.getpid():
                    if self._archive is not None:
                        self._archive.close()
                    self._archive = zipfile.ZipFile(self.source)
                    self._pid = os.getpid()
                return self._archive.read(name)
        path = (self.source / name).resolve()
        if not path.is_relative_to(self.source.resolve()):
            raise ValueError("Path escapes data root")
        return path.read_bytes()
    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_archive=None, _pid=None)
        state.pop("_read_lock",None)
        return state
    def __setstate__(self,state):
        self.__dict__.update(state)
        self._read_lock = threading.RLock()
    def unique(self, suffix):
        found = [n for n in self.names if n == suffix or n.endswith("/" + suffix)]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {suffix}; found {len(found)}")
        return found[0]

@dataclass(frozen=True)
class Record:
    image_id: str
    image_path: str
    modality: str
    role: str
    ship_id: str | None = None
    query_type: str | None = None

def nonempty(value, field):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Invalid {field}")
    return value

def load_labels(store):
    member = store.unique("labels.csv")
    base = PurePosixPath(member).parent
    reader = csv.DictReader(io.StringIO(store.read(member).decode("utf-8-sig")))
    if reader.fieldnames != ["image_id", "ship_id", "modality", "image_path"]:
        raise ValueError(f"Unexpected label fields: {reader.fieldnames}")
    records = []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed CSV row")
        for k, v in row.items():
            nonempty(v, k)
        if row["modality"] not in {"optical", "sar"}:
            raise ValueError("Unknown training modality")
        rel = safe_relative(row["image_path"])
        if rel.parts[0] != row["modality"] or rel.stem != row["image_id"]:
            raise ValueError("Label modality, path or image ID mismatch")
        records.append(Record(row["image_id"], (base / rel).as_posix(), row["modality"],
                              "train", row["ship_id"]))
    validate_records(records, store)
    return records, member

def load_task(store):
    member = store.unique("task.json")
    base = PurePosixPath(member).parent
    task = strict_json(store.read(member).decode("utf-8-sig"))
    if not isinstance(task, dict) or set(task) != {"queries", "gallery"}:
        raise ValueError("Expected task object containing queries and gallery")
    if not all(isinstance(task[k], list) and task[k] for k in task):
        raise ValueError("Empty or malformed task")
    queries, gallery = [], []
    for q in task["queries"]:
        if not isinstance(q, dict) or set(q) != {"query_id", "image_path", "query_type"}:
            raise ValueError("Unexpected query fields")
        if q["query_type"] not in DIRECTIONS:
            raise ValueError("Unknown query direction")
        nonempty(q["query_id"], "query_id")
        rel = safe_relative(q["image_path"])
        if rel.parts[0] != "query" or rel.stem != q["query_id"]:
            raise ValueError("Query path or ID mismatch")
        queries.append(Record(q["query_id"], (base / rel).as_posix(),
                              DIRECTIONS[q["query_type"]][0], "query", query_type=q["query_type"]))
    for g in task["gallery"]:
        if not isinstance(g, dict) or set(g) != {"image_id", "image_path", "modality"}:
            raise ValueError("Unexpected gallery fields")
        nonempty(g["image_id"], "image_id")
        if g["modality"] not in {"optical", "sar"}:
            raise ValueError("Unknown gallery modality")
        rel = safe_relative(g["image_path"])
        if rel.parts[0] != "gallery" or rel.stem != g["image_id"]:
            raise ValueError("Gallery path or ID mismatch")
        gallery.append(Record(g["image_id"], (base / rel).as_posix(), g["modality"], "gallery"))
    validate_records(queries + gallery, store)
    counts = Counter(g.modality for g in gallery)
    if any(counts[DIRECTIONS[q.query_type][1]] < 10 for q in queries):
        raise ValueError("Fewer than ten candidates in a target modality")
    return queries, gallery, member

def validate_records(records, store):
    if not records:
        raise ValueError("Empty record set")
    for field in ["image_id", "image_path"]:
        values = [getattr(r, field) for r in records]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field}")
    available = set(store.names)
    if any(r.image_path not in available for r in records):
        raise ValueError("Metadata references a missing image")

def decode_image(payload):
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "TIFF" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Expected a single-frame TIFF")
        image.load()
        array = np.asarray(image).copy()
        mode = image.mode
    if array.dtype != np.uint8 or array.shape != (256, 256, 3):
        raise ValueError(f"Expected uint8 256x256x3, got {array.dtype} {array.shape} ({mode})")
    return array
