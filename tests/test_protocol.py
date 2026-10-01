"""Synthetic regression checks for leakage, official schema and truncated AP."""
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import pickle
import shutil
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from track6.audit import audit, near_duplicates
from track6.data import Record, Store, decode_image, load_labels, load_task, safe_relative, strict_json, write_json
from track6.metrics import average_precision_at_k, cosine_top10, direction_metrics, evaluate
from track6.protocol import load_split, make_split
from track6.submission import validate_submission
from tools.inspect_archive import inspect

def tiff(rng, sar=False):
    pixels = rng.integers(0,256,size=(256,256,3),dtype=np.uint8)
    if sar:
        pixels[:] = pixels[...,:1]
    f = io.BytesIO()
    Image.fromarray(pixels).save(f, format="TIFF", compression="tiff_deflate")
    return f.getvalue()

def fixture(path):
    rng = np.random.default_rng(77)
    labels, queries, gallery = [], [], []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for identity in range(30):
            for modality, copies in [("optical",2),("sar",1)]:
                for copy in range(copies):
                    image = f"{modality}_{identity:03}_{copy}"
                    rel = f"{modality}/{image}.tif"
                    labels.append([image,f"ship_{identity}",modality,rel])
                    z.writestr("训练数据/"+rel,tiff(rng,modality=="sar"))
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["image_id","ship_id","modality","image_path"])
        writer.writerows(labels)
        z.writestr("训练数据/labels.csv",buffer.getvalue())
        for i,direction in enumerate(["O2S","S2O","O2O"]):
            q = {"query_id":f"q{i}","image_path":f"query/q{i}.tif","query_type":direction}
            queries.append(q)
            z.writestr("初赛测试数据/"+q["image_path"],tiff(rng,direction=="S2O"))
        for modality in ["optical","sar"]:
            for i in range(10):
                g = {"image_id":f"g_{modality}_{i}","image_path":f"gallery/g_{modality}_{i}.tif","modality":modality}
                gallery.append(g)
                z.writestr("初赛测试数据/"+g["image_path"],tiff(rng,modality=="sar"))
        z.writestr("初赛测试数据/task.json",json.dumps({"queries":queries,"gallery":gallery}))

class MetricsTests(unittest.TestCase):
    def test_ap_known_ranks(self):
        # positives at ranks 2 and 4: (1/2 + 2/4)/2 = 1/2.
        self.assertEqual(average_precision_at_k([0,1,0,1],2,denominator="min_relevant_k"),0.5)
    def test_denominator_more_than_ten(self):
        self.assertEqual(average_precision_at_k([1]*10,20,denominator="min_relevant_k"),1)
        self.assertEqual(average_precision_at_k([1]*10,20,denominator="all_relevant"),0.5)
    def test_zero_relevance_rejected(self):
        with self.assertRaises(ValueError):
            average_precision_at_k([0]*10,0,denominator="all_relevant")
    def test_duplicate_rank_rejected(self):
        with self.assertRaises(ValueError):
            direction_metrics({"q":["a"]*10},{"q":{"a"}},denominator="all_relevant")
    def test_cosine_ties_and_nonfinite(self):
        ids = [str(i) for i in reversed(range(10))]
        self.assertEqual(cosine_top10([[1,0]],[[1,0]]*10,["q"],ids)["q"],sorted(ids))
        for bad in [[[np.nan,1]],[[0,0]]]:
            with self.assertRaises(ValueError):
                cosine_top10(bad,[[1,0]]*10,["q"],ids)
    def test_near_hash_requires_pixel_correlation(self):
        rows = [{"image_id":"a","passed":True,"constant":False,"gray_thumbnail_degenerate":False,
                 "dhash":"0"*16,"pixel_sha256":"a","_thumbnail":np.array([1.,0.])},
                {"image_id":"b","passed":True,"constant":False,"gray_thumbnail_degenerate":False,
                 "dhash":"0"*16,"pixel_sha256":"b","_thumbnail":np.array([0.,1.])}]
        self.assertEqual(near_duplicates(rows),[])

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / "official-shaped.zip"
        fixture(cls.source)
        cls.audit_dir = cls.root / "audit"
        cls.summary = audit(cls.source,cls.audit_dir,2)
        cls.split_path = cls.root / "split.json"
        cls.split_summary = make_split(cls.source,cls.audit_dir,cls.split_path,99,0.5)
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def test_all_images_and_crc(self):
        self.assertTrue(self.summary["passed"])
        self.assertEqual(self.summary["roles"]["train"]["images"],90)
        self.assertEqual(self.summary["train_both_modalities"],30)
        self.assertTrue(inspect(self.source,True)["passed"])
    def test_safe_paths(self):
        for path in ["../escape","/abs","C:/file","a/../b","a\\b","a/./b"]:
            with self.subTest(path=path),self.assertRaises(ValueError):
                safe_relative(path)
        path = self.root / "unsafe.zip"
        with zipfile.ZipFile(path,"w") as z:
            z.writestr("../escape",b"x")
        with self.assertRaises(ValueError):
            Store(path)
    def test_decode_does_not_hide_bad_dtype(self):
        f = io.BytesIO()
        Image.fromarray(np.zeros((256,256),dtype=np.uint16)).save(f,format="TIFF")
        with self.assertRaises(ValueError):
            decode_image(f.getvalue())
    def test_duplicate_json_keys(self):
        with self.assertRaises(ValueError):
            strict_json('{"q":[],"q":[]}')
    def test_json_artifacts_have_portable_newlines(self):
        path = self.root / "portable.json"
        write_json(path,{"note":"跨平台"})
        self.assertNotIn(b"\r\n",path.read_bytes())
    def test_parallel_archive_reads_match_serial_bytes(self):
        store = Store(self.source)
        records,_ = load_labels(store)
        expected = {r.image_path:store.read(r.image_path) for r in records}
        names = [r.image_path for r in records]*5
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(store.read,names))
        self.assertTrue(all(payload==expected[name] for name,payload in zip(names,actual)))
    def test_store_pickle_reopens_archive_in_worker(self):
        store = Store(self.source)
        member = store.unique("labels.csv")
        expected = store.read(member)
        worker_store = pickle.loads(pickle.dumps(store))
        self.assertEqual(worker_store.read(member),expected)
    def test_identity_split_and_repeatability(self):
        store, records, split = load_split(self.source,self.split_path)
        self.assertFalse(set(split["train_identities"]) & set(split["validation_identities"]))
        self.assertFalse(set(split["train_images"]) & set(split["validation_images"]))
        other = self.root / "same-split.json"
        again = make_split(self.source,self.audit_dir,other,99,0.5)
        self.assertEqual(self.split_summary["split_sha256"],again["split_sha256"])
    def test_audit_timing_does_not_change_split(self):
        other_audit = self.root / "audit-other-time"
        shutil.copytree(self.audit_dir,other_audit)
        summary = json.loads((other_audit / "summary.json").read_text(encoding="utf-8"))
        summary["elapsed_seconds"] = 999.0
        summary["checked_at_utc"] = "2030-01-01T00:00:00Z"
        write_json(other_audit / "summary.json",summary)
        other = self.root / "same-decisions.json"
        again = make_split(self.source,other_audit,other,99,0.5)
        self.assertEqual(self.split_summary["split_sha256"],again["split_sha256"])
    def test_tampered_identity_split_rejected(self):
        split = strict_json(self.split_path.read_text(encoding="utf-8"))
        split["train_images"].append(split["validation_images"][0])
        bad = self.root / "bad-split.json"
        write_json(bad,split)
        with self.assertRaises(ValueError):
            load_split(self.source,bad)
    def test_query_self_match_rejected(self):
        split = strict_json(self.split_path.read_text(encoding="utf-8"))
        split["tasks"]["O2O"]["gallery"].append(split["tasks"]["O2O"]["queries"][0])
        bad = self.root / "bad-self.json"
        write_json(bad,split)
        with self.assertRaises(ValueError):
            load_split(self.source,bad)
    def test_valid_submission(self):
        queries,gallery,_ = load_task(Store(self.source))
        prediction = {q.image_id:[g.image_id for g in gallery if g.modality == ("sar" if q.query_type=="O2S" else "optical")]
                      for q in queries}
        out = self.root / "prediction.json"
        write_json(out,prediction)
        self.assertTrue(validate_submission(self.source,out)["passed"])
    def test_invalid_submissions(self):
        queries,gallery,_ = load_task(Store(self.source))
        valid = {q.image_id:[g.image_id for g in gallery if g.modality == ("sar" if q.query_type=="O2S" else "optical")]
                 for q in queries}
        variants = []
        p = dict(valid); p.pop("q0"); variants.append(p)
        p = dict(valid); p["extra"] = valid["q0"]; variants.append(p)
        p = dict(valid); p["q0"] = valid["q0"][:9]; variants.append(p)
        p = dict(valid); p["q0"] = [valid["q0"][0]]*10; variants.append(p)
        p = dict(valid); p["q0"] = ["unknown"] + valid["q0"][1:]; variants.append(p)
        p = dict(valid); p["q0"] = valid["q1"]; variants.append(p)
        p = dict(valid); p["q0"] = {"scores":valid["q0"]}; variants.append(p)
        out = self.root / "bad-prediction.json"
        for i,p in enumerate(variants):
            with self.subTest(case=i):
                write_json(out,p)
                with self.assertRaises(ValueError):
                    validate_submission(self.source,out)
    def test_perfect_metrics_three_directions(self):
        _,records,split = load_split(self.source,self.split_path)
        by_id = {r.image_id:r for r in records}
        rankings = {}
        for d,task in split["tasks"].items():
            rankings[d] = {}
            for q in task["queries"]:
                positive = [g for g in task["gallery"] if by_id[g].ship_id==by_id[q].ship_id]
                negative = [g for g in task["gallery"] if g not in positive]
                rankings[d][q] = (positive+negative)[:10]
        result = evaluate(rankings,split,records,denominator="min_relevant_k")
        self.assertEqual(result["weighted_score"],1)

if __name__ == "__main__":
    unittest.main()
