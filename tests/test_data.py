"""Small exposure fixtures; no real KuaiRand download or model training."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "sska_rec" / "data.py"
FILES = [
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
]
LABELS = ["click", "like", "follow", "forward"]


class KuaiRandDataTests(unittest.TestCase):
    def setUp(self):
        # The Windows app sandbox forbids rename in its redirected default temp.
        self.temp = tempfile.TemporaryDirectory(dir=MODULE_PATH.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "encoded"
        self.config = {"log_files": FILES, "min_positive_per_split": 1, "chunksize": 7}
        self.records = []
        for day in range(1, 21):
            for j in range(4):
                self.records.append(
                    {
                        "date": f"202204{day:02d}",
                        "time_ms": pd.Timestamp(
                            f"2022-04-{day:02d} 12:00", tz="Asia/Shanghai"
                        ).value
                        // 1_000_000
                        + (3 - j),
                        "user_id": "known" if day <= 14 else "validation_only",
                        "video_id": f"video_{j}",
                        "tab": j % 2,
                        "hourmin": 1200 + j,
                        "duration_ms": float(day),
                        "play_time_ms": 10_000_000 + j,
                        "long_view": j % 2,
                        **{f"is_{name}": (j + k) % 2 for k, name in enumerate(LABELS)},
                    }
                )
        # An identical repeated exposure must survive, with its own row ID.
        self.records.append(dict(self.records[0]))
        self.write_logs()

    def write_logs(self):
        chunks = [self.records[i::3] for i in range(3)]
        for name, rows in zip(FILES, chunks):
            pd.DataFrame(rows).to_csv(self.raw / name, index=False)
        self.source = pd.concat(
            [pd.read_csv(self.raw / name) for name in FILES], ignore_index=True
        )

    def prepare(self, config=None):
        self.assertTrue(
            MODULE_PATH.exists(), "KuaiRand data preparation has not been implemented"
        )
        spec = importlib.util.spec_from_file_location(
            "continual_kuairand_data", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.prepare_kuairand(
            self.raw, self.output, self.config if config is None else config
        )

    def load(self):
        return json.loads(Path(self.prepare()).read_text(encoding="utf-8"))

    def test_all_exposures_and_labels_remain_aligned_in_chronological_splits(self):
        manifest = self.load()
        self.assertEqual(manifest["labels"], LABELS)
        self.assertEqual(
            manifest["features"]["names"],
            ["user_id", "video_id", "tab", "hour", "duration_ms"],
        )
        self.assertEqual(
            [manifest["splits"][s]["rows"] for s in ["train", "val", "test"]],
            [57, 12, 12],
        )
        all_ids = []
        for split, dates in [
            ("train", range(1, 15)),
            ("val", range(15, 18)),
            ("test", range(18, 21)),
        ]:
            info = manifest["splits"][split]
            x = np.load(self.output / info["x"], mmap_mode="r")
            y = np.load(self.output / info["y"], mmap_mode="r")
            rowid = np.load(self.output / info["rowid"])
            self.assertEqual(x.dtype, np.dtype("float32"))
            self.assertEqual(y.dtype, np.dtype("float32"))
            self.assertEqual(x.shape[1], 5)
            expected = self.source[
                self.source.date.astype(str).isin([f"202204{d:02d}" for d in dates])
            ]
            expected = expected.assign(rowid=expected.index).sort_values(
                ["date", "time_ms", "rowid"]
            )
            np.testing.assert_array_equal(rowid, expected.rowid.to_numpy())
            np.testing.assert_array_equal(
                y, self.source.loc[rowid, [f"is_{s}" for s in LABELS]].to_numpy()
            )
            for i, name in enumerate(LABELS):
                self.assertEqual(info["labels"][name]["positive"], int(y[:, i].sum()))
                self.assertEqual(
                    info["labels"][name]["negative"], len(y) - int(y[:, i].sum())
                )
            all_ids.extend(rowid.tolist())
        self.assertEqual(sorted(all_ids), list(range(81)))
        self.assertEqual(manifest["splits"]["train"]["date_max"], "2022-04-14")
        self.assertEqual(manifest["splits"]["val"]["date_min"], "2022-04-15")
        self.assertEqual(manifest["exposure_policy"]["negative_sampling"], False)
        self.assertEqual(manifest["exposure_policy"]["deduplicate"], False)

    def test_vocabularies_and_numeric_normalization_use_train_only(self):
        manifest = self.load()
        features = manifest["features"]
        user = features["categorical"]["user_id"]
        self.assertEqual(user["mapping"], {"known": 1})
        self.assertEqual(user["unknown_index"], 0)
        train = self.source[self.source.date <= 20220414]
        normalizer = features["numeric"]["duration_ms"]
        self.assertAlmostEqual(normalizer["mean"], train.duration_ms.mean())
        self.assertAlmostEqual(normalizer["std"], train.duration_ms.std(ddof=0))
        val_info = manifest["splits"]["val"]
        x = np.load(self.output / val_info["x"])
        rowid = np.load(self.output / val_info["rowid"])
        np.testing.assert_array_equal(x[:, 0], 0)
        expected = (
            self.source.loc[rowid, "duration_ms"] - normalizer["mean"]
        ) / normalizer["std"]
        np.testing.assert_allclose(x[:, -1], expected.to_numpy(), rtol=1e-6)
        self.assertNotIn("play_time_ms", features["names"])
        self.assertNotIn("long_view", features["names"])
        self.assertNotIn("date", features["names"])

    def test_feedback_or_unapproved_feature_is_rejected(self):
        for name in [
            "is_click",
            "play_time_ms",
            "profile_stay_time",
            "long_view",
            "date",
            "new_statistic",
        ]:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "allowlist|not allowed"
            ):
                self.prepare({**self.config, "numeric_features": [name]})

    def test_single_class_split_blocks_before_encoded_files_are_written(self):
        for row in self.records:
            if int(row["date"]) >= 20220418:
                row["is_forward"] = 0
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "class|positive"):
            self.prepare()
        audit = json.loads((self.output / "audit.json").read_text())
        self.assertEqual(audit["status"], "blocked")
        self.assertEqual(audit["splits"]["test"]["labels"]["forward"]["positive"], 0)
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertFalse(list(self.output.glob("*.npy")))

    def test_minimum_positive_requirement_is_not_silently_relaxed(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.prepare({**self.config, "min_positive_per_split": 20})
        audit = json.loads((self.output / "audit.json").read_text())
        self.assertEqual(audit["status"], "blocked")
        self.assertEqual(audit["min_positive_per_split"], 20)

    def test_invalid_hhmm_or_binary_label_fails_closed(self):
        self.records[0]["hourmin"] = 1267
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "hourmin|HHMM"):
            self.prepare()
        self.records[0]["hourmin"] = 1200
        self.records[0]["is_like"] = 2
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "is_like|binary"):
            self.prepare()

    def test_input_hashes_and_cache_identity_prevent_reencoding(self):
        manifest = self.load()
        self.assertEqual(len(manifest["inputs"]), 3)
        for item in manifest["inputs"]:
            p = Path(item["path"])
            self.assertEqual(item["sha256"], hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertEqual(len(manifest["preproc_hash"]), 64)
        times = {p.name: p.stat().st_mtime_ns for p in self.output.glob("*.npy")}
        again = self.load()
        self.assertEqual(manifest["preproc_hash"], again["preproc_hash"])
        self.assertEqual(
            times, {p.name: p.stat().st_mtime_ns for p in self.output.glob("*.npy")}
        )
        self.records[0]["tab"] = 9
        self.write_logs()
        with self.assertRaisesRegex(ValueError, "identity|different|changed"):
            self.prepare()

    def test_missing_input_never_creates_a_success_manifest(self):
        (self.raw / FILES[1]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_csv_reader_is_closed_when_a_label_is_invalid(self):
        self.records[0]["is_click"] = 3
        self.write_logs()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ResourceWarning)
            with self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse(
            [w for w in observed if issubclass(w.category, ResourceWarning)]
        )

    def test_missing_values_and_constant_numeric_feature_have_finite_encoding(self):
        for row in self.records:
            row["duration_ms"] = 5 if int(row["date"]) <= 20220414 else None
        self.records[0]["duration_ms"] = None
        self.records[0]["user_id"] = None
        self.write_logs()
        manifest = self.load()
        normalizer = manifest["features"]["numeric"]["duration_ms"]
        self.assertEqual(normalizer["mean"], 5)
        self.assertEqual(normalizer["std"], 1)
        for split in ["train", "val", "test"]:
            x = np.load(self.output / manifest["splits"][split]["x"])
            self.assertTrue(np.isfinite(x).all())
            np.testing.assert_array_equal(x[:, -1], 0)

    def test_cache_corruption_blocks_reuse(self):
        manifest = self.load()
        array = self.output / manifest["splits"]["val"]["x"]
        with array.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"X")
        with self.assertRaisesRegex(ValueError, "changed|corrupt"):
            self.prepare()

    def test_real_epoch_day_is_authoritative_when_raw_date_crosses_midnight(self):
        # Move the ordinary fixture to Apr 16-May 5; its train boundary is Apr 29.
        for row in self.records:
            day = pd.to_datetime(row["date"], format="%Y%m%d") + pd.Timedelta(days=15)
            row["date"] = day.strftime("%Y%m%d")
            row["time_ms"] += 15 * 86_400_000
        # Observed official example: raw date=Apr 30, epoch=Apr 29 23:09 UTC+8.
        # Repeat that exact timestamp under Apr 29 as well: they MUST stay together.
        self.records[52]["time_ms"] = 1651244944324
        self.records[56]["time_ms"] = 1651244944324
        self.write_logs()
        manifest = self.load()
        self.assertEqual(manifest["time_definition"]["timezone"], "Asia/Shanghai")
        self.assertEqual(manifest["time_definition"]["source"], "time_ms")
        self.assertEqual(manifest["raw_date_audit"]["mismatch_rows"], 1)
        self.assertEqual(manifest["splits"]["train"]["rows"], 58)
        self.assertEqual(manifest["splits"]["train"]["date_max"], "2022-04-29")
        train_ids = set(
            np.load(self.output / manifest["splits"]["train"]["rowid"]).tolist()
        )
        duplicate_ids = set(
            self.source.index[self.source.time_ms == 1651244944324].tolist()
        )
        self.assertEqual(len(duplicate_ids), 2)
        self.assertTrue(duplicate_ids <= train_ids)
        for left, right in [("train", "val"), ("val", "test")]:
            self.assertLess(
                manifest["splits"][left]["time_ms_max"],
                manifest["splits"][right]["time_ms_min"],
            )


if __name__ == "__main__":
    unittest.main()
