"""Leakage-controlled KuaiRand-Pure exposure preparation.

Raw CSVs are read in place. Only encoded NumPy arrays and provenance are cached.
``rowid`` identifies the original row in the configured file-concatenation order;
arrays are ordered by (epoch-derived local day, time_ms, rowid). The raw ``date``
column is an audited logging bucket, never the authoritative split key.
"""

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


LABELS = ("click", "like", "follow", "forward")
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
)
CATEGORICAL_ALLOWLIST = ("user_id", "video_id", "tab", "hour")
NUMERIC_ALLOWLIST = ("duration_ms",)
SPLITS = ("train", "val", "test")
TIME_DEFINITION = {
    "source": "time_ms",
    "unit": "Unix epoch milliseconds",
    "timezone": "Asia/Shanghai",
    "split_key": "complete local calendar day",
    "raw_date_role": "audit_only",
    "hour_feature": "original hourmin // 100 logging bucket",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _configuration(config):
    config = dict(config or {})
    allowed_keys = {
        "log_files",
        "categorical_features",
        "numeric_features",
        "split_fractions",
        "min_positive_per_split",
        "chunksize",
    }
    if set(config) - allowed_keys:
        raise ValueError(
            f"Unknown preprocessing options: {sorted(set(config) - allowed_keys)}"
        )
    result = {
        "log_files": list(config.get("log_files", LOG_FILES)),
        "categorical_features": list(
            config.get("categorical_features", CATEGORICAL_ALLOWLIST)
        ),
        "numeric_features": list(config.get("numeric_features", NUMERIC_ALLOWLIST)),
        "split_fractions": list(config.get("split_fractions", [0.7, 0.15, 0.15])),
        "min_positive_per_split": config.get("min_positive_per_split", 20),
        "chunksize": config.get("chunksize", 100_000),
    }
    for key, allowlist in [
        ("categorical_features", CATEGORICAL_ALLOWLIST),
        ("numeric_features", NUMERIC_ALLOWLIST),
    ]:
        values = result[key]
        if len(set(values)) != len(values) or not set(values).issubset(allowlist):
            raise ValueError(
                f"{key} not allowed by pre-exposure feature allowlist: {values}"
            )
    if not result["categorical_features"] and not result["numeric_features"]:
        raise ValueError("At least one allowlist feature is required")
    if result["split_fractions"] != [0.7, 0.15, 0.15]:
        raise ValueError(
            "This protocol fixes whole-date split_fractions to [0.7, 0.15, 0.15]"
        )
    for key in ["min_positive_per_split", "chunksize"]:
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], int)
            or result[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if not result["log_files"] or any(
        not isinstance(s, str) or not s for s in result["log_files"]
    ):
        raise ValueError("log_files must explicitly identify nonempty exposure logs")
    return result


def _inputs(input_root, config):
    root = Path(input_root).resolve()
    items = []
    seen = set()
    for name in config["log_files"]:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Input is outside the specified input_root: {path}")
        if path in seen:
            raise ValueError(f"Repeated log file in config: {path}")
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"KuaiRand-Pure exposure log missing: {path}")
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed while hashing: {path}")
        items.append(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "logging_policy": (
                    "random" if path.name.startswith("log_random_") else "standard"
                ),
            }
        )
    return items


def _chunks(inputs, config):
    required = set(
        [
            "date",
            "time_ms",
            "hourmin",
            *[f"is_{s}" for s in LABELS],
            *config["numeric_features"],
            *[s for s in config["categorical_features"] if s != "hour"],
        ]
    )
    offset = 0
    for item in inputs:
        path = item["path"]
        columns = list(pd.read_csv(path, nrows=0).columns)
        if not required.issubset(columns):
            raise ValueError(
                f"Missing required columns in {path}: {sorted(required - set(columns))}"
            )
        with pd.read_csv(
            path,
            usecols=sorted(required),
            dtype="string",
            chunksize=config["chunksize"],
        ) as reader:
            for frame in reader:
                raw_date = frame["date"].str.replace("-", "", regex=False)
                dates = pd.to_datetime(raw_date, format="%Y%m%d", errors="coerce")
                raw_days = (
                    dates.dt.strftime("%Y%m%d").fillna("0").to_numpy(dtype=np.int32)
                )
                times = pd.to_numeric(frame["time_ms"], errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                if (
                    not np.isfinite(times).all()
                    or (times < 0).any()
                    or (times != np.floor(times)).any()
                    or (times > 2**53).any()
                ):
                    raise ValueError(
                        f"time_ms must be finite nonnegative integer timestamps: {path}"
                    )
                times = times.astype(np.int64)
                local_time = pd.to_datetime(times, unit="ms", utc=True).tz_convert(
                    TIME_DEFINITION["timezone"]
                )
                days = local_time.strftime("%Y%m%d").to_numpy(dtype=np.int32)
                frame["_raw_date_key"] = raw_days
                frame.attrs["source_path"] = path
                hhmm = pd.to_numeric(frame["hourmin"], errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                if (
                    not np.isfinite(hhmm).all()
                    or (hhmm != np.floor(hhmm)).any()
                    or (hhmm < 0).any()
                    or (hhmm > 2359).any()
                    or (hhmm % 100 >= 60).any()
                ):
                    raise ValueError(
                        f"Invalid hourmin: expected HHMM with hours 00-23 and minutes 00-59: {path}"
                    )
                frame["hour"] = pd.Series(
                    (hhmm.astype(np.int64) // 100).astype(str),
                    index=frame.index,
                    dtype="string",
                )
                ys = []
                for label in LABELS:
                    column = f"is_{label}"
                    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                        dtype=np.float64, na_value=np.nan
                    )
                    if not np.isin(values, [0.0, 1.0]).all():
                        raise ValueError(
                            f"{column} must contain observed binary 0/1 labels; missing labels are not negatives: {path}"
                        )
                    ys.append(values.astype(np.float32))
                for feature in config["numeric_features"]:
                    numeric = pd.to_numeric(frame[feature], errors="coerce").to_numpy(
                        dtype=np.float64, na_value=np.nan
                    )
                    invalid_text = frame[feature].notna().to_numpy() & np.isnan(numeric)
                    if (
                        invalid_text.any()
                        or np.isinf(numeric).any()
                        or (numeric[np.isfinite(numeric)] < 0).any()
                    ):
                        raise ValueError(
                            f"{feature} must be a nonnegative numeric value or missing: {path}"
                        )
                    frame[feature] = numeric
                yield frame, days, times, np.column_stack(ys), offset
                offset += len(frame)


def _date_partition(day_counts):
    days = sorted(day_counts)
    if len(days) < 3:
        raise ValueError(
            "At least three distinct dates are needed for chronological train/val/test"
        )
    first = min(max(1, math.floor(len(days) * 0.7)), len(days) - 2)
    second = min(max(first + 1, math.floor(len(days) * 0.85)), len(days) - 1)
    return {"train": days[:first], "val": days[first:second], "test": days[second:]}


def _date_text(value):
    text = str(value)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _reuse(manifest_path, preproc_hash):
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("preproc_hash") != preproc_hash:
        raise ValueError(
            "Cached preprocessing identity is different; use a new output_root rather than overwrite evidence"
        )
    for info in manifest["splits"].values():
        for kind in ["x", "y", "rowid"]:
            path = manifest_path.parent / info[kind]
            if not path.is_file() or _sha256(path) != info["sha256"][kind]:
                raise ValueError(f"Cached array missing or changed: {path}")
    return True


def prepare_kuairand(input_root, output_root, config=None):
    """Prepare all configured exposures, or reuse an exactly matching cache.

    Epoch timestamps define Asia/Shanghai calendar days. Whole days are
    partitioned by unique-day index: floor(.70 * D) and
    floor(.85 * D), subject to three nonempty splits. This does not promise
    exact exposure-count fractions. Single-class or underpowered labels block
    preparation and leave audit.json; targets and split boundaries never change
    automatically in response to their prevalence.
    """
    output = Path(output_root).resolve()
    audit = {"dataset": "KuaiRand-Pure", "status": "blocked"}
    try:
        settings = _configuration(config)
        inputs = _inputs(input_root, settings)
        audit.update(
            {
                "config": settings,
                "inputs": inputs,
                "min_positive_per_split": settings["min_positive_per_split"],
            }
        )
        source_hash = _sha256(Path(__file__))
        fingerprint = {
            "version": "kuairand-pure-exposures-v2-epoch-days",
            "source_sha256": source_hash,
            "time_definition": TIME_DEFINITION,
            "config": settings,
            "inputs": [{k: item[k] for k in ["path", "sha256"]} for item in inputs],
        }
        preproc_hash = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()
        audit["preproc_hash"] = preproc_hash
        manifest_path = output / "manifest.json"
        if _reuse(manifest_path, preproc_hash):
            return manifest_path

        audit["time_definition"] = TIME_DEFINITION
        day_counts = Counter()
        day_positive = {}
        day_times = {}
        raw_date_audit = {
            "rows": 0,
            "mismatch_rows": 0,
            "invalid_rows": 0,
            "examples": [],
            "by_file": {
                item["path"]: {"rows": 0, "mismatch_rows": 0, "invalid_rows": 0}
                for item in inputs
            },
        }
        for frame, days, times, y, offset in _chunks(inputs, settings):
            raw_days = frame["_raw_date_key"].to_numpy(dtype=np.int32)
            invalid = raw_days == 0
            mismatch = (raw_days != days) & ~invalid
            counters = raw_date_audit["by_file"][frame.attrs["source_path"]]
            for target in [raw_date_audit, counters]:
                target["rows"] += len(days)
                target["mismatch_rows"] += int(mismatch.sum())
                target["invalid_rows"] += int(invalid.sum())
            for i in np.flatnonzero(mismatch | invalid)[
                : max(0, 10 - len(raw_date_audit["examples"]))
            ]:
                raw_date_audit["examples"].append(
                    {
                        "source": frame.attrs["source_path"],
                        "rowid": offset + int(i),
                        "raw_date": str(frame["date"].iloc[i]),
                        "time_ms": int(times[i]),
                        "epoch_local_day": _date_text(int(days[i])),
                    }
                )
            for day in np.unique(days):
                mask = days == day
                key = int(day)
                day_counts[key] += int(mask.sum())
                day_positive.setdefault(key, np.zeros(len(LABELS), dtype=np.int64))
                day_positive[key] += y[mask].sum(axis=0).astype(np.int64)
                low, high = int(times[mask].min()), int(times[mask].max())
                prior = day_times.get(key, [low, high])
                day_times[key] = [min(low, prior[0]), max(high, prior[1])]
        audit["raw_date_audit"] = raw_date_audit
        partition = _date_partition(day_counts)
        rows = sum(day_counts.values())
        split_info = {}
        violations = []
        for split, dates in partition.items():
            size = sum(day_counts[d] for d in dates)
            positives = sum(
                (day_positive[d] for d in dates), np.zeros(4, dtype=np.int64)
            )
            split_info[split] = {
                "rows": size,
                "fraction_of_exposures": size / rows,
                "dates": [_date_text(d) for d in dates],
                "date_min": _date_text(dates[0]),
                "date_max": _date_text(dates[-1]),
                "time_ms_min": min(day_times[d][0] for d in dates),
                "time_ms_max": max(day_times[d][1] for d in dates),
                "labels": {
                    label: {
                        "positive": int(positives[i]),
                        "negative": size - int(positives[i]),
                    }
                    for i, label in enumerate(LABELS)
                },
            }
            for label, counts in split_info[split]["labels"].items():
                if counts["positive"] == 0 or counts["negative"] == 0:
                    violations.append(f"{split}/{label} is single-class: {counts}")
                elif counts["positive"] < settings["min_positive_per_split"]:
                    violations.append(
                        f'{split}/{label} positive count {counts["positive"]} is below minimum {settings["min_positive_per_split"]}'
                    )
        audit["splits"] = split_info
        for earlier, later in [("train", "val"), ("val", "test")]:
            if split_info[earlier]["time_ms_max"] >= split_info[later]["time_ms_min"]:
                violations.append(
                    f"Overlapping time_ms across {earlier}/{later}; identical timestamps cannot cross splits"
                )
        if violations:
            audit["violations"] = violations
            raise ValueError("; ".join(violations))

        categorical = settings["categorical_features"]
        numeric = settings["numeric_features"]
        vocab = {name: set() for name in categorical}
        moments = {name: [0, 0.0, 0.0] for name in numeric}
        all_days = np.empty(rows, dtype=np.int32)
        all_times = np.empty(rows, dtype=np.int64)
        for frame, days, times, _, offset in _chunks(inputs, settings):
            all_days[offset : offset + len(days)] = days
            all_times[offset : offset + len(days)] = times
            train = np.isin(days, partition["train"])
            for name in categorical:
                vocab[name].update(
                    str(x) for x in frame.loc[train, name].dropna().unique()
                )
            for name in numeric:
                values = frame.loc[train, name].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                count, mean, m2 = moments[name]
                batch_mean = float(values.mean())
                delta = batch_mean - mean
                combined = count + len(values)
                m2 += (
                    float(np.square(values - batch_mean).sum())
                    + delta * delta * count * len(values) / combined
                )
                moments[name] = [combined, mean + delta * len(values) / combined, m2]
        features = {
            "names": categorical + numeric,
            "categorical": {},
            "numeric": {},
            "fit_split": "train",
        }
        for name, values in vocab.items():
            if len(values) >= 2**24:
                raise ValueError(
                    f"{name} category IDs exceed exact float32 integer representation"
                )
            features["categorical"][name] = {
                "mapping": {v: i + 1 for i, v in enumerate(sorted(values))},
                "unknown_index": 0,
                "missing_index": 0,
                "cardinality": len(values) + 1,
            }
        for name, (count, mean, m2) in moments.items():
            if not count:
                raise ValueError(
                    f"{name} has no observed training values for normalization"
                )
            features["numeric"][name] = {
                "mean": mean,
                "std": math.sqrt(max(0.0, m2 / count)) or 1.0,
                "observed_train_rows": count,
                "missing": "training_mean",
            }

        # Metadata is small relative to the raw CSV; model arrays are written as memmaps.
        rank = np.empty(rows, dtype=np.int64)
        order = {}
        for split, dates in partition.items():
            ids = np.flatnonzero(np.isin(all_days, dates))
            ids = ids[np.lexsort((ids, all_times[ids], all_days[ids]))]
            order[split] = ids
            rank[ids] = np.arange(len(ids), dtype=np.int64)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as staging:
            staging = Path(staging)
            arrays = {}
            try:
                for split, info in split_info.items():
                    paths = {
                        kind: f"{split}-{kind}.npy" for kind in ["x", "y", "rowid"]
                    }
                    info.update(paths)
                    shapes = {
                        "x": (info["rows"], len(features["names"])),
                        "y": (info["rows"], 4),
                        "rowid": (info["rows"],),
                    }
                    arrays[split] = {
                        kind: np.lib.format.open_memmap(
                            staging / filename,
                            mode="w+",
                            dtype=np.int64 if kind == "rowid" else np.float32,
                            shape=shapes[kind],
                        )
                        for kind, filename in paths.items()
                    }
                    arrays[split]["rowid"][:] = order[split]
                for frame, days, _, y, offset in _chunks(inputs, settings):
                    columns = []
                    for name in categorical:
                        columns.append(
                            frame[name]
                            .map(features["categorical"][name]["mapping"])
                            .fillna(0)
                            .to_numpy(dtype=np.float32)
                        )
                    for name in numeric:
                        normalizer = features["numeric"][name]
                        values = frame[name].to_numpy(dtype=np.float64)
                        values = np.where(np.isnan(values), normalizer["mean"], values)
                        columns.append(
                            ((values - normalizer["mean"]) / normalizer["std"]).astype(
                                np.float32
                            )
                        )
                    x = np.column_stack(columns)
                    if not np.isfinite(x).all():
                        raise ValueError(
                            "Encoded features are not finite float32 values"
                        )
                    for split, dates in partition.items():
                        mask = np.isin(days, dates)
                        positions = rank[offset + np.flatnonzero(mask)]
                        arrays[split]["x"][positions] = x[mask]
                        arrays[split]["y"][positions] = y[mask]
            finally:
                for mapping in arrays.values():
                    for array in mapping.values():
                        array.flush()
                        array._mmap.close()
            for item in inputs:
                current = Path(item["path"]).stat()
                if (current.st_size, current.st_mtime_ns) != (
                    item["bytes"],
                    item["mtime_ns"],
                ):
                    raise ValueError(
                        f'Input changed during preprocessing: {item["path"]}'
                    )
            for info in split_info.values():
                info["sha256"] = {
                    kind: _sha256(staging / info[kind]) for kind in ["x", "y", "rowid"]
                }
                for kind in ["x", "y", "rowid"]:
                    os.replace(staging / info[kind], output / info[kind])
        manifest = {
            **fingerprint,
            "dataset": "KuaiRand-Pure",
            "preproc_hash": preproc_hash,
            "input_root": str(Path(input_root).resolve()),
            "inputs": inputs,
            "labels": list(LABELS),
            "label_columns": [f"is_{label}" for label in LABELS],
            "features": features,
            "splits": split_info,
            "raw_date_audit": raw_date_audit,
            "rowid": "zero-based row in configured input-file concatenation order; repeated exposures retained",
            "array_order": ["epoch_local_day", "time_ms", "rowid"],
            "split_policy": "sorted complete Asia/Shanghai days from time_ms; floor(0.70*D), floor(0.85*D); three nonempty splits",
            "exposure_policy": {
                "negative_sampling": False,
                "deduplicate": False,
                "feedback_filtered": False,
                "logging_policies_combined": True,
                "unbiased_estimation_claimed": False,
            },
        }
        _json_write(manifest_path, manifest)
        _json_write(output / "audit.json", {**audit, "status": "ready"})
        return manifest_path
    except (ValueError, FileNotFoundError) as error:
        output.mkdir(parents=True, exist_ok=True)
        _json_write(output / "audit.json", {**audit, "error": str(error)})
        raise
