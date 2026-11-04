"""Bounded MIMIC-IV extraction and train-only preprocessing for the paper."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "respiratory_rate",
    "spo2",
    "temperature_c",
)
STATIC = ("age", "female")
# Identifiers were checked against the local MIMIC-IV 3.1 d_items table.
# The resolved source labels are retained in the extraction manifest.
ITEMS = {
    220045: 0,
    220050: 1,
    220179: 1,
    220051: 2,
    220180: 2,
    220052: 3,
    220181: 3,
    220210: 4,
    220277: 5,
    223761: 6,
    223762: 6,
}
BOUNDS = np.array(
    [(20, 250), (30, 300), (10, 200), (15, 250), (1, 80), (20, 100), (25, 45)],
    dtype=float,
)
SPLITS = ("nuisance", "second_stage", "test")


def subject_partition(subject_id: int, seed: int = 2027) -> str:
    key = hashlib.sha256(f"mimiciv-v1:{seed}:{int(subject_id)}".encode()).digest()
    bucket = int.from_bytes(key[:8], "big") % 10
    return SPLITS[0 if bucket < 4 else 1 if bucket < 8 else 2]


def eligible_stays(
    stays: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    hours: int = 24,
) -> pd.DataFrame:
    if hours != 24:
        raise ValueError("protocol v2 fixes the landmark at 24 hours")
    s = stays.copy()
    for col in ("intime", "outtime"):
        s[col] = pd.to_datetime(s[col], errors="raise")
    p = patients[["subject_id", "gender", "anchor_age", "anchor_year"]]
    a = admissions[["subject_id", "hadm_id", "deathtime"]].copy()
    a["deathtime"] = pd.to_datetime(a.deathtime, errors="raise")
    s = s.merge(p, on="subject_id", validate="many_to_one")
    s = s.merge(a, on=["subject_id", "hadm_id"], validate="many_to_one")
    s["landmark"] = s.intime + pd.Timedelta(hours=hours)
    s["extraction_end"] = s.landmark + pd.Timedelta(hours=24)
    s["age"] = s.anchor_age + s.intime.dt.year - s.anchor_year
    keep = (
        (s.age >= 18)
        & (s.outtime > s.landmark)
        & (s.deathtime.isna() | (s.deathtime > s.landmark))
    )
    s = s.loc[keep].sort_values(["subject_id", "intime", "stay_id"])
    s = s.drop_duplicates("subject_id", keep="first").copy()
    if not s.gender.isin(["F", "M"]).all():
        raise ValueError("unexpected sex encoding; explicit handling required")
    s["female"] = (s.gender == "F").astype(float)
    return s.reset_index(drop=True)


def accumulate_events(chunk, stays, sums, counts, audit):
    """Use availability time, not backdated chart time, in strict left-closed bins."""
    c = chunk.loc[chunk.itemid.isin(ITEMS) & chunk.stay_id.isin(stays.stay_id)].copy()
    audit["selected_rows"] += len(c)
    if c.empty:
        return
    selected_count = len(c)
    c = c.merge(
        stays[
            [
                "stay_id",
                "subject_id",
                "intime",
                "landmark",
                "extraction_end",
                "outtime",
                "row",
            ]
        ],
        on=["stay_id", "subject_id"],
        validate="many_to_one",
    )
    if len(c) != selected_count:
        raise ValueError("source stay/subject identifiers are inconsistent")
    chart = pd.to_datetime(c.charttime, errors="coerce")
    store = pd.to_datetime(c.storetime, errors="coerce")
    available = pd.concat([chart, store], axis=1).max(axis=1)
    valid_time = chart.notna() & store.notna() & (chart >= c.intime)
    valid_time &= (
        (available >= c.intime)
        & (available < c.extraction_end)
        & (available < c.outtime)
    )
    audit["excluded_time_rows"] += int((~valid_time).sum())
    c = c.loc[valid_time].copy()
    available = available.loc[valid_time]
    if c.empty:
        return
    channel = c.itemid.map(ITEMS).to_numpy(int)
    # pandas 3 returns read-only arrays here, so ask for a writable copy before
    # the in-place Fahrenheit conversion below.
    values = pd.to_numeric(c.valuenum, errors="coerce").to_numpy(float, copy=True)
    fahrenheit = c.itemid.to_numpy() == 223761
    values[fahrenheit] = (values[fahrenheit] - 32) * (5 / 9)
    unit = c.valueuom.fillna("").astype(str).str.strip().str.lower().to_numpy()
    allowed = {
        0: {"bpm"},
        1: {"mmhg"},
        2: {"mmhg"},
        3: {"mmhg"},
        4: {"insp/min"},
        5: {"%"},
        6: {"°c", "°f", "deg c", "deg f"},
    }
    unit_ok = np.array([u in allowed[f] for u, f in zip(unit, channel)])
    unit_ok &= np.where(fahrenheit, np.isin(unit, ["°f", "deg f"]), True)
    unit_ok &= np.where(
        c.itemid.to_numpy() == 223762, np.isin(unit, ["°c", "deg c"]), True
    )
    good = np.isfinite(values) & unit_ok
    good &= (values >= BOUNDS[channel, 0]) & (values <= BOUNDS[channel, 1])
    audit["excluded_value_rows"] += int((~good).sum())
    bins = ((available - c.intime).dt.total_seconds() // (3 * 3600)).to_numpy(int)
    rows = c.row.to_numpy(int)
    np.add.at(sums, (rows[good], bins[good], channel[good]), values[good])
    np.add.at(counts, (rows[good], bins[good], channel[good]), 1)
    audit["accepted_rows"] += int(good.sum())
    pre = (available < c.landmark).to_numpy()
    audit["pre_landmark_rows"] = audit.get("pre_landmark_rows", 0) + int(
        (good & pre).sum()
    )
    audit["post_landmark_rows"] = audit.get("post_landmark_rows", 0) + int(
        (good & ~pre).sum()
    )
    # Bin right edge is strictly after each accepted record's availability time.
    boundary = c.intime + pd.to_timedelta((bins + 1) * 3, unit="h")
    violations = (available >= boundary).to_numpy()[good]
    audit["cutoff_violations"] += int(violations.sum())
    if violations.any():
        raise AssertionError("a record crossed its prediction boundary")


def extract_histories(
    raw_root, output, split_seed=2027, max_subjects=6000, chunksize=100_000
):
    """Stream source tables. Never stop a chunk scan early based on test outcomes."""
    root, out = Path(raw_root), Path(output)
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= max_subjects <= 50_000 or not 1 <= chunksize <= 200_000:
        raise ValueError("subject/chunk budget outside supported bounds")
    files = {
        "stays": root / "icu/icustays.csv.gz",
        "patients": root / "hosp/patients.csv.gz",
        "admissions": root / "hosp/admissions.csv.gz",
        "charts": root / "icu/chartevents.csv.gz",
        "items": root / "icu/d_items.csv.gz",
    }
    s = eligible_stays(
        pd.read_csv(
            files["stays"],
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        ),
        pd.read_csv(
            files["patients"],
            usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        ),
        pd.read_csv(
            files["admissions"], usecols=["subject_id", "hadm_id", "deathtime"]
        ),
    )
    s["selection_key"] = s.subject_id.map(
        lambda v: hashlib.sha256(f"cohort-v1:{split_seed}:{v}".encode()).hexdigest()
    )
    s = s.sort_values("selection_key").head(max_subjects).reset_index(drop=True)
    if s.empty:
        raise ValueError("no eligible subjects")
    s["row"] = np.arange(len(s))
    s["split"] = s.subject_id.map(lambda v: subject_partition(v, split_seed))
    items = pd.read_csv(files["items"]).set_index("itemid")
    if not set(ITEMS) <= set(items.index):
        raise ValueError("source item dictionary is incompatible")
    item_labels = {str(i): str(items.loc[i, "label"]) for i in ITEMS}
    sums = np.zeros((len(s), 16, len(FEATURES)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    audit = dict(
        selected_rows=0,
        excluded_time_rows=0,
        excluded_value_rows=0,
        accepted_rows=0,
        cutoff_violations=0,
        chunks=0,
    )
    columns = [
        "subject_id",
        "stay_id",
        "itemid",
        "charttime",
        "storetime",
        "valuenum",
        "valueuom",
    ]
    for chunk in pd.read_csv(files["charts"], usecols=columns, chunksize=chunksize):
        accumulate_events(chunk, s, sums, counts, audit)
        audit["chunks"] += 1
        if audit["chunks"] % 50 == 0:
            print(json.dumps(audit), flush=True)
    values = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 2,
        "dataset": "MIMIC-IV 3.1",
        "landmark_hours": 24,
        "extraction_hours": 48,
        "history_steps": 8,
        "grid_hours": 3,
        "features": FEATURES,
        "static_features": STATIC,
        "split_seed": split_seed,
        "audit": audit,
        "item_labels": item_labels,
        "source_files": {
            k: {
                "path": str(p.resolve()),
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for k, p in files.items()
        },
    }
    with out.open("xb") as handle:
        np.savez_compressed(
            handle,
            values=values,
            counts=counts,
            baseline=s[list(STATIC)].to_numpy(float),
            subject_id=s.subject_id.to_numpy(np.int64),
            stay_id=s.stay_id.to_numpy(np.int64),
            split=s.split.to_numpy(str),
            manifest=np.array(json.dumps(manifest)),
        )
    return manifest


@dataclass
class HistoryPreprocessor:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    static_mean: np.ndarray
    static_scale: np.ndarray

    @classmethod
    def fit(cls, values, baseline):
        values, baseline = np.asarray(values, float), np.asarray(baseline, float)
        if values.ndim != 3 or values.shape[-1] != len(FEATURES):
            raise ValueError("unexpected history dimensions")
        if len(values) == 0 or baseline.shape != (len(values), len(STATIC)):
            raise ValueError("empty or misaligned baseline cohort")
        if not np.isfinite(baseline).all() or np.isinf(values).any():
            raise ValueError("invalid source values")
        median = np.nanmedian(values, axis=(0, 1))
        if not np.isfinite(median).all():
            raise ValueError("a feature has no training observations")
        filled = cls.fill(values, median)
        return cls(
            median,
            filled.mean(axis=(0, 1)),
            np.maximum(filled.std(axis=(0, 1)), 1e-6),
            baseline.mean(axis=0),
            np.maximum(baseline.std(axis=0), 1e-6),
        )

    @staticmethod
    def fill(values, median):
        result = np.array(values, dtype=float, copy=True)
        previous = np.broadcast_to(median, (len(result), len(median))).copy()
        for k in range(result.shape[1]):
            previous = np.where(np.isnan(result[:, k]), previous, result[:, k])
            result[:, k] = previous
        return result

    def transform(self, values, baseline):
        x = (self.fill(values, self.median) - self.mean) / self.scale
        w = (np.asarray(baseline, float) - self.static_mean) / self.static_scale
        if not np.isfinite(x).all() or not np.isfinite(w).all():
            raise ValueError("nonfinite transformed history")
        return np.clip(x, -5, 5), np.clip(w, -5, 5)

    def as_dict(self):
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}


def main():
    import argparse

    from src.protocols import load_protocol

    parser = argparse.ArgumentParser(
        description="Prepare the 6,000-patient MIMIC-IV history archive."
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    dataset = load_protocol("semi_synthetic")["dataset"]
    manifest = extract_histories(
        args.raw_root,
        args.output,
        split_seed=dataset["selection_seed"],
        max_subjects=dataset["subjects"],
        chunksize=args.chunksize,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
