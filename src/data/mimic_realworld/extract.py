from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.mimic_semisynthetic.histories import ITEMS, eligible_stays

VASOPRESSORS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    222315: "vasopressin",
    221749: "phenylephrine",
    221662: "dopamine",
}
NOREPINEPHRINE = 221906
GRID_HOURS = 3
N_BINS = 32
ENTRY = 7
COHORT_WINDOW = (21.0, 24.0)
CHART_COLUMNS = [
    "subject_id",
    "stay_id",
    "itemid",
    "charttime",
    "storetime",
    "valuenum",
    "valueuom",
]
INPUT_COLUMNS = [
    "subject_id",
    "stay_id",
    "itemid",
    "starttime",
    "endtime",
    "storetime",
    "rate",
    "rateuom",
]


def raw_files(raw_root):
    root = Path(raw_root)
    return {
        "stays": root / "icu/icustays.csv.gz",
        "patients": root / "hosp/patients.csv.gz",
        "admissions": root / "hosp/admissions.csv.gz",
        "charts": root / "icu/chartevents.csv.gz",
        "inputs": root / "icu/inputevents.csv.gz",
        "items": root / "icu/d_items.csv.gz",
    }


def read_vasopressor_rows(path, stay_ids=None):
    rows = pd.read_csv(path, usecols=INPUT_COLUMNS)
    rows = rows.loc[rows.itemid.isin(VASOPRESSORS)].copy()
    if stay_ids is not None:
        rows = rows.loc[rows.stay_id.isin(stay_ids)].copy()
    for column in ("starttime", "endtime", "storetime"):
        rows[column] = pd.to_datetime(rows[column], errors="raise")
    if (rows.endtime < rows.starttime).any():
        raise ValueError("an infusion ends before it starts")
    return rows.reset_index(drop=True)


def overlapping(rows, intime, lower_hours, upper_hours):
    start = intime + pd.to_timedelta(lower_hours, unit="h")
    end = intime + pd.to_timedelta(upper_hours, unit="h")
    return (rows.starttime < end) & (rows.endtime > start)


def build_cohort(raw_root):
    files = raw_files(raw_root)
    stays = eligible_stays(
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
    vaso = read_vasopressor_rows(files["inputs"], set(stays.stay_id))
    vaso = vaso.merge(
        stays[["stay_id", "intime"]], on="stay_id", validate="many_to_one"
    )
    hit = vaso.loc[overlapping(vaso, vaso.intime, *COHORT_WINDOW), "stay_id"].unique()
    cohort = (
        stays.loc[stays.stay_id.isin(hit)]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    patients = pd.read_csv(files["patients"], usecols=["subject_id", "dod"])
    cohort = cohort.merge(patients, on="subject_id", validate="one_to_one")
    cohort["dod"] = pd.to_datetime(cohort.dod, errors="raise")
    cohort["row"] = np.arange(len(cohort))
    return cohort, vaso.loc[vaso.stay_id.isin(cohort.stay_id)].drop(
        columns="intime"
    ).reset_index(drop=True)


def filter_chartevents(path, stay_ids, output, block_size=1 << 26):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    convert = pacsv.ConvertOptions(
        include_columns=CHART_COLUMNS,
        column_types={
            "subject_id": pa.int64(),
            "stay_id": pa.int64(),
            "itemid": pa.int64(),
            "charttime": pa.string(),
            "storetime": pa.string(),
            "valuenum": pa.float64(),
            "valueuom": pa.string(),
        },
    )
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_size),
        convert_options=convert,
    )
    items = pa.array(sorted(ITEMS), pa.int64())
    stays = pa.array(sorted(int(s) for s in stay_ids), pa.int64())
    audit = {"batches": 0, "rows_read": 0, "rows_kept": 0}
    tmp = output.with_suffix(".partial")
    writer = None
    try:
        for batch in reader:
            audit["batches"] += 1
            audit["rows_read"] += batch.num_rows
            keep = pc.and_(
                pc.is_in(batch.column("itemid"), value_set=items),
                pc.is_in(batch.column("stay_id"), value_set=stays),
            )
            kept = pa.Table.from_batches([batch]).filter(keep)
            if kept.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(tmp, kept.schema)
                writer.write_table(kept)
                audit["rows_kept"] += kept.num_rows
            if audit["batches"] % 50 == 0:
                print(json.dumps(audit), flush=True)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(
            "no eligible chart rows found; verify the cohort, item IDs, and input tables"
        )
    tmp.rename(output)
    return audit


def main():
    import argparse
    import time

    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["cohort", "charts"])
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    started = time.monotonic()
    if a.command == "cohort":
        out.mkdir(parents=True, exist_ok=False)
        cohort, vaso = build_cohort(a.raw_root)
        cohort.to_parquet(out / "cohort.parquet", index=False)
        vaso.to_parquet(out / "vasopressor_rows.parquet", index=False)
        print(
            json.dumps(
                {
                    "patients": len(cohort),
                    "vasopressor_rows": len(vaso),
                    "seconds": round(time.monotonic() - started, 1),
                }
            )
        )
    else:
        cohort = pd.read_parquet(out / "cohort.parquet")
        target = out / "charts.parquet"
        audit = filter_chartevents(
            raw_files(a.raw_root)["charts"], cohort.stay_id, target
        )
        audit["seconds"] = round(time.monotonic() - started, 1)
        print(json.dumps(audit))


if __name__ == "__main__":
    main()
