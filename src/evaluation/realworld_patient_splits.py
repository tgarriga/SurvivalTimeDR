"""Combine the submission's four patient hashes without pooling fitting repetitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.protocols import load_protocol, protocol_snapshot


def validate_split_scores(per_seed, paired):
    protocol = load_protocol("realworld")
    splits = load_protocol("realworld_splits")["split_seeds"]
    scenarios = {scenario["name"]: scenario for scenario in protocol["scenarios"]}
    methods = ("pi", "ra", "iptw_ipcw", "dr", "tv_survcaus")
    expected_scores = {
        (split, scenario, method, seed)
        for split in splits
        for scenario in scenarios
        for method in methods
        for seed in range(protocol["study"]["n_replicates"])
    }
    expected_pairs = {
        (split, scenario, f"dr minus {method}")
        for split in splits
        for scenario in scenarios
        for method in methods
        if method != "dr"
    }
    datasets = (
        (
            per_seed,
            ["split", "scenario", "method", "seed"],
            ["weighted_brier"],
            expected_scores,
        ),
        (
            paired,
            ["split", "scenario", "contrast"],
            ["brier_difference", "ci_low", "ci_high"],
            expected_pairs,
        ),
    )
    for frame, keys, measures, expected in datasets:
        required = set(keys + measures + ["cap"])
        if not required.issubset(frame.columns):
            raise ValueError(
                f"missing patient-split columns: {sorted(required.difference(frame.columns))}"
            )
        if (
            frame[keys].isna().any().any()
            or frame.duplicated(keys).any()
            or set(frame[keys].itertuples(index=False, name=None)) != expected
        ):
            raise ValueError(
                "patient-split results must contain every configured scenario, method, "
                "and repetition exactly once"
            )
        if not np.isfinite(frame[measures].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite patient-split scores or bootstrap results")
        if not frame["cap"].eq(protocol["weight_cap"]).all():
            raise ValueError(f"patient splits require weight cap {protocol['weight_cap']}")
        for field in ("treatment_length", "tau"):
            expected_values = frame["scenario"].map(
                {name: scenario[field] for name, scenario in scenarios.items()}
            )
            if field in frame and not frame[field].eq(expected_values).all():
                raise ValueError(f"patient-split results have incorrect {field}")
    if (paired["ci_low"] > paired["ci_high"]).any():
        raise ValueError("reversed patient-split bootstrap interval")


def report(original_runs, runs, output):
    protocol = load_protocol("realworld_splits")
    per_seed, paired, weights = [], [], []
    trajectory_sha = None
    for split_seed in protocol["split_seeds"]:
        directory = (
            Path(original_runs)
            if split_seed == protocol["original_split"]
            else Path(runs) / f"split_{split_seed}"
        )
        raw = directory / "report"
        completion = raw / "complete.json"
        done = json.loads(completion.read_text(encoding="utf-8"))
        if (
            done["status"] != "complete"
            or done["replicates"] != 5
            or done["split_seed"] != split_seed
        ):
            raise ValueError("missing complete five-seed patient-split score")
        if (
            done["configuration"]["protocol"] != load_protocol("realworld")
            or done["configuration"]["training"] != load_protocol("training")
        ):
            raise ValueError(
                "patient splits were scored with different experiment/training settings"
            )
        if trajectory_sha is not None and trajectory_sha != done["trajectories_sha256"]:
            raise ValueError("patient splits use different trajectory inputs")
        trajectory_sha = done["trajectories_sha256"]
        per_seed.append(pd.read_csv(raw / "per_seed.csv").assign(split=split_seed))
        paired.append(pd.read_csv(raw / "paired.csv").assign(split=split_seed))
        weights.append(
            dict(split=split_seed, **json.loads((raw / "weights.json").read_text()))
        )
    per_seed = pd.concat(per_seed, ignore_index=True)
    paired = pd.concat(paired, ignore_index=True)
    validate_split_scores(per_seed, paired)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    paired.to_csv(output / "paired.csv", index=False)
    (output / "weights.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )
    (output / "complete.json").write_text(
        json.dumps(
            dict(
                status="complete",
                split_seeds=protocol["split_seeds"],
                fitting_repetitions_per_split=5,
                trajectories_sha256=trajectory_sha,
                configuration=protocol_snapshot("realworld_splits"),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-runs", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report(args.original_runs, args.runs, args.output)


if __name__ == "__main__":
    main()
