"""Command-line entry points for the manuscript experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.protocols import (
    PROTOCOLS,
    ROOT,
    file_sha256,
    load_protocol,
    protocol_snapshot,
    runtime_versions,
    synthetic_config,
)


def command(module, *arguments):
    return [sys.executable, "-B", "-m", module, *(str(value) for value in arguments)]


def digest_argument(path, dry_run):
    return "COMPUTED_SHA256" if dry_run else file_sha256(path)


def patient_split_commands(args, parser, output, stages, protocol):
    if args.data is None:
        parser.error(
            "realworld-splits requires --data pointing to the same trajectories.npz as the first split"
        )
    if "report" in stages and args.original_runs is None:
        parser.error(
            "realworld-splits reporting requires --original-runs from the completed primary realworld run"
        )
    if (args.seed is not None or args.split_seed is not None) and args.stage != "fit":
        parser.error("--seed/--split-seed are only valid with --stage fit")
    split_seeds = [
        value
        for value in protocol["split_seeds"]
        if value != protocol["original_split"]
    ]
    if args.split_seed is not None:
        if args.split_seed not in split_seeds:
            parser.error(
                "additional patient split must be 2028, 2029, or 2030; the original 2027 run is reused"
            )
        split_seeds = [args.split_seed]
    plans = []
    if "fit" in stages:
        seeds = (
            range(load_protocol("realworld")["study"]["n_replicates"])
            if args.seed is None
            else [args.seed]
        )
        for split_seed in split_seeds:
            for seed in seeds:
                if seed not in range(5):
                    parser.error("--seed must be between 0 and 4")
                plans.append(
                    command(
                        "src.evaluation.mimic_realworld_factual",
                        "--data",
                        args.data.resolve(),
                        "--out",
                        output / f"split_{split_seed}",
                        "--seed",
                        seed,
                        "--split-seed",
                        split_seed,
                    )
                )
    if "report" in stages:
        for split_seed in split_seeds:
            plans.append(
                command(
                    "src.evaluation.realworld_factual_metrics",
                    "--data",
                    args.data.resolve(),
                    "--runs",
                    output / f"split_{split_seed}",
                    "--out",
                    output / f"split_{split_seed}" / "report",
                    "--split-seed",
                    split_seed,
                )
            )
        plans.append(
            command(
                "src.evaluation.realworld_patient_splits",
                "--original-runs",
                args.original_runs.resolve(),
                "--runs",
                output,
                "--output",
                output / "report",
            )
        )
    return plans


def experiment_commands(args, parser):
    name = args.command.replace("-", "_")
    output = args.output.resolve()
    stages = ("init", "fit", "report") if args.stage == "all" else (args.stage,)
    plans = []
    protocol = load_protocol(name)
    if name == "realworld_splits":
        return patient_split_commands(args, parser, output, stages, protocol)
    if name == "synthetic":
        for stage in ("fit", "report"):
            if stage in stages:
                plans.append(
                    command(
                        "src.run",
                        "synthetic",
                        "--stage",
                        stage,
                        "--output",
                        output,
                        "--device",
                        args.device,
                    )
                )
        return plans
    if name == "semi_synthetic":
        module = "src.evaluation.mimic_history_cate_study"
        if "init" in stages:
            if args.archive is None:
                parser.error("semi-synthetic initialization requires --archive")
            archive = args.archive.resolve()
            plans.append(
                command(
                    module,
                    "init",
                    "--root",
                    output,
                    "--archive",
                    archive,
                    "--expected-archive-sha256",
                    digest_argument(archive, args.dry_run),
                )
            )
        if args.replicate is not None and args.stage != "fit":
            parser.error(
                "--replicate is only valid with --stage fit; full runs always use five repetitions"
            )
        if "fit" in stages:
            repetitions = (
                range(protocol["study"]["n_replicates"])
                if args.replicate is None
                else [args.replicate]
            )
            for replicate in repetitions:
                if replicate not in range(protocol["study"]["n_replicates"]):
                    parser.error("--replicate must be between 0 and 4")
                plans.append(
                    command(module, "case", "--root", output, "--replicate", replicate)
                )
    elif name == "stress":
        module = "src.evaluation.survival_stress"
        if "init" in stages:
            if args.prepared is None:
                parser.error(
                    "stress initialization requires --prepared from semi-synthetic initialization"
                )
            prepared = args.prepared.resolve()
            plans.append(
                command(
                    module,
                    "init",
                    "--root",
                    output,
                    "--prepared",
                    prepared,
                    "--expected-prepared-sha256",
                    digest_argument(prepared, args.dry_run),
                )
            )
        total = len(protocol["conditions"]) * protocol["study"]["n_replicates"]
        if args.index is not None and args.stage != "fit":
            parser.error(
                "--index is only valid with --stage fit; full runs always execute all 35 cases"
            )
        if "fit" in stages:
            indices = range(total) if args.index is None else [args.index]
            for index in indices:
                if index not in range(total):
                    parser.error(f"--index must be between 0 and {total - 1}")
                plans.append(
                    command(module, "case", "--root", output, "--index", index)
                )
    else:
        if args.data is None:
            parser.error("realworld requires --data pointing to trajectories.npz")
        if args.seed is not None and args.stage != "fit":
            parser.error(
                "--seed is only valid with --stage fit; full runs always use five seeds"
            )
        if "fit" in stages:
            seeds = (
                range(protocol["study"]["n_replicates"])
                if args.seed is None
                else [args.seed]
            )
            for seed in seeds:
                if seed not in range(protocol["study"]["n_replicates"]):
                    parser.error("--seed must be between 0 and 4")
                plans.append(
                    command(
                        "src.evaluation.mimic_realworld_factual",
                        "--data",
                        args.data.resolve(),
                        "--out",
                        output,
                        "--seed",
                        seed,
                    )
                )
        if "report" in stages:
            plans.append(
                command(
                    "src.evaluation.realworld_factual_metrics",
                    "--data",
                    args.data.resolve(),
                    "--runs",
                    output,
                    "--out",
                    output / "report",
                )
            )
    if "report" in stages:
        if name != "realworld":
            plans.append(command(module, "report", "--root", output))
    return plans


def fit_synthetic(args):
    from src.evaluation.experiments import run_synthetic_study

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = protocol_snapshot("synthetic")
    result = run_synthetic_study(
        synthetic_config("synthetic", args.device), output_dir=output
    )
    completion = dict(
        status="complete",
        smoke=False,
        replicates=5,
        device=args.device,
        configuration=snapshot,
        environment=runtime_versions(),
    )
    (output / "complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(result.summary.to_string(index=False))


def verify_synthetic_output(output):
    completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    if (
        completion["status"] != "complete"
        or completion["smoke"]
        or completion["replicates"] != 5
    ):
        raise ValueError(
            "synthetic reporting requires a completed five-repetition study"
        )
    if (
        completion["configuration"]["protocol"] != load_protocol("synthetic")
        or completion["configuration"]["training"] != load_protocol("training")
    ):
        raise ValueError(
            "synthetic output was produced with different experiment/training settings"
        )


def report_synthetic(output):
    import pandas as pd

    verify_synthetic_output(output)
    print(pd.read_csv(output / "summary.csv").to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    protocol = commands.add_parser(
        "protocol", help="Print experiment and training configuration."
    )
    protocol.add_argument("experiment", choices=PROTOCOLS)
    prepare = commands.add_parser(
        "prepare", help="Prepare credentialed MIMIC-IV 3.1 data."
    )
    prepare.add_argument("experiment", choices=("semi-synthetic", "realworld"))
    prepare.add_argument("--raw-root", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--dry-run", action="store_true")
    for name in (
        "synthetic",
        "semi-synthetic",
        "stress",
        "realworld",
        "realworld-splits",
    ):
        study = commands.add_parser(name)
        study.add_argument("--output", required=True, type=Path)
        stages = (
            ("all", "fit", "report")
            if name in ("synthetic", "realworld", "realworld-splits")
            else ("all", "init", "fit", "report")
        )
        study.add_argument("--stage", choices=stages, default="all")
        study.add_argument(
            "--dry-run",
            action="store_true",
            help="Print commands without reading data or fitting models.",
        )
        if name == "synthetic":
            study.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        elif name == "semi-synthetic":
            study.add_argument("--archive", type=Path)
            study.add_argument("--replicate", type=int)
        elif name == "stress":
            study.add_argument("--prepared", type=Path)
            study.add_argument("--index", type=int)
        else:
            study.add_argument("--data", type=Path)
            study.add_argument("--seed", type=int)
            if name == "realworld-splits":
                study.add_argument("--original-runs", type=Path)
                study.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    if args.command == "protocol":
        print(json.dumps(protocol_snapshot(args.experiment), indent=2))
        return
    if args.command == "prepare":
        raw, output = args.raw_root.resolve(), args.output.resolve()
        if args.experiment == "semi-synthetic":
            plans = [
                command(
                    "src.data.mimic_semisynthetic.histories",
                    "--raw-root",
                    raw,
                    "--output",
                    output,
                )
            ]
        else:
            plans = [
                command(
                    "src.data.mimic_realworld.extract",
                    stage,
                    "--raw-root",
                    raw,
                    "--out",
                    output,
                )
                for stage in ("cohort", "charts")
            ]
            plans.append(command("src.data.mimic_realworld.tensors", "--data", output))
    else:
        if (
            args.command == "synthetic"
            and args.stage in ("fit", "report")
            and not args.dry_run
        ):
            if args.stage == "fit":
                fit_synthetic(args)
            else:
                report_synthetic(args.output.resolve())
            return
        if args.stage in ("all", "init") and args.output.exists() and not args.dry_run:
            parser.error(
                "use a new output directory; existing runs are never overwritten"
            )
        plans = experiment_commands(args, parser)
    if args.dry_run:
        print(json.dumps({"commands": plans}, indent=2))
        return
    for planned in plans:
        subprocess.run(planned, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
