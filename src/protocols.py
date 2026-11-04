"""Fixed experiment settings loaded relative to the repository root."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = (
    "synthetic",
    "semi_synthetic",
    "stress",
    "realworld",
    "realworld_splits",
)


def load_protocol(name):
    if name not in (*PROTOCOLS, "training"):
        raise ValueError(f"unknown protocol: {name}")
    return json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def protocol_snapshot(name):
    protocol = load_protocol(name)
    snapshot = {
        "protocol": protocol,
        "training": load_protocol("training"),
    }
    if "base_protocol" in protocol:
        snapshot["base_protocol"] = load_protocol(protocol["base_protocol"])
    return snapshot


def runtime_versions():
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "scikit-learn",
        "pyarrow",
        "matplotlib",
        "tqdm",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def estimator_settings(name, device):
    from src.evaluation.experiments import EstimatorSettings

    protocol = load_protocol(name)
    if name == "realworld":
        from src.data.mimic_history_cate import VALIDATION_SEED

        training = load_protocol("training")
        if (
            training["validation_fraction"] != 0.15
            or training["validation_seed"] != VALIDATION_SEED
        ):
            raise ValueError(
                "validation settings differ from the preserved clinical preprocessing split"
            )
    return EstimatorSettings(
        **load_protocol("training"),
        device=device,
        probability_clip=protocol["probability_floor"],
        cumulative_weight_clip=protocol["weight_cap"],
    )


def synthetic_config(name, device, cohort_sizes=None, simulation=None):
    from src.data.survival_sim.dgp import SurvivalSimConfig
    from src.evaluation.experiments import StudyScenario, SyntheticStudyConfig

    protocol = load_protocol(name)
    if name in ("semi_synthetic", "stress"):
        from src.data.mimic_history_cate import ENTRY, VALIDATION_SEED

        training = load_protocol("training")
        if protocol["study"]["evaluation_landmark"] != ENTRY:
            raise ValueError(
                "the prepared real-history archive fixes the landmark at index 7"
            )
        if (
            training["validation_fraction"] != 0.15
            or training["validation_seed"] != VALIDATION_SEED
        ):
            raise ValueError(
                "validation settings differ from the preserved preprocessing split"
            )
    expected = protocol["cohorts"]
    if cohort_sizes is not None and dict(cohort_sizes) != expected:
        raise ValueError(
            f"cohort sizes do not match the paper: expected {expected}, got {cohort_sizes}"
        )
    return SyntheticStudyConfig(
        **load_protocol("training"),
        **protocol["study"],
        **expected,
        scenarios=tuple(
            StudyScenario(**scenario) for scenario in protocol["scenarios"]
        ),
        simulation=(
            simulation
            if simulation is not None
            else SurvivalSimConfig(**protocol["simulation"])
        ),
        probability_clip=protocol["probability_floor"],
        cumulative_weight_clip=protocol["weight_cap"],
        device=device,
    )
