from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.mimic_history_cate import inner_rows
from src.data.mimic_realworld.extract import ENTRY
from src.data.mimic_realworld.tensors import make_splits
from src.data.mimic_semisynthetic.adapter import (
    HistoryCollection,
    landmark_prediction_view,
)
from src.data.mimic_semisynthetic.histories import subject_partition
from src.evaluation.experiments import (
    StudyScenario,
    _seed_everything,
    fit_probability_nuisances,
    fit_sequence_regressor,
    predict_sequence_regressor,
)
from src.models.survival_ct_nuisance import (
    NuisanceTrainingConfig,
    SurvivalIterativeGComputationLearner,
)
from src.models.survival_dr import FittedNuisanceView, compute_regime_signals
from src.models.tv_survcaus import (
    TreatmentNumeratorLearner,
    TVSurvCausLearner,
    prepare_treatment_probabilities,
)
from src.protocols import (
    estimator_settings,
    file_sha256,
    load_protocol,
    protocol_snapshot,
    runtime_versions,
)

PROTOCOL = load_protocol("realworld")
SCENARIOS = tuple(StudyScenario(**scenario) for scenario in PROTOCOL["scenarios"])
N_SEEDS = PROTOCOL["study"]["n_replicates"]
STUDY_SEED = PROTOCOL["study"]["seed"]
MODEL_SEED_OFFSET = 10
METHODS = ("pi", "ra", "iptw_ipcw", "dr", "tv_survcaus")
SPLIT_SEEDS = tuple(load_protocol("realworld_splits")["split_seeds"])
if PROTOCOL["study"]["evaluation_landmark"] != ENTRY:
    raise ValueError("clinical extraction fixes the landmark at index 7")


def split_roles(patient_id, split_seed):
    if split_seed not in SPLIT_SEEDS:
        raise ValueError("split seed outside the protocol")
    return np.array(
        [subject_partition(int(patient), seed=split_seed) for patient in patient_id]
    )


def load_trajectories(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    roles = data.pop("roles")
    evaluation = {k[5:]: data.pop(k) for k in list(data) if k.startswith("eval_")}
    return data, roles, evaluation


def landmark_window_only(raw, landmark=ENTRY):
    raw = {k: v.copy() for k, v in raw.items()}
    raw["at_risk"][:, :landmark] = 0.0
    raw["censoring_label_observed"][:, :landmark] = 0.0
    raw["event_next"][:, :landmark] = np.nan
    raw["event_free_next"][:, :landmark] = np.nan
    raw["censoring_continue"][:, :landmark] = np.nan
    return raw


def build_collection(data, roles):
    splits, prep = make_splits(data, roles, inner_rows)
    splits = {name: landmark_window_only(raw) for name, raw in splits.items()}
    return HistoryCollection.from_raw(splits), splits, prep


def fit_arm_learner(collection, probability_learner, scenario, settings, seed):
    max_time = collection.train_f.raw["treatments"].shape[1]
    cfg = NuisanceTrainingConfig(
        tau=scenario.tau,
        treatment_sequence=scenario.regime_one,
        training_landmarks=tuple(range(ENTRY, max_time - scenario.tau + 1)),
        epochs=settings.nuisance_epochs,
        batch_size=settings.batch_size,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        device=settings.device,
        model_kwargs=dict(settings.transformer),
        seed=seed,
        validation_fraction=settings.validation_fraction,
        validation_seed=settings.validation_seed,
        early_stopping_patience=settings.early_stopping_patience,
        early_stopping_min_delta=settings.early_stopping_min_delta,
        early_stopping_min_epochs=settings.early_stopping_min_epochs,
    )
    _seed_everything(seed)
    return SurvivalIterativeGComputationLearner(cfg).fit(
        collection, probability_nuisance_learner=probability_learner
    )


def arm_signal_targets(learner, dataset, scenario, settings, cache):
    raw = dataset.raw
    n, max_time = raw["treatments"].shape
    regime = scenario.regime_one
    targets = {
        m: np.zeros((n, max_time, 1), np.float32) for m in ("ra", "iptw_ipcw", "dr")
    }
    mask = np.zeros((n, max_time, 1), np.float32)
    diagnostics = []
    for t in range(ENTRY, max_time - scenario.tau + 1):
        view = FittedNuisanceView(
            learner,
            dataset,
            t,
            t + scenario.treatment_length - 1,
            t + scenario.tau,
            regime,
            prediction_cache=cache,
        )
        s = compute_regime_signals(
            raw,
            regime,
            view,
            view,
            probability_clip=settings.probability_clip,
            cumulative_weight_clip=settings.cumulative_weight_clip,
        )
        eligible = s["eligible"]
        mask[:, t, 0] = eligible
        targets["ra"][:, t, 0] = np.nan_to_num(s["gcomp"])
        targets["iptw_ipcw"][:, t, 0] = np.nan_to_num(s["ipw"])
        targets["dr"][:, t, 0] = np.nan_to_num(s["dr"])
        w = s["ipw_weight"][eligible]
        w = w[w > 0]
        diagnostics.append(
            dict(
                landmark=t,
                n_eligible=int(eligible.sum()),
                n_positive_ipw=int(len(w)),
                ipw_ess=float(w.sum() ** 2 / np.square(w).sum()) if len(w) else 0.0,
                weight_clipped_fraction=float(s["weight_clipped"][eligible].mean()),
                dr_mean=float(np.nanmean(s["dr"][eligible])),
                dr_sd=float(np.nanstd(s["dr"][eligible])),
            )
        )
    return targets, mask, pd.DataFrame(diagnostics)


def tv_learner(scenario, settings, seed):
    return TVSurvCausLearner(
        scenario.treatment_length,
        scenario.tau,
        alpha=1.0,
        representation_dim=32,
        regime_embedding_dim=16,
        epochs=settings.second_stage_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.second_stage_weight_decay,
        device=settings.device,
        probability_clip=settings.probability_clip,
        cumulative_weight_clip=settings.cumulative_weight_clip,
        mmd_bandwidth=None,
        transformer_kwargs=dict(settings.transformer),
        seed=seed,
        validation_fraction=settings.validation_fraction,
        validation_seed=settings.validation_seed,
        early_stopping_patience=settings.early_stopping_patience,
        early_stopping_min_delta=settings.early_stopping_min_delta,
        early_stopping_min_epochs=settings.early_stopping_min_epochs,
    )


def run_seed(data, roles, seed_index, device="cuda"):
    collection, splits, prep = build_collection(data, roles)
    widths = dict(
        dim_covariates=data["states"].shape[-1] + 1,
        dim_static_features=data["baseline"].shape[-1],
    )
    base = estimator_settings("realworld", device)
    settings = replace(
        base,
        transformer={**base.transformer, **widths},
        probability_transformer={**base.probability_transformer, **widths},
    )
    seed = STUDY_SEED + 100_003 * seed_index
    max_time = collection.train_f.raw["treatments"].shape[1]
    probability = fit_probability_nuisances(
        collection,
        settings,
        max_time,
        seed + 101,
        training_landmarks=range(ENTRY, max_time),
    )
    numerator = TreatmentNumeratorLearner(
        transformer_kwargs=dict(settings.probability_transformer),
        epochs=settings.nuisance_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        device=settings.device,
        seed=seed + 307,
        validation_fraction=settings.validation_fraction,
        validation_seed=settings.validation_seed,
        early_stopping_patience=settings.early_stopping_patience,
        early_stopping_min_delta=settings.early_stopping_min_delta,
        early_stopping_min_epochs=settings.early_stopping_min_epochs,
    ).fit(collection.train_f)
    denominator_p, numerator_p = prepare_treatment_probabilities(
        collection.val_f, probability, numerator
    )
    test_view = landmark_prediction_view(collection.test_f, ENTRY)
    leaked = int(
        np.isfinite(test_view.raw["states"][:, ENTRY + 1 :]).sum()
        + np.isfinite(test_view.raw["treatments"][:, ENTRY:]).sum()
        + (test_view.data["covariates"][:, ENTRY + 1 :] != 0).sum()
    )
    if leaked:
        raise AssertionError(
            f"{leaked} post-landmark test inputs survived the prediction view"
        )
    rows = np.flatnonzero(test_view.raw["at_risk"][:, ENTRY] > 0)
    if len(rows) != len(test_view.raw["patient_id"]):
        raise AssertionError("every test patient must be at risk at the landmark")
    predictions, diagnostics = [], []
    for index, scenario in enumerate(SCENARIOS):
        scenario_seed = seed + 10_007 * (index + 1)
        learner = fit_arm_learner(
            collection, probability, scenario, settings, scenario_seed
        )
        pi_view = FittedNuisanceView(
            learner,
            test_view,
            ENTRY,
            ENTRY + scenario.treatment_length - 1,
            ENTRY + scenario.tau,
            scenario.regime_one,
        )
        pi = pi_view.m(test_view.raw, ENTRY, rows)
        cache = {}
        cap = float(settings.cumulative_weight_clip)
        cfg = replace(settings, cumulative_weight_clip=cap)
        targets, mask, diag = arm_signal_targets(
            learner, collection.val_f, scenario, cfg, cache
        )
        diagnostics.append(
            diag.assign(scenario=scenario.name, cap=cap, seed=seed_index)
        )
        ra_model = fit_sequence_regressor(
            collection.val_f, targets["ra"], mask, cfg, scenario_seed + 50_003
        )
        heads = {"ra": ra_model}
        for m_index, method in enumerate(("iptw_ipcw", "dr"), start=1):
            heads[method] = fit_sequence_regressor(
                collection.val_f,
                targets[method],
                mask,
                cfg,
                scenario_seed + 50_003 + 1009 * m_index + MODEL_SEED_OFFSET,
            )
        tv = tv_learner(scenario, cfg, scenario_seed + 70_001 + MODEL_SEED_OFFSET).fit(
            collection.val_f, denominator_p, numerator_p
        )
        frame = pd.DataFrame(
            {"patient_id": test_view.raw["patient_id"][rows], "pi": pi}
        )
        for method, model in heads.items():
            frame[method] = predict_sequence_regressor(model, test_view, cfg)[
                rows, ENTRY
            ]
        frame["tv_survcaus"] = tv.predict_survival(
            test_view, ENTRY, scenario.regime_one
        )[rows]
        predictions.append(
            frame.assign(scenario=scenario.name, cap=cap, seed=seed_index)
        )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(diagnostics, ignore_index=True),
        prep,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--split-seed", type=int, choices=SPLIT_SEEDS, default=SPLIT_SEEDS[0]
    )
    a = p.parse_args()
    if not 0 <= a.seed < N_SEEDS:
        raise ValueError("seed outside protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out) / f"seed_{a.seed}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trajectory_sha = file_sha256(a.data)
    configuration = protocol_snapshot("realworld")
    data, roles, _ = load_trajectories(a.data)
    stored_roles = roles
    roles = split_roles(data["patient_id"], a.split_seed)
    if a.split_seed == SPLIT_SEEDS[0] and not np.array_equal(roles, stored_roles):
        raise ValueError("original patient hash differs from stored trajectory roles")
    if len(data["patient_id"]) != PROTOCOL["dataset"]["expected_patients"] or (
        a.split_seed == SPLIT_SEEDS[0]
        and int((roles == "test").sum())
        != PROTOCOL["dataset"]["expected_test_patients"]
    ):
        raise ValueError(
            "cohort counts do not match the manuscript; verify the MIMIC-IV source and extraction"
        )
    predictions, diagnostics, prep = run_seed(data, roles, a.seed)
    if file_sha256(a.data) != trajectory_sha:
        raise ValueError("input changed during clinical fitting")
    predictions.to_csv(out / "predictions.csv", index=False)
    diagnostics.to_csv(out / "weight_diagnostics.csv", index=False)
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                status="complete",
                seed=a.seed,
                split_seed=a.split_seed,
                smoke=False,
                elapsed_seconds=time.monotonic() - started,
                roles={role: int((roles == role).sum()) for role in np.unique(roles)},
                trajectories_sha256=trajectory_sha,
                device=torch.cuda.get_device_name(),
                configuration=configuration,
                environment=runtime_versions(),
                predictions_sha256=file_sha256(out / "predictions.csv"),
                preprocessing=prep.as_dict(),
                settings=asdict(estimator_settings("realworld", "cuda")),
                input_dimensions=dict(
                    states=int(data["states"].shape[-1]),
                    baseline=int(data["baseline"].shape[-1]),
                ),
            ),
            indent=2,
        )
    )
    print(
        json.dumps(
            dict(
                status="complete",
                out=str(out),
                seconds=round(time.monotonic() - started),
            )
        )
    )


if __name__ == "__main__":
    main()
