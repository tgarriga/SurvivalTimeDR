"""Canonical simple-split synthetic survival study.

The study has one implementation path:

* fit every nuisance on an independent nuisance-training cohort;
* pool every valid landmark when fitting the sequence nuisances;
* construct RA, IPTW/IPCW and DR contrast signals on a second-stage cohort;
* fit identity-output Transformer regressions to those contrast signals;
* fit one pooled-landmark Transformer TV-SurvCaus model per scenario; and
* evaluate at one fixed landmark on a third, untouched cohort.

The command-line entry point is a front end to this module. The generic estimator steps
(``fit_probability_nuisances``, ``fit_contrast_nuisances``, ``contrast_signal_targets`` and
``predict_landmark_contrasts``) are shared with the MIMIC pipelines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.survival_sim.dataset import SurvivalSyntheticDatasetCollection
from src.data.survival_sim.dgp import SurvivalSimConfig, simulate_split
from src.data.survival_sim.oracle import oracle_pair
from src.models.survival_ct_nuisance import (
    NuisanceTrainingConfig,
    SurvivalIterativeGComputationLearner,
    SurvivalProbabilityNuisanceLearner,
    SurvivalSequenceRegressor,
)
from src.models.survival_dr import (
    FittedNuisanceView,
    OracleProbabilityView,
    compute_regime_signals,
)
from src.models.transformer_training import (
    ValidationEarlyStopping,
    patient_train_validation_loaders,
    training_progress,
    update_training_progress,
)
from src.models.tv_survcaus import (
    TreatmentNumeratorLearner,
    TVSurvCausLearner,
    prepare_treatment_probabilities,
)

from .metrics import effect_metrics, summarize_study


@dataclass(frozen=True)
class StudyScenario:
    """One finite intervention comparison at one prediction horizon."""

    name: str
    treatment_length: int
    tau: int

    def __post_init__(self):
        if not self.name:
            raise ValueError("scenario name cannot be empty")
        if self.treatment_length < 1:
            raise ValueError("treatment_length must be positive")
        if self.tau < self.treatment_length:
            raise ValueError("tau must be at least the treatment length")

    @property
    def regime_one(self) -> tuple[int, ...]:
        return (1,) * self.treatment_length

    @property
    def regime_zero(self) -> tuple[int, ...]:
        return (0,) * self.treatment_length


DEFAULT_SCENARIOS = (
    StudyScenario("L2_tau2", 2, 2),
    StudyScenario("L2_tau3", 2, 3),
    StudyScenario("L2_tau4", 2, 4),
    StudyScenario("L3_tau8", 3, 8),
)

_MODEL_BATCH_KEYS = {
    "active_entries",
    "covariates",
    "current_treatments",
    "outputs",
    "prev_outputs",
    "prev_treatments",
    "static_features",
    "target_mask",
}


def _default_transformer() -> dict[str, Any]:
    """WO-learner outcome/meta-learner Transformer hyperparameters."""
    return {
        "d_model": 30,
        "num_heads": 3,
        "num_encoder_layers": 1,
        "dim_feedforward": 20,
        "dropout": 0.1,
        "dim_output_hidden_units": 20,
        "max_seq_length": 64,
    }


def _default_probability_transformer() -> dict[str, Any]:
    """WO-learner propensity/weight-model Transformer hyperparameters."""
    return {
        "d_model": 40,
        "num_heads": 4,
        "num_encoder_layers": 1,
        "dim_feedforward": 20,
        "dropout": 0.0,
        "dim_output_hidden_units": 20,
        "max_seq_length": 64,
    }


TRAINING_REGIME = {
    "nuisance_epochs": 30,
    "second_stage_epochs": 30,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "second_stage_weight_decay": 1e-2,
    "validation_fraction": 0.15,
    "validation_seed": 20260917,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-4,
    "early_stopping_min_epochs": 10,
}


@dataclass(frozen=True)
class EstimatorSettings:
    """Optimization and weighting settings read by the generic estimator helpers.

    :class:`SyntheticStudyConfig` exposes the same attributes, so either may be passed.
    """

    nuisance_epochs: int = TRAINING_REGIME["nuisance_epochs"]
    second_stage_epochs: int = TRAINING_REGIME["second_stage_epochs"]
    batch_size: int = TRAINING_REGIME["batch_size"]
    learning_rate: float = TRAINING_REGIME["learning_rate"]
    weight_decay: float = TRAINING_REGIME["weight_decay"]
    second_stage_weight_decay: float = TRAINING_REGIME["second_stage_weight_decay"]
    device: str = "cpu"
    probability_clip: float = 0.03
    cumulative_weight_clip: float | None = 20.0
    validation_fraction: float = TRAINING_REGIME["validation_fraction"]
    validation_seed: int = TRAINING_REGIME["validation_seed"]
    early_stopping_patience: int = TRAINING_REGIME["early_stopping_patience"]
    early_stopping_min_delta: float = TRAINING_REGIME["early_stopping_min_delta"]
    early_stopping_min_epochs: int = TRAINING_REGIME["early_stopping_min_epochs"]
    probability_transformer: Mapping[str, Any] = field(
        default_factory=_default_probability_transformer
    )
    transformer: Mapping[str, Any] = field(default_factory=_default_transformer)

    def __post_init__(self):
        if min(self.nuisance_epochs, self.second_stage_epochs, self.batch_size) < 1:
            raise ValueError("epochs and batch_size must be positive")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.second_stage_weight_decay < 0
        ):
            raise ValueError("invalid optimizer configuration")
        if not 0.0 <= self.probability_clip < 0.5:
            raise ValueError("probability_clip must be in [0, 0.5)")
        if self.cumulative_weight_clip is not None and self.cumulative_weight_clip <= 0:
            raise ValueError("cumulative_weight_clip must be positive or None")
        _validate_internal_validation(self)


@dataclass(frozen=True)
class SyntheticStudyConfig:
    """Locked protocol for the headline synthetic experiment."""

    simulation: SurvivalSimConfig = field(default_factory=SurvivalSimConfig)
    scenarios: tuple[StudyScenario, ...] = DEFAULT_SCENARIOS
    n_nuisance: int = 1800
    n_second_stage: int = 1800
    n_test: int = 1200
    evaluation_landmark: int = 10
    n_replicates: int = 5
    oracle_rollouts: int = 1000
    nuisance_epochs: int = TRAINING_REGIME["nuisance_epochs"]
    second_stage_epochs: int = TRAINING_REGIME["second_stage_epochs"]
    batch_size: int = TRAINING_REGIME["batch_size"]
    learning_rate: float = TRAINING_REGIME["learning_rate"]
    weight_decay: float = TRAINING_REGIME["weight_decay"]
    second_stage_weight_decay: float = TRAINING_REGIME["second_stage_weight_decay"]
    device: str = "cpu"
    probability_clip: float = 0.03
    cumulative_weight_clip: float | None = 20.0
    validation_fraction: float = TRAINING_REGIME["validation_fraction"]
    validation_seed: int = TRAINING_REGIME["validation_seed"]
    early_stopping_patience: int = TRAINING_REGIME["early_stopping_patience"]
    early_stopping_min_delta: float = TRAINING_REGIME["early_stopping_min_delta"]
    early_stopping_min_epochs: int = TRAINING_REGIME["early_stopping_min_epochs"]
    tv_survcaus_alpha: float = 1.0
    tv_survcaus_representation_dim: int = 32
    tv_survcaus_regime_embedding_dim: int = 16
    tv_survcaus_epochs: int | None = None
    tv_survcaus_mmd_bandwidth: float | None = None
    probability_transformer: Mapping[str, Any] = field(
        default_factory=_default_probability_transformer
    )
    transformer: Mapping[str, Any] = field(default_factory=_default_transformer)
    seed: int = 2027

    def __post_init__(self):
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        if not self.scenarios:
            raise ValueError("at least one study scenario is required")
        if len({scenario.name for scenario in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario names must be unique")
        if min(self.n_nuisance, self.n_second_stage, self.n_test) < 1:
            raise ValueError("all three patient splits must be non-empty")
        if self.evaluation_landmark < 0:
            raise ValueError("evaluation_landmark cannot be negative")
        if self.n_replicates < 1:
            raise ValueError("n_replicates must be positive")
        if self.oracle_rollouts < 2:
            raise ValueError("oracle_rollouts must be at least two")
        largest_tau = max(s.tau for s in self.scenarios)
        if self.evaluation_landmark + largest_tau > self.simulation.max_seq_length:
            raise ValueError("evaluation landmark plus largest tau exceeds follow-up")
        if min(self.nuisance_epochs, self.second_stage_epochs, self.batch_size) < 1:
            raise ValueError("epochs and batch_size must be positive")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.second_stage_weight_decay < 0
        ):
            raise ValueError("invalid optimizer configuration")
        if not 0.0 <= self.probability_clip < 0.5:
            raise ValueError("probability_clip must be in [0, 0.5)")
        if self.cumulative_weight_clip is not None and self.cumulative_weight_clip <= 0:
            raise ValueError("cumulative_weight_clip must be positive or None")
        _validate_internal_validation(self)
        if not np.isfinite(self.tv_survcaus_alpha) or self.tv_survcaus_alpha < 0:
            raise ValueError("tv_survcaus_alpha must be finite and non-negative")
        if (
            min(
                self.tv_survcaus_representation_dim,
                self.tv_survcaus_regime_embedding_dim,
            )
            < 1
        ):
            raise ValueError("TV-SurvCaus representation dimensions must be positive")
        if self.tv_survcaus_epochs is not None and self.tv_survcaus_epochs < 1:
            raise ValueError("tv_survcaus_epochs must be positive or None")
        if self.tv_survcaus_mmd_bandwidth is not None and (
            not np.isfinite(self.tv_survcaus_mmd_bandwidth)
            or self.tv_survcaus_mmd_bandwidth <= 0
        ):
            raise ValueError("tv_survcaus_mmd_bandwidth must be positive or None")
        for name, settings in (
            ("outcome", self.transformer),
            ("probability", self.probability_transformer),
        ):
            transformer_length = int(
                settings.get("max_seq_length", self.simulation.max_seq_length)
            )
            if transformer_length < self.simulation.max_seq_length:
                raise ValueError(
                    f"{name} Transformer maximum is shorter than simulated follow-up"
                )


@dataclass
class SyntheticStudyResult:
    metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    config: SyntheticStudyConfig

    def save(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.metrics.assign(cap=self.config.cumulative_weight_clip).to_csv(
            root / "metrics.csv", index=False
        )
        self.summary.assign(cap=self.config.cumulative_weight_clip).to_csv(
            root / "summary.csv", index=False
        )
        self.predictions.assign(cap=self.config.cumulative_weight_clip).to_csv(
            root / "predictions.csv", index=False
        )
        self.diagnostics.assign(cap=self.config.cumulative_weight_clip).to_csv(
            root / "diagnostics.csv", index=False
        )
        diagnostic_kind = (
            self.diagnostics["diagnostic"]
            if "diagnostic" in self.diagnostics
            else pd.Series("", index=self.diagnostics.index, dtype=str)
        )
        training = self.diagnostics.loc[diagnostic_kind.eq("training_history")]
        if len(training):
            training.to_csv(root / "training_history.csv", index=False)
        oracle = self.diagnostics.loc[
            diagnostic_kind.eq("oracle_probability_benchmark")
        ]
        if len(oracle):
            oracle.to_csv(root / "oracle_probability_benchmark.csv", index=False)
        (root / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2),
            encoding="utf-8",
        )


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_history_frame(
    model,
    *,
    model_role: str,
    training_task: str,
    **context,
) -> pd.DataFrame:
    """Return one tidy per-epoch training record for any maintained Transformer."""
    history = pd.DataFrame(getattr(model, "training_history_", []))
    if history.empty:
        return pd.DataFrame(
            columns=(
                "diagnostic",
                "model_role",
                "training_task",
                "epoch",
                "train_loss",
                "validation_loss",
                "best_epoch",
                "stopped_epoch",
            )
        )
    for column in ("epoch", "train_loss", "validation_loss"):
        if column not in history:
            history[column] = np.nan
    history = history[["epoch", "train_loss", "validation_loss"]].copy()
    history.insert(0, "training_task", str(training_task))
    history.insert(0, "model_role", str(model_role))
    history.insert(0, "diagnostic", "training_history")
    history["best_epoch"] = int(getattr(model, "best_epoch_", len(history)))
    history["stopped_epoch"] = int(getattr(model, "stopped_epoch_", len(history)))
    for name, value in context.items():
        history[name] = value
    return history


def _loader_batch(batch, device):
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.items()
        if key in _MODEL_BATCH_KEYS
    }


def _validate_internal_validation(config) -> None:
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if min(config.early_stopping_patience, config.early_stopping_min_epochs) < 1:
        raise ValueError("early-stopping patience and minimum epochs must be positive")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative")
    if config.early_stopping_min_epochs > min(
        config.nuisance_epochs, config.second_stage_epochs
    ):
        raise ValueError("early_stopping_min_epochs cannot exceed configured epochs")


def fit_sequence_regressor(
    dataset,
    targets: np.ndarray,
    target_mask: np.ndarray,
    config: SyntheticStudyConfig,
    seed: int,
    *,
    progress_description: str = "train second-stage regression",
) -> SurvivalSequenceRegressor:
    """Fit one identity-output Transformer to a pooled landmark signal."""
    _seed_everything(seed)
    training = dataset.with_task("outcome_nuisance", targets, target_mask)
    model = SurvivalSequenceRegressor(**dict(config.transformer)).to(config.device)
    train_loader, validation_loader = patient_train_validation_loaders(
        training,
        batch_size=config.batch_size,
        validation_fraction=float(getattr(config, "validation_fraction", 0.0)),
        validation_seed=int(getattr(config, "validation_seed", 1)),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=float(
            getattr(config, "second_stage_weight_decay", config.weight_decay)
        ),
    )
    stopper = None
    if validation_loader is not None:
        stopper = ValidationEarlyStopping(
            patience=int(getattr(config, "early_stopping_patience", 7)),
            min_delta=float(getattr(config, "early_stopping_min_delta", 0.0)),
            min_epochs=int(getattr(config, "early_stopping_min_epochs", 1)),
        )
    history = []
    with training_progress(
        config.second_stage_epochs, progress_description
    ) as progress:
        for epoch in progress:
            model.train()
            train_sum, train_weight = 0.0, 0.0
            for batch in train_loader:
                tensor_batch = _loader_batch(batch, config.device)
                batch_weight = float(tensor_batch["target_mask"].sum().detach().cpu())
                if batch_weight <= 0:
                    continue
                optimizer.zero_grad()
                loss = model.loss(tensor_batch)
                loss.backward()
                optimizer.step()
                train_sum += float(loss.detach().cpu()) * batch_weight
                train_weight += batch_weight
            row = {"epoch": epoch, "train_loss": train_sum / max(train_weight, 1.0)}
            should_stop = False
            if validation_loader is not None:
                model.eval()
                validation_sum, validation_weight = 0.0, 0.0
                with torch.no_grad():
                    for batch in validation_loader:
                        tensor_batch = _loader_batch(batch, config.device)
                        batch_weight = float(tensor_batch["target_mask"].sum().cpu())
                        if batch_weight <= 0:
                            continue
                        loss = model.loss(tensor_batch)
                        validation_sum += float(loss.cpu()) * batch_weight
                        validation_weight += batch_weight
                validation_loss = validation_sum / max(validation_weight, 1.0)
                row["validation_loss"] = validation_loss
                history.append(row)
                should_stop = stopper.update(model, epoch, validation_loss)
            else:
                row["validation_loss"] = np.nan
                history.append(row)
            update_training_progress(progress, row)
            if should_stop:
                break
    if stopper is not None:
        stopper.restore(model)
        model.best_epoch_ = stopper.best_epoch
    else:
        model.best_epoch_ = len(history)
    model.stopped_epoch_ = len(history)
    model.training_history_ = history
    return model.eval()


@torch.no_grad()
def predict_sequence_regressor(
    model, dataset, config: SyntheticStudyConfig
) -> np.ndarray:
    """Return the unbounded conditional contrast prediction at every time."""
    loader = DataLoader(
        dataset.with_task("survival"), batch_size=config.batch_size, shuffle=False
    )
    predictions = []
    model.eval()
    for batch in loader:
        predictions.append(model(_loader_batch(batch, config.device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)[:, :, 0]


def _independent_splits(config: SyntheticStudyConfig, replicate: int):
    if replicate < 0:
        raise ValueError("replicate cannot be negative")
    base = config.seed + 100_003 * replicate
    nuisance = simulate_split(config.n_nuisance, base + 11, config.simulation)
    second = simulate_split(config.n_second_stage, base + 20_011, config.simulation)
    test = simulate_split(config.n_test, base + 40_009, config.simulation)
    # Globally unique identifiers make accidental split mixing visible in exported data.
    second["patient_id"] = second["patient_id"] + config.n_nuisance
    test["patient_id"] = test["patient_id"] + config.n_nuisance + config.n_second_stage
    return nuisance, second, test


def contrast_signal_targets(
    learners,
    dataset,
    scenario,
    config,
    *,
    probability_source: str = "fitted",
    training_landmarks=None,
):
    """Pooled-landmark RA, IPTW/IPCW and DR contrast targets on a second-stage cohort.

    ``scenario`` is a :class:`StudyScenario` and ``config`` a
    :class:`SyntheticStudyConfig` or :class:`EstimatorSettings`. Returns targets by method, their
    shared eligibility mask and per-landmark weight diagnostics.
    """
    if probability_source not in {"fitted", "oracle"}:
        raise ValueError("probability_source must be 'fitted' or 'oracle'")
    raw = dataset.raw
    n, max_time = raw["treatments"].shape
    targets = {
        "iptw_ipcw": np.zeros((n, max_time, 1), dtype=np.float32),
        "dr": np.zeros((n, max_time, 1), dtype=np.float32),
        "ra": np.zeros((n, max_time, 1), dtype=np.float32),
    }
    mask = np.zeros((n, max_time, 1), dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []
    cache: dict[tuple, np.ndarray] = {}
    landmarks = _pooled_landmarks(max_time, scenario.tau, training_landmarks)
    for landmark in landmarks:
        arm_signals = {}
        for arm, regime in (
            ("one", scenario.regime_one),
            ("zero", scenario.regime_zero),
        ):
            outcome_view = FittedNuisanceView(
                learners[arm],
                dataset,
                landmark,
                landmark + scenario.treatment_length - 1,
                landmark + scenario.tau,
                regime,
                prediction_cache=cache,
            )
            probability_view = (
                outcome_view
                if probability_source == "fitted"
                else OracleProbabilityView(
                    dataset,
                    landmark,
                    landmark + scenario.treatment_length - 1,
                    landmark + scenario.tau,
                )
            )
            arm_signals[arm] = compute_regime_signals(
                raw,
                regime,
                probability_view,
                outcome_view,
                probability_clip=config.probability_clip,
                cumulative_weight_clip=config.cumulative_weight_clip,
            )
            eligible = arm_signals[arm]["eligible"]
            positive_weight = arm_signals[arm]["ipw_weight"][eligible]
            positive_weight = positive_weight[positive_weight > 0]
            diagnostics.append(
                {
                    "landmark": landmark,
                    "arm": arm,
                    "probability_source": probability_source,
                    "n_eligible": int(eligible.sum()),
                    "n_positive_ipw": int(len(positive_weight)),
                    "ipw_ess": (
                        float(
                            positive_weight.sum() ** 2
                            / np.square(positive_weight).sum()
                        )
                        if len(positive_weight)
                        else 0.0
                    ),
                    "ipw_max": (
                        float(positive_weight.max()) if len(positive_weight) else 0.0
                    ),
                    "dr_sd": float(np.nanstd(arm_signals[arm]["dr"][eligible])),
                    "prob_clipped_low_fraction": (
                        float(np.mean(arm_signals[arm]["prob_clipped_low"][eligible]))
                        if eligible.any()
                        else np.nan
                    ),
                    "treatment_probability_floor_fraction": (
                        float(
                            np.mean(
                                arm_signals[arm]["treatment_prob_floored"][eligible]
                            )
                        )
                        if eligible.any()
                        else np.nan
                    ),
                    "censoring_probability_floor_fraction": (
                        float(
                            np.mean(
                                arm_signals[arm]["censoring_prob_floored"][eligible]
                            )
                        )
                        if eligible.any()
                        else np.nan
                    ),
                    "weight_clipped_fraction": (
                        float(np.mean(arm_signals[arm]["weight_clipped"][eligible]))
                        if eligible.any()
                        else np.nan
                    ),
                }
            )
        eligible = arm_signals["one"]["eligible"] & arm_signals["zero"]["eligible"]
        mask[:, landmark, 0] = eligible.astype(np.float32)
        ra_contrast = arm_signals["one"]["gcomp"] - arm_signals["zero"]["gcomp"]
        targets["ra"][:, landmark, 0] = np.nan_to_num(ra_contrast).astype(np.float32)
        for method, signal in (("iptw_ipcw", "ipw"), ("dr", "dr")):
            contrast = arm_signals["one"][signal] - arm_signals["zero"][signal]
            targets[method][:, landmark, 0] = np.nan_to_num(contrast).astype(np.float32)
    return targets, mask, pd.DataFrame(diagnostics)


def _pooled_landmarks(max_time, tau, supplied=None):
    landmarks = (
        tuple(range(max_time - tau + 1)) if supplied is None else tuple(supplied)
    )
    if not landmarks or len(set(landmarks)) != len(landmarks):
        raise ValueError("training landmarks must be nonempty and unique")
    if any(
        not isinstance(t, (int, np.integer)) or t < 0 or t + tau > max_time
        for t in landmarks
    ):
        raise ValueError("training landmark outside available follow-up")
    return landmarks


def fit_contrast_nuisances(
    collection,
    probability_learner,
    scenario,
    config,
    replicate_seed: int,
    *,
    training_landmarks=None,
    trainer_factory=None,
):
    """Fit one pooled-landmark g-computation learner per arm, sharing probability nuisances.

    ``scenario`` supplies ``tau``, ``regime_one`` and ``regime_zero``; ``config`` supplies the
    optimization settings (:class:`SyntheticStudyConfig` or :class:`EstimatorSettings`).
    """
    max_time = collection.train_f.raw["treatments"].shape[1]
    landmarks = _pooled_landmarks(max_time, scenario.tau, training_landmarks)
    learners = {}
    for arm_index, (arm, regime) in enumerate(
        (("one", scenario.regime_one), ("zero", scenario.regime_zero))
    ):
        _seed_everything(replicate_seed + 1009 * arm_index)
        nuisance_config = NuisanceTrainingConfig(
            tau=scenario.tau,
            treatment_sequence=regime,
            training_landmarks=landmarks,
            epochs=config.nuisance_epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            device=config.device,
            model_kwargs=dict(config.transformer),
            seed=replicate_seed + 1009 * arm_index,
            validation_fraction=float(getattr(config, "validation_fraction", 0.0)),
            validation_seed=int(getattr(config, "validation_seed", 1)),
            early_stopping_patience=int(getattr(config, "early_stopping_patience", 7)),
            early_stopping_min_delta=float(
                getattr(config, "early_stopping_min_delta", 0.0)
            ),
            early_stopping_min_epochs=int(
                getattr(config, "early_stopping_min_epochs", 1)
            ),
        )
        learner = SurvivalIterativeGComputationLearner(nuisance_config)
        if trainer_factory is not None:
            learner._trainer = trainer_factory(nuisance_config, arm)
        learners[arm] = learner.fit(
            collection, probability_nuisance_learner=probability_learner
        )
    return learners


def fit_probability_nuisances(
    collection,
    config,
    n_decisions: int,
    seed: int,
    *,
    training_landmarks=None,
    trainer_factory=None,
) -> SurvivalProbabilityNuisanceLearner:
    """Fit the pooled propensity and censoring nuisances shared by every regime and horizon."""
    probability_config = NuisanceTrainingConfig(
        tau=1,
        treatment_sequence=(1,),
        training_landmarks=_pooled_landmarks(n_decisions, 1, training_landmarks),
        epochs=config.nuisance_epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        model_kwargs=dict(
            getattr(config, "probability_transformer", config.transformer)
        ),
        seed=seed,
        validation_fraction=float(getattr(config, "validation_fraction", 0.0)),
        validation_seed=int(getattr(config, "validation_seed", 1)),
        early_stopping_patience=int(getattr(config, "early_stopping_patience", 7)),
        early_stopping_min_delta=float(
            getattr(config, "early_stopping_min_delta", 0.0)
        ),
        early_stopping_min_epochs=int(getattr(config, "early_stopping_min_epochs", 1)),
    )
    _seed_everything(seed)
    learner = SurvivalProbabilityNuisanceLearner(probability_config)
    if trainer_factory is not None:
        learner._trainer = trainer_factory(probability_config, "probability")
    return learner.fit(collection)


def predict_landmark_contrasts(
    dataset,
    learners,
    second_stage_models,
    scenario,
    landmark: int,
    config,
):
    """Predictions at one landmark for every patient at risk there.

    Returns the at-risk row indices and a dictionary holding arm-specific g-computation survival
    (``gcomp_one``, ``gcomp_zero``), the g-computation contrast (``gcomp``) and each second-stage
    regression's prediction under its method name.
    """
    raw = dataset.raw
    rows = np.flatnonzero(raw["at_risk"][:, landmark] > 0)
    cache: dict[tuple, np.ndarray] = {}
    views = {
        arm: FittedNuisanceView(
            learners[arm],
            dataset,
            landmark,
            landmark + scenario.treatment_length - 1,
            landmark + scenario.tau,
            regime,
            prediction_cache=cache,
        )
        for arm, regime in (
            ("one", scenario.regime_one),
            ("zero", scenario.regime_zero),
        )
    }
    survival_one = views["one"].m(raw, landmark, rows)
    survival_zero = views["zero"].m(raw, landmark, rows)
    predictions = {
        "gcomp_one": survival_one,
        "gcomp_zero": survival_zero,
        "gcomp": survival_one - survival_zero,
    }
    for method, model in second_stage_models.items():
        predictions[method] = predict_sequence_regressor(model, dataset, config)[
            rows, landmark
        ]
    return rows, predictions


def _evaluate_scenario(
    collection,
    learners,
    second_stage_models,
    tv_survcaus,
    scenario: StudyScenario,
    config: SyntheticStudyConfig,
    oracle_seed: int,
):
    dataset, raw = collection.test_f, collection.test_f.raw
    landmark = config.evaluation_landmark
    test_rows, estimates = predict_landmark_contrasts(
        dataset, learners, second_stage_models, scenario, landmark, config
    )
    tv_survcaus_effect = (
        tv_survcaus.predict_survival(dataset, landmark, scenario.regime_one)
        - tv_survcaus.predict_survival(dataset, landmark, scenario.regime_zero)
    )[test_rows]
    truth = oracle_pair(
        raw,
        landmark_time=landmark,
        horizons=scenario.tau,
        regime_a=scenario.regime_one,
        regime_b=scenario.regime_zero,
        n_rollouts=config.oracle_rollouts,
        seed=oracle_seed,
        config=config.simulation,
    )
    patient_to_row = {
        int(raw["patient_id"][row]): pos for pos, row in enumerate(test_rows)
    }
    order = truth["patient_id"].map(patient_to_row).to_numpy(int)
    predictions = truth.copy()
    predictions["gcomp_effect"] = np.asarray(estimates["gcomp"])[order]
    for method in second_stage_models:
        predictions[f"{method}_effect"] = np.asarray(estimates[method])[order]
    predictions["tv_survcaus_effect"] = np.asarray(tv_survcaus_effect)[order]
    return predictions


def run_study_replicate(
    config: SyntheticStudyConfig,
    replicate: int,
    *,
    raw_splits=None,
    scaling_params=None,
    training_start=0,
    artifact_callback=None,
    nuisance_trainer_factory=None,
    target_validator=None,
):
    """Run every configured scenario on one nuisance/stage/test split."""
    replicate_seed = config.seed + 100_003 * replicate
    if not 0 <= training_start <= config.evaluation_landmark:
        raise ValueError("training start must not follow evaluation landmark")
    nuisance_raw, second_raw, test_raw = (
        _independent_splits(config, replicate)
        if raw_splits is None
        else (raw_splits[k] for k in ("train", "val", "test"))
    )
    collection = SurvivalSyntheticDatasetCollection.from_raw(
        {
            "train": nuisance_raw,
            "val": second_raw,
            "test": test_raw,
        },
        scaling_params=scaling_params,
    )
    probability_learner = fit_probability_nuisances(
        collection,
        config,
        config.simulation.max_seq_length,
        replicate_seed + 101,
        training_landmarks=range(training_start, config.simulation.max_seq_length),
        trainer_factory=nuisance_trainer_factory,
    )
    if target_validator is not None:
        numerator_data = collection.train_f.with_task("propensity")
        target_validator(
            numerator_data,
            config,
            numerator_data.data["outputs"],
            numerator_data.data["target_mask"],
            "treatment_numerator",
        )
    numerator_learner = TreatmentNumeratorLearner(
        transformer_kwargs=dict(config.probability_transformer),
        epochs=config.nuisance_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        seed=replicate_seed + 307,
        validation_fraction=config.validation_fraction,
        validation_seed=config.validation_seed,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_min_epochs=config.early_stopping_min_epochs,
    ).fit(collection.train_f)
    denominator_probability, numerator_probability = prepare_treatment_probabilities(
        collection.val_f,
        probability_learner,
        numerator_learner,
    )

    if artifact_callback is not None:
        artifact_callback(
            "shared",
            collection=collection,
            probability=probability_learner,
            numerator=numerator_learner,
            config=config,
        )
    metrics, predictions = [], []
    diagnostics = [
        pd.concat(
            [
                training_history_frame(
                    probability_learner.propensity_model,
                    model_role="probability_nuisance",
                    training_task="propensity",
                    replicate=replicate,
                    scenario="shared",
                ),
                training_history_frame(
                    probability_learner.censoring_model,
                    model_role="probability_nuisance",
                    training_task="censoring",
                    replicate=replicate,
                    scenario="shared",
                ),
                training_history_frame(
                    numerator_learner.model,
                    model_role="treatment_numerator",
                    training_task="propensity",
                    replicate=replicate,
                    scenario="shared",
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    ]
    scenario_progress = tqdm(
        config.scenarios,
        desc=f"replicate {replicate + 1}: scenarios",
        unit="scenario",
        leave=False,
        dynamic_ncols=True,
    )
    for scenario_index, scenario in enumerate(scenario_progress):
        scenario_progress.set_postfix_str(scenario.name, refresh=False)
        scenario_seed = replicate_seed + 10_007 * (scenario_index + 1)
        learners = fit_contrast_nuisances(
            collection,
            probability_learner,
            scenario,
            config,
            scenario_seed,
            training_landmarks=range(
                training_start, config.simulation.max_seq_length - scenario.tau + 1
            ),
            trainer_factory=nuisance_trainer_factory,
        )
        targets, target_mask, diagnostic = contrast_signal_targets(
            learners,
            collection.val_f,
            scenario,
            config,
            training_landmarks=range(
                training_start, config.simulation.max_seq_length - scenario.tau + 1
            ),
        )
        oracle_targets, oracle_target_mask, oracle_diagnostic = contrast_signal_targets(
            learners,
            collection.val_f,
            scenario,
            config,
            probability_source="oracle",
            training_landmarks=range(
                training_start, config.simulation.max_seq_length - scenario.tau + 1
            ),
        )
        if target_validator is not None:
            for method, target in targets.items():
                target_validator(
                    collection.val_f,
                    config,
                    target,
                    target_mask,
                    scenario.name + "/" + method,
                )
            for method in ("iptw_ipcw", "dr"):
                target_validator(
                    collection.val_f,
                    config,
                    oracle_targets[method],
                    oracle_target_mask,
                    scenario.name + "/" + method + "_oracle_probability",
                )
            from src.models.tv_survcaus import _build_training_arrays

            tv_arrays, _ = _build_training_arrays(
                collection.val_f,
                tuple(range(config.simulation.max_seq_length - scenario.tau + 1)),
                scenario.treatment_length,
                scenario.tau,
                denominator_probability,
                numerator_probability,
                config.probability_clip,
                config.cumulative_weight_clip,
            )
            target_validator(
                collection.val_f,
                config,
                tv_arrays["event_targets"],
                tv_arrays["interval_observed"],
                scenario.name + "/tv_survcaus",
            )
        second_stage_models = {
            method: fit_sequence_regressor(
                collection.val_f,
                target,
                target_mask,
                config,
                scenario_seed + 50_003 + 1009 * method_index,
                progress_description=f"train {scenario.name} {method}",
            )
            for method_index, (method, target) in enumerate(targets.items())
        }
        oracle_models = {
            f"{method}_oracle_probability": fit_sequence_regressor(
                collection.val_f,
                oracle_targets[method],
                oracle_target_mask,
                config,
                scenario_seed + 80_003 + 1009 * method_index,
                progress_description=(
                    f"train {scenario.name} {method} (oracle probabilities)"
                ),
            )
            for method_index, method in enumerate(("iptw_ipcw", "dr"))
        }
        second_stage_models.update(oracle_models)
        tv_survcaus = TVSurvCausLearner(
            scenario.treatment_length,
            scenario.tau,
            alpha=config.tv_survcaus_alpha,
            representation_dim=config.tv_survcaus_representation_dim,
            regime_embedding_dim=config.tv_survcaus_regime_embedding_dim,
            epochs=(
                config.second_stage_epochs
                if config.tv_survcaus_epochs is None
                else config.tv_survcaus_epochs
            ),
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.second_stage_weight_decay,
            device=config.device,
            probability_clip=config.probability_clip,
            cumulative_weight_clip=config.cumulative_weight_clip,
            mmd_bandwidth=config.tv_survcaus_mmd_bandwidth,
            transformer_kwargs=dict(config.transformer),
            seed=scenario_seed + 70_001,
            validation_fraction=config.validation_fraction,
            validation_seed=config.validation_seed,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            early_stopping_min_epochs=config.early_stopping_min_epochs,
        ).fit(
            collection.val_f,
            denominator_probability,
            numerator_probability,
        )
        if artifact_callback is not None:
            artifact_callback(
                "scenario",
                collection=collection,
                learners=learners,
                heads=second_stage_models,
                tv=tv_survcaus,
                scenario=scenario,
                config=config,
            )
        prediction = _evaluate_scenario(
            collection,
            learners,
            second_stage_models,
            tv_survcaus,
            scenario,
            config,
            oracle_seed=scenario_seed + 90_001,
        ).assign(scenario=scenario.name, replicate=replicate)
        metric = effect_metrics(prediction).assign(
            scenario=scenario.name,
            treatment_length=scenario.treatment_length,
            tau=scenario.tau,
            replicate=replicate,
        )
        oracle_benchmark = effect_metrics(
            prediction,
            methods=tuple(oracle_models),
        ).assign(
            diagnostic="oracle_probability_benchmark",
            scenario=scenario.name,
            treatment_length=scenario.treatment_length,
            tau=scenario.tau,
            replicate=replicate,
        )
        history_frames = []
        for arm, learner in learners.items():
            for relative_step, model in learner.q_models.items():
                history_frames.append(
                    training_history_frame(
                        model,
                        model_role="outcome_nuisance",
                        training_task="q",
                        arm=arm,
                        relative_step=relative_step,
                        scenario=scenario.name,
                        replicate=replicate,
                    )
                )
            for relative_step, model in learner.m_models.items():
                history_frames.append(
                    training_history_frame(
                        model,
                        model_role="outcome_nuisance",
                        training_task="m",
                        arm=arm,
                        relative_step=relative_step,
                        scenario=scenario.name,
                        replicate=replicate,
                    )
                )
        for method, model in second_stage_models.items():
            history_frames.append(
                training_history_frame(
                    model,
                    model_role="second_stage",
                    training_task=method,
                    scenario=scenario.name,
                    replicate=replicate,
                )
            )
        history_frames.append(
            training_history_frame(
                tv_survcaus.model,
                model_role="tv_survcaus",
                training_task="tv_survcaus",
                scenario=scenario.name,
                replicate=replicate,
            )
        )
        diagnostic = pd.concat(
            [
                diagnostic.assign(diagnostic="weight_support"),
                oracle_diagnostic.assign(
                    diagnostic="oracle_probability_weight_support"
                ),
                pd.DataFrame([tv_survcaus.diagnostics_]),
                oracle_benchmark,
                *history_frames,
            ],
            ignore_index=True,
            sort=False,
        ).assign(
            scenario=scenario.name,
            treatment_length=scenario.treatment_length,
            tau=scenario.tau,
            replicate=replicate,
        )
        metrics.append(metric)
        predictions.append(prediction)
        diagnostics.append(diagnostic)
    return (
        pd.concat(metrics, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
        pd.concat(diagnostics, ignore_index=True),
    )


def run_synthetic_study(
    config: SyntheticStudyConfig | None = None,
    output_dir: str | Path | None = None,
) -> SyntheticStudyResult:
    """Run the full study, optionally persisting tidy tables after completion."""
    config = config or SyntheticStudyConfig()
    metric_frames, prediction_frames, diagnostic_frames = [], [], []
    replicates = tqdm(
        range(config.n_replicates),
        desc="synthetic study replicates",
        unit="replicate",
        dynamic_ncols=True,
    )
    for replicate in replicates:
        metric, prediction, diagnostic = run_study_replicate(config, replicate)
        metric_frames.append(metric)
        prediction_frames.append(prediction)
        diagnostic_frames.append(diagnostic)
    metrics = pd.concat(metric_frames, ignore_index=True)
    result = SyntheticStudyResult(
        metrics=metrics,
        summary=summarize_study(metrics),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        diagnostics=pd.concat(diagnostic_frames, ignore_index=True),
        config=config,
    )
    if output_dir is not None:
        result.save(output_dir)
    return result
