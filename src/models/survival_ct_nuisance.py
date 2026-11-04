"""Causal-Transformer models and pooled-landmark nuisance recursion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

_PROBABILITY_TASKS = {"propensity", "censoring"}
_OUTCOME_TASKS = {"survival", "outcome_nuisance", "q", "m", "continuation"}
_NUISANCE_TASKS = _PROBABILITY_TASKS | _OUTCOME_TASKS


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positions used by the WO-learner Transformer."""

    def __init__(self, max_seq_length: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        positions = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        rates = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_seq_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * rates)
        encoding[:, 1::2] = torch.cos(positions * rates)
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs + self.encoding[:, : inputs.shape[1]])


class SurvivalHistoryTransformer(nn.Module):
    """Small causal encoder matching the WO-learner reference architecture.

    The reference concatenates covariates, previous outcomes and previous treatments,
    applies one input projection, sinusoidal positions and a standard causal
    ``TransformerEncoder``.  This repository additionally repeats its separately stored
    static baseline features at each time point; that preserves the information set used
    by the survival estimators without changing the encoder architecture.
    """

    dim_treatments = 2
    dim_covariates = 4
    dim_static_features = 3
    dim_outcome = 1

    def __init__(
        self,
        dim_treatments: int = 2,
        dim_covariates: int = 4,
        dim_static_features: int = 3,
        dim_outcome: int = 1,
        d_model: int = 30,
        num_heads: int = 3,
        num_encoder_layers: int = 1,
        dim_feedforward: int = 20,
        dropout: float = 0.1,
        dim_output_hidden_units: int = 20,
        max_seq_length: int = 64,
        positional_dropout: float = 0.1,
    ):
        super().__init__()
        d_model = int(d_model)
        num_encoder_layers = int(num_encoder_layers)
        dropout = float(dropout)
        if d_model % 2:
            raise ValueError("d_model must be even for sinusoidal positions")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if min(dim_treatments, dim_covariates, dim_static_features, dim_outcome) < 1:
            raise ValueError("all input dimensions must be positive")
        if (
            min(
                d_model,
                num_encoder_layers,
                num_heads,
                dim_feedforward,
                dim_output_hidden_units,
                max_seq_length,
            )
            < 1
        ):
            raise ValueError("Transformer sizes must be positive")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= positional_dropout < 1.0:
            raise ValueError("dropout rates must be in [0, 1)")
        self.dim_treatments = dim_treatments
        self.dim_covariates = dim_covariates
        self.dim_static_features = dim_static_features
        self.dim_outcome = dim_outcome
        self.d_model = d_model
        self.br_size = d_model
        self.dim_output_hidden_units = dim_output_hidden_units
        self.max_seq_length = max_seq_length
        input_dim = dim_covariates + dim_outcome + dim_treatments + dim_static_features
        self.input_layer = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionalEncoding(
            max_seq_length, d_model, positional_dropout
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def output_head(self, input_dim: int | None = None) -> nn.Sequential:
        """Return the two-layer ReLU head used throughout the reference code."""
        width = self.d_model if input_dim is None else int(input_dim)
        return nn.Sequential(
            nn.Linear(width, self.dim_output_hidden_units),
            nn.ReLU(),
            nn.Linear(self.dim_output_hidden_units, 1),
        )

    def build_br(
        self,
        prev_treatments,
        covariates,
        prev_outputs,
        static_features,
        active_entries,
    ):
        if covariates.shape[1] > self.max_seq_length:
            raise ValueError(
                f"sequence length {covariates.shape[1]} exceeds transformer maximum "
                f"{self.max_seq_length}"
            )
        length = covariates.shape[1]
        static = static_features.float().unsqueeze(1).expand(-1, length, -1)
        history = torch.cat(
            (
                covariates.float(),
                prev_outputs.float(),
                prev_treatments.float(),
                static,
            ),
            dim=-1,
        )
        encoded = self.position(self.input_layer(history))
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=encoded.device),
            diagonal=1,
        )
        padding_mask = active_entries.squeeze(-1) <= 0
        encoded = self.transformer_encoder(
            encoded,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return encoded * active_entries.float()


class SurvivalCTNuisanceModel(nn.Module):
    """Task-specific probability or continuation head over transformer histories."""

    def __init__(
        self,
        nuisance_task="survival",
        **encoder_kwargs,
    ):
        super().__init__()
        if nuisance_task not in _NUISANCE_TASKS:
            raise ValueError(f"unknown nuisance task: {nuisance_task!r}")
        self.ct_model = SurvivalHistoryTransformer(**encoder_kwargs)
        self.nuisance_task = nuisance_task
        br_size = self.ct_model.br_size
        dim_t = self.ct_model.dim_treatments
        output_head = self.ct_model.output_head

        self.propensity_head = output_head(br_size)
        self.censoring_head = output_head(br_size)
        self.survival_head = output_head(br_size + dim_t)
        self.continuation_head = output_head(br_size)

    def build_br(self, batch):
        return self.ct_model.build_br(
            batch["prev_treatments"],
            batch["covariates"],
            batch["prev_outputs"],
            batch["static_features"],
            batch["active_entries"],
        )

    def forward(self, batch, task=None, current_treatments=None):
        task = task or self.nuisance_task
        br = self.build_br(batch)
        if task == "propensity":
            return self.propensity_head(br)
        if task == "censoring":
            return self.censoring_head(br)
        if task in {"survival", "outcome_nuisance", "q"}:
            treatment = (
                batch["current_treatments"]
                if current_treatments is None
                else current_treatments
            )
            return self.survival_head(torch.cat([br, treatment.float()], dim=-1))
        if task in {"m", "continuation"}:
            return self.continuation_head(br)
        raise ValueError(f"unknown nuisance task: {task!r}")

    def predict_proba(self, batch, task=None, current_treatments=None):
        return torch.sigmoid(self.forward(batch, task, current_treatments))

    def loss(self, batch, task=None):
        task = task or self.nuisance_task
        prediction = self.forward(batch, task)
        target = batch["outputs"].float()
        mask = batch["target_mask"].float()
        if task in _OUTCOME_TASKS:
            elementwise = F.mse_loss(
                torch.sigmoid(prediction), target, reduction="none"
            )
        else:
            elementwise = F.binary_cross_entropy_with_logits(
                prediction,
                target,
                reduction="none",
            )
        return (elementwise * mask).sum() / mask.sum().clamp_min(1.0)


class SurvivalSequenceRegressor(nn.Module):
    """Identity-output second-stage regression at every eligible landmark."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = SurvivalHistoryTransformer(**encoder_kwargs)
        self.head = self.encoder.output_head()

    def forward(self, batch):
        br = self.encoder.build_br(
            batch["prev_treatments"],
            batch["covariates"],
            batch["prev_outputs"],
            batch["static_features"],
            batch["active_entries"],
        )
        return self.head(br)

    def loss(self, batch):
        prediction = self.forward(batch)
        target = batch["outputs"].float()
        mask = batch["target_mask"].float()
        elementwise = F.mse_loss(prediction, target, reduction="none")
        return (elementwise * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass(frozen=True)
class NuisanceTrainingConfig:
    tau: int
    treatment_sequence: Sequence[int]
    landmark_time: int = 0
    training_landmarks: Sequence[int] | None = None
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 1
    validation_fraction: float = 0.0
    validation_seed: int = 1
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 1
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        treatment_sequence = tuple(int(action) for action in self.treatment_sequence)
        object.__setattr__(self, "treatment_sequence", treatment_sequence)
        if self.training_landmarks is not None:
            landmarks = tuple(int(time) for time in self.training_landmarks)
            object.__setattr__(self, "training_landmarks", landmarks)
        if self.tau < 1:
            raise ValueError("tau must be positive")
        if len(self.treatment_sequence) == 0:
            raise ValueError("treatment_sequence must be non-empty")
        if len(self.treatment_sequence) > self.tau:
            raise ValueError("treatment_sequence cannot be longer than tau")
        if any(int(action) not in (0, 1) for action in self.treatment_sequence):
            raise ValueError("treatment_sequence must be binary")
        if self.landmark_time < 0:
            raise ValueError("landmark_time cannot be negative")
        if self.training_landmarks is not None:
            if not self.training_landmarks:
                raise ValueError("training_landmarks cannot be empty")
            if any(landmark < 0 for landmark in self.training_landmarks):
                raise ValueError("training landmarks cannot be negative")
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0:
            raise ValueError("invalid optimization configuration")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if min(self.early_stopping_patience, self.early_stopping_min_epochs) < 1:
            raise ValueError(
                "early-stopping patience and minimum epochs must be positive"
            )
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.early_stopping_min_epochs > self.epochs:
            raise ValueError("early_stopping_min_epochs cannot exceed epochs")


def _mask_before(dataset, first_time: int):
    if not 0 <= first_time < dataset.data["target_mask"].shape[1]:
        raise ValueError("first training time is outside the dataset")
    dataset.data["target_mask"][:, :first_time, :] = 0.0
    return dataset


class SurvivalProbabilityNuisanceLearner:
    """Pooled treatment and censoring nuisances, shared by every regime and horizon."""

    def __init__(self, config: NuisanceTrainingConfig):
        self.config = config
        self.propensity_model = None
        self.censoring_model = None

    @property
    def trainer(self):
        if getattr(self, "_trainer", None) is None:
            from .transformer_training import TransformerTrainer

            self._trainer = TransformerTrainer(self.config)
        return self._trainer

    def _new_model(self, task):
        return self.trainer.new_model(task)

    def _fit(self, model, dataset, task, *, show_progress=True):
        return self.trainer.fit(model, dataset, task, show_progress=show_progress)

    def _predict_array(self, model, dataset, task, treatment_value=None):
        return self.trainer.predict_array(model, dataset, task, treatment_value)

    def fit(self, collection):
        cfg = self.config
        if cfg.training_landmarks is not None:
            first_time = min(int(t) for t in cfg.training_landmarks)
        else:
            # Preserve the fixed-landmark API for focused diagnostics. The canonical
            # study passes ``training_landmarks`` and therefore pools all valid times.
            first_time = int(cfg.landmark_time)
        propensity = _mask_before(
            collection.with_task("propensity").train_f, first_time
        )
        censoring = _mask_before(
            collection.with_task("censoring").train_f, first_time + 1
        )
        self.propensity_model = self._fit(
            self._new_model("propensity"),
            propensity,
            "propensity",
        )
        self.censoring_model = self._fit(
            self._new_model("censoring"),
            censoring,
            "censoring",
        )
        return self


class SurvivalIterativeGComputationLearner:
    """Backward Q/m recursion pooled over patient-disjoint training landmarks.

    Models are indexed by relative step ``r`` before the horizon. One model therefore
    learns from all absolute calendar times ``landmark + r``. Within the intervention
    window, continuation is exactly Q evaluated at the prescribed action; a separate
    m-regression is fitted only after treatment returns to the natural course.
    """

    def __init__(self, config: NuisanceTrainingConfig):
        self.config = config
        self.propensity_model = None
        self.censoring_model = None
        self.q_models: dict[int, Any] = {}
        self.m_models: dict[int, Any] = {}
        self.training_landmarks_: tuple[int, ...] = ()

    @property
    def trainer(self):
        if getattr(self, "_trainer", None) is None:
            from .transformer_training import TransformerTrainer

            self._trainer = TransformerTrainer(self.config)
        return self._trainer

    def _new_model(self, task):
        return self.trainer.new_model(task)

    def _fit(self, model, dataset, task, *, show_progress=True):
        return self.trainer.fit(model, dataset, task, show_progress=show_progress)

    def _predict_array(self, model, dataset, task, treatment_value=None):
        return self.trainer.predict_array(model, dataset, task, treatment_value)

    def _fit_and_predict(self, collection, targets, mask, task, treatment_values):
        dataset = collection.with_task(
            "outcome_nuisance", train_targets=targets, train_target_mask=mask
        ).train_f
        model = self._fit(self._new_model(task), dataset, task, show_progress=False)
        survival = collection.train_f.with_task("survival")
        predictions = {
            value: self._predict_array(model, survival, task, value)
            for value in treatment_values
        }
        return model, predictions

    def _resolve_landmarks(self, max_time: int) -> tuple[int, ...]:
        cfg = self.config
        landmarks = (
            tuple(sorted({int(t) for t in cfg.training_landmarks}))
            if cfg.training_landmarks is not None
            else (int(cfg.landmark_time),)
        )
        if not landmarks:
            raise ValueError("at least one training landmark is required")
        if min(landmarks) < 0 or max(landmarks) + cfg.tau > max_time:
            raise ValueError("training landmark plus tau exceeds available follow-up")
        return landmarks

    def fit(self, collection, probability_nuisance_learner=None):
        cfg = self.config
        base_train = collection.train_f.with_task("survival")
        n, max_time = base_train.data["outputs"].shape[:2]
        landmarks = self._resolve_landmarks(max_time)
        self.training_landmarks_ = landmarks

        if probability_nuisance_learner is None:
            probability_nuisance_learner = SurvivalProbabilityNuisanceLearner(cfg).fit(
                collection
            )
        if (
            probability_nuisance_learner.propensity_model is None
            or probability_nuisance_learner.censoring_model is None
        ):
            raise ValueError("probability_nuisance_learner must be fitted")
        self.propensity_model = probability_nuisance_learner.propensity_model
        self.censoring_model = probability_nuisance_learner.censoring_model

        continuation_next = np.ones((n, len(landmarks)), dtype=np.float32)
        regime_label = "".join(map(str, cfg.treatment_sequence))
        n_models = 2 * cfg.tau - len(cfg.treatment_sequence)
        model_progress = tqdm(
            total=n_models,
            desc=f"train Q/m nuisances tau={cfg.tau} regime={regime_label}",
            unit="model",
            leave=False,
            dynamic_ncols=True,
        )
        for r in range(cfg.tau - 1, -1, -1):
            q_targets = np.zeros_like(base_train.data["outputs"], dtype=np.float32)
            q_mask = np.zeros_like(base_train.data["target_mask"], dtype=np.float32)
            for position, landmark in enumerate(landmarks):
                k = landmark + r
                at_risk = base_train.raw["at_risk"][:, k].astype(np.float32)
                survived = np.nan_to_num(
                    base_train.raw["event_free_next"][:, k], nan=0.0
                ).astype(np.float32)
                q_targets[:, k, 0] = survived * continuation_next[:, position]
                q_mask[:, k, 0] = at_risk

            prescribed = r < len(cfg.treatment_sequence)
            treatment_values = (
                [int(cfg.treatment_sequence[r])] if prescribed else [None]
            )
            q_model, q_prediction = self._fit_and_predict(
                collection,
                q_targets,
                q_mask,
                "q",
                treatment_values,
            )
            self.q_models[r] = q_model
            model_progress.update(1)
            model_progress.set_postfix(
                model=f"Q[{r}]",
                best=int(q_model.best_epoch_),
                stopped=int(q_model.stopped_epoch_),
                refresh=False,
            )

            continuation_current = np.full_like(continuation_next, np.nan)
            if prescribed:
                action = int(cfg.treatment_sequence[r])
                for position, landmark in enumerate(landmarks):
                    continuation_current[:, position] = q_prediction[action][
                        :, landmark + r, 0
                    ]
            else:
                m_targets = np.zeros_like(q_targets)
                m_mask = np.zeros_like(q_mask)
                for position, landmark in enumerate(landmarks):
                    k = landmark + r
                    m_targets[:, k, 0] = q_prediction[None][:, k, 0]
                    m_mask[:, k, 0] = q_mask[:, k, 0]
                m_model, m_prediction = self._fit_and_predict(
                    collection,
                    m_targets,
                    m_mask,
                    "m",
                    [None],
                )
                self.m_models[r] = m_model
                model_progress.update(1)
                model_progress.set_postfix(
                    model=f"m[{r}]",
                    best=int(m_model.best_epoch_),
                    stopped=int(m_model.stopped_epoch_),
                    refresh=False,
                )
                for position, landmark in enumerate(landmarks):
                    continuation_current[:, position] = m_prediction[None][
                        :, landmark + r, 0
                    ]
            continuation_next = continuation_current
        model_progress.close()
        return self

    def predict(self, dataset, landmark_time: int | None = None) -> np.ndarray:
        if 0 not in self.q_models:
            raise RuntimeError("fit the g-computation learner before prediction")
        landmark = (
            int(landmark_time)
            if landmark_time is not None
            else int(self.training_landmarks_[0])
        )
        if (
            landmark < 0
            or landmark + self.config.tau > dataset.raw["treatments"].shape[1]
        ):
            raise ValueError("prediction landmark plus tau exceeds available follow-up")
        survival = dataset.with_task("survival")
        action = int(self.config.treatment_sequence[0])
        prediction = self._predict_array(
            self.q_models[0], survival, "q", treatment_value=action
        )
        return prediction[:, landmark, 0]
