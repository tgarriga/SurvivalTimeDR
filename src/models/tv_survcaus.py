"""Transformer adaptation of TV-SurvCaus (Abraich, 2025).

This is not an exact reproduction of unpublished TV-SurvCaus code. It preserves
longitudinal history encoding, a learned balanced representation, RBF-MMD across
factual treatment sequences, stabilized sequential treatment weighting, a
censoring-aware survival likelihood, and treatment-sequence-conditioned
counterfactual prediction.

The deliberate adaptations are that the repository's causal Transformer replaces the
proposed RNN/LSTM/GRU encoder, interventions have finite length and return to the
natural course afterwards, hazards receive only causal treatment prefixes, and
independent sigmoid discrete hazards are used. The latter makes the paper's stated
hazard/product-survival formulation internally coherent; hazards are not constrained
to sum to one.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.survival_sim.timing import observed_survival_next

from .survival_ct_nuisance import SurvivalHistoryTransformer
from .transformer_training import (
    ValidationEarlyStopping,
    patient_train_validation_loaders,
    training_progress,
    update_training_progress,
)

_HISTORY_BATCH_KEYS = {
    "active_entries",
    "covariates",
    "prev_outputs",
    "prev_treatments",
    "static_features",
}
_TRAINING_BATCH_KEYS = _HISTORY_BATCH_KEYS | {
    "complete_regime",
    "event_targets",
    "factual_regimes",
    "interval_observed",
    "stabilized_weights",
}


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_batch(batch, device: str, keys) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.items()
        if key in keys
    }


def _validate_probability_array(
    probability: np.ndarray,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if probability.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {probability.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"{name} must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return probability


def survival_interval_targets(
    raw: Mapping[str, Any],
    landmarks: Sequence[int],
    tau: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build discrete event targets and observed-interval masks.

    Interval ``r`` starts at ``k = landmark + r`` and ends at ``k + 1``. The
    repository's timing helper treats censoring exactly at ``k + 1`` as observed
    survival through that endpoint. ``at_risk[:, k]`` therefore identifies exactly
    the intervals whose binary event outcome is observed.
    """
    landmarks = tuple(int(landmark) for landmark in landmarks)
    if tau < 1 or not landmarks:
        raise ValueError("tau and landmarks must be non-empty and positive")
    at_risk = np.asarray(raw["at_risk"], dtype=float)
    if at_risk.ndim != 2:
        raise ValueError("at_risk must be a patient-by-time array")
    n_patients, max_time = at_risk.shape
    if min(landmarks) < 0 or max(landmarks) + tau > max_time:
        raise ValueError("landmark plus tau exceeds available follow-up")

    event_targets = np.zeros((n_patients, len(landmarks), tau), dtype=np.float32)
    interval_observed = np.zeros_like(event_targets)
    for position, landmark in enumerate(landmarks):
        for relative_time in range(tau):
            decision_time = landmark + relative_time
            observed = at_risk[:, decision_time] > 0
            interval_observed[:, position, relative_time] = observed
            event_targets[:, position, relative_time] = observed * (
                1.0 - observed_survival_next(raw, decision_time)
            )
    return event_targets, interval_observed


def stabilized_treatment_weights(
    raw: Mapping[str, Any],
    denominator_probability: np.ndarray,
    numerator_probability: np.ndarray,
    landmarks: Sequence[int],
    treatment_length: int,
    tau: int,
    probability_clip: float,
    cumulative_weight_clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return treatment-only stabilized prefix weights and clipping indicators.

    The supplied arrays are probabilities of treatment one. For future interval
    ``r = 1, ..., tau``, the weight is

    ``prod_{j=0}^{min(r, L)-1} p(A_{t+j}|A_{<t+j}) / e(A_{t+j}|H_{t+j})``.

    Thus early failures need only their observed treatment prefix, and the complete
    intervention-window weight is held fixed after ``L``. No censoring probability
    appears in this function.
    """
    landmarks = tuple(int(landmark) for landmark in landmarks)
    if treatment_length < 1 or tau < treatment_length:
        raise ValueError("treatment_length must be positive and no larger than tau")
    if not landmarks:
        raise ValueError("at least one landmark is required")
    if not 0.0 <= probability_clip < 0.5:
        raise ValueError("probability_clip must be in [0, 0.5)")
    if cumulative_weight_clip is not None and cumulative_weight_clip <= 0:
        raise ValueError("cumulative_weight_clip must be positive or None")

    treatments = np.asarray(raw["treatments"], dtype=float)
    treatment_observed = np.asarray(raw["treatment_observed"], dtype=float)
    at_risk = np.asarray(raw["at_risk"], dtype=float)
    if treatments.ndim != 2 or treatment_observed.shape != treatments.shape:
        raise ValueError("treatment arrays must share patient-by-time shape")
    if at_risk.shape != treatments.shape:
        raise ValueError("at_risk and treatments must share shape")
    n_patients, max_time = treatments.shape
    if min(landmarks) < 0 or max(landmarks) + tau > max_time:
        raise ValueError("landmark plus tau exceeds available follow-up")

    denominator_one = _validate_probability_array(
        denominator_probability, treatments.shape, "denominator_probability"
    )
    numerator_one = _validate_probability_array(
        numerator_probability, treatments.shape, "numerator_probability"
    )
    weights = np.zeros((n_patients, len(landmarks), tau), dtype=np.float32)
    clipped = np.zeros_like(weights, dtype=bool)
    numerical_floor = np.finfo(np.float64).eps
    lower = max(float(probability_clip), numerical_floor)

    for position, landmark in enumerate(landmarks):
        cumulative = np.ones(n_patients, dtype=float)
        prefix_observed = at_risk[:, landmark] > 0
        clipped_so_far = np.zeros(n_patients, dtype=bool)
        for relative_time in range(tau):
            if relative_time < treatment_length:
                decision_time = landmark + relative_time
                observed_now = (treatment_observed[:, decision_time] > 0) & np.isfinite(
                    treatments[:, decision_time]
                )
                prefix_observed &= observed_now
                action = np.nan_to_num(treatments[:, decision_time], nan=0.0).astype(
                    int
                )
                if np.any(prefix_observed & ~np.isin(action, (0, 1))):
                    raise ValueError("observed treatments must be binary")
                denominator = np.where(
                    action == 1,
                    denominator_one[:, decision_time],
                    1.0 - denominator_one[:, decision_time],
                )
                numerator = np.where(
                    action == 1,
                    numerator_one[:, decision_time],
                    1.0 - numerator_one[:, decision_time],
                )
                # This is an inverse-probability denominator. Near-one values are
                # well supported and must not be reduced by a symmetric upper clip.
                factor = numerator / np.maximum(denominator, lower)
                updated = cumulative * factor
                if cumulative_weight_clip is not None:
                    clipped_now = prefix_observed & (updated > cumulative_weight_clip)
                    clipped_so_far |= clipped_now
                    updated = np.minimum(updated, cumulative_weight_clip)
                cumulative = np.where(prefix_observed, updated, 0.0)
            weights[:, position, relative_time] = cumulative.astype(np.float32)
            clipped[:, position, relative_time] = clipped_so_far & prefix_observed

    if not np.isfinite(weights).all():
        raise ValueError("stabilized weights must be finite")
    return weights, clipped


def _resolve_mmd_bandwidth(
    points: torch.Tensor,
    bandwidth: float | torch.Tensor | None,
) -> torch.Tensor:
    if bandwidth is not None:
        value = torch.as_tensor(
            bandwidth, dtype=points.dtype, device=points.device
        ).detach()
        if value.numel() != 1 or not torch.isfinite(value) or value <= 0:
            raise ValueError("MMD bandwidth must be a finite positive scalar")
        return value

    detached = points.detach()
    if detached.shape[0] > 512:
        stride = ceil(detached.shape[0] / 512)
        detached = detached[::stride]
    distances = torch.pdist(detached)
    positive = distances[distances > 0]
    if positive.numel() == 0:
        return detached.new_tensor(1.0)
    return positive.median().clamp_min(torch.finfo(detached.dtype).eps ** 0.5)


def rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Biased RBF-kernel MMD squared, with a detached median heuristic by default."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must be non-empty matrices with equal width")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("MMD groups cannot be empty")
    scale = _resolve_mmd_bandwidth(torch.cat([x, y], dim=0), bandwidth)

    def kernel(left, right):
        squared_distance = torch.cdist(left, right).square()
        return torch.exp(-squared_distance / (2.0 * scale.square()))

    value = kernel(x, x).mean() + kernel(y, y).mean()
    value = value - 2.0 * kernel(x, y).mean()
    return value.clamp_min(0.0)


def _sequence_balance_mmd(
    representations: torch.Tensor,
    factual_regimes: torch.Tensor,
    complete_regime: torch.Tensor,
    bandwidth: float | None,
) -> tuple[torch.Tensor, int]:
    """Average MMD over all usable factual sequence-group pairs in a batch."""
    flat_z = representations.reshape(-1, representations.shape[-1])
    flat_regime = factual_regimes.reshape(-1, factual_regimes.shape[-1]).long()
    complete = complete_regime.reshape(-1).bool()
    if complete.sum() < 4:
        return flat_z.sum() * 0.0, 0

    powers = 2 ** torch.arange(
        flat_regime.shape[-1], device=flat_regime.device, dtype=torch.long
    )
    codes = (flat_regime * powers).sum(dim=-1)
    groups = [
        flat_z[complete & (codes == code)] for code in torch.unique(codes[complete])
    ]
    groups = [group for group in groups if group.shape[0] >= 2]
    if len(groups) < 2:
        return flat_z.sum() * 0.0, len(groups)

    common_bandwidth = _resolve_mmd_bandwidth(torch.cat(groups, dim=0), bandwidth)
    pairwise = [
        rbf_mmd2(groups[left], groups[right], common_bandwidth)
        for left in range(len(groups))
        for right in range(left + 1, len(groups))
    ]
    return torch.stack(pairwise).mean(), len(groups)


class TreatmentNumeratorModel(nn.Module):
    """Estimate ``P(A_k=1 | A_0,...,A_{k-1}, k)`` on the at-risk set.

    The causal Transformer receives the shifted treatment history and positional
    encodings, while all covariate and baseline values are zeroed. It therefore cannot
    use patient covariates in the stabilized-weight numerator.
    """

    def __init__(self, **transformer_kwargs):
        super().__init__()
        self.encoder = SurvivalHistoryTransformer(**transformer_kwargs)
        self.head = nn.Linear(self.encoder.br_size, 1)

    def forward(self, batch):
        representation = self.encoder.build_br(
            batch["prev_treatments"],
            torch.zeros_like(batch["covariates"]),
            torch.zeros_like(batch["prev_outputs"]),
            torch.zeros_like(batch["static_features"]),
            batch["active_entries"],
        )
        return self.head(representation)

    def loss(self, batch):
        elementwise = F.binary_cross_entropy_with_logits(
            self(batch), batch["outputs"].float(), reduction="none"
        )
        mask = batch["target_mask"].float()
        return (elementwise * mask).sum() / mask.sum().clamp_min(1.0)


class TreatmentNumeratorLearner:
    """Fit the stabilized-weight numerator once on the nuisance cohort."""

    def __init__(
        self,
        *,
        transformer_kwargs: Mapping[str, Any],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        device: str,
        seed: int,
        validation_fraction: float = 0.0,
        validation_seed: int = 1,
        early_stopping_patience: int = 7,
        early_stopping_min_delta: float = 0.0,
        early_stopping_min_epochs: int = 1,
    ):
        if min(epochs, batch_size) < 1 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("invalid numerator optimization configuration")
        self.transformer_kwargs = dict(transformer_kwargs)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.seed = int(seed)
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.early_stopping_min_epochs = int(early_stopping_min_epochs)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if min(self.early_stopping_patience, self.early_stopping_min_epochs) < 1:
            raise ValueError("invalid early-stopping configuration")
        if self.early_stopping_min_epochs > self.epochs:
            raise ValueError("early_stopping_min_epochs cannot exceed epochs")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        self.model: TreatmentNumeratorModel | None = None
        self.training_history_: list[dict[str, float]] = []

    def fit(self, dataset):
        _seed_everything(self.seed)
        training = dataset.with_task("propensity")
        self.model = TreatmentNumeratorModel(**self.transformer_kwargs).to(self.device)
        train_loader, validation_loader = patient_train_validation_loaders(
            training,
            batch_size=self.batch_size,
            validation_fraction=self.validation_fraction,
            validation_seed=self.validation_seed,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        stopper = None
        if validation_loader is not None:
            stopper = ValidationEarlyStopping(
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
                min_epochs=self.early_stopping_min_epochs,
            )
        with training_progress(self.epochs, "train treatment numerator") as progress:
            for epoch in progress:
                self.model.train()
                train_sum, train_weight = 0.0, 0.0
                for batch in train_loader:
                    tensor_batch = _tensor_batch(
                        batch,
                        self.device,
                        _HISTORY_BATCH_KEYS | {"outputs", "target_mask"},
                    )
                    batch_weight = float(
                        tensor_batch["target_mask"].sum().detach().cpu()
                    )
                    if batch_weight <= 0:
                        continue
                    optimizer.zero_grad()
                    loss = self.model.loss(tensor_batch)
                    loss.backward()
                    optimizer.step()
                    train_sum += float(loss.detach().cpu()) * batch_weight
                    train_weight += batch_weight
                row = {"epoch": epoch, "train_loss": train_sum / max(train_weight, 1.0)}
                should_stop = False
                if validation_loader is not None:
                    self.model.eval()
                    validation_sum, validation_weight = 0.0, 0.0
                    with torch.no_grad():
                        for batch in validation_loader:
                            tensor_batch = _tensor_batch(
                                batch,
                                self.device,
                                _HISTORY_BATCH_KEYS | {"outputs", "target_mask"},
                            )
                            batch_weight = float(
                                tensor_batch["target_mask"].sum().cpu()
                            )
                            if batch_weight <= 0:
                                continue
                            loss = self.model.loss(tensor_batch)
                            validation_sum += float(loss.cpu()) * batch_weight
                            validation_weight += batch_weight
                    validation_loss = validation_sum / max(validation_weight, 1.0)
                    row["validation_loss"] = validation_loss
                    self.training_history_.append(row)
                    should_stop = stopper.update(self.model, epoch, validation_loss)
                else:
                    row["validation_loss"] = np.nan
                    self.training_history_.append(row)
                update_training_progress(progress, row)
                if should_stop:
                    break
        if stopper is not None:
            stopper.restore(self.model)
            self.model.best_epoch_ = stopper.best_epoch
        else:
            self.model.best_epoch_ = len(self.training_history_)
        self.model.stopped_epoch_ = len(self.training_history_)
        self.model.training_history_ = self.training_history_
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, dataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit the treatment numerator before prediction")
        prediction = []
        loader = DataLoader(
            dataset.with_task("propensity"),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.model.eval()
        for batch in loader:
            tensor_batch = _tensor_batch(batch, self.device, _HISTORY_BATCH_KEYS)
            prediction.append(torch.sigmoid(self.model(tensor_batch)).cpu().numpy())
        return np.concatenate(prediction, axis=0)[:, :, 0]


def prepare_treatment_probabilities(
    dataset,
    probability_learner,
    numerator_learner: TreatmentNumeratorLearner,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict denominator and numerator probabilities on a second-stage cohort."""
    if probability_learner.propensity_model is None:
        raise ValueError("probability nuisance learner must be fitted")
    propensity_data = dataset.with_task("propensity")
    denominator = probability_learner._predict_array(
        probability_learner.propensity_model,
        propensity_data,
        "propensity",
    )[:, :, 0]
    numerator = numerator_learner.predict(dataset)
    return denominator, numerator


class TVSurvCausModel(nn.Module):
    """Balanced Transformer representation with a causal-prefix hazard head."""

    def __init__(
        self,
        treatment_length: int,
        tau: int,
        representation_dim: int,
        regime_embedding_dim: int,
        **transformer_kwargs,
    ):
        super().__init__()
        if treatment_length < 1 or tau < treatment_length:
            raise ValueError("treatment_length must be positive and no larger than tau")
        if min(representation_dim, regime_embedding_dim) < 1:
            raise ValueError("representation dimensions must be positive")
        self.treatment_length = int(treatment_length)
        self.tau = int(tau)
        self.encoder = SurvivalHistoryTransformer(**transformer_kwargs)
        self.representation = nn.Sequential(
            nn.Linear(self.encoder.br_size, representation_dim),
            nn.ELU(),
            nn.Linear(representation_dim, representation_dim),
        )
        self.regime_embedding = nn.Sequential(
            nn.Linear(2 * treatment_length, regime_embedding_dim),
            nn.ELU(),
            nn.Linear(regime_embedding_dim, regime_embedding_dim),
        )
        self.horizon_embedding = nn.Embedding(tau, regime_embedding_dim)
        self.hazard_head = nn.Sequential(
            nn.Linear(
                representation_dim + 2 * regime_embedding_dim,
                representation_dim,
            ),
            nn.ELU(),
            nn.Linear(representation_dim, 1),
        )
        prefix_mask = (
            torch.arange(treatment_length)[None, :] < torch.arange(1, tau + 1)[:, None]
        ).float()
        self.register_buffer("regime_prefix_mask", prefix_mask)

    def history_representations(self, batch, landmarks: torch.Tensor):
        """Return ``z_t`` at selected landmarks from causal factual histories."""
        encoded = self.encoder.build_br(
            batch["prev_treatments"],
            batch["covariates"],
            batch["prev_outputs"],
            batch["static_features"],
            batch["active_entries"],
        )
        landmarks = torch.as_tensor(landmarks, dtype=torch.long, device=encoded.device)
        return self.representation(encoded.index_select(1, landmarks))

    def hazard_logits(self, z: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        """Evaluate shared future hazards using only each action's causal prefix."""
        leading_shape = z.shape[:-1]
        regime = torch.as_tensor(regime, dtype=z.dtype, device=z.device)
        if regime.ndim == 1:
            if regime.shape[0] != self.treatment_length:
                raise ValueError("regime length does not match treatment_length")
            regime = regime.view(
                *([1] * len(leading_shape)), self.treatment_length
            ).expand(*leading_shape, self.treatment_length)
        if regime.shape != (*leading_shape, self.treatment_length):
            raise ValueError(
                "regime shape must match representations and treatment length"
            )

        prefix_mask = self.regime_prefix_mask.to(dtype=z.dtype)
        mask_shape = (*([1] * len(leading_shape)), self.tau, self.treatment_length)
        expanded_mask = prefix_mask.view(mask_shape).expand(
            *leading_shape, self.tau, self.treatment_length
        )
        actions = regime.unsqueeze(-2).expand_as(expanded_mask) * expanded_mask
        regime_features = torch.cat([actions, expanded_mask], dim=-1)
        regime_embedding = self.regime_embedding(regime_features)

        horizon = self.horizon_embedding(torch.arange(self.tau, device=z.device))
        horizon = horizon.view(
            *([1] * len(leading_shape)), self.tau, horizon.shape[-1]
        ).expand(*leading_shape, self.tau, horizon.shape[-1])
        expanded_z = z.unsqueeze(-2).expand(*leading_shape, self.tau, z.shape[-1])
        return self.hazard_head(
            torch.cat([expanded_z, regime_embedding, horizon], dim=-1)
        ).squeeze(-1)

    def hazard_probabilities(self, z: torch.Tensor, regime: torch.Tensor):
        return torch.sigmoid(self.hazard_logits(z, regime))

    def survival_curve_from_representation(
        self, z: torch.Tensor, regime: torch.Tensor
    ) -> torch.Tensor:
        hazards = self.hazard_probabilities(z, regime)
        return torch.cumprod(1.0 - hazards, dim=-1)

    def survival_curve(self, batch, landmark: int, regime: Sequence[int]):
        landmarks = torch.as_tensor(
            [landmark], dtype=torch.long, device=batch["covariates"].device
        )
        z = self.history_representations(batch, landmarks)[:, 0]
        return self.survival_curve_from_representation(
            z,
            torch.as_tensor(regime, dtype=z.dtype, device=z.device),
        )

    def loss_components(
        self,
        batch,
        landmarks: torch.Tensor,
        alpha: float,
        mmd_bandwidth: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        z = self.history_representations(batch, landmarks)
        logits = self.hazard_logits(z, batch["factual_regimes"])
        elementwise = F.binary_cross_entropy_with_logits(
            logits, batch["event_targets"].float(), reduction="none"
        )
        observed = batch["interval_observed"].float()
        weighted = elementwise * batch["stabilized_weights"].float() * observed
        observed_count = int(observed.sum().detach().cpu())
        survival_loss = weighted.sum() / observed.sum().clamp_min(1.0)
        balance_loss, n_groups = _sequence_balance_mmd(
            z,
            batch["factual_regimes"],
            batch["complete_regime"],
            mmd_bandwidth,
        )
        total = survival_loss + float(alpha) * balance_loss
        return total, survival_loss, balance_loss, n_groups, observed_count


class _TVSurvCausTrainingDataset(Dataset):
    def __init__(self, base_dataset, training_arrays: Mapping[str, np.ndarray]):
        self.base_dataset = base_dataset
        self.training_arrays = dict(training_arrays)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        item.update({key: value[index] for key, value in self.training_arrays.items()})
        return item


def _build_training_arrays(
    dataset,
    landmarks: Sequence[int],
    treatment_length: int,
    tau: int,
    denominator_probability: np.ndarray,
    numerator_probability: np.ndarray,
    probability_clip: float,
    cumulative_weight_clip: float | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    raw = dataset.raw
    treatments = np.asarray(raw["treatments"], dtype=float)
    treatment_observed = np.asarray(raw["treatment_observed"], dtype=float)
    at_risk = np.asarray(raw["at_risk"], dtype=float)
    n_patients = treatments.shape[0]
    landmarks = tuple(int(landmark) for landmark in landmarks)

    events, observed_intervals = survival_interval_targets(raw, landmarks, tau)
    factual_regimes = np.zeros(
        (n_patients, len(landmarks), treatment_length), dtype=np.float32
    )
    complete_regime = np.zeros((n_patients, len(landmarks)), dtype=np.float32)
    for position, landmark in enumerate(landmarks):
        regime_slice = treatments[:, landmark : landmark + treatment_length]
        observed_slice = (
            treatment_observed[:, landmark : landmark + treatment_length] > 0
        )
        factual_regimes[:, position] = np.nan_to_num(regime_slice).astype(np.float32)
        complete_regime[:, position] = (at_risk[:, landmark] > 0) & observed_slice.all(
            axis=1
        )
        prefix_observed = at_risk[:, landmark] > 0
        for relative_time in range(tau):
            if relative_time < treatment_length:
                prefix_observed &= observed_slice[:, relative_time]
            observed_intervals[:, position, relative_time] *= prefix_observed

    weights, weight_clipped = stabilized_treatment_weights(
        raw,
        denominator_probability,
        numerator_probability,
        landmarks,
        treatment_length,
        tau,
        probability_clip,
        cumulative_weight_clip,
    )
    arrays = {
        "event_targets": events,
        "interval_observed": observed_intervals,
        "factual_regimes": factual_regimes,
        "complete_regime": complete_regime,
        "stabilized_weights": weights,
    }
    return arrays, weight_clipped


class TVSurvCausLearner:
    """Fit one pooled-landmark TV-SurvCaus model for a study scenario."""

    def __init__(
        self,
        treatment_length: int,
        tau: int,
        *,
        alpha: float,
        representation_dim: int,
        regime_embedding_dim: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        device: str,
        probability_clip: float,
        cumulative_weight_clip: float | None,
        mmd_bandwidth: float | None,
        transformer_kwargs: Mapping[str, Any],
        seed: int,
        validation_fraction: float = 0.0,
        validation_seed: int = 1,
        early_stopping_patience: int = 7,
        early_stopping_min_delta: float = 0.0,
        early_stopping_min_epochs: int = 1,
    ):
        if alpha < 0 or not np.isfinite(alpha):
            raise ValueError("alpha must be finite and non-negative")
        if min(epochs, batch_size) < 1 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("invalid TV-SurvCaus optimization configuration")
        if mmd_bandwidth is not None and (
            not np.isfinite(mmd_bandwidth) or mmd_bandwidth <= 0
        ):
            raise ValueError("mmd_bandwidth must be positive or None")
        self.treatment_length = int(treatment_length)
        self.tau = int(tau)
        self.alpha = float(alpha)
        self.representation_dim = int(representation_dim)
        self.regime_embedding_dim = int(regime_embedding_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.probability_clip = float(probability_clip)
        self.cumulative_weight_clip = cumulative_weight_clip
        self.mmd_bandwidth = mmd_bandwidth
        self.transformer_kwargs = dict(transformer_kwargs)
        self.seed = int(seed)
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.early_stopping_min_epochs = int(early_stopping_min_epochs)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if min(self.early_stopping_patience, self.early_stopping_min_epochs) < 1:
            raise ValueError("invalid early-stopping configuration")
        if self.early_stopping_min_epochs > self.epochs:
            raise ValueError("early_stopping_min_epochs cannot exceed epochs")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        self.model: TVSurvCausModel | None = None
        self.training_landmarks_: tuple[int, ...] = ()
        self.diagnostics_: dict[str, Any] = {}
        self.training_history_: list[dict[str, float]] = []

    def fit(
        self,
        dataset,
        denominator_probability: np.ndarray,
        numerator_probability: np.ndarray,
    ):
        max_time = dataset.raw["treatments"].shape[1]
        self.training_landmarks_ = tuple(range(max_time - self.tau + 1))
        arrays, weight_clipped = _build_training_arrays(
            dataset,
            self.training_landmarks_,
            self.treatment_length,
            self.tau,
            denominator_probability,
            numerator_probability,
            self.probability_clip,
            self.cumulative_weight_clip,
        )
        training = _TVSurvCausTrainingDataset(dataset.with_task("survival"), arrays)

        _seed_everything(self.seed)
        self.model = TVSurvCausModel(
            self.treatment_length,
            self.tau,
            self.representation_dim,
            self.regime_embedding_dim,
            **self.transformer_kwargs,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_loader, validation_loader = patient_train_validation_loaders(
            training,
            batch_size=self.batch_size,
            validation_fraction=self.validation_fraction,
            validation_seed=self.validation_seed,
            patient_ids=dataset.raw["patient_id"],
        )
        landmarks = torch.as_tensor(
            self.training_landmarks_, dtype=torch.long, device=self.device
        )
        final_survival_sum = 0.0
        final_observed_count = 0
        final_mmd_losses: list[float] = []
        final_group_counts: list[int] = []
        stopper = None
        if validation_loader is not None:
            stopper = ValidationEarlyStopping(
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
                min_epochs=self.early_stopping_min_epochs,
            )
        description = f"train TV-SurvCaus L={self.treatment_length} tau={self.tau}"
        with training_progress(self.epochs, description) as progress:
            for epoch in progress:
                self.model.train()
                epoch_total_sum, epoch_observed = 0.0, 0
                epoch_survival_sum = 0.0
                epoch_mmd_losses: list[float] = []
                epoch_group_counts: list[int] = []
                for batch in train_loader:
                    tensor_batch = _tensor_batch(
                        batch, self.device, _TRAINING_BATCH_KEYS
                    )
                    optimizer.zero_grad()
                    total, survival, balance, n_groups, n_observed = (
                        self.model.loss_components(
                            tensor_batch,
                            landmarks,
                            self.alpha,
                            self.mmd_bandwidth,
                        )
                    )
                    total.backward()
                    optimizer.step()
                    epoch_total_sum += float(total.detach().cpu()) * n_observed
                    epoch_survival_sum += float(survival.detach().cpu()) * n_observed
                    epoch_observed += n_observed
                    epoch_mmd_losses.append(float(balance.detach().cpu()))
                    epoch_group_counts.append(n_groups)
                final_survival_sum = epoch_survival_sum
                final_observed_count = epoch_observed
                final_mmd_losses = epoch_mmd_losses
                final_group_counts = epoch_group_counts
                row = {
                    "epoch": epoch,
                    "train_loss": epoch_total_sum / max(epoch_observed, 1),
                    "train_survival_loss": epoch_survival_sum / max(epoch_observed, 1),
                    "train_mmd_loss": float(np.mean(epoch_mmd_losses)),
                }
                should_stop = False
                if validation_loader is not None:
                    self.model.eval()
                    validation_sum, validation_observed = 0.0, 0
                    with torch.no_grad():
                        for batch in validation_loader:
                            tensor_batch = _tensor_batch(
                                batch, self.device, _TRAINING_BATCH_KEYS
                            )
                            total, _, _, _, n_observed = self.model.loss_components(
                                tensor_batch,
                                landmarks,
                                self.alpha,
                                self.mmd_bandwidth,
                            )
                            validation_sum += float(total.cpu()) * n_observed
                            validation_observed += n_observed
                    validation_loss = validation_sum / max(validation_observed, 1)
                    row["validation_loss"] = validation_loss
                    self.training_history_.append(row)
                    should_stop = stopper.update(self.model, epoch, validation_loss)
                else:
                    row["validation_loss"] = np.nan
                    self.training_history_.append(row)
                update_training_progress(progress, row)
                if should_stop:
                    break
        if stopper is not None:
            stopper.restore(self.model)
            self.model.best_epoch_ = stopper.best_epoch
        else:
            self.model.best_epoch_ = len(self.training_history_)
        self.model.stopped_epoch_ = len(self.training_history_)
        self.model.training_history_ = self.training_history_
        self.model.eval()

        observed = arrays["interval_observed"].astype(bool)
        observed_weights = arrays["stabilized_weights"][observed]
        complete = arrays["complete_regime"].astype(bool)
        complete_regimes = arrays["factual_regimes"][complete].astype(int)
        if len(complete_regimes):
            powers = 2 ** np.arange(self.treatment_length)
            _, counts = np.unique(complete_regimes @ powers, return_counts=True)
            n_sequence_groups = int(np.sum(counts >= 2))
        else:
            n_sequence_groups = 0
        self.diagnostics_ = {
            "method": "tv_survcaus",
            "diagnostic": "training",
            "arm": "pooled",
            "landmark": np.nan,
            "mean_stabilized_weight": float(observed_weights.mean()),
            "max_stabilized_weight": float(observed_weights.max()),
            "weight_clipped_fraction": float(weight_clipped[observed].mean()),
            "mean_survival_loss": (final_survival_sum / max(final_observed_count, 1)),
            "mean_mmd_loss": float(np.mean(final_mmd_losses)),
            "mean_mmd_groups": float(np.mean(final_group_counts)),
            "n_factual_sequence_groups": n_sequence_groups,
            "n_training_landmarks": len(self.training_landmarks_),
            "n_observed_intervals": int(observed.sum()),
            "best_epoch": int(self.model.best_epoch_),
            "stopped_epoch": int(self.model.stopped_epoch_),
        }
        return self

    @torch.no_grad()
    def predict_survival(
        self,
        dataset,
        landmark: int,
        regime: Sequence[int],
        *,
        return_curve: bool = False,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit TV-SurvCaus before prediction")
        regime = tuple(int(action) for action in regime)
        if len(regime) != self.treatment_length or any(
            action not in (0, 1) for action in regime
        ):
            raise ValueError(
                "regime must be a binary treatment window of fitted length"
            )
        max_time = dataset.raw["treatments"].shape[1]
        if landmark < 0 or landmark + self.tau > max_time:
            raise ValueError("landmark plus tau exceeds available follow-up")

        curves = []
        loader = DataLoader(
            dataset.with_task("survival"),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.model.eval()
        for batch in loader:
            tensor_batch = _tensor_batch(batch, self.device, _HISTORY_BATCH_KEYS)
            curves.append(
                self.model.survival_curve(tensor_batch, landmark, regime).cpu().numpy()
            )
        survival = np.concatenate(curves, axis=0)
        return survival if return_curve else survival[:, -1]


__all__ = [
    "TVSurvCausLearner",
    "TVSurvCausModel",
    "TreatmentNumeratorLearner",
    "TreatmentNumeratorModel",
    "prepare_treatment_probabilities",
    "rbf_mmd2",
    "stabilized_treatment_weights",
    "survival_interval_targets",
]
