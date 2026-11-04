"""Optimization and prediction helpers for the study's Transformer models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .survival_ct_nuisance import SurvivalCTNuisanceModel

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


def training_progress(epochs: int, description: str, *, enabled: bool = True):
    """Return one notebook/terminal-aware progress bar for a model fit."""
    return tqdm(
        range(1, int(epochs) + 1),
        total=int(epochs),
        desc=str(description),
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
        disable=not enabled,
    )


def update_training_progress(progress, row: dict[str, float]) -> None:
    """Display the latest finite train/validation losses without changing history."""
    postfix = {"train": f"{float(row['train_loss']):.4g}"}
    validation_loss = float(row.get("validation_loss", np.nan))
    if np.isfinite(validation_loss):
        postfix["val"] = f"{validation_loss:.4g}"
    progress.set_postfix(postfix, refresh=False)


def _tensor_batch(batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.items()
        if key in _MODEL_BATCH_KEYS
    }


def patient_train_validation_indices(
    dataset,
    validation_fraction: float,
    seed: int,
    *,
    patient_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic patient-disjoint internal validation split.

    The external nuisance/second-stage/test split remains unchanged.  This split is
    only used to select the epoch within the cohort on which a model is fitted.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    n_rows = len(dataset)
    if validation_fraction == 0.0:
        return np.arange(n_rows), np.empty(0, dtype=int)
    if n_rows < 2:
        raise ValueError("internal validation requires at least two patients")
    if patient_ids is None:
        if not hasattr(dataset, "raw") or "patient_id" not in dataset.raw:
            raise ValueError("dataset must expose raw['patient_id']")
        patient_ids = dataset.raw["patient_id"]
    patient_ids = np.asarray(patient_ids)
    if patient_ids.shape != (n_rows,):
        raise ValueError("patient_ids must contain one identifier per dataset row")

    unique_ids = np.unique(patient_ids)
    if len(unique_ids) < 2:
        raise ValueError("internal validation requires at least two distinct patients")
    shuffled = np.sort(unique_ids).copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)
    n_validation = int(round(validation_fraction * len(unique_ids)))
    n_validation = min(max(n_validation, 1), len(unique_ids) - 1)
    validation_ids = shuffled[:n_validation]
    validation_mask = np.isin(patient_ids, validation_ids)
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def patient_train_validation_loaders(
    dataset,
    *,
    batch_size: int,
    validation_fraction: float,
    validation_seed: int,
    patient_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """Build shuffled training and non-shuffled validation loaders."""
    train_rows, validation_rows = patient_train_validation_indices(
        dataset,
        validation_fraction,
        validation_seed,
        patient_ids=patient_ids,
    )
    training = Subset(dataset, train_rows.tolist())
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    validation_loader = None
    if len(validation_rows):
        validation = Subset(dataset, validation_rows.tolist())
        validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


class ValidationEarlyStopping:
    """Track and restore the epoch with the lowest internal validation loss."""

    def __init__(self, *, patience: int, min_delta: float, min_epochs: int):
        if patience < 1 or min_epochs < 1 or min_delta < 0:
            raise ValueError("invalid early-stopping configuration")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.best_loss = np.inf
        self.best_epoch = 0
        self.bad_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def update(self, model, epoch: int, validation_loss: float) -> bool:
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = float(validation_loss)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _mean_masked_loss(model, loader, device: str, task: str) -> float:
    total, weight = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensor_batch = _tensor_batch(batch, device)
            batch_weight = float(tensor_batch["target_mask"].sum().cpu())
            if batch_weight <= 0:
                continue
            total += float(model.loss(tensor_batch, task).cpu()) * batch_weight
            weight += batch_weight
    return total / max(weight, 1.0)


class TransformerTrainer:
    """Fit and evaluate the causal-Transformer nuisance models."""

    def __init__(self, config):
        self.config = config
        self._model_index = 0

    def new_model(self, task: str) -> SurvivalCTNuisanceModel:
        seed = int(self.config.seed) + self._model_index
        self._model_index += 1
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return SurvivalCTNuisanceModel(
            nuisance_task=task,
            **dict(self.config.model_kwargs),
        ).to(self.config.device)

    def fit(self, model, dataset, task: str, *, show_progress: bool = True):
        validation_fraction = float(getattr(self.config, "validation_fraction", 0.0))
        train_loader, validation_loader = patient_train_validation_loaders(
            dataset,
            batch_size=self.config.batch_size,
            validation_fraction=validation_fraction,
            validation_seed=int(getattr(self.config, "validation_seed", 1)),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        stopper = None
        if validation_loader is not None:
            stopper = ValidationEarlyStopping(
                patience=int(getattr(self.config, "early_stopping_patience", 7)),
                min_delta=float(getattr(self.config, "early_stopping_min_delta", 0.0)),
                min_epochs=int(getattr(self.config, "early_stopping_min_epochs", 1)),
            )
        history = []
        description = f"train {task} model {self._model_index}"
        with training_progress(
            self.config.epochs, description, enabled=show_progress
        ) as progress:
            for epoch in progress:
                model.train()
                total, weight = 0.0, 0.0
                for batch in train_loader:
                    tensor_batch = _tensor_batch(batch, self.config.device)
                    batch_weight = float(
                        tensor_batch["target_mask"].sum().detach().cpu()
                    )
                    if batch_weight <= 0:
                        continue
                    optimizer.zero_grad()
                    loss = model.loss(tensor_batch, task)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach().cpu()) * batch_weight
                    weight += batch_weight
                row = {"epoch": epoch, "train_loss": total / max(weight, 1.0)}
                should_stop = False
                if validation_loader is not None:
                    validation_loss = _mean_masked_loss(
                        model, validation_loader, self.config.device, task
                    )
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
    def predict_array(
        self,
        model,
        dataset,
        task: str,
        treatment_value: int | None = None,
    ) -> np.ndarray:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        predictions = []
        for batch in loader:
            tensor_batch = _tensor_batch(batch, self.config.device)
            current_treatments = None
            if treatment_value is not None:
                current_treatments = torch.zeros_like(
                    tensor_batch["current_treatments"]
                )
                current_treatments[..., int(treatment_value)] = 1.0
            prediction = model.predict_proba(
                tensor_batch,
                task,
                current_treatments=current_treatments,
            )
            predictions.append(prediction.cpu().numpy())
        return np.concatenate(predictions, axis=0)


__all__ = [
    "TransformerTrainer",
    "ValidationEarlyStopping",
    "patient_train_validation_indices",
    "patient_train_validation_loaders",
]
