"""PyTorch dataset views for the synthetic longitudinal survival process."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from torch.utils.data import Dataset

from .dgp import SurvivalSimConfig, simulate_split

TASKS = frozenset({"propensity", "censoring", "survival", "outcome_nuisance"})


def _one_hot_treatment(
    treatment: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Encode observed binary treatments; unobserved cells remain zero padded."""
    encoded = np.zeros(treatment.shape + (2,), dtype=np.float32)
    valid = (observed > 0) & ~np.isnan(treatment)
    encoded[..., 0] = valid & (treatment == 0)
    encoded[..., 1] = valid & (treatment == 1)
    return encoded


class SurvivalSyntheticDataset(Dataset):
    """One split exposed in the tensor layout used by the Transformer."""

    def __init__(
        self,
        raw_data: Mapping[str, Any],
        task: str = "survival",
        scaling_params: Mapping[str, np.ndarray] | None = None,
    ):
        if task not in TASKS:
            raise ValueError(f"unknown task: {task!r}")

        self.raw = raw_data
        self.task = task
        self.scaling_params = scaling_params
        self.data: dict[str, np.ndarray] = {}

        if scaling_params is not None:
            self.process_data(scaling_params)

    def __len__(self) -> int:
        return int(self.raw["baseline"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            key: value[index]
            for key, value in self.data.items()
            if hasattr(value, "__len__") and len(value) == len(self)
        }

    def get_scaling_params(self) -> dict[str, np.ndarray]:
        """Estimate input normalization from observed cells in this split.

        The state width is read from the data, so any number of longitudinal features
        is supported; the canonical synthetic process has three.
        """
        baseline = np.asarray(self.raw["baseline"], dtype=float)
        states = np.asarray(self.raw["states"], dtype=float)
        states = states.reshape(-1, states.shape[-1])
        observed_states = states[~np.isnan(states).any(axis=1)]
        if len(observed_states) == 0:
            raise ValueError("cannot scale a split with no observed states")
        return {
            "static_mean": baseline.mean(axis=0).astype(np.float32),
            "static_std": (baseline.std(axis=0) + 1e-6).astype(np.float32),
            "covariate_mean": observed_states.mean(axis=0).astype(np.float32),
            "covariate_std": (observed_states.std(axis=0) + 1e-6).astype(np.float32),
        }

    @staticmethod
    def _time_channel(
        n_patients: int,
        length: int,
        denominator: int,
    ) -> np.ndarray:
        values = np.arange(length, dtype=np.float32) / max(denominator, 1)
        return np.broadcast_to(values[None, :, None], (n_patients, length, 1))

    def _standard_inputs(
        self,
        scaling: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        raw = self.raw
        n_patients, max_time = raw["treatments"].shape
        state = np.nan_to_num(
            (raw["states"][:, :max_time] - scaling["covariate_mean"])
            / scaling["covariate_std"]
        )
        covariates = np.concatenate(
            [state, self._time_channel(n_patients, max_time, max_time - 1)],
            axis=-1,
        )
        static_features = (raw["baseline"] - scaling["static_mean"]) / scaling[
            "static_std"
        ]
        current_treatments = _one_hot_treatment(
            raw["treatments"],
            raw["treatment_observed"],
        )
        previous_treatments = np.zeros_like(current_treatments)
        previous_treatments[:, 1:] = current_treatments[:, :-1]

        # Censoring at k leaves X_k observed but removes A_k and its target.
        active_entries = raw["state_observed"][:, :max_time, None].astype(np.float32)
        covariates *= active_entries
        return (
            covariates.astype(np.float32),
            static_features.astype(np.float32),
            current_treatments,
            previous_treatments,
            active_entries,
        )

    def _censoring_view(
        self,
        scaling: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        raw = self.raw
        n_patients, max_time = raw["treatments"].shape
        length = max_time + 1
        state = np.nan_to_num(
            (raw["states"] - scaling["covariate_mean"]) / scaling["covariate_std"]
        ).astype(np.float32)
        covariates = np.concatenate(
            [state, self._time_channel(n_patients, length, max_time)],
            axis=-1,
        ).astype(np.float32)

        current_treatments = np.zeros((n_patients, length, 2), dtype=np.float32)
        current_treatments[:, :max_time] = _one_hot_treatment(
            raw["treatments"],
            raw["treatment_observed"],
        )
        previous_treatments = np.zeros_like(current_treatments)
        previous_treatments[:, 1:] = current_treatments[:, :max_time]
        active_entries = raw["state_observed"][..., None].astype(np.float32)
        covariates *= active_entries

        outputs = np.zeros((n_patients, length, 1), dtype=np.float32)
        target_mask = np.zeros_like(outputs)
        for successor_time in range(1, length):
            outputs[:, successor_time, 0] = raw["censoring_time"] > successor_time
            target_mask[:, successor_time, 0] = (
                (raw["event_time"] > successor_time)
                & (raw["censoring_time"] > successor_time - 1)
                & (raw["state_observed"][:, successor_time] > 0)
            )
        return (
            covariates,
            current_treatments,
            previous_treatments,
            active_entries,
            outputs,
            target_mask,
        )

    def process_data(
        self,
        scaling_params: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Build the tensor view for the configured nuisance task."""
        scaling = scaling_params or self.get_scaling_params()
        self.scaling_params = scaling
        raw = self.raw
        n_patients, max_time = raw["treatments"].shape
        (
            covariates,
            static_features,
            current_treatments,
            previous_treatments,
            active_entries,
        ) = self._standard_inputs(scaling)

        if self.task == "censoring":
            (
                covariates,
                current_treatments,
                previous_treatments,
                active_entries,
                outputs,
                target_mask,
            ) = self._censoring_view(scaling)
        else:
            raw_outputs = (
                raw["treatments"]
                if self.task == "propensity"
                else raw["event_free_next"]
            )
            outputs = np.nan_to_num(raw_outputs[..., None]).astype(np.float32)
            target_mask = raw["at_risk"][..., None].astype(np.float32)

        sequence_length = covariates.shape[1]
        self.data = {
            "prev_treatments": previous_treatments,
            "covariates": covariates,
            "prev_outputs": np.zeros(
                (n_patients, sequence_length, 1),
                dtype=np.float32,
            ),
            "static_features": static_features,
            "current_treatments": current_treatments,
            "outputs": outputs,
            "active_entries": active_entries,
            "target_mask": target_mask,
        }
        return self.data

    def with_task(
        self,
        task: str,
        targets: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
    ) -> "SurvivalSyntheticDataset":
        """Create a lightweight task view while sharing immutable raw arrays."""
        scaling = self.scaling_params or self.get_scaling_params()
        dataset = SurvivalSyntheticDataset(
            self.raw,
            task=task,
            scaling_params=scaling,
        )
        if task == "outcome_nuisance":
            if targets is None or target_mask is None:
                raise ValueError("outcome_nuisance requires targets and target_mask")
            if targets.shape != dataset.data["outputs"].shape:
                raise ValueError(f"bad targets shape: {targets.shape}")
            if target_mask.shape != dataset.data["target_mask"].shape:
                raise ValueError(f"bad target_mask shape: {target_mask.shape}")
            dataset.data["outputs"] = targets.astype(np.float32, copy=True)
            dataset.data["target_mask"] = target_mask.astype(
                np.float32,
                copy=True,
            )
        return dataset


class SurvivalSyntheticDatasetCollection:
    """Independent train/validation/test datasets sharing train-set scaling."""

    def __init__(
        self,
        seed: int = 1,
        task: str = "survival",
        num_patients: Mapping[str, int] | None = None,
        max_seq_length: int = 32,
        overlap_strength: float = 0.75,
        heterogeneity_strength: float = 3.0,
        feedback_strength: float = 0.8,
        censoring_intercept: float = 5.2,
        event_intercept: float = -4.0,
        treatment_effect_scale: float = 1.0,
        propensity_clip: float = 0.08,
        censoring_enabled: bool = True,
    ):
        sizes = num_patients or {
            "train": 1800,
            "val": 1800,
            "test": 1200,
        }
        if "train" not in sizes or "val" not in sizes:
            raise ValueError("num_patients must contain train and val splits")
        config = SurvivalSimConfig(
            max_seq_length=max_seq_length,
            overlap_strength=overlap_strength,
            heterogeneity_strength=heterogeneity_strength,
            feedback_strength=feedback_strength,
            censoring_intercept=censoring_intercept,
            event_intercept=event_intercept,
            treatment_effect_scale=treatment_effect_scale,
            propensity_clip=propensity_clip,
            censoring_enabled=censoring_enabled,
        )
        raw_splits = {
            name: simulate_split(count, seed + 1009 * offset, config)
            for offset, (name, count) in enumerate(sizes.items())
        }
        self._initialize(
            raw_splits,
            task=task,
        )

    @staticmethod
    def _validated_scaling_params(
        scaling_params: Mapping[str, np.ndarray],
        train_raw: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Check a supplied normalization against the training split's feature widths."""
        widths = {
            "static": np.asarray(train_raw["baseline"]).shape[-1],
            "covariate": np.asarray(train_raw["states"]).shape[-1],
        }
        validated = {}
        for prefix, width in widths.items():
            for statistic in ("mean", "std"):
                key = f"{prefix}_{statistic}"
                if key not in scaling_params:
                    raise ValueError(f"scaling_params lacks {key!r}")
                value = np.asarray(scaling_params[key], dtype=np.float32)
                if value.shape != (width,):
                    raise ValueError(
                        f"{key} must have shape ({width},), got {value.shape}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(f"{key} must be finite")
                if statistic == "std" and np.any(value <= 0):
                    raise ValueError(f"{key} must be positive")
                validated[key] = value
        return validated

    def _initialize(
        self,
        raw_splits: Mapping[str, Mapping[str, Any]],
        *,
        task: str,
        scaling_params: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        train = SurvivalSyntheticDataset(
            raw_splits["train"],
            task,
        )
        self.train_scaling_params = (
            train.get_scaling_params()
            if scaling_params is None
            else self._validated_scaling_params(scaling_params, raw_splits["train"])
        )
        train.process_data(self.train_scaling_params)
        validation_raw = raw_splits.get("val", raw_splits["train"])
        test_raw = raw_splits.get("test", validation_raw)
        self.train_f = train
        self.val_f = SurvivalSyntheticDataset(
            validation_raw,
            task,
            self.train_scaling_params,
        )
        self.test_f = SurvivalSyntheticDataset(
            test_raw,
            task,
            self.train_scaling_params,
        )

    @classmethod
    def from_raw(
        cls,
        raw_splits: Mapping[str, Mapping[str, Any]],
        task: str = "survival",
        scaling_params: Mapping[str, np.ndarray] | None = None,
    ) -> "SurvivalSyntheticDatasetCollection":
        """Build a collection from existing patient-disjoint raw splits.

        By default the normalization is estimated from ``train``. ``scaling_params`` instead
        supplies the one normalization (static and covariate mean and std) applied unchanged to
        every split, for callers that must fix it before the raw arrays are generated.
        """
        if "train" not in raw_splits:
            raise ValueError("raw_splits must contain a train split")
        collection = cls.__new__(cls)
        supplied = {} if scaling_params is None else {"scaling_params": scaling_params}
        collection._initialize(
            raw_splits,
            task=task,
            **supplied,
        )
        return collection

    def with_task(
        self,
        task: str,
        train_targets: np.ndarray | None = None,
        train_target_mask: np.ndarray | None = None,
        val_targets: np.ndarray | None = None,
        val_target_mask: np.ndarray | None = None,
        test_targets: np.ndarray | None = None,
        test_target_mask: np.ndarray | None = None,
    ) -> "SurvivalSyntheticDatasetCollection":
        """Create task-specific split views without copying raw trajectories."""
        collection = self.__class__.__new__(self.__class__)
        collection.train_scaling_params = self.train_scaling_params

        if task == "outcome_nuisance":
            val_targets = (
                self.val_f.data["outputs"] if val_targets is None else val_targets
            )
            val_target_mask = (
                self.val_f.data["target_mask"]
                if val_target_mask is None
                else val_target_mask
            )
            test_targets = (
                self.test_f.data["outputs"] if test_targets is None else test_targets
            )
            test_target_mask = (
                self.test_f.data["target_mask"]
                if test_target_mask is None
                else test_target_mask
            )
            collection.train_f = self.train_f.with_task(
                task,
                train_targets,
                train_target_mask,
            )
            collection.val_f = self.val_f.with_task(
                task,
                val_targets,
                val_target_mask,
            )
            collection.test_f = self.test_f.with_task(
                task,
                test_targets,
                test_target_mask,
            )
        else:
            collection.train_f = self.train_f.with_task(task)
            collection.val_f = self.val_f.with_task(task)
            collection.test_f = self.test_f.with_task(task)

        return collection


__all__ = [
    "SurvivalSyntheticDataset",
    "SurvivalSyntheticDatasetCollection",
]
