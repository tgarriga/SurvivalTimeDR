"""Longitudinal inverse-probability and doubly robust signals.

This is the only implementation of the estimators' algebra. Model fitting lives in
``survival_ct_nuisance.py`` and study orchestration lives in
``src.evaluation.experiments``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.data.survival_sim.timing import (
    observed_survival_indicator,
    observed_survival_next,
)


def _floor_denominator(probability: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply a one-sided floor to an inverse-probability denominator.

    These values are denominators, rather than generic Bernoulli probabilities.  An
    upper clip would turn a well-supported censoring continuation probability near one
    into a *larger* inverse weight, which is the opposite of the intended variance
    protection.  The retained ``probability_clip`` public name is kept for configuration
    compatibility, but its estimator meaning is a lower denominator floor.
    """
    if not 0.0 <= epsilon < 0.5:
        raise ValueError("probability floor must be in [0, 0.5)")
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("nuisance probabilities must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("nuisance probabilities must lie in [0, 1]")
    return np.maximum(probability, epsilon)


def _update_cumulative_weight(
    weight: np.ndarray,
    factor: np.ndarray,
    cumulative_weight_clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiply one longitudinal factor into a single cumulative weight and clip it.

    Returns the updated weight and an indicator for observations whose *unclipped*
    updated weight exceeded the configured cap at this step.
    """
    updated = np.asarray(weight, dtype=float) * np.asarray(factor, dtype=float)
    if cumulative_weight_clip is None:
        return updated, np.zeros_like(updated, dtype=bool)
    if cumulative_weight_clip <= 0:
        raise ValueError("cumulative_weight_clip must be positive or None")
    clipped = updated > cumulative_weight_clip
    return np.minimum(updated, cumulative_weight_clip), clipped


def cumulative_regime_weight(
    raw: Mapping[str, Any],
    regime: Sequence[int],
    probability_view: Any,
    landmark: int,
    treatment_end: int,
    horizon: int,
    rows=None,
    probability_clip: float = 0.02,
    cumulative_weight_clip: float | None = 20.0,
    on_probability=None,
    on_weight_clip=None,
) -> np.ndarray:
    """Single cumulative treatment/censoring weight over one intervention window.

    Treatment and censoring factors are incorporated in chronological order. After each
    factor the combined cumulative weight is clipped, so stabilization acts on the same
    evolving treatment/censoring product used by the longitudinal DR signal. The
    censoring indicator itself is carried by the observed survival outcome. Before
    inversion, each treatment or censoring denominator receives the one-sided lower
    floor configured by ``probability_clip``; near-one denominators are unchanged.
    """
    regime = tuple(int(action) for action in regime)
    if any(action not in (0, 1) for action in regime):
        raise ValueError("the current implementation requires binary treatment")
    if treatment_end - landmark + 1 != len(regime):
        raise ValueError("regime length does not match the treatment window")
    if horizon <= treatment_end:
        raise ValueError("the survival horizon must follow the treatment window")
    if landmark < 0 or horizon > raw["treatments"].shape[1]:
        raise ValueError("requested window is outside available follow-up")
    if cumulative_weight_clip is not None and cumulative_weight_clip <= 0:
        raise ValueError("cumulative_weight_clip must be positive or None")

    selected = (
        np.arange(len(raw["baseline"]), dtype=int)
        if rows is None
        else np.asarray(rows, dtype=int)
    )
    if np.any((selected < 0) | (selected >= len(raw["baseline"]))):
        raise IndexError("patient row is outside the raw data")

    weight = np.ones(len(selected), dtype=float)
    for decision_time in range(landmark, horizon):
        if decision_time <= treatment_end:
            action = regime[decision_time - landmark]
            p_one = np.asarray(
                probability_view.pi(raw, decision_time, selected), dtype=float
            )
            denominator = p_one if action == 1 else 1.0 - p_one
            if on_probability is not None:
                on_probability("treatment", selected, denominator)
            denominator = _floor_denominator(denominator, probability_clip)
            adherent = raw["treatments"][selected, decision_time].astype(int) == action
            weight, clipped = _update_cumulative_weight(
                weight,
                adherent / denominator,
                cumulative_weight_clip,
            )
            if on_weight_clip is not None and np.any(clipped):
                on_weight_clip(selected, clipped)

        if decision_time <= horizon - 2:
            successor_time = decision_time + 1
            continuation = np.asarray(
                probability_view.g(raw, successor_time, selected), dtype=float
            )
            if on_probability is not None:
                on_probability("censoring", selected, continuation)
            continuation = _floor_denominator(continuation, probability_clip)
            weight, clipped = _update_cumulative_weight(
                weight,
                1.0 / continuation,
                cumulative_weight_clip,
            )
            if on_weight_clip is not None and np.any(clipped):
                on_weight_clip(selected, clipped)

    return weight


def compute_regime_signals(
    raw: Mapping[str, Any],
    regime: Sequence[int],
    probability_view: Any,
    outcome_view: Any,
    probability_clip: float = 0.02,
    cumulative_weight_clip: float | None = 20.0,
) -> dict[str, np.ndarray]:
    """Construct g-computation, IPTW/IPCW and longitudinal DR signals.

    Both inverse-probability estimators use one recursively updated cumulative weight
    containing treatment and censoring factors. The same cap is applied after every new
    factor for IPTW/IPCW and for the DR orthogonal weight. ``probability_clip`` is a
    one-sided lower floor on whichever treatment/censoring probability is inverted.

    Both views expose absolute-time ``pi``, ``g``, ``q`` and ``m`` accessors plus
    ``t``, ``k_end`` and ``j_end``. The fitted Transformer adapter below maps those
    absolute times onto relative-step outcome models.
    """
    regime = tuple(int(action) for action in regime)
    landmark = int(outcome_view.t)
    treatment_end = int(outcome_view.k_end)
    horizon = int(outcome_view.j_end)
    probability_window = (
        int(probability_view.t),
        int(probability_view.k_end),
        int(probability_view.j_end),
    )
    if probability_window != (landmark, treatment_end, horizon):
        raise ValueError("probability and outcome views must use the same window")
    if cumulative_weight_clip is not None and cumulative_weight_clip <= 0:
        raise ValueError("cumulative_weight_clip must be positive or None")
    if treatment_end - landmark + 1 != len(regime):
        raise ValueError("regime length does not match the outcome view")
    if landmark < 0 or horizon > raw["treatments"].shape[1]:
        raise ValueError("requested window is outside available follow-up")
    if horizon <= treatment_end:
        raise ValueError("the survival horizon must follow the treatment window")

    n_patients = len(raw["baseline"])
    eligible = raw["at_risk"][:, landmark].astype(bool)
    eligible_rows = np.flatnonzero(eligible)

    treatment_denominator_floored = np.zeros(n_patients, dtype=float)
    censoring_denominator_floored = np.zeros(n_patients, dtype=float)
    min_treatment_denominator = np.full(n_patients, np.nan, dtype=float)
    min_censoring_denominator = np.full(n_patients, np.nan, dtype=float)
    weight_clipped = np.zeros(n_patients, dtype=float)

    def note_probability(mechanism, rows, values):
        rows = np.asarray(rows, dtype=int)
        values = np.asarray(values, dtype=float)
        if mechanism == "treatment":
            affected = treatment_denominator_floored
            minimum = min_treatment_denominator
        elif mechanism == "censoring":
            affected = censoring_denominator_floored
            minimum = min_censoring_denominator
        else:
            raise ValueError("unknown probability mechanism " + repr(mechanism))
        affected[rows] = np.maximum(
            affected[rows], (values < probability_clip).astype(float)
        )
        unseen = np.isnan(minimum[rows])
        minimum[rows] = np.where(unseen, values, np.minimum(minimum[rows], values))

    def note_weight_clip(rows, clipped):
        rows = np.asarray(rows, dtype=int)
        clipped = np.asarray(clipped, dtype=bool)
        if np.any(clipped):
            weight_clipped[rows[clipped]] = 1.0

    gcomp = np.full(n_patients, np.nan, dtype=float)
    gcomp[eligible] = outcome_view.m(raw, landmark, eligible_rows)

    # Observed survival is zero when censoring occurs before the horizon. Censoring
    # exactly at the horizon still establishes survival and needs no G_horizon factor.
    observed_horizon = observed_survival_indicator(raw, horizon).astype(float)
    ipw = np.zeros(n_patients, dtype=float)
    ipw_weight = np.zeros(n_patients, dtype=float)
    supported_rows = np.flatnonzero(eligible & (observed_horizon > 0))
    if len(supported_rows):
        weights = cumulative_regime_weight(
            raw,
            regime,
            probability_view,
            landmark,
            treatment_end,
            horizon,
            supported_rows,
            probability_clip=probability_clip,
            cumulative_weight_clip=cumulative_weight_clip,
            on_probability=note_probability,
            on_weight_clip=note_weight_clip,
        )
        ipw[supported_rows] = weights
        ipw_weight[supported_rows] = weights

    dr = gcomp.copy()
    cumulative_orthogonal_weight = np.ones(n_patients, dtype=float)
    max_orthogonal_weight = np.zeros(n_patients, dtype=float)

    for decision_time in range(landmark, horizon):
        at_risk_rows = np.flatnonzero(raw["at_risk"][:, decision_time] > 0)

        if decision_time <= treatment_end:
            action = regime[decision_time - landmark]
            factor = np.ones(n_patients, dtype=float)
            if len(at_risk_rows):
                p_one = np.asarray(
                    probability_view.pi(raw, decision_time, at_risk_rows), dtype=float
                )
                denominator = p_one if action == 1 else 1.0 - p_one
                note_probability("treatment", at_risk_rows, denominator)
                denominator = _floor_denominator(denominator, probability_clip)
                adherent = (
                    raw["treatments"][at_risk_rows, decision_time].astype(int) == action
                )
                factor[at_risk_rows] = adherent / denominator
            cumulative_orthogonal_weight, clipped = _update_cumulative_weight(
                cumulative_orthogonal_weight,
                factor,
                cumulative_weight_clip,
            )
            note_weight_clip(np.arange(n_patients), clipped)

        omega = (
            raw["at_risk"][:, decision_time].astype(float)
            * cumulative_orthogonal_weight
        )
        max_orthogonal_weight = np.maximum(max_orthogonal_weight, np.abs(omega))

        survival_next = observed_survival_next(raw, decision_time).astype(float)
        continuation_next = np.zeros(n_patients, dtype=float)
        if decision_time + 1 == horizon:
            continuation_next[:] = 1.0
        else:
            observed_state_rows = np.flatnonzero(
                raw["state_observed"][:, decision_time + 1] > 0
            )
            if len(observed_state_rows):
                continuation_next[observed_state_rows] = outcome_view.m(
                    raw, decision_time + 1, observed_state_rows
                )

        bridge = np.zeros(n_patients, dtype=float)
        if len(at_risk_rows):
            if decision_time <= treatment_end:
                bridge[at_risk_rows] = outcome_view.q(
                    raw,
                    decision_time,
                    at_risk_rows,
                    regime[decision_time - landmark],
                )
            else:
                bridge[at_risk_rows] = outcome_view.m(raw, decision_time, at_risk_rows)
        dr += omega * (survival_next * continuation_next - bridge)

        if decision_time <= horizon - 2:
            successor_time = decision_time + 1
            factor = np.ones(n_patients, dtype=float)
            observed_state_rows = np.flatnonzero(
                raw["state_observed"][:, successor_time] > 0
            )
            if len(observed_state_rows):
                raw_continuation = np.asarray(
                    probability_view.g(raw, successor_time, observed_state_rows),
                    dtype=float,
                )
                note_probability("censoring", observed_state_rows, raw_continuation)
                continuation = _floor_denominator(raw_continuation, probability_clip)
                factor[observed_state_rows] = 1.0 / continuation
            cumulative_orthogonal_weight, clipped = _update_cumulative_weight(
                cumulative_orthogonal_weight,
                factor,
                cumulative_weight_clip,
            )
            note_weight_clip(np.arange(n_patients), clipped)

    dr[~eligible] = np.nan
    ipw[~eligible] = np.nan
    return {
        "eligible": eligible,
        "gcomp": gcomp,
        "ipw": ipw,
        "dr": dr,
        "ipw_weight": ipw_weight,
        "max_omega": max_orthogonal_weight,
        "prob_clipped_low": np.maximum(
            treatment_denominator_floored, censoring_denominator_floored
        ),
        "treatment_prob_floored": treatment_denominator_floored,
        "censoring_prob_floored": censoring_denominator_floored,
        "min_treatment_denominator": min_treatment_denominator,
        "min_censoring_denominator": min_censoring_denominator,
        "weight_clipped": weight_clipped,
    }


class FittedNuisanceView:
    """Adapt relative-step fitted models to the signal functions' time interface."""

    def __init__(
        self,
        learner,
        dataset,
        landmark: int,
        treatment_end: int,
        horizon: int,
        regime: Sequence[int],
        prediction_cache=None,
    ):
        self.t = int(landmark)
        self.k_end = int(treatment_end)
        self.j_end = int(horizon)
        self.regime = tuple(int(action) for action in regime)
        if self.k_end - self.t + 1 != len(self.regime):
            raise ValueError("regime length does not match the treatment window")
        if self.j_end - self.t != learner.config.tau:
            raise ValueError("view horizon does not match the fitted learner")
        if self.t < 0 or self.j_end > dataset.raw["treatments"].shape[1]:
            raise ValueError("view window is outside available follow-up")
        if learner.propensity_model is None or learner.censoring_model is None:
            raise ValueError("learner probability nuisances must be fitted")

        self._learner = learner
        self._survival = dataset.with_task("survival")
        self._censoring = dataset.with_task("censoring")
        self._cache = prediction_cache if prediction_cache is not None else {}
        self._pi = self._predict(
            learner.propensity_model, self._survival, "propensity"
        )[:, :, 0]
        self._g = self._predict(learner.censoring_model, self._censoring, "censoring")[
            :, :, 0
        ]

    def _predict(self, model, dataset, task, treatment_value=None):
        key = (id(model), id(dataset.raw), task, treatment_value)
        if key not in self._cache:
            self._cache[key] = self._learner._predict_array(
                model, dataset, task, treatment_value=treatment_value
            )
        return self._cache[key]

    def _q_at(self, decision_time: int, action: int) -> np.ndarray:
        if action not in (0, 1):
            raise ValueError("treatment must be binary")
        relative_step = decision_time - self.t
        try:
            model = self._learner.q_models[relative_step]
        except KeyError as error:
            raise IndexError(f"no Q model for relative step {relative_step}") from error
        return self._predict(model, self._survival, "q", action)[:, decision_time, 0]

    def pi(self, raw, decision_time, rows):
        return self._pi[np.asarray(rows, dtype=int), decision_time]

    def g(self, raw, successor_time, rows):
        return self._g[np.asarray(rows, dtype=int), successor_time]

    def q(self, raw, decision_time, rows, treatment):
        selected = np.asarray(rows, dtype=int)
        if np.isscalar(treatment):
            return self._q_at(decision_time, int(treatment))[selected]
        treatment = np.asarray(treatment, dtype=int)
        return np.where(
            treatment == 1,
            self._q_at(decision_time, 1)[selected],
            self._q_at(decision_time, 0)[selected],
        )

    def m(self, raw, decision_time, rows):
        selected = np.asarray(rows, dtype=int)
        if not self.t <= decision_time <= self.j_end:
            raise ValueError("decision time is outside the view window")
        if decision_time == self.j_end:
            return np.ones(len(selected))
        relative_step = decision_time - self.t
        if relative_step < len(self.regime):
            return self._q_at(decision_time, self.regime[relative_step])[selected]
        try:
            model = self._learner.m_models[relative_step]
        except KeyError as error:
            raise IndexError(
                f"no continuation model for relative step {relative_step}"
            ) from error
        return self._predict(model, self._survival, "m")[:, decision_time, 0][selected]


class OracleProbabilityView:
    """Expose the known simulation probabilities through the nuisance-view interface.

    This view changes only treatment and censoring probabilities. Outcome regressions
    remain fitted, so learned-versus-oracle weighting isolates probability-model error
    without changing the estimand or the outcome learner.
    """

    def __init__(self, dataset, landmark: int, treatment_end: int, horizon: int):
        self.t = int(landmark)
        self.k_end = int(treatment_end)
        self.j_end = int(horizon)
        raw = dataset.raw
        required = {"true_treatment_probability", "true_censoring_probability"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(
                "oracle probabilities are unavailable: " + repr(sorted(missing))
            )
        if self.t < 0 or self.j_end > raw["treatments"].shape[1]:
            raise ValueError("oracle probability window is outside available follow-up")

    def pi(self, raw, decision_time, rows):
        values = np.asarray(raw["true_treatment_probability"])[
            np.asarray(rows, dtype=int), int(decision_time)
        ]
        if not np.isfinite(values).all():
            raise ValueError(
                "oracle treatment probability is unavailable for a requested row"
            )
        return values

    def g(self, raw, successor_time, rows):
        # Simulators store G_{k+1} at decision index k.
        values = np.asarray(raw["true_censoring_probability"])[
            np.asarray(rows, dtype=int), int(successor_time) - 1
        ]
        if not np.isfinite(values).all():
            raise ValueError(
                "oracle censoring probability is unavailable for a requested row"
            )
        return values


__all__ = [
    "FittedNuisanceView",
    "OracleProbabilityView",
    "compute_regime_signals",
    "cumulative_regime_weight",
]
