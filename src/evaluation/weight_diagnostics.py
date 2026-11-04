"""Weight-support calculations used by the manuscript stress study."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data.survival_sim.timing import observed_survival_indicator
from src.models.survival_dr import cumulative_regime_weight


class _FittedProbabilityView:
    """Prediction adapter exposing only pi/g for an already-fitted probability learner."""

    def __init__(self, learner, dataset, prediction_cache=None):
        if learner.propensity_model is None or learner.censoring_model is None:
            raise ValueError("probability nuisance learner must be fitted")
        self._learner = learner
        self._cache = prediction_cache if prediction_cache is not None else {}
        survival = dataset.with_task("survival")
        censoring = dataset.with_task("censoring")
        self._pi = self._predict(learner.propensity_model, survival, "propensity")[
            :, :, 0
        ]
        self._g = self._predict(learner.censoring_model, censoring, "censoring")[
            :, :, 0
        ]

    def _predict(self, model, dataset, task):
        key = (id(model), id(dataset.raw), task)
        if key not in self._cache:
            self._cache[key] = self._learner._predict_array(model, dataset, task)
        return self._cache[key]

    def pi(self, raw, decision_time, rows):
        return self._pi[np.asarray(rows, dtype=int), decision_time]

    def g(self, raw, successor_time, rows):
        return self._g[np.asarray(rows, dtype=int), successor_time]


def _treatment_match_mask(raw, regime, landmark: int) -> np.ndarray:
    """Patients at risk at ``landmark`` whose full observed treatment path matches ``regime``.

    Treatments after death or censoring are NaN in the synthetic DGP, so incomplete
    intervention windows never count as matches.
    """
    regime = np.asarray(tuple(int(action) for action in regime), dtype=float)
    treatment_end = landmark + len(regime)
    observed = np.asarray(raw["treatments"][:, landmark:treatment_end], dtype=float)
    if observed.shape[1] != len(regime):
        raise ValueError("treatment regime extends beyond available follow-up")
    eligible = np.asarray(raw["at_risk"][:, landmark], dtype=bool)
    return eligible & np.all(observed == regime[None, :], axis=1)


def _apply_cumulative_clip(weight, factor, cumulative_weight_clip):
    """Mirror the cumulative clipping update used by the IPTW and DR estimators."""
    updated = np.asarray(weight, dtype=float) * np.asarray(factor, dtype=float)
    if cumulative_weight_clip is None:
        return updated, np.zeros_like(updated, dtype=bool)
    clipped = updated > cumulative_weight_clip
    return np.minimum(updated, cumulative_weight_clip), clipped


def _dr_weight_clip_flags(
    raw,
    regime,
    probability_view,
    landmark: int,
    horizon: int,
    probability_clip: float,
    cumulative_weight_clip: float | None,
) -> np.ndarray:
    """Flag patients whose DR orthogonal cumulative weight is clipped at any step."""
    regime = tuple(int(action) for action in regime)
    treatment_end = landmark + len(regime) - 1
    n_patients = len(raw["baseline"])
    cumulative_weight = np.ones(n_patients, dtype=float)
    clipped_any = np.zeros(n_patients, dtype=bool)

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
                denominator = np.maximum(denominator, probability_clip)
                adherent = (
                    raw["treatments"][at_risk_rows, decision_time].astype(int) == action
                )
                factor[at_risk_rows] = adherent / denominator
            cumulative_weight, clipped = _apply_cumulative_clip(
                cumulative_weight,
                factor,
                cumulative_weight_clip,
            )
            clipped_any |= clipped

        if decision_time <= horizon - 2:
            successor_time = decision_time + 1
            factor = np.ones(n_patients, dtype=float)
            observed_state_rows = np.flatnonzero(
                raw["state_observed"][:, successor_time] > 0
            )
            if len(observed_state_rows):
                continuation = np.asarray(
                    probability_view.g(raw, successor_time, observed_state_rows),
                    dtype=float,
                )
                continuation = np.maximum(continuation, probability_clip)
                factor[observed_state_rows] = 1.0 / continuation
            cumulative_weight, clipped = _apply_cumulative_clip(
                cumulative_weight,
                factor,
                cumulative_weight_clip,
            )
            clipped_any |= clipped

    return clipped_any


def _iptw_regime_signal(
    raw,
    regime,
    probability_view,
    landmark: int,
    horizon: int,
    probability_clip: float,
    cumulative_weight_clip: float | None,
):
    """IPTW/IPCW survival signal and clipping diagnostics for one regime."""
    regime = tuple(int(action) for action in regime)
    treatment_end = landmark + len(regime) - 1
    n_patients = len(raw["baseline"])
    eligible = raw["at_risk"][:, landmark].astype(bool)
    treatment_match = _treatment_match_mask(raw, regime, landmark)

    treatment_floored = np.zeros(n_patients, dtype=float)
    censoring_floored = np.zeros(n_patients, dtype=float)
    weight_clipped = np.zeros(n_patients, dtype=float)

    def note_probability(mechanism, rows, values):
        rows = np.asarray(rows, dtype=int)
        values = np.asarray(values, dtype=float)
        affected = treatment_floored if mechanism == "treatment" else censoring_floored
        affected[rows] = np.maximum(
            affected[rows], (values < probability_clip).astype(float)
        )

    def note_weight_clip(rows, clipped):
        rows = np.asarray(rows, dtype=int)
        clipped = np.asarray(clipped, dtype=bool)
        if np.any(clipped):
            weight_clipped[rows[clipped]] = 1.0

    observed_horizon = observed_survival_indicator(raw, horizon).astype(float)
    signal = np.zeros(n_patients, dtype=float)
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
        signal[supported_rows] = weights
        ipw_weight[supported_rows] = weights
    signal[~eligible] = np.nan
    return {
        "eligible": eligible,
        "treatment_match": treatment_match,
        "ipw": signal,
        "ipw_weight": ipw_weight,
        "prob_clipped_low": np.maximum(treatment_floored, censoring_floored),
        "treatment_prob_floored": treatment_floored,
        "censoring_prob_floored": censoring_floored,
        "weight_clipped": weight_clipped,
        "iptw_weight_clipped": weight_clipped,
    }


def _match_clip_record(
    arm_signal,
    dr_weight_clipped=None,
) -> dict[str, Any]:
    """Summarize regime support and estimator-specific clipping among full matches."""
    eligible = np.asarray(arm_signal["eligible"], dtype=bool)
    matched = np.asarray(arm_signal["treatment_match"], dtype=bool)
    iptw_clipped = np.asarray(arm_signal["iptw_weight_clipped"], dtype=bool)
    positive_ipw = np.asarray(arm_signal["ipw_weight"], dtype=float) > 0

    n_eligible = int(eligible.sum())
    n_match = int(matched.sum())
    n_positive = int((matched & positive_ipw).sum())
    n_iptw_clipped = int((matched & iptw_clipped).sum())

    if dr_weight_clipped is None:
        n_dr_clipped = np.nan
        dr_clip_fraction = np.nan
    else:
        dr_weight_clipped = np.asarray(dr_weight_clipped, dtype=bool)
        n_dr_clipped = int((matched & dr_weight_clipped).sum())
        dr_clip_fraction = float(n_dr_clipped / n_match) if n_match else np.nan

    return {
        "n_treatment_match": n_match,
        "treatment_match_fraction": (
            float(n_match / n_eligible) if n_eligible else np.nan
        ),
        "positive_ipw_among_match_fraction": (
            float(n_positive / n_match) if n_match else np.nan
        ),
        "n_iptw_weight_clipped_among_match": n_iptw_clipped,
        "iptw_weight_clipped_among_match_fraction": (
            float(n_iptw_clipped / n_match) if n_match else np.nan
        ),
        "iptw_weight_clipped_among_positive_fraction": (
            float(n_iptw_clipped / n_positive) if n_positive else np.nan
        ),
        "n_dr_weight_clipped_among_match": n_dr_clipped,
        "dr_weight_clipped_among_match_fraction": dr_clip_fraction,
    }
