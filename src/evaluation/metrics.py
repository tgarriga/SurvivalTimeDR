"""Pure metric and replicate-summary functions for the canonical study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PRIMARY_METHODS = ("gcomp", "ra", "iptw_ipcw", "dr", "tv_survcaus")


def effect_metrics(
    predictions: pd.DataFrame,
    methods=PRIMARY_METHODS,
) -> pd.DataFrame:
    """Compute raw conditional-effect error and signed marginal bias."""
    methods = tuple(methods)
    if not methods:
        raise ValueError("at least one method is required")
    if predictions.empty:
        raise ValueError("predictions cannot be empty")
    required = {"effect_true", "mcse_effect"} | {
        f"{method}_effect" for method in methods
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    truth = predictions["effect_true"].to_numpy(float)
    records = []
    for method in methods:
        estimate = predictions[f"{method}_effect"].to_numpy(float)
        error = estimate - truth
        records.append(
            {
                "method": method,
                "cate_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "cate_mae": float(np.mean(np.abs(error))),
                "ate_bias": float(np.mean(error)),
                "absolute_ate_bias": float(abs(np.mean(error))),
                "estimated_ate": float(np.mean(estimate)),
                "true_ate": float(np.mean(truth)),
                "true_cate_sd": float(np.std(truth)),
                "oracle_mcse": float(predictions["mcse_effect"].mean()),
                "n_test_eligible": int(len(predictions)),
            }
        )
    return pd.DataFrame(records)


def mean_confidence_interval(values, confidence: float = 0.95):
    """Replicate mean, standard error and two-sided Student-t interval.

    One implementation shared by the synthetic study summary, the MIMIC-IV path-metric
    summary and the MIMIC-IV report tables. Missing values are dropped, and a single
    replicate gives a NaN width.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    mean = float(values.mean()) if len(values) else np.nan
    standard_error = (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    )
    critical = (
        float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
        if len(values) > 1
        else np.nan
    )
    return (
        mean,
        standard_error,
        mean - critical * standard_error,
        mean + critical * standard_error,
    )


def summarize_study(
    metrics: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize replicate metrics with Student-t confidence intervals."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    group_columns = ["scenario", "treatment_length", "tau", "method"]
    measure_columns = ["cate_rmse", "ate_bias", "absolute_ate_bias"]
    required = set(group_columns + measure_columns + ["replicate"])
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metrics are missing columns: {sorted(missing)}")
    records = []
    for keys, group in metrics.groupby(group_columns, sort=False):
        record = dict(zip(group_columns, keys))
        record["n_replicates"] = int(group["replicate"].nunique())
        for measure in measure_columns:
            mean, standard_error, low, high = mean_confidence_interval(
                group[measure], confidence
            )
            record[f"{measure}_mean"] = mean
            record[f"{measure}_se"] = standard_error
            record[f"{measure}_ci_low"] = low
            record[f"{measure}_ci_high"] = high
        records.append(record)
    return pd.DataFrame(records)


__all__ = [
    "PRIMARY_METHODS",
    "effect_metrics",
    "mean_confidence_interval",
    "summarize_study",
]
