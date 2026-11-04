from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.mimic_semisynthetic.histories import (
    FEATURES,
    accumulate_events,
    subject_partition,
)

from .extract import ENTRY, GRID_HOURS, N_BINS, NOREPINEPHRINE, VASOPRESSORS

N_DECISIONS = N_BINS - 1
STATE_CHANNELS = FEATURES + ("norepinephrine_rate", "vasopressor_agents")
HOUR = pd.Timedelta(hours=1)


def _hours(times, intime):
    return ((times - intime) / HOUR).to_numpy(float)


def vital_bins(cohort, charts, chunk_rows=500_000):
    stays = cohort[["stay_id", "subject_id", "intime", "outtime", "row"]].copy()
    stays["landmark"] = stays.intime + ENTRY * GRID_HOURS * HOUR + GRID_HOURS * HOUR
    stays["extraction_end"] = stays.intime + N_BINS * GRID_HOURS * HOUR
    sums = np.zeros((len(cohort), N_BINS, len(FEATURES)))
    counts = np.zeros_like(sums, dtype=np.int64)
    audit = dict(
        selected_rows=0,
        excluded_time_rows=0,
        excluded_value_rows=0,
        accepted_rows=0,
        cutoff_violations=0,
    )
    for start in range(0, len(charts), chunk_rows):
        accumulate_events(
            charts.iloc[start : start + chunk_rows], stays, sums, counts, audit
        )
    values = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return values, counts, audit


def vasopressor_bins(cohort, vaso):
    rows = vaso.merge(
        cohort[["stay_id", "intime", "row"]], on="stay_id", validate="many_to_one"
    )
    start = _hours(rows.starttime, rows.intime)
    end = _hours(rows.endtime, rows.intime)
    store = _hours(rows.storetime, rows.intime)
    n = len(cohort)
    running = np.zeros((n, N_BINS), dtype=bool)
    rate = np.full((n, N_BINS), np.nan)
    agents = np.zeros((n, N_BINS), dtype=np.int64)
    known = np.zeros((n, N_BINS), dtype=bool)
    patient = rows.row.to_numpy(int)
    item = rows.itemid.to_numpy(int)
    item_index = {code: i for i, code in enumerate(sorted(VASOPRESSORS))}
    agent_seen = np.zeros((n, N_BINS, len(VASOPRESSORS)), dtype=bool)
    norepi = (item == NOREPINEPHRINE) & (
        rows.rateuom.fillna("").str.strip().str.lower().to_numpy() == "mcg/kg/min"
    )
    values = pd.to_numeric(rows.rate, errors="coerce").to_numpy(float)
    audit = dict(
        rows=len(rows),
        late_rows_excluded_from_dose=0,
        norepinephrine_rows_other_units=int(((item == NOREPINEPHRINE) & ~norepi).sum()),
    )
    for j in range(N_BINS):
        lo, hi = GRID_HOURS * j, GRID_HOURS * (j + 1)
        overlap = (start < hi) & (end > lo)
        running[patient[overlap], j] = True
        available = overlap & (store < hi)
        audit["late_rows_excluded_from_dose"] += int((overlap & ~available).sum())
        known[patient[available], j] = True
        agent_seen[patient[available], j, [item_index[c] for c in item[available]]] = (
            True
        )
        sel = available & norepi & np.isfinite(values)
        if sel.any():
            current = pd.Series(values[sel]).groupby(patient[sel]).max()
            rate[current.index.to_numpy(int), j] = current.to_numpy()
    agents[:] = agent_seen.sum(axis=2)
    audit["running_bins"] = int(running.sum())
    audit["running_bins_without_available_row"] = int((running & ~known).sum())
    treatments = running[:, 1:].astype(np.float32)
    return (
        treatments,
        np.nan_to_num(rate, nan=0.0),
        agents.astype(float),
        running,
        audit,
    )


def outcomes(cohort):
    death_h = _hours(cohort.deathtime, cohort.intime)
    out_h = _hours(cohort.outtime, cohort.intime)
    in_icu_death = (
        np.isfinite(death_h) & (cohort.deathtime <= cohort.outtime).to_numpy()
    )
    end_h = N_BINS * GRID_HOURS
    n = len(cohort)
    event_time = np.full(n, N_DECISIONS + 1, dtype=np.int32)
    censoring_time = np.full(n, N_DECISIONS + 1, dtype=np.int32)
    interval_death = np.floor(death_h / GRID_HOURS) - 1
    interval_out = np.floor(out_h / GRID_HOURS) - 1
    dies = in_icu_death & (death_h < end_h)
    event_time[dies] = (interval_death[dies] + 1).astype(np.int32)
    leaves = ~in_icu_death & (out_h < end_h)
    censoring_time[leaves] = (interval_out[leaves] + 1).astype(np.int32)
    if (event_time[dies] <= ENTRY).any() or (censoring_time[leaves] <= ENTRY).any():
        raise ValueError("an eligible patient left follow-up before the landmark")
    observed_time = np.minimum(
        np.minimum(event_time, censoring_time), N_DECISIONS
    ).astype(np.int32)
    delta = (event_time <= N_DECISIONS).astype(np.int32)
    after_discharge = np.isfinite(death_h) & ~in_icu_death
    same_interval = (
        after_discharge
        & leaves
        & (np.floor(death_h / GRID_HOURS) == np.floor(out_h / GRID_HOURS))
    )
    audit = dict(
        icu_deaths_in_follow_up=int(dies.sum()),
        icu_discharges_alive_in_follow_up=int(leaves.sum()),
        deaths_after_icu_discharge=int(after_discharge.sum()),
        deaths_after_discharge_within_same_interval=int(same_interval.sum()),
        administratively_censored=int((~dies & ~leaves).sum()),
    )
    return event_time, censoring_time, observed_time, delta, audit


def evaluation_survival(cohort, tau):
    horizon = cohort.intime + (GRID_HOURS * (ENTRY + 1 + tau)) * HOUR
    exact = cohort.deathtime.notna()
    survived = np.ones(len(cohort), dtype=float)
    known = np.ones(len(cohort), dtype=bool)
    survived[exact.to_numpy()] = (cohort.deathtime[exact] >= horizon[exact]).to_numpy(
        float
    )
    dated = (~exact & cohort.dod.notna()).to_numpy()
    day_start = cohort.dod.dt.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    before = (day_end <= horizon).to_numpy()
    after = (day_start >= horizon).to_numpy()
    survived[dated & before] = 0.0
    survived[dated & after] = 1.0
    known[dated & ~before & ~after] = False
    return survived, known


def build_raw(cohort, vaso, charts):
    vitals, vital_counts, vital_audit = vital_bins(cohort, charts)
    treatments_all, rate, agents, running, dose_audit = vasopressor_bins(cohort, vaso)
    event_time, censoring_time, observed_time, delta, outcome_audit = outcomes(cohort)
    n = len(cohort)
    states = np.concatenate(
        [vitals, rate[..., None], agents[..., None]], axis=2
    ).astype(np.float32)
    k = np.arange(N_DECISIONS)[None, :]
    at_risk = (k < observed_time[:, None]).astype(np.float32)
    event_next = np.where(
        at_risk > 0, (event_time[:, None] == k + 1).astype(float), np.nan
    )
    j = np.arange(N_BINS)[None, :]
    last_state = np.where(delta == 1, observed_time - 1, observed_time)
    state_observed = (j <= last_state[:, None]).astype(np.float32)
    states[state_observed == 0] = np.nan
    treatments = np.where(at_risk > 0, treatments_all, np.nan).astype(np.float32)
    survive_step = (at_risk > 0) & (event_next == 0)
    censoring_label_observed = survive_step.astype(np.float32)
    censoring_continue = np.where(
        survive_step, (censoring_time[:, None] != k + 1).astype(float), np.nan
    )
    raw = dict(
        baseline=cohort[["age", "female"]].to_numpy(np.float32),
        states=states,
        treatments=treatments,
        at_risk=at_risk,
        state_observed=state_observed,
        treatment_observed=at_risk.copy(),
        event_free_next=(1.0 - event_next).astype(np.float32),
        event_next=event_next.astype(np.float32),
        censoring_continue=censoring_continue.astype(np.float32),
        censoring_label_observed=censoring_label_observed,
        event_time=event_time,
        censoring_time=censoring_time,
        observed_time=observed_time,
        delta=delta,
        patient_id=cohort.subject_id.to_numpy(np.int64),
    )
    roles = np.array([subject_partition(int(s)) for s in cohort.subject_id])
    evaluation = {
        f"survived_tau{tau}": evaluation_survival(cohort, tau)[0] for tau in (8, 16)
    }
    evaluation.update(
        {f"known_tau{tau}": evaluation_survival(cohort, tau)[1] for tau in (8, 16)}
    )
    evaluation["running"] = running
    audit = dict(
        vitals=vital_audit,
        vasopressors=dose_audit,
        outcomes=outcome_audit,
        patients=n,
        roles={r: int((roles == r).sum()) for r in np.unique(roles)},
        vital_bins_observed=float(
            (vital_counts.sum(axis=2) > 0)[state_observed > 0].mean()
        ),
    )
    return raw, roles, evaluation, audit


@dataclass
class StatePreprocessor:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    static_mean: np.ndarray
    static_scale: np.ndarray

    @staticmethod
    def fill(values, median, observed):
        result = np.array(values, dtype=float, copy=True)
        previous = np.broadcast_to(median, (len(result), len(median))).copy()
        for step in range(result.shape[1]):
            previous = np.where(np.isnan(result[:, step]), previous, result[:, step])
            result[:, step] = previous
        result[observed == 0] = np.nan
        return result

    @classmethod
    def fit(cls, states, observed, baseline):
        masked = np.where(observed[..., None] > 0, states, np.nan)
        median = np.nanmedian(masked.reshape(-1, masked.shape[-1]), axis=0)
        if not np.isfinite(median).all():
            raise ValueError("a channel has no training observations")
        filled = cls.fill(states, median, observed)
        flat = filled[observed > 0]
        return cls(
            median,
            flat.mean(axis=0),
            np.maximum(flat.std(axis=0), 1e-6),
            baseline.mean(axis=0),
            np.maximum(baseline.std(axis=0), 1e-6),
        )

    def transform(self, states, observed, baseline):
        x = (self.fill(states, self.median, observed) - self.mean) / self.scale
        w = (np.asarray(baseline, float) - self.static_mean) / self.static_scale
        w[:, 1] = baseline[:, 1]
        return np.clip(x, -5, 5).astype(np.float32), np.clip(w, -5, 5).astype(
            np.float32
        )

    def as_dict(self):
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}


def select(raw, rows):
    return {key: value[rows].copy() for key, value in raw.items()}


def make_splits(raw, roles, inner_rows):
    names = (("train", "nuisance"), ("val", "second_stage"), ("test", "test"))
    splits = {name: select(raw, np.flatnonzero(roles == role)) for name, role in names}
    fit_rows, _ = inner_rows(splits["train"]["patient_id"])
    train = splits["train"]
    prep = StatePreprocessor.fit(
        train["states"][fit_rows],
        train["state_observed"][fit_rows],
        train["baseline"][fit_rows],
    )
    for split in splits.values():
        split["states"], split["baseline"] = prep.transform(
            split["states"], split["state_observed"], split["baseline"]
        )
    return splits, prep


def main():
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    a = p.parse_args()
    data = Path(a.data)
    out = data / "trajectories.npz"
    if out.exists():
        raise FileExistsError(out)
    cohort = pd.read_parquet(data / "cohort.parquet")
    vaso = pd.read_parquet(data / "vasopressor_rows.parquet")
    charts = pd.read_parquet(data / "charts.parquet")
    raw, roles, evaluation, audit = build_raw(cohort, vaso, charts)
    with out.open("xb") as handle:
        np.savez_compressed(
            handle,
            roles=roles,
            **raw,
            **{"eval_" + k: v for k, v in evaluation.items()},
        )
    (data / "trajectories_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
