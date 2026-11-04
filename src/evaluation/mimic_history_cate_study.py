"""Checkpointed real-history benchmark using the shared study code."""

import argparse
import gc
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.mimic_history_cate import ENTRY, make_splits, prepare_archive
from src.data.mimic_semisynthetic.checkpoints import save_model, sha256, write_json
from src.evaluation.experiments import (
    StudyScenario,
    run_study_replicate,
    training_history_frame,
)
from src.evaluation.metrics import effect_metrics, summarize_study
from src.evaluation.runtime import peak_rss_kib
from src.protocols import load_protocol, synthetic_config

PROTOCOL = load_protocol("semi_synthetic")
SCENARIOS = tuple(StudyScenario(**scenario) for scenario in PROTOCOL["scenarios"])


def configure():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()


def initialize(archive, root, expected_archive_sha256=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    arrays, prep = prepare_archive(archive, expected_archive_sha256)
    np.savez_compressed(root / "prepared.npz", **arrays)
    cohort_sizes = dict(
        n_nuisance=int((arrays["roles"] == "nuisance").sum()),
        n_second_stage=int((arrays["roles"] == "second_stage").sum()),
        n_test=int((arrays["roles"] == "test").sum()),
    )
    config = synthetic_config("semi_synthetic", "cuda", cohort_sizes)
    manifest = dict(
        schema=2,
        prepared_sha256=sha256(root / "prepared.npz"),
        preprocessing=prep,
        n_replicates=config.n_replicates,
        weight_cap=PROTOCOL["weight_cap"],
        config=asdict(config),
        oracle="unchanged paired Monte Carlo conditional-history oracle; 1000 rollouts",
        scientific_scope="semisynthetic abstract dynamics initialized by real histories",
    )
    write_json(root / "manifest.json", manifest)


def inputs(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if sha256(root / "prepared.npz") != manifest["prepared_sha256"]:
        raise ValueError("prepared input changed")
    with np.load(root / "prepared.npz", allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    return root, manifest, data


class Recorder:
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        self.root.mkdir(parents=True, exist_ok=False)
        self.count = 0

    def save(self, name, model, constructor):
        path = self.root / (name + ".pt")
        save_model(path, model, constructor, self.identity)
        history = training_history_frame(model, model_role=name, training_task=name)
        if history.empty or not np.isfinite(history.train_loss).all():
            raise ValueError("invalid training history")
        history.to_csv(path.with_suffix(".history.csv"), index=False)
        self.count += 1

    def __call__(self, phase, **kw):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                return self._record(phase, **kw)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def _record(self, phase, **kw):
        cfg = kw["config"]
        if phase == "shared":
            p, n = kw["probability"], kw["numerator"]
            for task, attr in [
                ("propensity", "propensity_model"),
                ("censoring", "censoring_model"),
            ]:
                self.save(
                    "shared/" + task,
                    getattr(p, attr),
                    {**cfg.probability_transformer, "nuisance_task": task},
                )
            self.save(
                "shared/numerator", n.model, cfg.probability_transformer
            )
            return
        learners, heads, tv, scenario = (
            kw["learners"],
            kw["heads"],
            kw["tv"],
            kw["scenario"],
        )
        for arm, learner in learners.items():
            for kind, models in [("Q", learner.q_models), ("m", learner.m_models)]:
                for step, model in models.items():
                    self.save(
                        f"{scenario.name}/{arm}.{kind}.{step}",
                        model,
                        {**cfg.transformer, "nuisance_task": kind.lower()},
                    )
        for name, model in heads.items():
            self.save(f"{scenario.name}/{name}", model, cfg.transformer)
        self.save(
            f"{scenario.name}/tv_survcaus",
            tv.model,
            dict(
                treatment_length=scenario.treatment_length,
                tau=scenario.tau,
                representation_dim=cfg.tv_survcaus_representation_dim,
                regime_embedding_dim=cfg.tv_survcaus_regime_embedding_dim,
                **cfg.transformer,
            ),
        )


def case(root, rep):
    configure()
    root, manifest, data = inputs(root)
    if not 0 <= rep < 5:
        raise ValueError("replicate outside frozen protocol")
    dest = root / f"replicate_{rep:02d}"
    dest.mkdir(exist_ok=False)
    started = time.monotonic()
    config = synthetic_config(
        "semi_synthetic",
        "cuda",
        dict(
            n_nuisance=int((data["roles"] == "nuisance").sum()),
            n_second_stage=int((data["roles"] == "second_stage").sum()),
            n_test=int((data["roles"] == "test").sum()),
        ),
    )
    splits, scaling = make_splits(data, rep, config.simulation, study_seed=config.seed)
    ids = [set(raw["patient_id"].tolist()) for raw in splits.values()]
    assert not any(ids[i] & ids[j] for i in range(3) for j in range(i))
    for name, raw in splits.items():
        np.savez_compressed(
            dest / f"{name}.npz", **{k: v for k, v in raw.items() if k != "config"}
        )
    write_json(dest / "scaling.json", {k: v.tolist() for k, v in scaling.items()})
    if json.loads(json.dumps(asdict(config))) != manifest["config"]:
        raise ValueError("runtime config differs from frozen protocol")
    cap = manifest["weight_cap"]
    if cap != config.cumulative_weight_clip:
        raise ValueError("weight cap differs from the frozen configuration")
    cfg = replace(config, cumulative_weight_clip=float(cap))
    write_json(dest / "config.json", asdict(cfg))
    rec = Recorder(
        dest / "checkpoints",
        dict(
            prepared_sha256=manifest["prepared_sha256"],
            replicate=rep,
            cap=cap,
            smoke=False,
        ),
    )
    metrics, predictions, diagnostics = run_study_replicate(
        cfg,
        rep,
        raw_splits=splits,
        scaling_params=scaling,
        training_start=ENTRY,
        artifact_callback=rec,
    )
    for scenario in cfg.scenarios:
        pred = predictions.query("scenario == @scenario.name")
        check = effect_metrics(pred)
        saved = metrics.query("scenario == @scenario.name")
        for method in check.method:
            err = pred[method + "_effect"].to_numpy() - pred.effect_true.to_numpy()
            actual = saved.set_index("method").loc[method]
            np.testing.assert_allclose(
                [actual.cate_rmse, actual.cate_mae, actual.ate_bias],
                [np.sqrt(np.mean(err**2)), np.mean(np.abs(err)), np.mean(err)],
                rtol=1e-12,
                atol=1e-12,
            )
    metrics.to_csv(dest / "metrics.csv", index=False)
    predictions.to_csv(dest / "predictions.csv", index=False)
    diagnostics.to_csv(dest / "diagnostics.csv", index=False)
    gc.collect()
    torch.cuda.empty_cache()
    write_json(
        dest / "complete.json",
        dict(
            status="complete",
            replicate=rep,
            smoke=False,
            cap=cap,
            component_fits=rec.count,
            metrics_checked=True,
            elapsed_seconds=time.monotonic() - started,
            device=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            dtype="float32 models; native NumPy simulation/oracle",
            max_rss_kib=peak_rss_kib(),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        ),
    )


def report(root):
    root, manifest, _ = inputs(root)
    out = root / "report"
    out.mkdir(exist_ok=False)
    cap = manifest["weight_cap"]
    frames = []
    predictions = []
    diagnostics = []
    checks = []
    for rep in range(5):
        dest = root / f"replicate_{rep:02d}"
        done = json.loads((dest / "complete.json").read_text())
        assert done["status"] == "complete" and not done["smoke"]
        assert done["metrics_checked"]
        assert done["cap"] == cap
        frames.append(pd.read_csv(dest / "metrics.csv").assign(cap=cap))
        predictions.append(pd.read_csv(dest / "predictions.csv").assign(cap=cap))
        diagnostics.append(pd.read_csv(dest / "diagnostics.csv").assign(cap=cap))
        checks.append(dict(replicate=rep, cap=cap, fits=done["component_fits"]))
    metrics = pd.concat(frames)
    pred = pd.concat(predictions)
    assert len(metrics) == 5 * 4 * 5
    summary = summarize_study(metrics).assign(cap=cap)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pred.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "checkpoints.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        out / "diagnostics.csv", index=False
    )
    oracle = (
        pred.groupby(["scenario", "replicate"])
        .agg(
            mean_mcse=("mcse_effect", "mean"),
            max_mcse=("mcse_effect", "max"),
            true_cate_sd=("effect_true", "std"),
        )
        .reset_index()
    )
    oracle.insert(0, "cap", cap)
    oracle.to_csv(out / "oracle_precision.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.plotting import points

    names = ("gcomp", "ra", "iptw_ipcw", "dr", "tv_survcaus")
    labels = ("g-computation", "RA", "IPTW/IPCW", "DR", "TV-SurvCaus")
    colors = ("#4477AA", "#66BBBB", "#EE7733", "#228833", "#AA77AA")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (method, label, color) in enumerate(zip(names, labels, colors)):
        for j, scenario in enumerate(SCENARIOS):
            values = metrics.query("method == @method and scenario == @scenario.name")
            for ax, key in zip(axes, ("cate_rmse", "ate_bias")):
                points(
                    ax,
                    values[key],
                    j + (i - 2) * 0.13,
                    color,
                    label if j == 0 else None,
                )
    for ax, title in zip(axes, ("CATE RMSE", "Signed ATE bias")):
        ax.set(
            title=title, xticks=range(4), xticklabels=[s.name for s in SCENARIOS]
        )
    axes[1].axhline(0, color="grey")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=180)
    fig.savefig(out / "summary.pdf")
    plt.close(fig)
    write_json(
        out / "complete.json",
        dict(
            status="complete",
            replicates=5,
            component_fits=sum(c["fits"] for c in checks),
        ),
    )
    print(json.dumps(dict(status="complete", report=str(out))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["init", "case", "report"])
    p.add_argument("--root", required=True)
    p.add_argument("--archive")
    p.add_argument("--expected-archive-sha256")
    p.add_argument("--replicate", type=int, default=0)
    a = p.parse_args()
    if a.command == "init":
        initialize(a.archive, a.root, a.expected_archive_sha256)
    elif a.command == "report":
        report(a.root)
    else:
        case(a.root, a.replicate)


if __name__ == "__main__":
    main()
