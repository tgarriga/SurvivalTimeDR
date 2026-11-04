# SurvivalTimeDR

Code for [Doubly Robust Learning of Heterogeneous Survival Treatment Effects
under Time-Varying Treatments](https://arxiv.org/abs/2610.26710).

**Authors:** Tomàs Garriga, Munib Mesinovic, Luke De Bretton-Gordon, Gerard Sanz,
Axel Brando, and Tingting Zhu.

The method estimates heterogeneous survival treatment effects from longitudinal
data with time-varying treatments and censoring using doubly robust learning.

Experiments cover fully synthetic data, MIMIC-IV semi-synthetic data, stress
conditions, and real-world survival evaluation with repeated patient partitions.

## Installation

Use Python 3.12 and install the dependencies:

```sh
pip install -r requirements.txt
```

Real-world preprocessing and evaluation also require:

```sh
pip install -r requirements/clinical.txt
```

## Reproduce the experiments

Run from the repository root, using a new output directory for each study.
Experiment settings are in [configs/](configs/). MIMIC-based model fitting requires
CUDA; synthetic runs also support `--device cpu`. Before CUDA runs, set the
environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Fully synthetic experiments require no external data:

```sh
python -m src.run synthetic --device cuda --output runs/synthetic
```

The remaining experiments require authorized access to
[MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/), including PhysioNet
credentialing, training, and the data-use agreement. Retain the original layout:
`hosp/patients.csv.gz`, `hosp/admissions.csv.gz`, `icu/icustays.csv.gz`,
`icu/chartevents.csv.gz`, and `icu/d_items.csv.gz`; real-world preparation also
requires `icu/inputevents.csv.gz`.

Prepare the data, replacing `PATH_TO_MIMIC_IV_3_1` with the raw dataset root:

```sh
python -m src.run prepare semi-synthetic --raw-root PATH_TO_MIMIC_IV_3_1 --output data/histories.npz
python -m src.run prepare realworld --raw-root PATH_TO_MIMIC_IV_3_1 --output data/realworld
```

Run the studies below once their inputs are ready. The semi-synthetic/stress
studies and the real-world studies can run independently. Stress uses data from
semi-synthetic initialization; the real-world split report reuses the completed
original real-world run.

```sh
python -m src.run semi-synthetic --archive data/histories.npz --output runs/semi_synthetic
python -m src.run stress --prepared runs/semi_synthetic/prepared.npz --output runs/stress
python -m src.run realworld --data data/realworld/trajectories.npz --output runs/realworld
python -m src.run realworld-splits --data data/realworld/trajectories.npz --original-runs runs/realworld --output runs/realworld_splits
```

Commands save metrics, predictions, and summaries. Semi-synthetic and stress
studies also produce diagnostic plots. Use `--stage report` to collect completed
runs; for synthetic experiments this verifies the saved run and prints its
summary. Append `--dry-run` to preview a command without reading data or fitting
models. Keep raw MIMIC-IV data, derived patient files, and generated outputs
(including `data/` and `runs/`) out of Git.

