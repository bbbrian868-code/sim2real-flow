# sim2real-flow

Flow matching for sim-to-real spatiotemporal prediction on [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench) (cylinder wake).
Research code, work in progress.

Current contents:
- **RealPDEBench 2.0.1 migration**: remote/local diff of the dataset, split and integrity checks, re-evaluation of the 2.0.0 checkpoints.
- **Baselines**: U-Net, DeepONet and Transolver retrained on 2.0.1 (numerical / real / finetune), evaluated through a mirror of the official evaluation loop.
- **Basic flow matching**: rectified flow (linear path, Gaussian source, Euler sampler) with U-Net and DiT backbones in three sizes; Phase 6 results, ablations (Euler steps N, number of samples K, sim noise on the condition only), training/eval curves and field visualizations.

Code only: no data, checkpoints or result files are tracked (see `.gitignore`).

## Layout

| Path | Contents |
|---|---|
| `fm/` | flow matching: `paths.py` (loss, Euler sampler), `embed.py`, `backbones/` (U-Net, DiT), `wrapper.py` (`FMPredictor`, official-model interface), `train_fm.py`, `configs/cylinder/`, `tests/` |
| `pilfm/` | shared tooling: fast zero-copy dataset, mirror of the official eval loop, result metadata |
| `scripts/` | evaluation, summaries, plots, data checks; `legacy/` holds the scripts of the 2.0.0 runs |
| `configs/cylinder/` | official baseline configs as used (paths and seeds changed) |
| `slurm/` | job scripts for the NCHC Nano4 cluster |
| `jobs/` | job lists and submission queues |
| `patches/` | local modification of the official package, as a diff |
| `docs/` | task reports (`reports/step_*.md`), `DECISIONS.md` (every non-obvious choice, with reasons), consolidated Phase 0-4 report |

## Setup

Requires the official RealPDEBench package (commit `62f4c80`, plus `patches/`) installed in the same environment, and the dataset from
[Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench) (version 2.0.1).

Paths to the data, checkpoints and results are currently hard-coded for our cluster (`/work/<user>/...`) and the
job scripts carry our cluster account; they need to be changed to run elsewhere.

## Data policy

This repository contains algorithm code and experiments on **public** datasets only (currently RealPDEBench).
**Partner or confidential data must never be committed here, even though the repository is private**, and
neither may anything derived from it: geometry, dimensions, operating conditions, material or device names,
preprocessing or hyperparameters tuned on it, results, figures, logs or notebook outputs. Code that works with
such data belongs in a separate repository on approved infrastructure, which imports this one, not the other way around.

Git hooks in `.githooks/` enforce part of this (data / model / CAD / CFD / office file types, files over 2 MB,
restricted directory names, credentials, project-specific sensitive terms) on commit, on the commit message,
and again on push. After cloning, enable them and create the local term list:

```bash
git config core.hooksPath .githooks
# one term per line, case-insensitive; kept inside .git/ so it is never committed (ask the maintainer for the list)
$EDITOR .git/sensitive-terms
```

Do not bypass the hooks with `--no-verify`; for a reviewed exception add a path glob to `.githooks/allowlist`.
The hooks are a safety net, not a guarantee: they cannot recognise every derived quantity.

## License and attribution

Released under [CC BY-NC 4.0](LICENSE) (non-commercial use only). Parts of this repository are adapted from
RealPDEBench (CC BY-NC 4.0); [NOTICE](NOTICE) lists the adapted files and what was changed.
The RealPDEBench dataset is not included and is distributed by its authors under CC BY-NC 4.0.

If you use the benchmark, please cite RealPDEBench:
"RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data", ICLR 2026, [arXiv:2601.01829](https://arxiv.org/abs/2601.01829).
