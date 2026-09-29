# pi-lfm-code

Code for RealPDEBench cylinder experiments (2.0.1 migration, baselines, flow matching).
Code only; no data or checkpoints.

| What | Path |
|---|---|
| Official RealPDEBench repo | /work/b314513067/RealPDEBench |
| Data, checkpoints, results | /work/b314513067/pi-lfm |

- `configs/cylinder/` — copied from `pi-lfm/configs/cylinder` (configs used for the 2.0.0 runs; originals left in place).
- `scripts/legacy/` — copied from the gitignored `RealPDEBench/scripts/` plus the untracked
  `download_cylinder.sh`, `env.sh`, `setup_env.sh` from the official repo root (originals left in place).
- `patches/` — local modification present in the official repo working tree, recorded as a diff.
