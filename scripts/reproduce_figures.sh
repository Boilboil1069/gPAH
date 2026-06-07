#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf results/replotted
mkdir -p results/replotted/policy results/replotted/rl results/replotted/ablation

cp data/raw/final_run/policy/*.csv results/replotted/policy/
cp data/raw/final_run/rl/*.csv results/replotted/rl/
cp data/raw/ablation_r50_conditional/*.csv results/replotted/ablation/

python experiment/replot_gpah_paper1_figures.py \
  --policy_dir results/replotted/policy \
  --ablation_dir results/replotted/ablation

python experiment/replot_gpah_paper1_rl_figures.py \
  --output_dir results/replotted/rl

