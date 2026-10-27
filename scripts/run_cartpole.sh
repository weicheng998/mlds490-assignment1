#!/usr/bin/env bash
# CartPole-v1: 5 seeds x {no baseline, value baseline}, 2000 episodes each, then 500-episode evals.
# Resumable: re-running skips finished work (training resumes from checkpoints).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p logs
for b in none value; do
  for s in 0 1 2 3 4; do
    python -m pg.train --env cartpole --baseline $b --seed $s --episodes 2000
    python -m pg.evaluate --env cartpole --run cartpole_${b}_s${s}
  done
done
echo "CARTPOLE ALL DONE"
