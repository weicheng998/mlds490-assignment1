#!/usr/bin/env bash
# Pong-v5, one run: $1 = none | value, $2 = time limit in hours (default 7.5).
# Adam lr 1e-3, 10 episodes per update, 10 rollout workers. Resumes from checkpoint if present.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
B=${1:?none|value}; H=${2:-7.5}
python -m pg.train --env pong --baseline "$B" --seed 0 --lr 1e-3 --batch 10 --workers 10 \
    --time-limit-hours "$H"
python -m pg.evaluate --env pong --run pong_${B}_s0 --workers 10
echo "PONG $B DONE"
