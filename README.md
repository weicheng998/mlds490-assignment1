# MLDS 490 — Assignment 1: Value/Policy Iteration and Policy Gradients

Author: Weicheng (GitHub `weicheng998`) · Repository: https://github.com/weicheng998/mlds490-assignment1

| Part | What | Code |
|---|---|---|
| 2 | Shuttle-dispatch MDP: enumeration (T=500), value iteration, policy iteration | `mdp/` |
| 3 | REINFORCE (reward-to-go) on CartPole-v1 and Pong-v5 | `pg/` |
| 3b | REINFORCE with a learned state-value baseline V(s) | `pg/` (`--baseline value`) |

The report (`report/report.pdf`) contains all figures, assumptions, edge cases, validation results and discussion.

## Setup

```bash
conda activate myenv            # deepdish; PyTorch with CUDA (cu126)
pip install -r requirements.txt # gymnasium[atari]==1.0.0, ale-py==0.10.1, matplotlib, scipy
```

Training and evaluation **require CUDA** and raise an error at startup otherwise. `--allow-cpu` exists only for
local smoke tests and the CPU-only validation scripts. Use one GPU: `CUDA_VISIBLE_DEVICES=0`.

## Part 2 — shuttle MDP (CPU, ~15 s)

```bash
python -m mdp.run        # solves with all three methods, writes results/mdp/fig_*.png and summary.json
python -m mdp.validate   # 9 validation experiments -> results/mdp/validation.{txt,json}
```

Modeling assumptions (see `mdp/shuttle.py`):
* State = number waiting, 0..200. Actions = hold / dispatch, both allowed in every state, including s=0.
* Dispatch carries min(s, K) customers and costs cf = 100.
* Holding cost ch = 2 per customer still waiting **after** the decision.
* Arrivals A ~ Unif{1..5} come after the decision. The next state is min(s_after + A, 200): customers beyond 200 are turned away.
* Reward = −cost, γ = 0.95, terminal value V_T = 0 for enumeration.

## Part 3 / 3b — policy gradient

```bash
# CartPole-v1: 5 seeds x {no baseline, value baseline}, 2000 episodes, then 500-episode evaluations
nohup bash scripts/run_cartpole.sh > logs/cartpole.log 2>&1 &
# Pong-v5 (time-limited; resumes from checkpoints/ if restarted)
nohup bash scripts/run_pong.sh none  7 > logs/pong_none.log  2>&1 &
nohup bash scripts/run_pong.sh value 7 > logs/pong_value.log 2>&1 &
python -m pg.plot        # figures -> results/pg/fig_*.png, cartpole_seed_summary.json
```

Single run: `python -m pg.train --env {cartpole,pong} --baseline {none,value} --seed 0 [--lr --batch --episodes --workers --time-limit-hours]`.
Evaluation: `python -m pg.evaluate --env pong --run pong_value_s0 --workers 10`.

Key design choices:
* **Loss.** −(1/B) Σ_episodes Σ_t log π(a_t|s_t)·G_t, with reward-to-go G_t and no return normalization. The baseline variant uses (G_t − V(s_t)).detach(). V is a separate network trained by MSE to G_t.
* **CartPole-v1.** γ = 0.95, MLP 4-128-2, Adam lr 0.01, one episode per update.
* **Pong-v5.** γ = 0.99, MLP 6400-200-2, Adam lr 1e-3, 10 episodes per update. The exact `preprocess()` from the handout is used. The policy picks only between ALE actions 2 (up) and 3 (down).
* **Pong motion.** The network input is the difference of consecutive preprocessed frames, cur − prev. It is all zeros on the first step.
* **Rollouts and updates.** Rollouts run on CPU copies of the policy (parallel worker processes for Pong). Gradient updates run on the GPU.
* **Checkpointing.** Checkpoints are written atomically every `--ckpt-every` updates. Restarting the same command resumes, and the CSV log is truncated to the checkpoint so it has no duplicate rows.

## Validation (CPU)

```bash
python -m pg.validate --allow-cpu                       # reward-to-go, exact-gradient toy MDP, reward-sign control,
                                                        # Pong preprocessing/action mapping, seeding, first-10-steps trace
python -m pg.validate --allow-cpu --tests variance      # needs checkpoints/cartpole_value_s*/ckpt.pt
python -m pg.validate --allow-cpu --tests variance --variance-env pong --variance-episodes 40
```

## Layout

```
mdp/shuttle.py      MDP model, enumeration, value iteration, policy iteration, independent MC simulator
mdp/run.py          Part 2 figures        mdp/validate.py   Part 2 validation experiments
pg/common.py        preprocess, frame-difference wrapper, networks, reward-to-go, rollout workers
pg/train.py         REINFORCE (+ baseline) training with checkpoint/resume
pg/evaluate.py      500-episode evaluation   pg/plot.py  figures   pg/validate.py  validation experiments
scripts/            deepdish launch scripts  results/  logs (CSV/JSON) and figures   report/  report source + PDF
viz.py              shared plot style
```
