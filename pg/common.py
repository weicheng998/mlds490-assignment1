"""Shared pieces for REINFORCE on CartPole-v1 and ALE/Pong-v5.

Design notes (also in README / report):
  * Pong: a single 80x80 frame has no motion information, so the policy input is
    the DIFFERENCE of consecutive preprocessed frames (cur - prev). At reset the
    previous frame is set to the current frame, so the first input is all zeros.
  * Pong actions: the network chooses between 2 outputs mapped to ALE actions
    2 (RIGHT) and 3 (LEFT) only.
  * Rollouts run on a CPU copy of the policy (optionally in worker processes);
    gradient updates run on CUDA.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn

PONG_ACTIONS = (2, 3)  # network output index -> ALE action id


# ---------------------------------------------------------------- device ----
def get_device(allow_cpu=False):
    """CUDA is required. `allow_cpu` exists only for local smoke tests."""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"[warn] {torch.cuda.device_count()} GPUs visible; using cuda:0 only. "
                  "Set CUDA_VISIBLE_DEVICES=0.")
        return torch.device("cuda:0")
    if allow_cpu:
        print("[warn] CUDA unavailable; running on CPU because --allow-cpu was given (smoke test only).")
        return torch.device("cpu")
    raise RuntimeError("CUDA is not available. This assignment run requires a GPU "
                       "(use --allow-cpu only for local smoke tests).")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------ preprocess ----
def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80, 80])


class PongDiff:
    """Turns raw Pong frames into frame-difference inputs (int8 in {-1,0,1}).

    preprocess() writes into its input in place, so we always pass a copy of the
    env observation (the env may reuse or freeze that buffer).
    """

    def __init__(self):
        self.prev = None

    def reset(self, frame):
        cur = preprocess(frame.copy())
        self.prev = cur
        return np.zeros(6400, dtype=np.int8)

    def step(self, frame):
        cur = preprocess(frame.copy())
        diff = (cur - self.prev).astype(np.int8).ravel()
        self.prev = cur
        return diff


# ------------------------------------------------------------------ envs ----
def make_env(env_name):
    import gymnasium as gym
    if env_name == "cartpole":
        return gym.make("CartPole-v1")
    if env_name == "pong":
        import ale_py
        gym.register_envs(ale_py)
        return gym.make("ALE/Pong-v5")  # v5 defaults: frameskip=4, sticky actions p=0.25
    raise ValueError(env_name)


# -------------------------------------------------------------- networks ----
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def build_nets(env_name):
    """Returns (policy, value) constructors' outputs for an env."""
    if env_name == "cartpole":
        return MLP(4, 128, 2), MLP(4, 128, 1)
    if env_name == "pong":
        return MLP(6400, 200, 2), MLP(6400, 200, 1)
    raise ValueError(env_name)


# ----------------------------------------------------------- reward-to-go ---
def reward_to_go(rewards, gamma):
    """G_t = sum_{k>=t} gamma^(k-t) r_k, computed backwards in O(T)."""
    g = 0.0
    out = np.zeros(len(rewards), dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


# --------------------------------------------------------------- rollout ----
@torch.no_grad()
def run_episode(env_name, policy, env, seed, greedy=False, trace=0):
    """Play one full episode with a CPU policy. Returns dict(obs, act, rew).

    obs: float32 (T,4) for CartPole; int8 (T,6400) frame differences for Pong.
    act: network output index (0/1). For Pong these map to ALE actions 2/3.
    trace>0 prints the first `trace` steps (state summary, probs, action, reward).
    """
    obs_l, act_l, rew_l = [], [], []
    raw, _ = env.reset(seed=int(seed))
    pong = env_name == "pong"
    if pong:
        diff = PongDiff()
        x = diff.reset(raw)
    else:
        x = np.asarray(raw, dtype=np.float32)
    done = False
    t = 0
    while not done:
        logits = policy(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))[0]
        probs = torch.softmax(logits, -1)
        a = int(torch.argmax(probs)) if greedy else int(torch.multinomial(probs, 1))
        env_a = PONG_ACTIONS[a] if pong else a
        raw, r, term, trunc, _ = env.step(env_a)
        if t < trace:
            s = f"nnz(diff)={int(np.count_nonzero(x))}" if pong else np.round(x, 3).tolist()
            print(f"  t={t:3d} state={s} p={np.round(probs.numpy(), 3).tolist()} a={env_a} r={r}")
        obs_l.append(x)
        act_l.append(a)
        rew_l.append(float(r))
        done = term or trunc
        x = diff.step(raw) if pong else np.asarray(raw, dtype=np.float32)
        t += 1
    return {"obs": np.stack(obs_l), "act": np.asarray(act_l, dtype=np.int64),
            "rew": np.asarray(rew_l, dtype=np.float32)}


# Worker-process plumbing (spawn context; each worker keeps its own env + CPU policy).
_W = {}


def _worker_init(env_name):
    torch.set_num_threads(1)
    _W["env_name"] = env_name
    _W["env"] = make_env(env_name)
    _W["policy"] = build_nets(env_name)[0]
    _W["version"] = -1


def _worker_run(args):
    version, state_dict, seed, greedy = args
    if version != _W["version"]:
        _W["policy"].load_state_dict(state_dict)
        _W["version"] = version
    return run_episode(_W["env_name"], _W["policy"], _W["env"], seed, greedy=greedy)


class Collector:
    """Collects B full episodes with the current policy weights (on-policy)."""

    def __init__(self, env_name, n_workers=0):
        self.env_name = env_name
        self.n_workers = n_workers
        if n_workers > 0:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(n_workers, initializer=_worker_init, initargs=(env_name,))
        else:
            self.env = make_env(env_name)
            self.cpu_policy = build_nets(env_name)[0]
        self.version = 0

    def collect(self, policy, seeds, greedy=False):
        sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        self.version += 1
        if self.n_workers > 0:
            return self.pool.map(_worker_run, [(self.version, sd, s, greedy) for s in seeds], chunksize=1)
        self.cpu_policy.load_state_dict(sd)
        return [run_episode(self.env_name, self.cpu_policy, self.env, s, greedy=greedy) for s in seeds]

    def close(self):
        if self.n_workers > 0:
            self.pool.close()
            self.pool.join()
        else:
            self.env.close()


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
