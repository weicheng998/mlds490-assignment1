"""REINFORCE (reward-to-go) with an optional learned state-value baseline.

Part 1 (--baseline none):    loss = -(1/B) sum_episodes sum_t log pi(a_t|s_t) * G_t
Part 2 (--baseline value):   loss = -(1/B) sum sum log pi(a_t|s_t) * (G_t - V(s_t)).detach()
                             + separate value loss mean_t (V(s_t) - G_t)^2
No return normalization in either mode, so the two are directly comparable.

Example:
  python -m pg.train --env cartpole --baseline none --seed 0
  python -m pg.train --env pong --baseline value --lr 1e-3 --batch 10 --workers 10
Runs resume automatically from <ckpt_dir>/<run>/ckpt.pt if it exists.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from pg.common import (Collector, atomic_torch_save, build_nets, get_device,
                       reward_to_go, seed_everything)

DEFAULTS = {
    "cartpole": dict(gamma=0.95, lr=1e-2, batch=1, episodes=2000, workers=0, ckpt_every=200),
    "pong": dict(gamma=0.99, lr=1e-3, batch=10, episodes=100000, workers=10, ckpt_every=10),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], required=True)
    p.add_argument("--baseline", choices=["none", "value"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--value-lr", type=float, help="defaults to --lr")
    p.add_argument("--batch", type=int, help="episodes per gradient update")
    p.add_argument("--episodes", type=int, help="total training episodes")
    p.add_argument("--workers", type=int, help="rollout worker processes (0 = in-process)")
    p.add_argument("--ckpt-every", type=int, help="updates between checkpoints")
    p.add_argument("--time-limit-hours", type=float, default=1e9)
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-dir", default="results/pg")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--chunk", type=int, default=8192, help="timesteps per backward chunk")
    p.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    a = p.parse_args()
    for k, v in DEFAULTS[a.env].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.value_lr is None:
        a.value_lr = a.lr
    if a.run_name is None:
        a.run_name = f"{a.env}_{a.baseline}_s{a.seed}"
    return a


def update(policy, value, opt_p, opt_v, episodes, gamma, device, chunk):
    """One REINFORCE step on a batch of complete episodes. Returns stats dict."""
    B = len(episodes)
    obs = np.concatenate([e["obs"] for e in episodes])
    act = np.concatenate([e["act"] for e in episodes])
    ret = np.concatenate([reward_to_go(e["rew"], gamma) for e in episodes]).astype(np.float32)
    N = len(act)
    opt_p.zero_grad(set_to_none=True)
    if value is not None:
        opt_v.zero_grad(set_to_none=True)
    v_loss_total, adv_sq, ret_sq = 0.0, 0.0, 0.0
    for i in range(0, N, chunk):
        o = torch.as_tensor(obs[i:i + chunk], dtype=torch.float32, device=device)
        a = torch.as_tensor(act[i:i + chunk], device=device)
        g = torch.as_tensor(ret[i:i + chunk], device=device)
        logp = torch.log_softmax(policy(o), -1).gather(1, a[:, None]).squeeze(1)
        if value is not None:
            v = value(o).squeeze(1)
            weight = (g - v).detach()                 # advantage, no gradient into V
            v_loss = ((v - g) ** 2).sum() / N
            v_loss.backward()
            v_loss_total += v_loss.item()
        else:
            weight = g
        adv_sq += float((weight ** 2).sum())
        ret_sq += float((g ** 2).sum())
        p_loss = -(logp * weight).sum() / B
        p_loss.backward()
    gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in policy.parameters())))
    opt_p.step()
    if value is not None:
        opt_v.step()
    return {"value_loss": v_loss_total, "grad_norm": gnorm,
            "rms_weight": (adv_sq / N) ** 0.5, "rms_return": (ret_sq / N) ** 0.5, "steps": N}


def main():
    args = parse()
    device = get_device(args.allow_cpu)
    run_dir = os.path.join(args.ckpt_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    csv_path = os.path.join(args.results_dir, f"{args.run_name}.csv")

    seed_everything(args.seed)
    policy, value = build_nets(args.env)
    policy.to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = None
    if args.baseline == "value":
        value.to(device)
        opt_v = torch.optim.Adam(value.parameters(), lr=args.value_lr)
    else:
        value = None

    episode, n_update, wall_prev = 0, 0, 0.0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy"])
        opt_p.load_state_dict(ck["opt_p"])
        if value is not None:
            value.load_state_dict(ck["value"])
            opt_v.load_state_dict(ck["opt_v"])
        episode, n_update, wall_prev = ck["episode"], ck["update"], ck["wall"]
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        # Drop CSV rows logged after the checkpoint so the log has no duplicates.
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            keep = [rows[0]] + [r for r in rows[1:] if int(r[0]) < episode]
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerows(keep)
        print(f"[resume] {args.run_name}: episode={episode} update={n_update}")
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "update", "return", "length", "wall_s",
                                    "grad_norm", "value_loss", "rms_weight", "rms_return"])
    print(f"[config] {vars(args)} device={device}", flush=True)

    collector = Collector(args.env, args.workers)
    t0 = time.time()

    def save():
        ck = {"policy": policy.state_dict(), "opt_p": opt_p.state_dict(),
              "episode": episode, "update": n_update, "wall": wall_prev + time.time() - t0,
              "py_rng": random.getstate(), "np_rng": np.random.get_state(),
              "torch_rng": torch.get_rng_state(), "args": vars(args)}
        if value is not None:
            ck["value"] = value.state_dict()
            ck["opt_v"] = opt_v.state_dict()
        atomic_torch_save(ck, ckpt_path)
        atomic_torch_save(policy.state_dict(), os.path.join(run_dir, "policy_final.pt"))

    try:
        while episode < args.episodes:
            if time.time() - t0 > args.time_limit_hours * 3600:
                print("[stop] time limit reached", flush=True)
                break
            B = min(args.batch, args.episodes - episode)
            # Unique, reproducible env seed for every training episode.
            seeds = [args.seed * 1_000_003 + episode + i for i in range(B)]
            eps = collector.collect(policy, seeds)
            stats = update(policy, value, opt_p, opt_v, eps, args.gamma, device, args.chunk)
            n_update += 1
            wall = wall_prev + time.time() - t0
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                for i, e in enumerate(eps):
                    w.writerow([episode + i, n_update, float(e["rew"].sum()), len(e["rew"]),
                                round(wall, 1), round(stats["grad_norm"], 4),
                                round(stats["value_loss"], 5), round(stats["rms_weight"], 4),
                                round(stats["rms_return"], 4)])
            episode += B
            if n_update % max(1, 100 // args.batch) == 0 or args.env == "pong":
                rets = [float(e["rew"].sum()) for e in eps]
                print(f"ep {episode:6d} upd {n_update:5d} ret {np.mean(rets):7.2f} "
                      f"len {stats['steps'] / B:6.0f} gnorm {stats['grad_norm']:.3g} "
                      f"vloss {stats['value_loss']:.3g} t {wall / 60:.1f}m", flush=True)
            if n_update % args.ckpt_every == 0:
                save()
    finally:
        save()
        collector.close()
        print(f"[done] {args.run_name} episodes={episode}", flush=True)


if __name__ == "__main__":
    main()
