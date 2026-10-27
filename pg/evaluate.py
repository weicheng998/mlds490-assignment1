"""Roll out N (default 500) episodes with a trained policy; save rewards, stats, histogram.

The policy is sampled stochastically (the learned policy itself); pass --greedy for argmax.
Evaluation env seeds (10^9 + i) never overlap with training seeds.

  python -m pg.evaluate --env cartpole --run cartpole_none_s0
"""
import argparse
import json
import os

import numpy as np
import torch

from pg.common import Collector, build_nets, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pong"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results/pg")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    get_device(a.allow_cpu)

    policy = build_nets(a.env)[0]
    policy.load_state_dict(torch.load(os.path.join(a.ckpt_dir, a.run, "policy_final.pt"),
                                      map_location="cpu"))
    col = Collector(a.env, a.workers)
    seeds = [10**9 + i for i in range(a.n)]
    eps = col.collect(policy, seeds, greedy=a.greedy)
    col.close()
    rets = np.array([float(e["rew"].sum()) for e in eps])
    lens = np.array([len(e["rew"]) for e in eps])
    tag = f"{a.run}_eval{'_greedy' if a.greedy else ''}"
    np.save(os.path.join(a.results_dir, f"{tag}.npy"), rets)
    stats = {"run": a.run, "n": int(a.n), "greedy": a.greedy, "mean": float(rets.mean()),
             "std": float(rets.std(ddof=1)), "min": float(rets.min()), "max": float(rets.max()),
             "mean_len": float(lens.mean())}
    with open(os.path.join(a.results_dir, f"{tag}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
