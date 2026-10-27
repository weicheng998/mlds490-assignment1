"""Figures for Part 3 / 3b from results/pg/*.csv and *_eval.npy.   python -m pg.plot"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from viz import plt, BLUE, ORANGE, GRAY, INK2

LABEL = {"none": "REINFORCE (no baseline)", "value": "REINFORCE + learned V(s) baseline"}
COLOR = {"none": BLUE, "value": ORANGE}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None


def ma(x, w=100):
    """Trailing moving average; the first w-1 points average over what is available."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def curve(ax, d, color, title, ylabel="episode reward"):
    ep = d["episode"] + 1
    ax.plot(ep, d["return"], color=color, alpha=0.25, lw=0.6, label="episode reward")
    ax.plot(ep, ma(d["return"]), color=color, lw=2, label="100-episode moving average")
    ax.set(xlabel="training episode", ylabel=ylabel, title=title)
    ax.legend(loc="upper left" if d["return"].mean() > 0 else "lower right", fontsize=8)


def hist(ax, r, color, title, bins):
    ax.hist(r, bins=bins, color=color, edgecolor="white", linewidth=0.8)
    ax.axvline(r.mean(), color=INK2, lw=1.2, ls="--")
    ax.set(xlabel="episode reward", ylabel="# episodes (of 500)", title=title)
    ax.text(0.02, 0.95, f"mean = {r.mean():.2f}\nstd = {r.std(ddof=1):.2f}\nn = {len(r)}",
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/pg")
    p.add_argument("--seed", type=int, default=0, help="seed shown in single-run figures")
    a = p.parse_args()
    R = a.results_dir
    figs = []

    for env, bins, ylim in (("cartpole", np.arange(0, 525, 25), (0, 520)), ("pong", np.arange(-21.5, 22, 1), (-22, 22))):
        for b in ("none", "value"):
            run = f"{env}_{b}_s{a.seed}"
            path = f"{R}/{run}.csv"
            if not os.path.exists(path):
                continue
            d = load(path)
            ev = f"{R}/{run}_eval.npy"
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
            name = "CartPole-v1" if env == "cartpole" else "Pong-v5"
            curve(axes[0], d, COLOR[b], f"{name} training: {LABEL[b]}")
            axes[0].set_ylim(*ylim)
            if os.path.exists(ev):
                hist(axes[1], np.load(ev), COLOR[b], f"500 evaluation episodes (trained policy)", bins)
            else:
                axes[1].text(0.5, 0.5, "evaluation pending", ha="center", transform=axes[1].transAxes)
            fig.tight_layout()
            out = f"{R}/fig_{run}.png"
            fig.savefig(out); plt.close(fig); figs.append(out)

    # CartPole: all seeds, both methods.
    runs = {b: sorted(glob.glob(f"{R}/cartpole_{b}_s[0-9].csv")) for b in ("none", "value")}
    if runs["none"] and runs["value"]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        summary = {}
        for b in ("none", "value"):
            M = [ma(load(f)["return"]) for f in runs[b]]
            n = min(len(m) for m in M)
            M = np.stack([m[:n] for m in M])
            x = np.arange(1, n + 1)
            for m in M:
                axes[0 if b == "none" else 1].plot(x, m, color=COLOR[b], lw=1, alpha=0.8)
            axes[2].plot(x, M.mean(0), color=COLOR[b], label=LABEL[b])
            axes[2].fill_between(x, M.min(0), M.max(0), color=COLOR[b], alpha=0.15, lw=0)
            raw = np.stack([load(f)["return"][:n] for f in runs[b]])
            evs = [np.load(f.replace(".csv", "_eval.npy")) for f in runs[b] if os.path.exists(f.replace(".csv", "_eval.npy"))]
            first = [int(np.argmax(m >= 475)) + 1 if (m >= 475).any() else None for m in M]
            summary[b] = {
                "seeds": len(runs[b]),
                "mean return over all training episodes (per seed)": raw.mean(1).round(1).tolist(),
                "first episode with MA100 >= 475 (per seed)": first,
                "final MA100 (per seed)": M[:, -1].round(1).tolist(),
                "across-seed std of MA100, averaged over training": float(M.std(0).mean()),
                "within-run std of episode reward over the last 500 episodes (mean over seeds)": float(raw[:, -500:].std(1).mean()),
                "eval mean per seed": [round(float(e.mean()), 1) for e in evs],
                "eval std per seed": [round(float(e.std(ddof=1)), 1) for e in evs],
            }
        for i, b in enumerate(("none", "value")):
            axes[i].set(title=f"{LABEL[b]}\n100-ep moving average, {len(runs[b])} seeds", xlabel="training episode",
                        ylabel="episode reward (MA100)", ylim=(0, 520))
        axes[2].set(title="mean over seeds (band = min–max)", xlabel="training episode", ylabel="episode reward (MA100)", ylim=(0, 520))
        axes[2].legend(loc="lower right", fontsize=8)
        fig.tight_layout(); out = f"{R}/fig_cartpole_seeds.png"; fig.savefig(out); plt.close(fig); figs.append(out)
        with open(f"{R}/cartpole_seed_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps(summary, indent=1))

    # Baseline effect on the size of the policy-gradient weights during training.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    drew = False
    for i, env in enumerate(("cartpole", "pong")):
        for f in sorted(glob.glob(f"{R}/{env}_value_s*.csv"))[:5]:
            d = load(f)
            _, idx = np.unique(d["update"], return_index=True)
            ratio = d["rms_weight"][idx] / np.maximum(d["rms_return"][idx], 1e-8)
            w = 100 if env == "cartpole" else 20
            axes[i].plot(d["episode"][idx] + 1, ma(ratio, w), color=ORANGE, lw=1.2, alpha=0.9)
            drew = True
        axes[i].axhline(1, color=GRAY, lw=1, ls="--")
        axes[i].set(title=f"{'CartPole-v1' if env == 'cartpole' else 'Pong-v5'}: RMS(G−V) / RMS(G) per update",
                    xlabel="training episode", ylabel="ratio (1 = no reduction)", ylim=(0, 1.2))
    if drew:
        fig.tight_layout(); out = f"{R}/fig_baseline_weight_ratio.png"; fig.savefig(out); figs.append(out)
    plt.close(fig)

    # Pong: both methods on one axis (moving averages only).
    pr = {b: f"{R}/pong_{b}_s0.csv" for b in ("none", "value")}
    if all(os.path.exists(v) for v in pr.values()):
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for b, f in pr.items():
            d = load(f)
            ax.plot(d["episode"] + 1, ma(d["return"]), color=COLOR[b], label=LABEL[b])
        ax.set(xlabel="training episode", ylabel="episode reward (MA100)", title="Pong-v5: 100-episode moving average",
               ylim=(-21.5, 21.5))
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); out = f"{R}/fig_pong_compare.png"; fig.savefig(out); plt.close(fig); figs.append(out)
    print("\n".join(figs))


if __name__ == "__main__":
    main()
