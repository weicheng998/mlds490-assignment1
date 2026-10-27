"""Solve the shuttle MDP three ways and make the Part 2 figures.  python -m mdp.run"""
import json
import os

import numpy as np

import viz
from viz import plt, BLUE, ORANGE, AQUA, GRAY, INK2
from mdp.shuttle import ShuttleMDP, threshold_of

OUT = "results/mdp"


def main():
    os.makedirs(OUT, exist_ok=True)
    m = ShuttleMDP()
    s = np.arange(m.S)
    Ve, pe = m.enumerate(T=500)
    Vv, pv, hist = m.value_iteration()
    Vp, pp, H = m.policy_iteration()
    tau, _ = threshold_of(pp)

    # (a) enumeration V_0
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(s, Ve[0], color=BLUE)
    ax.axvline(tau, color=GRAY, lw=1, ls="--")
    ax.text(tau + 3, Ve[0].min() * 0.95, f"dispatch threshold s = {tau}", color=INK2, fontsize=9)
    ax.set(xlabel="customers waiting, s", ylabel="optimal value $V_0(s)$ (= −expected discounted cost)",
           title="(a) Enumeration, T = 500: optimal value at time 0")
    fig.savefig(f"{OUT}/fig_a_enumeration_V0.png"); plt.close(fig)

    # (b) value iteration V*
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(s, Vv, color=ORANGE, label="value iteration $V^*$")
    ax.plot(s, Ve[0], color=BLUE, lw=1, ls=(0, (4, 3)), label="enumeration $V_0$ (T=500)")
    ax.legend(loc="lower left")
    ax.text(0.98, 0.92, f"max |V_VI − V_enum| = {np.abs(Vv - Ve[0]).max():.1e}\n"
            f"{len(hist)} iterations, tol ε=1e-8", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=INK2)
    ax.set(xlabel="customers waiting, s", ylabel="optimal value $V^*(s)$",
           title="(b) Value iteration (T = ∞): optimal value function")
    fig.savefig(f"{OUT}/fig_b_value_iteration_V.png"); plt.close(fig)

    # (c) policy iteration policy
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    for ax, lim in zip(axes, (200, 30)):
        ax.step(s[:lim + 1], pp[:lim + 1], where="mid", color=AQUA)
        ax.scatter(s[:lim + 1], pp[:lim + 1], s=10 if lim > 50 else 22, color=AQUA, zorder=3)
        ax.set_yticks([0, 1], ["hold (0)", "dispatch (1)"])
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("customers waiting, s")
    axes[0].set_title(f"(c) Policy iteration: optimal policy ({len(H)} PI iterations)")
    axes[1].set_title(f"zoom s ≤ 30: dispatch iff s ≥ {tau}")
    axes[1].axvline(tau, color=GRAY, lw=1, ls="--")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_c_policy_iteration_policy.png"); plt.close(fig)

    # (d) convergence diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].semilogy(hist, color=ORANGE)
    axes[0].semilogy(hist[0] * m.gamma ** np.arange(len(hist)), color=GRAY, lw=1, ls="--", label="$0.95^k$ reference")
    axes[0].legend(); axes[0].set(xlabel="VI iteration k", ylabel="$\\|V_{k+1}-V_k\\|_\\infty$", title="VI contraction")
    changes = [int((H[i + 1][0] != H[i][0]).sum()) for i in range(len(H) - 1)]
    thr = [threshold_of(h[0])[0] for h in H]
    axes[1].bar(range(1, len(changes) + 1), changes, color=AQUA, width=0.6)
    for i, t in enumerate(thr[1:]):
        axes[1].text(i + 1, changes[i], f"τ={t}", ha="center", va="bottom", fontsize=8, color=INK2)
    axes[1].set(xlabel="PI iteration", ylabel="# states whose action changed", title="PI (from always-hold)")
    ts = np.arange(470, 500)
    th = [threshold_of(pe[t])[0] for t in ts]
    axes[2].plot(ts, [np.nan if x is None else x for x in th], "o-", color=BLUE, ms=4)
    never = [t for t, x in zip(ts, th) if x is None]
    if never:
        axes[2].axvspan(min(never) - 0.5, max(never) + 0.5, color=GRAY, alpha=0.18, lw=0)
        axes[2].text(min(never), 13.5, "never\ndispatch", fontsize=8, color=INK2)
    axes[2].set_xlim(469, 500)
    axes[2].set(xlabel="time t (T = 500)", ylabel="dispatch threshold at t",
                title="Enumeration: horizon effect near T")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_d_convergence.png"); plt.close(fig)

    summary = {"threshold": tau, "VI_iterations": len(hist), "PI_iterations": len(H),
               "V*": {int(k): round(float(Vp[k]), 3) for k in (0, 5, 10, 13, 15, 20, 50, 100, 150, 200)},
               "V_enum0_minus_V*_max": float(np.abs(Ve[0] - Vp).max()),
               "thresholds_last_steps": {int(t): threshold_of(pe[t])[0] for t in range(490, 500)}}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
