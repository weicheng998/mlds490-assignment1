"""Validation experiments for the shuttle MDP. Each test states a prediction made BEFORE
looking at the solver output and checks it numerically. Run:  python -m mdp.validate
Writes results/mdp/validation.json and results/mdp/validation.txt.
"""
import json
import os
from functools import lru_cache

import numpy as np

from mdp.shuttle import ShuttleMDP, simulate, threshold_of

OUT = "results/mdp"
LOG = []
RESULTS = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def check(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), **detail}
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    os.makedirs(OUT, exist_ok=True)
    m = ShuttleMDP()
    T = 500
    Ve, pe = m.enumerate(T)
    Vv, pv, hist = m.value_iteration()
    Vp, pp, H = m.policy_iteration()
    tau, pure = threshold_of(pp)
    Vmax = np.abs(Vp).max()

    # 1. Model edge cases: stochastic rows, cap, dispatch at s=0, s>K.
    rows_ok = np.allclose(m.P.sum(2), 1)
    cap = m.P[0, 200, 200] == 1.0 and abs(m.P[0, 198, 200] - 0.8) < 1e-12
    check("T1 transition model edge cases", rows_ok and cap and m.R[1, 0] == -100 and m.R[1, 20] == -110,
          {"rows_sum_to_1": bool(rows_ok), "P_hold(200->200)": m.P[0, 200, 200],
           "P_hold(198->200) (pred 0.8: A in {2,3,4,5})": m.P[0, 198, 200],
           "R_dispatch(0) (pred -100)": m.R[1, 0], "R_dispatch(20) (pred -100-2*5=-110)": m.R[1, 20]})

    # 2. Three solvers agree; enumeration error obeys |V0^T - V*| <= gamma^T ||V*||.
    bound = m.gamma ** T * Vmax
    e_enum, e_vi = np.abs(Ve[0] - Vp).max(), np.abs(Vv - Vp).max()
    check("T2 enumeration / VI / PI agree", e_enum <= bound + 1e-9 and e_vi < 1e-7
          and (pe[0] == pp).all() and (pv == pp).all(),
          {"max|V_enum0 - V_PI|": float(e_enum), "theory bound gamma^T*||V*||": float(bound),
           "max|V_VI - V_PI|": float(e_vi), "policies identical": bool((pe[0] == pp).all() and (pv == pp).all()),
           "threshold": tau, "pure threshold": pure, "VI iterations": len(hist), "PI iterations": len(H)})

    # 3. Bellman optimality residual of the PI solution and VI contraction rate.
    resid = np.abs(m.q_values(Vp).max(0) - Vp).max()
    ratios = hist[1:50] / hist[:49]
    check("T3 Bellman residual and gamma-contraction", resid < 1e-8 and ratios.max() <= m.gamma + 1e-9,
          {"||TV - V||_inf": float(resid), "max VI residual ratio (pred <= 0.95)": float(ratios.max()),
           "VI residual ratio late (pred ~0.95)": float(hist[400] / hist[399])})

    # 4. Independent Monte Carlo simulator (no matrices) reproduces V*(s).
    rng = np.random.default_rng(0)
    mc = {}
    ok = True
    for s0 in (0, 13, 50, 120, 200):
        g = simulate(m.K, m.cf, m.ch, m.arrivals, m.smax, m.gamma, lambda s: int(pp[s]), s0, 600, 4000, rng)
        se = g.std(ddof=1) / np.sqrt(len(g))
        z = (g.mean() - Vp[s0]) / se
        ok &= abs(z) < 3.5
        mc[f"s0={s0}"] = {"MC mean": round(g.mean(), 2), "SE": round(se, 2), "V*": round(Vp[s0], 2), "z": round(z, 2)}
    check("T4 Monte Carlo simulation matches V* (|z|<3.5)", ok, mc)

    # 5. Brute force over all threshold policies (exact evaluation): best one == PI policy,
    #    and no threshold policy beats V* in any state.
    best_tau, best_v0 = None, -np.inf
    any_better = False
    for t in range(0, m.S + 1):
        pi_t = (np.arange(m.S) >= t).astype(int)
        Vt = m.evaluate(pi_t)
        any_better |= (Vt > Vp + 1e-7).any()
        if Vt[0] > best_v0 + 1e-12:
            best_tau, best_v0 = t, Vt[0]
    v_never = m.evaluate(np.zeros(m.S, dtype=int))
    v_always = m.evaluate(np.ones(m.S, dtype=int))
    check("T5 brute force over 202 threshold policies", best_tau == tau and not any_better,
          {"best threshold": best_tau, "PI threshold": tau, "some threshold beats V* somewhere": bool(any_better),
           "V_never-dispatch(0)": round(v_never[0], 1), "V_always-dispatch(0)": round(v_always[0], 1),
           "V*(0)": round(Vp[0], 1)})

    # 6. Analytic toy cases.
    m0 = ShuttleMDP(ch=0.0)
    V0, p0, _ = m0.policy_iteration()
    m1 = ShuttleMDP(cf=0.0)
    V1, p1, _ = m1.policy_iteration()
    # Deterministic arrivals a=5, huge shuttle: closed-form value of the cycle.
    a, cf, ch, g = 5, 100.0, 2.0, 0.95
    md = ShuttleMDP(K=200, arrivals={a: 1.0})
    Vd, pd, _ = md.policy_iteration()
    def cycle_value(mm):  # value at s=a of "dispatch when s reaches mm*a"
        hold = sum(g ** (k - 1) * ch * k * a for k in range(1, mm))
        return -(hold + g ** (mm - 1) * cf) / (1 - g ** mm)
    vals = {mm: cycle_value(mm) for mm in range(1, 40)}
    m_star = max(vals, key=vals.get)
    tau_d, _ = threshold_of(pd)
    check("T6 analytic toy cases",
          np.abs(V0).max() < 1e-9 and not p0.any() and np.abs(V1[:16]).max() < 1e-9 and p1[1:].all()
          and abs(Vd[a] - vals[m_star]) < 1e-6 and (m_star - 1) * a < tau_d <= m_star * a,
          {"ch=0: max|V*| (pred 0)": float(np.abs(V0).max()), "ch=0: any dispatch (pred False)": bool(p0.any()),
           "cf=0: max|V*(s<=15)| (pred 0)": float(np.abs(V1[:16]).max()),
           "cf=0: dispatch for all s>=1 (pred True)": bool(p1[1:].all()),
           "deterministic A=5,K=200: closed-form best cycle m*": m_star,
           "closed-form V(5)": round(vals[m_star], 6), "solver V(5)": round(Vd[a], 6),
           "solver threshold (pred in ((m*-1)a, m*a]; only multiples of a reachable)": tau_d})

    # 7. Comparative statics (predicted directions).
    ths = {}
    INF = 10**6  # "never dispatch"
    thr = lambda mdp: (lambda t: INF if t is None else t)(threshold_of(mdp.policy_iteration()[1])[0])
    mono = lambda x, inc: all((b >= a) if inc else (b <= a) for a, b in zip(x, x[1:]))
    for cfv in (25, 50, 100, 200, 400):
        ths[f"cf={cfv}"] = thr(ShuttleMDP(cf=cfv))
    for chv in (0.5, 1, 2, 4, 8):
        ths[f"ch={chv}"] = thr(ShuttleMDP(ch=chv))
    gammas = (0.5, 0.6, 0.69, 0.7, 0.72, 0.75, 0.8, 0.9, 0.95, 0.99)
    for gv in gammas:
        ths[f"gamma={gv}"] = thr(ShuttleMDP(gamma=gv))
    cf_t = [ths[f"cf={c}"] for c in (25, 50, 100, 200, 400)]
    ch_t = [ths[f"ch={c}"] for c in (0.5, 1, 2, 4, 8)]
    g_t = [ths[f"gamma={c}"] for c in gammas]
    # Analytic bound: dispatching saves at most ch*K/(1-gamma); if that is < cf, never dispatch.
    never_pred = [gv for gv in gammas if m.ch * m.K / (1 - gv) < m.cf]
    never_ok = all(ths[f"gamma={gv}"] == INF for gv in never_pred)
    ths["gammas where ch*K/(1-gamma) < cf (pred never dispatch)"] = never_pred
    check("T7 comparative statics: threshold up in cf, down in ch, down in gamma (10^6 = never)",
          mono(cf_t, True) and mono(ch_t, False) and mono(g_t, False) and never_ok, ths)

    # 8. Finite-horizon edge cases: last step never dispatches (saving <= ch*K = 30 < cf = 100),
    #    an independent memo-free expectimax recursion matches backward induction for T=6,
    #    and far from the end the time-0 policy equals the stationary one.
    def expectimax(s, steps):
        if steps == 0:
            return 0.0
        best = -np.inf
        for act in (0, 1):
            left = s - min(s, m.K) if act else s
            c = (m.cf if act else 0) + m.ch * left
            ev = sum(p * expectimax(min(left + A, m.smax), steps - 1) for A, p in m.arrivals.items())
            best = max(best, -c + m.gamma * ev)
        return best
    Ve6, _ = m.enumerate(6)
    em = {s: (round(expectimax(s, 6), 6), round(Ve6[0, s], 6)) for s in (0, 7, 13, 30)}
    first_diff = int(min(t for t in range(T) if (pe[t] != pp).any()))
    check("T8 finite-horizon checks",
          not pe[T - 1].any() and all(abs(a_ - b_) < 1e-9 for a_, b_ in em.values()),
          {"dispatch anywhere at t=T-1 (pred False)": bool(pe[T - 1].any()),
           "expectimax vs enumeration, T=6, (indep, enum)": em,
           "first t where policy differs from stationary": first_diff,
           "thresholds at t=T-5..T-1": [threshold_of(pe[t])[0] for t in range(T - 5, T)]})

    # 9. Cap sensitivity: raising smax to 400 should not change the policy or V on low states.
    mbig = ShuttleMDP(smax=400)
    Vb, pb, _ = mbig.policy_iteration()
    dv = np.abs(Vb[:101] - Vp[:101]).max()
    check("T9 cap 200 vs 400", (pb[:201] == pp).all() and dv < 1e-6,
          {"policy identical on 0..200": bool((pb[:201] == pp).all()), "max|dV| on s<=100": float(dv),
           "V(200) cap200 vs cap400": (round(Vp[200], 2), round(Vb[200], 2))})

    # 10. Trace the first 10 steps under the optimal policy from s=0 (and from s=40).
    rng = np.random.default_rng(1)
    log("\nFirst 10 steps under the optimal policy (s0=0 and s0=40):")
    trace = []
    for s0 in (0, 40):
        s = s0
        for t in range(10):
            a = int(pp[s])
            left = s - min(s, m.K) if a else s
            c = (m.cf if a else 0) + m.ch * left
            A = int(rng.integers(1, 6))
            row = {"t": t, "s": s, "V*(s)": round(Vp[s], 1), "action": "dispatch" if a else "hold",
                   "cost": c, "arrivals": A, "s_next": min(left + A, m.smax)}
            trace.append(row)
            log("  ", row)
            s = row["s_next"]
    RESULTS["trace"] = trace
    RESULTS["summary"] = {"threshold": tau, "V*(0)": Vp[0], "V*(200)": Vp[200]}

    with open(f"{OUT}/validation.json", "w") as f:
        json.dump(RESULTS, f, indent=1, default=float)
    with open(f"{OUT}/validation.txt", "w") as f:
        f.write("\n".join(LOG) + "\n")
    n_fail = sum(1 for k, v in RESULTS.items() if isinstance(v, dict) and v.get("pass") is False)
    log(f"\n{n_fail} failing checks")


if __name__ == "__main__":
    main()
