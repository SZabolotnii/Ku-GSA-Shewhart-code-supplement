"""
exp_steadystate.py — Gap C: steady-state ARL₁ (а не лише zero-state).

Усі ARL₁ у рукописі — zero-state (зсув присутній з t=1, детектор «свіжий»). Рецензент
QREI/JQT майже завжди просить STEADY-STATE: зсув настає ПІСЛЯ того, як детектор довго
працював у контролі й статистика досягла стаціонарного розподілу. Zero-state ЗАВИЩУЄ
перевагу кумулятивних карт (свіжий CUSUM=0 близький до порога знизу). Тут міряємо обидва.

Steady-state оцінка (циклічна): burn-in під H₀ з renewal (скидання стану на хибну
тривогу) → стаціонарний conditional-on-no-alarm розподіл статистики; далі вмикаємо
зсув і міряємо затримку. EWMA отримує асимптотичну межу (глобальний час великий).

Очікування: SS-ARL₁ > ZS-ARL₁ для кумулятивних карт (CUSUM/EWMA/GSA), Shewhart
без пам'яті → SS=ZS; перевага GSA над Shewhart/Page має ВЦІЛІТИ, хоч і меншою.

Артефакт: results/steadystate.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import distributions as D
import detectors as det
import gsa
import robust_cusum as RC

TARGET = 370.0
N_CAL = 30_000
N_EVAL = 60_000
MAX_STEPS = 8000
BURNIN = 100
DELTAS = [0.5, 1.0]


def ss_run_lengths(sampler_h0, sampler_h1, step_fn, n_state, N, burnin, max_steps, rng):
    """Steady-state: burn-in під H₀ з renewal, потім зсув; повертає затримки."""
    state = np.zeros((N, n_state)) if n_state > 0 else np.zeros((N, 0))
    for t in range(1, burnin + 1):
        z = sampler_h0(rng, N)
        state, crossed = step_fn(state, z, t)
        if n_state > 0 and crossed.any():
            state[crossed] = 0.0  # renewal на хибну тривогу
    active = np.arange(N)
    run_len = np.full(N, max_steps, dtype=np.int64)
    st = state
    for s in range(1, max_steps + 1):
        m = active.size
        if m == 0:
            break
        z = sampler_h1(rng, m)
        st, crossed = step_fn(st, z, burnin + s)  # глобальний час (для EWMA-межі)
        if crossed.any():
            run_len[active[crossed]] = s
            keep = ~crossed
            active = active[keep]
            st = st[keep]
    return run_len


def zs_arl1(dist, step, ns, d, seed=515):
    rng = np.random.default_rng(seed + int(d * 1000))
    rl = det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL, MAX_STEPS, rng)
    return float(rl.mean())


def ss_arl1(dist, step, ns, d, seed=616):
    rng = np.random.default_rng(seed + int(d * 1000))
    rl = ss_run_lengths(det.make_sampler(dist, 0.0), det.make_sampler(dist, d), step, ns,
                        N_EVAL, BURNIN, MAX_STEPS, rng)
    return float(rl.mean())


def detectors_for(dist, gsa_basis, gsa_s):
    out = {}
    L, _ = det.calibrate_shewhart(dist, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    out["Shewhart"] = det.shewhart_step(L)
    Le, _ = det.calibrate_ewma(dist, 0.2, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    out["EWMA"] = det.ewma_step(0.2, Le)
    hp, _ = det.calibrate_page(dist, 0.5, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    out["Page"] = det.page_cusum_step(0.5, hp)
    exps = [0.5, 1.0, 1.5] if gsa_basis == "frac" else None
    up = gsa.build_empirical(dist, +1.0, gsa_basis, gsa_s, exponents=exps,
                             n_cal=300_000, rng=np.random.default_rng(41))
    lo = gsa.build_empirical(dist, -1.0, gsa_basis, gsa_s, exponents=exps,
                             n_cal=300_000, rng=np.random.default_rng(42))
    hg, _ = det.calibrate_gsa(dist, up, lo, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    out[f"GSA {gsa_basis} s{gsa_s}"] = det.generic_cusum_step(up.increment, lo.increment, hg)
    wu, wl = RC.robust_builder(RC.huber_score(1.5), n_cal=300_000)(dist, 1.0)
    hw, _ = det.calibrate_gsa(dist, wu, wl, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    out["Winsor c=1.5"] = det.generic_cusum_step(wu.increment, wl.increment, hw)
    return out


def run_case(name, dist, gsa_basis, gsa_s):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    dets = detectors_for(dist, gsa_basis, gsa_s)
    rows = []
    for label, (step, ns) in dets.items():
        row = dict(method=label)
        for d in DELTAS:
            zs = zs_arl1(dist, step, ns, d)
            ss = ss_arl1(dist, step, ns, d)
            row[f"ZS@{d}"] = round(zs, 2)
            row[f"SS@{d}"] = round(ss, 2)
            row[f"SS/ZS@{d}"] = round(ss / zs, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    df.insert(0, "case", name.split(" (")[0])
    return df


def main():
    import os
    cases = [
        ("Gaussian", D.Gaussian(), "poly", 2),
        ("skew-normal (γ3=0.78)", D.SkewNormal(4.0), "poly", 3),
        ("Student t5", D.StudentT(5.0), "frac", 3),
    ]
    alld = [run_case(*c) for c in cases]
    out = pd.concat(alld, ignore_index=True)
    path = os.path.join(os.path.dirname(__file__), "..", "results", "steadystate.csv")
    out.to_csv(path, index=False)
    print(f"\nЗбережено: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
