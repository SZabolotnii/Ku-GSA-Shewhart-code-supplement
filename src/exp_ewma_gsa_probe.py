"""
exp_ewma_gsa_probe.py — ПРОТОТИП гібрида GSA-EWMA (скор Кунченка в EWMA-акумуляторі).

Ідея: ваш «GSA-алгоритм» = GSA-CUSUM (інкремент Λ(z)−r у CUSUM-рекурсії). EWMA — той
самий клас, але інший акумулятор на СИРОМУ лінійному z. λ-тюнінг (§5.2) показав, що
перевага EWMA — це memory-length ефект, ОРТОГОНАЛЬНИЙ до адаптації форми. Гібрид
композує обидва: подаємо стандартизований GSA-скор у EWMA-рекурсію

    W_t = λ·g(z) + (1−λ)·W_{t−1},   g(z) = (Λ(z) − E0[Λ]) / σ0[Λ],   |W_t| > L·σ_W(t)

Двосторонність — як у GSA-CUSUM: дві руки (up/lo) з окремих детекторів det_up/det_lo,
кожна одностороння (мірою своєї спрямованості зсуву). Скор стандартизований до Var0=1,
тож σ_W(t) ІДЕНТИЧНИЙ вашому EWMA → пряме, чесне порівняння «що дає заміна z→g».

Міряємо ОБИДВА режими (zero-state і steady-state, SS-протокол точно як exp_steadystate.py),
бо гібрид успадковує EWMA-акумулятор, а отже й її SS-ваду (Table 11). Порівнюємо проти:
  - чистого EWMA при тому ж λ  (чи б'є гібрид «6.97» на skew-normal?)
  - GSA-CUSUM (ваш поточний алгоритм)

Артефакт: results/ewma_gsa_probe.csv
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import distributions as D
import detectors as det
import gsa

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

TARGET = 370.0
N_CAL = 20_000
N_EVAL = 40_000
MAX_STEPS = 6000
BURNIN = 100
N_H0_STD = 500_000        # вибірка для оцінки E0[Λ], σ0[Λ]
N_CAL_GSA = 200_000       # вибірка для побудови GSA-детекторів
LAMBDAS = [0.05, 0.1, 0.2]
SEED_CAL = 12345

CASES = {
    "skewnormal": dict(dist=D.SkewNormal(4.0), title="Skew-normal (α=4)",
                       basis="poly", s=3, exps=None),
    "tpn": dict(dist=D.TwoPieceNormal(2.0), title="Two-piece normal (r=2)",
                basis="poly", s=2, exps=None),
    "student_t5": dict(dist=D.StudentT(5.0), title="Student t5 (heavy tails)",
                       basis="frac", s=3, exps=[0.5, 1.0, 1.5]),
}


# ---------- гібрид: стандартизована рука та крок ----------

def standardized_arm(detector, dist, seed):
    """g(z)=(Λ(z)−E0[Λ])/σ0[Λ], де E0,σ0 оцінені на свіжій H0-вибірці."""
    rng = np.random.default_rng(seed)
    x0 = dist.sample(rng, N_H0_STD, delta=0.0)
    L0 = detector.Lambda(x0)
    m0, s0 = float(L0.mean()), float(L0.std(ddof=1))
    return (lambda z: (detector.Lambda(z) - m0) / s0), m0, s0


def ewma_gsa_step(g_up, g_lo, lam, L):
    """Двосторонній GSA-EWMA: дві одно-сторонні руки на стандартизованих скорах."""
    ss = lam / (2.0 - lam)

    def step(state, z, t):
        Wu = lam * g_up(z) + (1.0 - lam) * state[:, 0]
        Wl = lam * g_lo(z) + (1.0 - lam) * state[:, 1]
        thr = L * np.sqrt(ss * (1.0 - (1.0 - lam) ** (2 * t)))
        crossed = (Wu > thr) | (Wl > thr)
        return np.column_stack([Wu, Wl]), crossed
    return step, 2


# ---------- steady-state (точно як exp_steadystate.ss_run_lengths) ----------

def ss_run_lengths(s_h0, s_h1, step, ns, N, burnin, max_steps, rng):
    state = np.zeros((N, ns)) if ns > 0 else np.zeros((N, 0))
    for t in range(1, burnin + 1):
        z = s_h0(rng, N)
        state, crossed = step(state, z, t)
        if ns > 0 and crossed.any():
            state[crossed] = 0.0
    active = np.arange(N)
    rl = np.full(N, max_steps, dtype=np.int64)
    st = state
    for s in range(1, max_steps + 1):
        m = active.size
        if m == 0:
            break
        z = s_h1(rng, m)
        st, crossed = step(st, z, burnin + s)
        if crossed.any():
            rl[active[crossed]] = s
            keep = ~crossed
            active = active[keep]
            st = st[keep]
    return rl


def zs_arl1(dist, step, ns, d, seed=515):
    rng = np.random.default_rng(seed + int(round(d * 1000)))
    rl = det.simulate_run_lengths(det.make_sampler(dist, float(d)), step, ns,
                                  N_EVAL, MAX_STEPS, rng)
    return float(rl.mean())


def ss_arl1(dist, step, ns, d, seed=616):
    rng = np.random.default_rng(seed + int(round(d * 1000)))
    rl = ss_run_lengths(det.make_sampler(dist, 0.0), det.make_sampler(dist, float(d)),
                        step, ns, N_EVAL, BURNIN, MAX_STEPS, rng)
    return float(rl.mean())


def evaluate(dist, step, ns):
    return dict(zs05=zs_arl1(dist, step, ns, 0.5), zs10=zs_arl1(dist, step, ns, 1.0),
                ss05=ss_arl1(dist, step, ns, 0.5), ss10=ss_arl1(dist, step, ns, 1.0))


def main():
    rows = []
    for key, cfg in CASES.items():
        dist = cfg["dist"]
        print(f"\n{'='*82}\n{cfg['title']}  [basis={cfg['basis']} s={cfg['s']}]\n{'='*82}", flush=True)

        # --- будуємо GSA-детектори (ті самі, що дають «best GSA» у M3A) ---
        up = gsa.build_empirical(dist, +1.0, cfg["basis"], cfg["s"], exponents=cfg["exps"],
                                 n_cal=N_CAL_GSA, rng=np.random.default_rng(41))
        lo = gsa.build_empirical(dist, -1.0, cfg["basis"], cfg["s"], exponents=cfg["exps"],
                                 n_cal=N_CAL_GSA, rng=np.random.default_rng(42))
        g_up, *_ = standardized_arm(up, dist, seed=7)
        g_lo, *_ = standardized_arm(lo, dist, seed=8)

        # --- референс 1: GSA-CUSUM (ваш поточний алгоритм; λ-незалежний) ---
        hg, a0g = det.calibrate_gsa(dist, up, lo, TARGET, N=N_CAL, max_steps=MAX_STEPS)
        step_g, ns_g = det.generic_cusum_step(up.increment, lo.increment, hg)
        eg = evaluate(dist, step_g, ns_g)
        rows.append(dict(dist=key, method="GSA-CUSUM", lam=np.nan, thr=round(hg, 3),
                         arl0=round(a0g, 1), J=round(up.J, 3), **eg))
        print(f"  GSA-CUSUM            h={hg:5.3f} ARL0={a0g:5.1f} J={up.J:.3f} | "
              f"ZS@1={eg['zs10']:6.2f}  SS@1={eg['ss10']:6.2f}  SS/ZS={eg['ss10']/eg['zs10']:.2f}", flush=True)

        # --- по сітці λ: чистий EWMA vs GSA-EWMA ---
        for lam in LAMBDAS:
            # чистий EWMA (сирий z) — референс 2; має відтворити вашу таблицю λ-тюнінгу
            Le, a0e = det.calibrate_ewma(dist, lam, TARGET, N=N_CAL, max_steps=MAX_STEPS)
            step_e, ns_e = det.ewma_step(lam, Le)
            ee = evaluate(dist, step_e, ns_e)
            rows.append(dict(dist=key, method="EWMA", lam=lam, thr=round(Le, 3),
                             arl0=round(a0e, 1), J=np.nan, **ee))

            # GSA-EWMA (гібрид)
            Lh, a0h = det.calibrate_threshold(
                det.make_sampler(dist, 0.0),
                lambda L, _gu=g_up, _gl=g_lo, _lm=lam: ewma_gsa_step(_gu, _gl, _lm, L),
                2, target_arl0=TARGET, N=N_CAL, max_steps=MAX_STEPS,
                lo=1.0, hi=5.0, tol=0.01, seed=SEED_CAL)
            step_h, ns_h = ewma_gsa_step(g_up, g_lo, lam, Lh)
            eh = evaluate(dist, step_h, ns_h)
            rows.append(dict(dist=key, method="GSA-EWMA", lam=lam, thr=round(Lh, 3),
                             arl0=round(a0h, 1), J=round(up.J, 3), **eh))

            d_zs = ee["zs10"] - eh["zs10"]
            print(f"  λ={lam:<4g}  EWMA   L={Le:5.3f} ARL0={a0e:5.1f} | ZS@.5={ee['zs05']:6.2f} "
                  f"ZS@1={ee['zs10']:6.2f}  SS@1={ee['ss10']:6.2f}  SS/ZS={ee['ss10']/ee['zs10']:.2f}", flush=True)
            print(f"  λ={lam:<4g}  GSA-EWMA L={Lh:5.3f} ARL0={a0h:5.1f} | ZS@.5={eh['zs05']:6.2f} "
                  f"ZS@1={eh['zs10']:6.2f}  SS@1={eh['ss10']:6.2f}  SS/ZS={eh['ss10']/eh['zs10']:.2f}"
                  f"   → ΔZS@1 vs EWMA = {d_zs:+.2f}", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "ewma_gsa_probe.csv")
    df.to_csv(path, index=False)

    # ---- вердикт ----
    print(f"\n{'='*82}\nВЕРДИКТ\n{'='*82}", flush=True)
    for key, cfg in CASES.items():
        sub = df[df.dist == key]
        ew = sub[sub.method == "EWMA"]
        he = sub[sub.method == "GSA-EWMA"]
        gc = sub[sub.method == "GSA-CUSUM"].iloc[0]
        # найкращий чистий EWMA (ZS@1) — аналог «6.97»
        be = ew.loc[ew.zs10.idxmin()]
        bh = he.loc[he.zs10.idxmin()]
        print(f"\n{cfg['title']}:", flush=True)
        print(f"  GSA-CUSUM (ваш):        ZS@1={gc.zs10:6.2f}   SS@1={gc.ss10:6.2f}", flush=True)
        print(f"  best EWMA  (λ={be.lam:<4g}):    ZS@1={be.zs10:6.2f}   SS@1={be.ss10:6.2f}", flush=True)
        print(f"  best GSA-EWMA (λ={bh.lam:<4g}): ZS@1={bh.zs10:6.2f}   SS@1={bh.ss10:6.2f}", flush=True)
        print(f"  → гібрид б'є чистий EWMA у ZS@1? {'ТАК' if bh.zs10 < be.zs10 else 'ні'}  "
              f"(Δ={be.zs10 - bh.zs10:+.2f})", flush=True)
        print(f"  → гібрид б'є GSA-CUSUM у ZS@1?   {'ТАК' if bh.zs10 < gc.zs10 else 'ні'}  "
              f"(Δ={gc.zs10 - bh.zs10:+.2f})", flush=True)
        print(f"  → SS-засторога: гібрид у SS@1 проти GSA-CUSUM: "
              f"{bh.ss10:6.2f} vs {gc.ss10:6.2f} ({'гірше' if bh.ss10 > gc.ss10 else 'краще'})", flush=True)

    print(f"\nЗбережено: {os.path.abspath(path)}", flush=True)


if __name__ == "__main__":
    main()
