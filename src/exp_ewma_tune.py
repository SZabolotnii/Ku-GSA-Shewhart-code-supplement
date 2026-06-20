"""
exp_ewma_tune.py — EWMA λ-tuning fairness check.

Зауваження рецензента: EWMA зафіксовано на λ=0.2 без поканального тюнінгу, що може
«гандикапити» карту. Тут перевіряємо емпірично: для кожного розподілу калібруємо
EWMA до СПІЛЬНОЇ ARL₀=370 при кожному λ зі сітки {0.05,0.1,0.2,0.3,0.5} і міряємо
ARL₁ при δ=0.5 і δ=1.0. Питання: чи найкраще-тюнінгована EWMA коли-небудь обганяє
GSA-адапт на АСИМЕТРИЧНИХ розподілах (skew-normal, two-piece normal)?

GSA-адапт числа НЕ перераховуємо — читаємо з results/M3A_detection_table.csv
(best-GSA per dist: skewnormal→gsa3, student_t5→gsafrac, tpn→gsa2). Gaussian у M3A
немає (там GSA≈Page≈oracle) → GSA для нього позначаємо NaN.

Артефакт: results/EWMA_tune_table.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import distributions as D
import benchmark as B

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

TARGET = 370.0
N_CAL = 20_000
N_EVAL = 40_000
MAX_STEPS = 6000
SEED = 4242
LAMBDAS = [0.05, 0.1, 0.2, 0.3, 0.5]
DELTAS = [0.5, 1.0]

# Розподіли — ті самі конструктори, що в exp_m3_detection.py.
CASES = {
    "gaussian": dict(dist=D.Gaussian(), title="Gaussian", best_gsa_key=None),
    "skewnormal": dict(dist=D.SkewNormal(4.0), title="Skew-normal (α=4)", best_gsa_key="gsa3"),
    "student_t5": dict(dist=D.StudentT(5.0), title="Student t5 (heavy tails)", best_gsa_key="gsafrac"),
    "tpn": dict(dist=D.TwoPieceNormal(2.0), title="Two-piece normal (r=2)", best_gsa_key="gsa2"),
}


def gsa_adapted_arl1():
    """Читає GSA-адапт ARL₁(δ=0.5,1.0) з M3A_detection_table.csv (без перерахунку)."""
    path = os.path.join(OUT, "M3A_detection_table.csv")
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    out = {}
    for key, cfg in CASES.items():
        gk = cfg["best_gsa_key"]
        if gk is None:
            continue
        sub = d[(d["dist"] == key) & (d["method"] == gk)]
        out[key] = {float(r.delta): float(r.arl) for r in sub.itertuples()}
    return out


def main():
    gsa = gsa_adapted_arl1()
    rows = []
    for key, cfg in CASES.items():
        dist = cfg["dist"]
        print(f"\n=== {cfg['title']} ===")
        for lam in LAMBDAS:
            # include=("ewma",) — лише EWMA, щоб тримати швидко; run_benchmark калібрує
            # EWMA до спільної ARL₀=370 при цьому λ і повертає ARL₁(δ).
            df, calib = B.run_benchmark(
                dist, DELTAS, include=("ewma",), ewma_lambda=lam,
                target_arl0=TARGET, N_cal=N_CAL, N_eval=N_EVAL,
                max_steps=MAX_STEPS, seed=SEED, verbose=False)
            arl0 = calib["ewma"]["arl0"]
            piv = df.set_index("delta")["arl"]
            rows.append(dict(dist=key, title=cfg["title"], lam=lam, arl0=round(arl0, 1),
                             arl1_d05=round(float(piv.loc[0.5]), 3),
                             arl1_d10=round(float(piv.loc[1.0]), 3)))
            print(f"  λ={lam:<4g} ARL0={arl0:6.1f}  ARL1@0.5={piv.loc[0.5]:7.3f}  "
                  f"ARL1@1.0={piv.loc[1.0]:7.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "EWMA_tune_table.csv"), index=False)

    # ---- підсумок: best-λ vs λ=0.2 vs GSA-адапт ----
    print(f"\n{'='*78}\nПІДСУМОК: best-tuned EWMA vs λ=0.2 vs GSA-адапт\n{'='*78}")
    for key, cfg in CASES.items():
        sub = table[table.dist == key]
        for d, col in [(1.0, "arl1_d10"), (0.5, "arl1_d05")]:
            best_i = sub[col].idxmin()
            best_lam = sub.loc[best_i, "lam"]
            best_arl = sub.loc[best_i, col]
            lam02 = float(sub[sub.lam == 0.2][col].iloc[0])
            g = gsa.get(key, {}).get(d, np.nan)
            beats = (not np.isnan(g)) and (best_arl < g)
            print(f"  {cfg['title']:24s} δ={d}: best λ={best_lam:<4g} ARL1={best_arl:7.3f} | "
                  f"λ=0.2 ARL1={lam02:7.3f} | GSA-adapt={g if not np.isnan(g) else float('nan'):7.3f}"
                  f"  → EWMA-best beats GSA? {'YES' if beats else 'no'}")

    print(f"\nЗбережено: {os.path.abspath(os.path.join(OUT, 'EWMA_tune_table.csv'))}")


if __name__ == "__main__":
    main()
