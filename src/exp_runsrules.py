"""
exp_runsrules.py — Gap E: порівняння з sensitizing run rules (Western Electric).

Підручниковий ПЕРШИЙ засіб проти нечутливості Шухарта до малих зсувів — додаткові
правила (Western Electric / Nelson). Рецензент SPC обов'язково спитає, чому їх нема.
Тут додаємо повний набір WE-правил як детектор і калібруємо до СПІЛЬНОЇ ARL₀=370
(масштабуванням усіх зон на λ), щоб ARL₁ був порівнянний із GSA.

Правила Western Electric (двосторонні):
  R1: одна точка за межею 3σ;
  R2: 2 з 3 послідовних за 2σ з одного боку;
  R3: 4 з 5 послідовних за 1σ з одного боку;
  R4: 8 послідовних з одного боку центру.

Ключова теза: WE-правила прискорюють виявлення малих зсувів, АЛЕ ціною ARL₀ —
у нативному вигляді (λ=1) ARL₀≈92 на Гауссі (~4× більше хибних тривог за номінал 370).
Калібрування до 370 послаблює зони (λ>1) і з'їдає частину виграшу; GSA лишається кращим.

Артефакт: results/runsrules.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import distributions as D
import detectors as det
import gsa

TARGET = 370.0
N_CAL = 30_000
N_EVAL = 60_000
MAX_STEPS = 8000
DELTAS = [0.5, 0.75, 1.0, 1.5, 2.0]


def western_electric_step(scale: float, include_r4: bool = True):
    """Двосторонні WE-правила, зони масштабовано на `scale` (1σ→scale, 2σ→2·scale, 3σ→3·scale).

    УВАГА: R4 (8 поспіль з одного боку) НЕ залежить від масштабу зон, тож повний набір
    не можна калібрувати до довільного ARL₀ — R4 сам обмежує ARL₀ стелею (~253 на Гауссі).
    Для зіставлення за спільної ARL₀=370 використовуємо підмножину R1–R3 (include_r4=False),
    яка масштабовна. Повний набір показуємо на нативних межах (λ=1)."""
    L1, L2, L3 = scale * 1.0, scale * 2.0, scale * 3.0

    def step(state, z, t):
        buf = np.concatenate([state[:, 1:], z[:, None]], axis=1)  # зсув уліво + новий z (вікно 8)
        r1 = np.abs(z) > L3
        last3 = buf[:, -3:]
        r2 = ((last3 > L2).sum(1) >= 2) | ((last3 < -L2).sum(1) >= 2)
        last5 = buf[:, -5:]
        r3 = ((last5 > L1).sum(1) >= 4) | ((last5 < -L1).sum(1) >= 4)
        crossed = r1 | r2 | r3
        if include_r4:
            crossed = crossed | (buf > 0).all(1) | (buf < 0).all(1)
        return buf, crossed
    return step, 8


def calibrate_we(dist, N=N_CAL):
    """Бісекція по λ (масштаб зон) до ARL₀=370 для підмножини R1–R3 (масштабовна)."""
    s_h0 = det.make_sampler(dist, 0.0)
    return det.calibrate_threshold(s_h0, lambda lam: western_electric_step(lam, include_r4=False),
                                   8, target_arl0=TARGET, N=N, max_steps=MAX_STEPS,
                                   lo=0.9, hi=2.5, tol=0.01)


def arl_curve_step(dist, step, ns, deltas, seed=515):
    out = {}
    for d in deltas:
        rng = np.random.default_rng(seed + int(d * 1000))
        rl = det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL, MAX_STEPS, rng)
        out[d] = float(rl.mean())
    return out


def run_case(name, dist, gsa_basis, gsa_s):
    print(f"\n{'='*66}\n{name}\n{'='*66}")
    rows = []

    # Shewhart, калібрований до 370
    L, a0 = det.calibrate_shewhart(dist, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    step, ns = det.shewhart_step(L)
    arl = arl_curve_step(dist, step, ns, DELTAS)
    rows.append(dict(method="Shewhart", thr=round(L, 3), ARL0=round(a0, 1),
                     **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))

    # повний WE (R1–R4) на НАТИВНИХ межах λ=1 — реальна підручникова конфігурація;
    # ARL₀ значно нижче 370 (R4 не калібровна) → режим ВИЩИХ хибних тривог
    step, ns = western_electric_step(1.0, include_r4=True)
    a0_native = float(det.simulate_run_lengths(det.make_sampler(dist, 0.0), step, ns,
                      60_000, MAX_STEPS, np.random.default_rng(4242)).mean())
    arl = arl_curve_step(dist, step, ns, DELTAS)
    rows.append(dict(method="WE full native (λ=1)", thr=1.0, ARL0=round(a0_native, 1),
                     **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))
    print(f"  WE full native ARL₀={a0_native:.1f} (номінал 370 → {370/a0_native:.1f}× більше хибних тривог)")

    # WE R1–R3 (без R4), калібровано до спільної ARL₀=370 (matched-ARL₀ зіставлення)
    lam, a0 = calibrate_we(dist)
    step, ns = western_electric_step(lam, include_r4=False)
    arl = arl_curve_step(dist, step, ns, DELTAS)
    rows.append(dict(method="WE R1-R3 (calib. 370)", thr=round(lam, 3), ARL0=round(a0, 1),
                     **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))

    # GSA-адапт
    exps = [0.5, 1.0, 1.5] if gsa_basis == "frac" else None
    up = gsa.build_empirical(dist, +1.0, gsa_basis, gsa_s, exponents=exps,
                             n_cal=300_000, rng=np.random.default_rng(41))
    lo = gsa.build_empirical(dist, -1.0, gsa_basis, gsa_s, exponents=exps,
                             n_cal=300_000, rng=np.random.default_rng(42))
    h, a0 = det.calibrate_gsa(dist, up, lo, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    step, ns = det.generic_cusum_step(up.increment, lo.increment, h)
    arl = arl_curve_step(dist, step, ns, DELTAS)
    rows.append(dict(method=f"GSA {gsa_basis} s{gsa_s}", thr=round(h, 3), ARL0=round(a0, 1),
                     **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.insert(0, "case", name.split(" (")[0])
    df["WE_native_ARL0"] = round(a0_native, 1)
    return df


def main():
    import os
    cases = [
        ("Gaussian (subgroup mean n=1)", D.Gaussian(), "poly", 2),
        ("skew-normal (γ3=0.78)", D.SkewNormal(4.0), "poly", 3),
    ]
    alld = [run_case(*c) for c in cases]
    out = pd.concat(alld, ignore_index=True)
    path = os.path.join(os.path.dirname(__file__), "..", "results", "runsrules.csv")
    out.to_csv(path, index=False)
    print(f"\nЗбережено: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
