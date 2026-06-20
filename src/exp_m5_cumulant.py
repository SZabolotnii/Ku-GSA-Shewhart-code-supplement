"""
exp_m5_cumulant.py — M5 (напрямок C): кумулянтний монітор форми розподілу.

Сценарій, у якому карта Шухарта СЛІПА за побудовою: середнє μ та дисперсія σ²
лишаються незмінними, але розподіл набуває АСИМЕТРІЇ (з'являється γ₃≠0). Так буває
при зміні режиму процесу, що зсуває «хвіст», не чіпаючи перших двох моментів.

Шухарт реагує лише на |z|>L, тож для незмінних μ,σ дає ARL≈370 незалежно від форми.
GSA-монітор з кубічним базисом φ(z)=z³ (E[z³]=γ₃) накопичує зсув третього моменту й
виявляє зміну форми. Це той самий апарат Кунченка, лише націлений на інший момент.

Артефакти: results/M5_shape_table.csv, results/M5_cumulant_chart.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import optimize, stats

import distributions as D
import detectors as det
import plotting as P

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def skewnorm_for_skewness(target_g1):
    """Знаходить α skew-normal із заданою асиметрією g1 (стандартизований)."""
    if target_g1 == 0:
        return 0.0
    def f(a):
        return D.SkewNormal(a).skewness - target_g1
    return optimize.brentq(f, 1e-4, 60)


def cubic_cusum_step(k, h):
    """Односторонній CUSUM на кубічному базисі: u = z³ − k (виявлення γ₃>0)."""
    def step(state, z, t):
        c = np.maximum(0.0, state[:, 0] + (z ** 3 - k))
        crossed = c > h
        return c[:, None], crossed
    return step, 1


def subgroup_mean_sampler(dist, n=5, delta=0.0):
    """Стандартизоване середнє підгрупи розміру n: Z̄=√n·mean (σ індивід.=1)."""
    def sampler(rng, size):
        x = dist.sample(rng, (size, n), delta=delta)
        return x.mean(axis=1) * np.sqrt(n)
    return sampler


def main():
    # цільові рівні асиметрії (μ,σ незмінні!)
    targets = [0.0, 0.15, 0.3, 0.45, 0.6, 0.78]
    alphas = [skewnorm_for_skewness(g) for g in targets]

    # калібрування під Гаусс (in-control): усі карти ARL0=370
    g_dist = D.Gaussian()
    NSUB = 5  # розмір підгрупи, як у базовій статті
    # Shewhart на ІНДИВІДУАЛЬНИХ значеннях
    L_ind, a0_ind = det.calibrate_shewhart(g_dist, 370.0, N=60_000)
    # Shewhart на СЕРЕДНІХ підгруп (n=5) — калібруємо межу L на стандартизованому Z̄
    sub_h0 = subgroup_mean_sampler(g_dist, NSUB, 0.0)
    L_sub, a0_sub = det.calibrate_threshold(sub_h0, lambda L: det.shewhart_step(L), 0,
                                            target_arl0=370.0, lo=1.5, hi=6.0, N=60_000)
    # cubic CUSUM на індивідуальних значеннях: k = γ3_design/2, γ3_design=0.4
    k = 0.4 / 2.0
    s_h0 = det.make_sampler(g_dist, 0.0)
    h, a0_cu = det.calibrate_threshold(s_h0, lambda h: cubic_cusum_step(k, h), 1,
                                       target_arl0=370.0, N=60_000, lo=0.5, hi=60.0)
    print(f"Калібрування під ARL0=370: Shewhart-індив L={L_ind:.3f} (ARL0={a0_ind:.0f}); "
          f"Shewhart-середні(n={NSUB}) L={L_sub:.3f} (ARL0={a0_sub:.0f}); "
          f"cubic-CUSUM k={k} h={h:.2f} (ARL0={a0_cu:.0f})\n")

    rows = []
    for g1, a in zip(targets, alphas):
        dist = D.Gaussian() if a == 0 else D.SkewNormal(a)
        # перевірка: μ≈0, σ≈1, тільки форма змінюється
        chk = dist.sample(np.random.default_rng(5), 2_000_000)
        sampler = det.make_sampler(dist, 0.0)
        sub_sampler = subgroup_mean_sampler(dist, NSUB, 0.0)
        rl_ind = det.simulate_run_lengths(sampler, *det.shewhart_step(L_ind)[:1], 0,
                                          60_000, 8000, np.random.default_rng(100))
        rl_sub = det.simulate_run_lengths(sub_sampler, *det.shewhart_step(L_sub)[:1], 0,
                                          60_000, 8000, np.random.default_rng(150))
        rl_cu = det.simulate_run_lengths(
            sampler, *cubic_cusum_step(k, h)[:1], 1, 60_000, 8000,
            np.random.default_rng(200))
        rows.append(dict(
            target_skew=g1, alpha=a,
            actual_mean=chk.mean(), actual_std=chk.std(), actual_skew=stats.skew(chk),
            ARL_shewhart_ind=rl_ind.mean(), ARL_shewhart_sub=rl_sub.mean(),
            ARL_cubic=rl_cu.mean()))
        print(f"γ₃≈{g1:.2f} (α={a:5.2f}): skew={stats.skew(chk):+.3f} | "
              f"Шухарт-індив ARL={rl_ind.mean():6.1f}  "
              f"Шухарт-середні ARL={rl_sub.mean():6.1f}  "
              f"кубіч.CUSUM ARL={rl_cu.mean():6.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "M5_shape_table.csv"), index=False)

    # фігура
    fig, ax = P.newfig(7.3, 4.6)
    ax.plot(df.actual_skew, df.ARL_shewhart_sub, "o-", color=P.PALETTE["shewhart"],
            label=f"Карта Шухарта, середні (n=5)")
    ax.plot(df.actual_skew, df.ARL_shewhart_ind, "s--", color=P.PALETTE["ewma"],
            label="Карта Шухарта, індивід.")
    ax.plot(df.actual_skew, df.ARL_cubic, "D-", color=P.PALETTE["gsa2"],
            label="GSA-монітор форми (φ=z³)")
    ax.axhline(370, color="gray", ls=":", lw=1.0, label="ARL₀=370")
    ax.set_yscale("log")
    ax.set_xlabel(r"Асиметрія, що з'являється $\gamma_3$ (за незмінних $\mu,\sigma$)")
    ax.set_ylabel("ARL до виявлення зміни форми")
    ax.set_title("M5. Зміна форми розподілу: Шухарт сліпий, кумулянтний монітор бачить")
    ax.legend()
    fig.savefig(os.path.join(OUT, "M5_cumulant_chart.png"))
    print(f"\nАртефакти збережено у {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
