"""
exp_m1.py — M1: відтворення базової лінії карти Шухарта.

Відтворює аналітичний вираз імовірності сигналу P(Δ) (формули 10-11 базової статті)
та підтверджує його Монте-Карло. Контрольний результат: P(1σ)=0.02, P(2σ)=0.16,
P(3σ)=0.50. Будує ARL1_Shewhart(δ)=1/P(δ) — базову лінію для подальших порівнянь.

Артефакти: results/M1_shewhart_table.csv, results/M1_pshift.png, results/M1_arl.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

import distributions as D
import detectors as det
import plotting as P

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42
L = 3.0
N_MC = 200_000


def p_signal_analytic(delta, L=3.0):
    """Імовірність сигналу за один такт (двостороння карта ±L), формули 10-11."""
    return 1.0 - (stats.norm.cdf(L - delta) - stats.norm.cdf(-L - delta))


def main():
    g = D.Gaussian()
    deltas = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])

    rows = []
    for d in deltas:
        p_an = p_signal_analytic(d, L)
        # MC: частка тактів, що дають сигнал (одне спостереження = один такт)
        rng = np.random.default_rng(SEED + int(d * 100))
        z = g.sample(rng, N_MC, delta=float(d))
        p_mc = np.mean(np.abs(z) > L)
        se = np.sqrt(p_mc * (1 - p_mc) / N_MC)
        arl1_an = 1.0 / p_an
        rows.append(dict(delta=d, P_analytic=p_an, P_mc=p_mc, P_mc_se=se,
                         ARL1_analytic=arl1_an, ARL1_mc=(1.0 / p_mc if p_mc > 0 else np.inf)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "M1_shewhart_table.csv"), index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(df.to_string(index=False))

    # перевірка контрольних точок
    print("\nКонтроль базової статті:")
    for d, target in [(1.0, 0.02), (2.0, 0.16), (3.0, 0.50)]:
        pa = p_signal_analytic(d, L)
        ok = abs(pa - target) <= 0.01
        print(f"  P({d:.0f}σ)={pa:.4f}  ціль≈{target}  {'OK' if ok else 'MISS'}")

    # ---- Фігура 1: P(Δ) аналітика vs MC (відтворення Рис.3/Рис.6) ----
    fig, ax = P.newfig()
    dd = np.linspace(0, 3.2, 200)
    ax.plot(dd, [p_signal_analytic(x, L) for x in dd], color=P.PALETTE["analytic"],
            label="Аналітика, ф-ли (10)-(11)")
    ax.errorbar(df.delta, df.P_mc, yerr=1.96 * df.P_mc_se, fmt="o",
                color=P.PALETTE["mc"], label="Монте-Карло (95% ДІ)", capsize=3, zorder=5)
    for d, target in [(1, 0.02), (2, 0.16), (3, 0.50)]:
        ax.annotate(f"P({d}σ)={target}", (d, target), textcoords="offset points",
                    xytext=(6, 10), fontsize=9)
        ax.scatter([d], [target], color="black", zorder=6, s=18)
    ax.set_xlabel(r"Зсув середнього $\Delta$, в одиницях $\sigma_{\bar X}$")
    ax.set_ylabel(r"Імовірність сигналу за такт, $P(\Delta)$")
    ax.set_title("M1. Карта Шухарта: імовірність сигналу при зсуві середнього")
    ax.legend()
    fig.savefig(os.path.join(OUT, "M1_pshift.png"))

    # ---- Фігура 2: ARL1(Δ) Шухарта ----
    fig, ax = P.newfig()
    dd = np.linspace(0.25, 3.2, 200)
    ax.plot(dd, [1.0 / p_signal_analytic(x, L) for x in dd],
            color=P.PALETTE["shewhart"], label="Шухарт, ARL₁=1/P(Δ)")
    ax.scatter(df.delta[df.delta > 0], df.ARL1_mc[df.delta > 0],
               color=P.PALETTE["mc"], zorder=5, label="MC")
    ax.axhline(370, color="gray", ls=":", lw=1.2, label="ARL₀=370")
    ax.set_yscale("log")
    ax.set_xlabel(r"Зсув середнього $\Delta$, $\sigma_{\bar X}$")
    ax.set_ylabel("ARL₁ (середня кількість тактів до сигналу)")
    ax.set_title("M1. Швидкість виявлення зсуву картою Шухарта")
    ax.legend()
    fig.savefig(os.path.join(OUT, "M1_arl.png"))

    print(f"\nАртефакти збережено у {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
