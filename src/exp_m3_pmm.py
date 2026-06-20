"""
exp_m3_pmm.py — M3-B: калібрування меж за негаусівськості (напрямок B).

Два сюжети:
  (1) ARL0-дефляція при наївних гаусівських межах ±3σ. Для важкохвостових/
      асиметричних даних справжня частота хибних тривог ≠ 0.0027, тож фактичне
      ARL0 << 370. Відновлення — межі за КВАНТИЛЯМИ фактичного розподілу (shape-
      aware), що повертає ARL0=370 (часто це АСИМЕТРИЧНІ та ширші межі).
  (2) PMM2-ефективність: оцінка центральної лінії (положення) у Фазі I має
      дисперсію g₂·σ²/n < σ²/n для асиметричних залишків → стабільніші межі.
      Коефіцієнт g₂ = 1 − γ₃²/(2+γ₄) кількісно визначає виграш.

Артефакти: results/M3B_calibration_table.csv, results/M3B_arl0_deflation.png,
           results/M3B_pmm_efficiency.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

import distributions as D
import pmm
import plotting as P

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

TARGET_P = 0.0027  # двостороння частота хибних тривог для ARL0≈370
N_MC = 20_000_000

CASES = [
    ("gaussian", D.Gaussian(), "N(0,1)"),
    ("laplace", D.Laplace(), "Лаплас"),
    ("student_t5", D.StudentT(5.0), "Стьюдент t₅"),
    ("skewnormal", D.SkewNormal(4.0), "skew-normal α=4"),
    ("tpn", D.TwoPieceNormal(2.0), "two-piece normal r=2"),
]


def main():
    rng = np.random.default_rng(2025)
    rows = []
    for key, dist, label in CASES:
        x = dist.sample(rng, N_MC)
        g3, g4 = stats.skew(x), stats.kurtosis(x)

        # (1) наївні гаусівські межі ±3σ -> фактична частота хибних тривог
        p_naive = np.mean(np.abs(x) > 3.0)
        arl0_naive = 1.0 / p_naive if p_naive > 0 else np.inf

        # shape-aware межі за квантилями (двостороння хвостова маса = TARGET_P)
        lcl = np.quantile(x, TARGET_P / 2)
        ucl = np.quantile(x, 1 - TARGET_P / 2)
        # перевірка відновлення ARL0
        p_recal = np.mean((x > ucl) | (x < lcl))
        arl0_recal = 1.0 / p_recal if p_recal > 0 else np.inf
        # симетричний еквівалент L*
        L_sym = np.quantile(np.abs(x), 1 - TARGET_P)

        # (2) PMM2 коефіцієнт ефективності
        g2 = pmm.g2_coefficient(g3, g4)

        rows.append(dict(
            dist=key, label=label, gamma3=g3, gamma4=g4,
            p_naive=p_naive, ARL0_naive=arl0_naive,
            LCL_recal=lcl, UCL_recal=ucl, L_symmetric=L_sym, ARL0_recal=arl0_recal,
            g2_PMM2=g2, var_reduction_pct=100 * (1 - g2)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "M3B_calibration_table.csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("M3-B. Калібрування меж за негаусівськості:\n")
    print(df[["label", "gamma3", "gamma4", "p_naive", "ARL0_naive",
              "LCL_recal", "UCL_recal", "ARL0_recal", "g2_PMM2", "var_reduction_pct"]].to_string(index=False))

    # ---- MC-перевірка PMM2 на скінченних вибірках (Фаза I, n=50) ----
    print("\nPMM2 vs вибіркове середнє (n=50, 20000 повторів):")
    pmm_rows = []
    rng2 = np.random.default_rng(11)
    for key, dist, label in CASES:
        info = pmm.theoretical_g_for(dist)
        n, reps = 50, 20000
        em = np.empty(reps); ep = np.empty(reps)
        for r in range(reps):
            xx = dist.sample(rng2, n)
            em[r] = xx.mean()
            ep[r] = pmm.pmm2_location(xx, sigma=1.0, gamma3=info["gamma3"], gamma4=info["gamma4"])
        ratio = ep.var() / em.var()
        pmm_rows.append(dict(dist=key, label=label, g2_theory=info["g2"], ratio_mc=ratio))
        print(f"  {label:22s} g₂(теор)={info['g2']:.4f}  Var(PMM2)/Var(mean)={ratio:.4f}")
    pmm_df = pd.DataFrame(pmm_rows)

    # ---- Фігура 1: ARL0-дефляція ----
    fig, ax = P.newfig(7.6, 4.6)
    labels = df.label.tolist()
    xpos = np.arange(len(labels))
    bars = ax.bar(xpos, df.ARL0_naive, color=[P.PALETTE["shewhart"] if a < 300 else "#4c8c4a"
                                              for a in df.ARL0_naive])
    ax.axhline(370, color="gray", ls="--", lw=1.3, label="Ціль ARL₀=370")
    for i, a in enumerate(df.ARL0_naive):
        ax.text(i, a + 8, f"{a:.0f}", ha="center", fontsize=9)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Фактичне ARL₀ при наївних межах ±3σ")
    ax.set_title("M3-B. Руйнування ARL₀ гаусівськими межами за негаусівськості")
    ax.legend()
    fig.savefig(os.path.join(OUT, "M3B_arl0_deflation.png"))

    # ---- Фігура 2: PMM2 ефективність ----
    fig, ax = P.newfig(7.2, 4.4)
    xpos = np.arange(len(pmm_df))
    ax.bar(xpos - 0.2, pmm_df.g2_theory, 0.4, color=P.PALETTE["gsa2"], label="g₂ (теорія)")
    ax.bar(xpos + 0.2, pmm_df.ratio_mc, 0.4, color=P.PALETTE["mc"], label="Var(PMM2)/Var(серед.), MC")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_xticks(xpos); ax.set_xticklabels(pmm_df.label, rotation=18, ha="right")
    ax.set_ylabel("Відносна дисперсія оцінки положення")
    ax.set_ylim(0, 1.15)
    ax.set_title("M3-B. PMM2: зменшення дисперсії оцінки центральної лінії")
    ax.legend()
    fig.savefig(os.path.join(OUT, "M3B_pmm_efficiency.png"))

    pmm_df.to_csv(os.path.join(OUT, "M3B_pmm_efficiency.csv"), index=False)
    print(f"\nАртефакти збережено у {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
