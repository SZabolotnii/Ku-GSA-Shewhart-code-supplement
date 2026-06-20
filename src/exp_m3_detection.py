"""
exp_m3_detection.py — M3-A: виграш Кунченка на негаусівських даних.

Для негаусівських розподілів точний LLR зсуву нелінійний: для важких хвостів
(Стьюдент) він «перевизначається» (redescending) — гасить викиди; для асиметрії
(skew-normal, TPN) — несиметричний. Лінійний Page-CUSUM цього не враховує. GSA з
адаптованим базисом (poly s=2 для асиметрії, frac sign|z|^p для важких хвостів)
наближає оптимальний оракул і випереджає Page за СПІЛЬНОЇ ARL0=370.

Моніторяться ОКРЕМІ спостереження (n=1, карта індивідуальних значень) — саме там
форма розподілу повністю проявляється (усереднення n=5 її б згладило за ЦГТ).

Артефакти: results/M3A_<dist>_arl.png, results/M3A_detection_table.csv,
           results/M3A_summary.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import distributions as D
import benchmark as B
import plotting as P
import robust_cusum as RC

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

CASES = {
    "skewnormal": dict(
        dist=D.SkewNormal(4.0),
        title="Skew-normal (α=4, асиметрія)",
        gsa=[("gsa2", "GSA-CUSUM (poly s=2)", B.empirical_gsa_builder("poly", 2), "gsa2"),
             ("gsa3", "GSA-CUSUM (poly s=3)", B.empirical_gsa_builder("poly", 3), "gsa3")],
    ),
    "student_t5": dict(
        dist=D.StudentT(5.0),
        title="Стьюдент t₅ (важкі хвости)",
        gsa=[("gsa2", "GSA-CUSUM (poly s=2)", B.empirical_gsa_builder("poly", 2), "gsa2"),
             ("gsafrac", "GSA-CUSUM (frac |z|^{0.5,1,1.5})",
              B.empirical_gsa_builder("frac", 3, exponents=[0.5, 1.0, 1.5], winsor=0.0), "gsa_frac")],
    ),
    "tpn": dict(
        dist=D.TwoPieceNormal(2.0),
        title="Two-piece normal (r=2, асиметрія+платикуртоз)",
        gsa=[("gsa2", "GSA-CUSUM (poly s=2)", B.empirical_gsa_builder("poly", 2), "gsa2"),
             ("gsa3", "GSA-CUSUM (poly s=3)", B.empirical_gsa_builder("poly", 3), "gsa3")],
    ),
}

DELTAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
DELTA_DESIGN = 1.0

# Сучасні робастні/непараметричні CUSUM-конкуренти (розрив A): спільні для всіх
# розподілів. Той самий формат (key,label,builder,plot_key), що й GSA-специфікації.
ROBUST_SPECS = [
    ("winsor", "Winsorized CUSUM (c=1.5)", RC.robust_builder(RC.huber_score(1.5)), "winsor"),
    ("sign", "Sign CUSUM", RC.robust_builder(RC.sign_score()), "sign"),
]


def main():
    all_rows = []
    summary = []
    for key, cfg in CASES.items():
        print(f"\n=== {key}: {cfg['title']} ===")
        df, calib = B.run_benchmark(
            cfg["dist"], DELTAS, delta_design=DELTA_DESIGN,
            gsa_specs=cfg["gsa"] + ROBUST_SPECS,
            include=("shewhart", "ewma", "page", "oracle"),
            N_cal=40_000, N_eval=80_000, seed=3030)
        df["dist"] = key
        all_rows.append(df)

        piv = df.pivot(index="delta", columns="method", values="arl")
        gsa_keys = [g[0] for g in cfg["gsa"]]
        robust_keys = [r[0] for r in ROBUST_SPECS]
        order = ["shewhart", "ewma", "page"] + robust_keys + gsa_keys + ["oracle"]
        order = [c for c in order if c in piv.columns]
        print(piv[order].round(2).to_string())

        # summary @ δ=1: наскільки GSA ближче до оракула, ніж Page і РОБАСТНІ конкуренти
        if 1.0 in piv.index:
            row = piv.loc[1.0]
            best_gsa = min(gsa_keys, key=lambda g: row[g])
            best_robust = min(robust_keys, key=lambda r: row[r])
            summary.append(dict(
                dist=key, title=cfg["title"],
                ARL1_shewhart=row.get("shewhart"), ARL1_ewma=row.get("ewma"),
                ARL1_page=row["page"], ARL1_winsor=row.get("winsor"),
                ARL1_sign=row.get("sign"), ARL1_oracle=row["oracle"],
                best_gsa=best_gsa, ARL1_best_gsa=row[best_gsa],
                best_robust=best_robust, ARL1_best_robust=row[best_robust],
                page_over_oracle=row["page"] / row["oracle"],
                gsa_over_oracle=row[best_gsa] / row["oracle"],
                gsa_improve_vs_page=row["page"] / row[best_gsa],
                gsa_improve_vs_best_robust=row[best_robust] / row[best_gsa]))

        # фігура для цього розподілу
        fig, ax = P.newfig(7.4, 4.8)
        for m in order:
            sub = df[df.method == m].sort_values("delta")
            pk = calib[m]["plot_key"]
            ax.errorbar(sub.delta, sub.arl, yerr=1.96 * sub.se, color=P.PALETTE[pk],
                        marker=P.MARKERS.get(pk, "o"), label=calib[m]["label"], capsize=2)
        ax.set_yscale("log")
        ax.axhline(370, color="gray", ls=":", lw=1.0)
        ax.set_xlabel(r"Справжній зсув $\Delta$, $\sigma$")
        ax.set_ylabel("ARL₁ (тактів до виявлення)")
        ax.set_title(f"M3. {cfg['title']} — спільна ARL₀=370")
        ax.legend(ncol=2)
        fig.savefig(os.path.join(OUT, f"M3A_{key}_arl.png"))

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(os.path.join(OUT, "M3A_detection_table.csv"), index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUT, "M3A_summary.csv"), index=False)
    print("\n=== ПІДСУМОК @ δ=1σ ===")
    pd.set_option("display.width", 200)
    print(sdf.round(3).to_string(index=False))
    print(f"\nАртефакти збережено у {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
