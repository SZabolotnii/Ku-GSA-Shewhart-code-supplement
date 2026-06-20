"""
exp_m2.py — M2 (флагман): GSA-CUSUM проти Шухарта, EWMA, Page-CUSUM, оракула
на гаусівському зсуві середнього, за СПІЛЬНОЇ ARL0=370.

Очікування (з теорії): для Гаусса LLR лінійний, тому Page-CUSUM = оракул, а
GSA(s≥2) не дає виграшу над Page (J(s) майже не зростає). Натомість УВЕСЬ клас
накопичувальних карт (Page/EWMA/GSA) кардинально випереджає Шухарта на малих
зсувах — це і є головне повідомлення відносно базової статті.

Артефакти: results/M2_arl_table.csv, results/M2_gsa_vs_shewhart_arl.png,
           results/M2_speedup.png, results/M2_calibration.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import distributions as D
import benchmark as B
import plotting as P

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def main():
    dist = D.Gaussian()
    deltas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    delta_design = 1.0

    gsa_specs = [
        ("gsa2", "GSA-CUSUM (poly s=2)", B.gaussian_gsa_builder(2), "gsa2"),
        ("gsa3", "GSA-CUSUM (poly s=3)", B.gaussian_gsa_builder(3), "gsa3"),
    ]

    print("M2: калібрування під ARL0=370 (Гаусс, δ_design=1.0)")
    df, calib = B.run_benchmark(
        dist, deltas, delta_design=delta_design, gsa_specs=gsa_specs,
        include=("shewhart", "ewma", "page", "oracle"),
        N_cal=40_000, N_eval=80_000, seed=2024)

    df.to_csv(os.path.join(OUT, "M2_arl_table.csv"), index=False)
    cal_df = pd.DataFrame([{"method": k, **v} for k, v in calib.items()])
    cal_df.to_csv(os.path.join(OUT, "M2_calibration.csv"), index=False)

    # таблиця ARL1
    piv = df.pivot(index="delta", columns="method", values="arl")
    order = ["shewhart", "ewma", "page", "gsa2", "gsa3", "oracle"]
    piv = piv[[c for c in order if c in piv.columns]]
    print("\nARL1(δ):")
    print(piv.round(2).to_string())

    # speedup vs Shewhart
    speed = piv.copy()
    for c in speed.columns:
        speed[c] = piv["shewhart"] / piv[c]
    print("\nПрискорення відносно Шухарта (ARL1_Shewhart / ARL1_метод):")
    print(speed.round(2).to_string())

    # ---- Фігура: ARL1(δ) усі методи ----
    fig, ax = P.newfig(7.6, 4.9)
    label_map = {m: calib[m]["label"] for m in calib}
    plotkey = {m: calib[m]["plot_key"] for m in calib}
    for m in order:
        if m not in piv.columns:
            continue
        sub = df[df.method == m].sort_values("delta")
        ax.errorbar(sub.delta, sub.arl, yerr=1.96 * sub.se,
                    color=P.PALETTE[plotkey[m]], marker=P.MARKERS.get(plotkey[m], "o"),
                    label=label_map[m], capsize=2)
    ax.axhline(370, color="gray", ls=":", lw=1.0)
    ax.text(2.55, 400, "ARL₀=370", color="gray", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel(r"Справжній зсув $\Delta$, $\sigma$")
    ax.set_ylabel("ARL₁ (тактів до виявлення)")
    ax.set_title("M2. Швидкість виявлення зсуву (Гаусс, спільна ARL₀=370)")
    ax.legend(ncol=2)
    fig.savefig(os.path.join(OUT, "M2_gsa_vs_shewhart_arl.png"))

    # ---- Фігура: прискорення відносно Шухарта ----
    fig, ax = P.newfig()
    for m in ["page", "gsa2", "ewma", "oracle"]:
        if m not in piv.columns:
            continue
        ax.plot(speed.index, speed[m], color=P.PALETTE[plotkey[m]],
                marker=P.MARKERS.get(plotkey[m], "o"), label=label_map[m])
    ax.axhline(1.0, color=P.PALETTE["shewhart"], ls="--", label="Шухарт (база)")
    ax.set_xlabel(r"Справжній зсув $\Delta$, $\sigma$")
    ax.set_ylabel("Прискорення виявлення, разів")
    ax.set_title("M2. У скільки разів накопичувальні карти швидші за Шухарта")
    ax.legend()
    fig.savefig(os.path.join(OUT, "M2_speedup.png"))

    print(f"\nАртефакти збережено у {os.path.abspath(OUT)}")
    return piv, speed, calib


if __name__ == "__main__":
    main()
