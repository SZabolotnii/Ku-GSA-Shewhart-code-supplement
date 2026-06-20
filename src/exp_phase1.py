"""
exp_phase1.py — Gap D: вплив оцінених (Phase-I) параметрів на ARL₀.

Усі ARL у рукописі припускають ВІДОМИЙ центр μ=0. На практиці центр оцінюється зі
скінченної Phase-I вибірки розміру m; за μ̂≠0 карта зміщена → реальний ARL₀ відхиляється
від номіналу 370 і має «practitioner-to-practitioner» розкид (Jensen2006, Goedhart2017).
Оскільки стаття пропонує PMM2-оцінку центру (Var=g₂·σ²/n, g₂<1 на асиметрії), природна
теза: PMM2 ЗВУЖУЄ розкид реального ARL₀ проти вибіркового середнього — тобто дає
надійніший контроль in-control. Тут це кількісно.

Метод (дворівневий MC, ефективно): один раз будуємо lookup реального ARL₀(offset) для
зміщеного центру через MC; потім R Phase-I вибірок → μ̂_mean і μ̂_PMM2 → мапимо через
lookup у реальний ARL₀. Порівнюємо розкид (SD) і частку «поганих» карт (ARL₀<300).

Артефакт: results/phase1.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import distributions as D
import detectors as det
import pmm

TARGET = 370.0
N_CAL = 30_000
N_ARL0 = 40_000
MAX_STEPS = 8000
R = 3000                 # Phase-I вибірок на кожне m
M_GRID = [25, 50, 100]   # розміри Phase-I
K = 0.5                  # Page reference


def main():
    import os
    dist = D.SkewNormal(4.0)
    big = dist.sample(np.random.default_rng(0), 4_000_000, delta=0.0)
    g3 = float(stats.skew(big))
    g4 = float(stats.kurtosis(big))   # надлишковий ексцес
    g2 = pmm.g2_coefficient(g3, g4)
    print(f"skew-normal: γ3={g3:.3f}  γ4={g4:.3f}  g2(PMM2)={g2:.3f}  "
          f"(теор. SD-зменшення √g2={np.sqrt(g2):.3f})")

    # ---- lookup: реальний ARL₀ як функція зміщення центру ----
    h, a0c = det.calibrate_page(dist, K, TARGET, N=N_CAL, max_steps=MAX_STEPS)
    step, ns = det.page_cusum_step(K, h)
    offsets = np.linspace(-0.6, 0.6, 49)
    arl0_grid = []
    for i, off in enumerate(offsets):
        # карта зміщена на μ̂=off → бачить процес із середнім −off
        def sampler(rng, size, _o=off):
            return dist.sample(rng, size, delta=0.0) - _o
        rl = det.simulate_run_lengths(sampler, step, ns, N_ARL0, MAX_STEPS,
                                      np.random.default_rng(5000 + i))
        arl0_grid.append(float(rl.mean()))
    arl0_grid = np.array(arl0_grid)
    print(f"Page калібровано: h={h:.3f}, ARL₀(центр)={a0c:.1f}; "
          f"lookup ARL₀(offset) від {arl0_grid.min():.0f} до {arl0_grid.max():.0f}")

    # ---- Phase-I: μ̂_mean vs μ̂_PMM2 → реальний ARL₀ ----
    rows = []
    for m in M_GRID:
        rng = np.random.default_rng(100 + m)
        mu_mean = np.empty(R)
        mu_pmm = np.empty(R)
        for r in range(R):
            x = dist.sample(rng, m, delta=0.0)
            mu_mean[r] = x.mean()
            mu_pmm[r] = pmm.pmm2_location(x, sigma=1.0, gamma3=g3, gamma4=g4)
        a0_mean = np.interp(mu_mean, offsets, arl0_grid)
        a0_pmm = np.interp(mu_pmm, offsets, arl0_grid)
        for tag, muhat, a0 in [("sample-mean", mu_mean, a0_mean), ("PMM2", mu_pmm, a0_pmm)]:
            rows.append(dict(
                m=m, center=tag,
                SD_muhat=round(float(muhat.std()), 4),
                meanARL0=round(float(a0.mean()), 1),
                SD_ARL0=round(float(a0.std()), 1),
                p_ARL0_lt300=round(float((a0 < 300).mean()), 3),
                p5_ARL0=round(float(np.percentile(a0, 5)), 1)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    # зведення виграшу PMM2
    print("\nВиграш PMM2 (звуження розкиду реального ARL₀):")
    for m in M_GRID:
        sm = df[(df.m == m) & (df.center == "sample-mean")].iloc[0]
        pm = df[(df.m == m) & (df.center == "PMM2")].iloc[0]
        red_sd = (1 - pm.SD_ARL0 / sm.SD_ARL0) * 100
        print(f"  m={m}: SD(ARL₀) {sm.SD_ARL0}→{pm.SD_ARL0} ({red_sd:+.1f}%); "
              f"P(ARL₀<300) {sm.p_ARL0_lt300}→{pm.p_ARL0_lt300}")

    path = os.path.join(os.path.dirname(__file__), "..", "results", "phase1.csv")
    df.to_csv(path, index=False)
    print(f"\nЗбережено: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
