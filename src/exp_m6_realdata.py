"""
exp_m6_realdata.py — M6-real: валідація GSA-Shewhart на ДВОХ реальних рядах.

Закриває обмеження (iii) §6.5: «усі дані синтетичні… валідації на реальних
метрологічних рядах не виконано». Тут — well-log (важкі хвости) та Nile (чистий
single change-point), через EmpiricalDistribution (bootstrap реального in-control
marginal + зсув location), що підставляється у наявний синтетичний пайплайн.

Чотири блоки на кожен набір (дзеркало M3A/M3B/M2 на реальних даних):
  (1) Негаусівськість    — skew, надлишк. ексцес, D'Agostino-нормальність.
  (2) Дефляція ARL₀       — наївні ±3σ vs квантильне калібрування (M3B).
  (3) ARL₁-бенчмарк       — Шухарт/EWMA/Page/GSA(адапт.) за спільної ARL₀=370 (M2/M3A).
  (4) PMM2 g₂             — зменшення дисперсії оцінки центру (M3B).
Плюс фігура виявлення реальної розладки на самій часовій послідовності.

Артефакти: results/M6_nongaussianity.csv, M6_calibration.csv,
           M6_arl_<ds>.csv, M6_<ds>_arl.png, M6_<ds>_detection.png,
           M6_welllog_hist.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

import detectors as det
import benchmark as B
import gsa
import pmm
import plotting as P
import realdata as R
import robust_cusum as RC

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

DELTAS = [0.5, 0.75, 1.0, 1.5, 2.0]
DELTA_DESIGN = 1.0
TARGET = 370.0
N_CAL, N_EVAL, MAX_STEPS = 25_000, 40_000, 6000

# Сучасні робастні CUSUM-конкуренти (розрив A) — той самий формат, що й GSA-специфікації.
ROBUST_SPECS = [
    ("winsor", "Winsorized CUSUM (c=1.5)", RC.robust_builder(RC.huber_score(1.5)), "winsor"),
    ("sign", "Sign CUSUM", RC.robust_builder(RC.sign_score()), "sign"),
]


# ============================================================
#  допоміжні
# ============================================================

def arl0_at_shewhart_L(dist, L, N=40_000, max_steps=8000, seed=4242):
    """Фактичне ARL₀ карти Шухарта з фіксованою межею ±L на marginal dist."""
    rng = np.random.default_rng(seed)
    sampler = det.make_sampler(dist, 0.0)
    step, ns = det.shewhart_step(L)
    rl = det.simulate_run_lengths(sampler, step, ns, N, max_steps, rng)
    return float(rl.mean())


def run_shewhart_series(z, L):
    crossed = np.abs(z) > L
    first = int(np.argmax(crossed)) if crossed.any() else None
    return first


def run_cusum_series(z, inc_up, inc_lo, h):
    """Двосторонній CUSUM по фіксованому ряду z: перший перетин + трасу max(C+,C-)."""
    cp = cm = 0.0
    trace = np.empty(len(z))
    first = None
    for t, zt in enumerate(z):
        cp = max(0.0, cp + float(inc_up(zt)))
        cm = max(0.0, cm + float(inc_lo(zt)))
        trace[t] = max(cp, cm)
        if first is None and (cp > h or cm > h):
            first = t
    return first, trace


def pmm2_var_ratio(dist, n=50, reps=4000, seed=11):
    """Емпіричне Var(μ̂_PMM2)/Var(X̄) на bootstrap-вибірках реального marginal."""
    rng = np.random.default_rng(seed)
    g3, g4 = dist.skew, dist.exkurt
    em = np.empty(reps)
    ep = np.empty(reps)
    for r in range(reps):
        x = dist.sample(rng, n, delta=0.0)
        em[r] = x.mean()
        ep[r] = pmm.pmm2_location(x, sigma=1.0, gamma3=g3, gamma4=g4)
    return float(ep.var() / em.var())


def gsa_specs_for(case):
    """Список (key,label,builder,plot_key) GSA-карт для набору."""
    k = case["key"]
    specs = []
    if k == "welllog":
        # poly s=2 не схоплює важкі хвости (J ≈ гаусів) — контраст із frac (M3A-t5)
        specs.append((f"{k}_poly2", "GSA-CUSUM (poly s=2)",
                      B.empirical_gsa_builder("poly", 2), "gsa2"))
        specs.append((f"{k}_frac", case["gsa_label"],
                      B.empirical_gsa_builder("frac", case["gsa_s"],
                                              exponents=case["gsa_exponents"]),
                      case["gsa_plot_key"]))
    else:
        specs.append((f"{k}_gsa", case["gsa_label"],
                      B.empirical_gsa_builder(case["gsa_basis"], case["gsa_s"],
                                              exponents=case["gsa_exponents"]),
                      case["gsa_plot_key"]))
    return specs


# ============================================================
#  основний прогін на один набір
# ============================================================

def analyze(case):
    key = case["key"]
    dist = case["dist"]
    print(f"\n{'='*64}\n{case['title']}\n{'='*64}")

    # ---- (1) негаусівськість ----
    base = dist.base
    dag = stats.normaltest(base)  # D'Agostino K² (skew+kurt) тест нормальності
    nong = dict(dataset=key, n_incontrol=dist.n, skew=dist.skew,
                excess_kurtosis=dist.exkurt,
                dagostino_p=float(dag.pvalue))
    print(f"(1) n={dist.n}  skew={dist.skew:+.3f}  exkurt={dist.exkurt:+.3f}  "
          f"D'Agostino p={dag.pvalue:.2e}  -> "
          f"{'НЕ гаусів' if dag.pvalue < 0.05 else 'сумісний з гаусовим'}")

    # ---- (3) спільне калібрування + ARL₁-бенчмарк ----
    specs_gsa = gsa_specs_for(case)
    specs = specs_gsa + ROBUST_SPECS
    df, calib = B.run_benchmark(
        dist, DELTAS, delta_design=DELTA_DESIGN, gsa_specs=specs,
        include=("shewhart", "ewma", "page"),   # oracle недоступний (щільність невідома)
        target_arl0=TARGET, N_cal=N_CAL, N_eval=N_EVAL, max_steps=MAX_STEPS, seed=606)
    df["dataset"] = key
    df.to_csv(os.path.join(OUT, f"M6_arl_{key}.csv"), index=False)

    gsa_keys = [s[0] for s in specs_gsa]
    robust_keys = [r[0] for r in ROBUST_SPECS]
    det_keys = gsa_keys + robust_keys
    order = ["shewhart", "ewma", "page"] + robust_keys + gsa_keys
    piv = df.pivot(index="delta", columns="method", values="arl")[order]
    print("(3) ARL₁(δ) за спільної ARL₀=370:")
    print(piv.round(2).to_string())

    # ---- (2) дефляція ARL₀ під наївними ±3σ ----
    arl0_naive = arl0_at_shewhart_L(dist, 3.0)
    L_cal = calib["shewhart"]["thr"]
    q_lo, q_hi = np.quantile(base, [0.00135, 0.99865])  # асиметричні квантильні межі
    deflation = TARGET / arl0_naive
    print(f"(2) наївні ±3σ: ARL₀={arl0_naive:.1f}  (дефляція {deflation:.2f}×)  ->  "
          f"калібр. симетр. L={L_cal:.2f}; квантильні межі LCL={q_lo:.2f}/UCL={q_hi:.2f}")

    # ---- (4) PMM2 ----
    g2_theory = pmm.g2_coefficient(dist.skew, dist.exkurt)
    g2_emp = pmm2_var_ratio(dist)
    print(f"(4) PMM2: g₂(теор)={g2_theory:.3f}  Var(PMM2)/Var(X̄)(emp)={g2_emp:.3f}  "
          f"-> {'-' + format((1-g2_emp)*100, '.1f') + '% дисперсії' if g2_emp < 0.99 else 'без виграшу (симетрія)'}")

    calib_row = dict(dataset=key, arl0_naive_3sigma=arl0_naive, deflation=deflation,
                     L_calibrated=L_cal, quantile_LCL=float(q_lo), quantile_UCL=float(q_hi),
                     g2_theory=g2_theory, g2_empirical=g2_emp,
                     real_shift_sigma=case.get("real_shift", np.nan))
    for gk in det_keys:
        calib_row[f"J_{gk}"] = calib[gk].get("J")
        calib_row[f"condF_{gk}"] = calib[gk].get("condF")

    # ---- фігура ARL₁(δ) ----
    fig, ax = P.newfig(7.4, 4.7)
    for m in order:
        sub = df[df.method == m].sort_values("delta")
        pk = calib[m]["plot_key"]
        ax.errorbar(sub.delta, sub.arl, yerr=1.96 * sub.se, color=P.PALETTE[pk],
                    marker=P.MARKERS.get(pk, "o"), label=calib[m]["label"], capsize=2)
    if not np.isnan(case.get("real_shift", np.nan)):
        ax.axvline(case["real_shift"], color="k", ls="-.", lw=1.2,
                   label=f"реальний зсув ≈{case['real_shift']:.2f}σ")
    ax.set_yscale("log")
    ax.axhline(370, color="gray", ls=":", lw=1.0)
    ax.set_xlabel(r"Зсув середнього $\delta$, $\sigma$ (на реальному розподілі)")
    ax.set_ylabel("ARL₁ (тактів до виявлення)")
    ax.set_title(f"M6. {case['title']} — спільна ARL₀=370")
    ax.legend(ncol=2)
    fig.savefig(os.path.join(OUT, f"M6_{key}_arl.png"))

    # ---- фігура виявлення на реальному ряді (для чистих single-CP кейсів) ----
    if case.get("detection_figure", False):
        detection_summary = make_detection_figure(case, calib, specs_gsa)
        calib_row.update(detection_summary)

    # ---- well-log: гістограма in-control vs Гаусс (обґрунтування важких хвостів) ----
    if key == "welllog":
        fig, ax = P.newfig(7.0, 4.4)
        ax.hist(base, bins=80, density=True, color="#8c564b", alpha=0.6,
                label=f"in-control (exkurt={dist.exkurt:+.1f})")
        xs = np.linspace(base.min(), base.max(), 400)
        ax.plot(xs, stats.norm.pdf(xs), color="#1f77b4", lw=2.0, label=r"$\mathcal{N}(0,1)$")
        ax.set_yscale("log")
        ax.set_xlabel("стандартизований відгук z")
        ax.set_ylabel("щільність (log)")
        ax.set_title("M6. Well-log in-control: важкі хвости проти Гаусса")
        ax.legend()
        fig.savefig(os.path.join(OUT, "M6_welllog_hist.png"))

        # ---- дефляція ARL0 на самому in-control ряді: наївні ±3σ vs квантильні межі ----
        # Чесна on-series ілюстрація M3B: на реальному важкохвостому пласті наївні
        # гаусові межі перетинаються викидами багато разів (хибні тривоги), тоді як
        # квантильно-калібровані межі тримають близьку до номінальної частоту.
        L_naive = 3.0
        q_lo, q_hi = np.quantile(base, [0.00135, 0.99865])
        ipts = np.arange(base.size)
        naive_hits = np.where(np.abs(base) > L_naive)[0]
        quant_hits = np.where((base < q_lo) | (base > q_hi))[0]
        expected = base.size * 0.0027
        print(f"    дефляція in-situ ({base.size} pts): наївні ±3σ -> {len(naive_hits)} тривог "
              f"(очік. {expected:.1f}); квантильні [{q_lo:.1f},{q_hi:.1f}] -> {len(quant_hits)}")
        fig, ax = P.newfig(8.2, 4.4)
        ax.plot(ipts, base, color="#444", lw=0.7, alpha=0.8, label="in-control пласт (well-log)")
        ax.axhline(L_naive, color="#d62728", ls="--", lw=1.0)
        ax.axhline(-L_naive, color="#d62728", ls="--", lw=1.0,
                   label=f"наївні ±3σ ({len(naive_hits)} тривог)")
        ax.axhline(q_hi, color="#2ca02c", ls="-", lw=1.3)
        ax.axhline(q_lo, color="#2ca02c", ls="-", lw=1.3,
                   label=f"квантильні межі ({len(quant_hits)} тривог)")
        if len(naive_hits):
            ax.scatter(naive_hits, base[naive_hits], color="#d62728", s=16, zorder=5)
        ax.set_xlabel("індекс виміру (in-control пласт)")
        ax.set_ylabel("стандартизований відгук z")
        ax.set_title(f"M6. Well-log in-control: дефляція ARL₀ — наївні ±3σ ({len(naive_hits)} тривог) "
                     f"проти квантильних ({len(quant_hits)}; очік. {expected:.0f})")
        ax.legend(ncol=2, fontsize=8.5)
        fig.savefig(os.path.join(OUT, "M6_welllog_falsealarm.png"))
        calib_row.update(dict(fa_n_naive3sigma=int(len(naive_hits)),
                              fa_n_quantile=int(len(quant_hits)),
                              fa_expected=round(float(expected), 1)))

    return df, nong, calib_row


def make_detection_figure(case, calib, specs):
    """Прогін Шухарта/Page/адапт-GSA по САМОМУ реальному ряду; перший сигнал."""
    z = case["series_full"][case["view"]]
    x = case["x_index"][case["view"]]
    cp = case["change_point"] - (case["view"].start or 0)

    # Карта Шухарта на реальному ряді — конвенційні гаусові ±3σ (in-control сумісний з Гауссом).
    L_naive = 3.0
    first_shew = run_shewhart_series(z, L_naive)

    k = DELTA_DESIGN / 2.0
    h_page = calib["page"]["thr"]
    first_page, _ = run_cusum_series(z, lambda v: v - k, lambda v: -v - k, h_page)

    # адаптований GSA: останній у specs (frac для well-log, poly2 для Nile)
    gk, glabel, builder, gpk = specs[-1]
    det_up, det_lo = builder(case["dist"], DELTA_DESIGN)
    h_gsa = calib[gk]["thr"]
    first_gsa, _ = run_cusum_series(z, det_up.increment, det_lo.increment, h_gsa)

    # робастні конкуренти на тому самому ряді (перший сигнал)
    first_robust = {}
    for rk, _rlabel, rbuilder, _rpk in ROBUST_SPECS:
        r_up, r_lo = rbuilder(case["dist"], DELTA_DESIGN)
        first_robust[rk], _ = run_cusum_series(z, r_up.increment, r_lo.increment,
                                               calib[rk]["thr"])

    cp_x = x[cp] if 0 <= cp < len(x) else case["change_point"]

    def lab(first):
        return "—" if first is None else str(x[first])

    def delay(first):
        return "—" if first is None else str(int(x[first] - cp_x))

    print(f"    виявлення на ряді (x): Шухарт(±3σ)={lab(first_shew)} (затримка {delay(first_shew)})  "
          f"Page={lab(first_page)} (затримка {delay(first_page)})  "
          f"Winsor={lab(first_robust['winsor'])} (затримка {delay(first_robust['winsor'])})  "
          f"Sign={lab(first_robust['sign'])} (затримка {delay(first_robust['sign'])})  "
          f"GSA={lab(first_gsa)} (затримка {delay(first_gsa)}) | розладка @ {cp_x}")

    # фігура
    fig, ax = P.newfig(8.2, 4.4)
    ax.plot(x, z, color="#444", lw=0.8, alpha=0.8, label="стандартизований ряд")
    ax.axhline(L_naive, color="#d62728", ls="--", lw=1.0)
    ax.axhline(-L_naive, color="#d62728", ls="--", lw=1.0, label=f"Шухарт ±{L_naive:.0f}σ")
    if 0 <= cp < len(x):
        ax.axvline(x[cp], color="k", ls="-.", lw=1.4, label="точка розладки (відома)")
    marks = [("Page", first_page, P.PALETTE["page"], "^"),
             (glabel.split(" (")[0], first_gsa, P.PALETTE[gpk], "P"),
             ("Шухарт", first_shew, P.PALETTE["shewhart"], "o")]
    for name, fi, col, mk in marks:
        if fi is not None:
            ax.scatter([x[fi]], [z[fi]], color=col, marker=mk, s=110, zorder=5,
                       edgecolor="k", linewidth=0.6, label=f"{name} → {x[fi]}")
    ax.set_xlabel(case["x_label"])
    ax.set_ylabel("z")
    ax.set_title(f"M6. Виявлення реальної розладки: {case['title']}")
    ax.legend(ncol=2, fontsize=8.5)
    fig.savefig(os.path.join(OUT, f"M6_{case['key']}_detection.png"))

    return dict(detect_changepoint=str(cp_x),
                detect_shewhart=lab(first_shew), delay_shewhart=delay(first_shew),
                detect_page=lab(first_page), delay_page=delay(first_page),
                detect_winsor=lab(first_robust["winsor"]), delay_winsor=delay(first_robust["winsor"]),
                detect_sign=lab(first_robust["sign"]), delay_sign=delay(first_robust["sign"]),
                detect_gsa=lab(first_gsa), delay_gsa=delay(first_gsa))


def main():
    nong_rows, calib_rows, all_arl = [], [], []
    for key in ("welllog", "nile"):
        case = R.CASES[key]()
        df, nong, calib_row = analyze(case)
        nong_rows.append(nong)
        calib_rows.append(calib_row)
        all_arl.append(df)

    pd.DataFrame(nong_rows).to_csv(os.path.join(OUT, "M6_nongaussianity.csv"), index=False)
    pd.DataFrame(calib_rows).to_csv(os.path.join(OUT, "M6_calibration.csv"), index=False)
    pd.concat(all_arl, ignore_index=True).to_csv(os.path.join(OUT, "M6_arl_all.csv"), index=False)

    print(f"\n{'='*64}\nПІДСУМОК M6 (реальні дані)\n{'='*64}")
    print(pd.DataFrame(nong_rows).round(3).to_string(index=False))
    print()
    cdf = pd.DataFrame(calib_rows)
    cols = ["dataset", "arl0_naive_3sigma", "deflation", "L_calibrated",
            "g2_theory", "g2_empirical", "detect_changepoint", "detect_shewhart",
            "detect_page", "detect_winsor", "detect_sign", "detect_gsa",
            "delay_shewhart", "delay_page", "delay_winsor", "delay_sign", "delay_gsa"]
    cols = [c for c in cols if c in cdf.columns]
    print(cdf[cols].round(3).to_string(index=False))
    print(f"\nАртефакти збережено у {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
