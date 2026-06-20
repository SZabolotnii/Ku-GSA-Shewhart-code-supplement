"""
exp_opmm_probe.py — чи відіграє oPMM-концепція важкі хвости проти Winsorized CUSUM?

Контекст: GSA-frac (фіксовані експоненти [0.5,1,1.5]) йде в нічию з Winsorized
CUSUM на важких хвостах (t₅, well-log). Гіпотеза користувача: oPMM-конструкція
(/Users/docua/Project/Research/oPMM) як альтернатива PATP могла б пробити цю стелю.

oPMM_α = k₁·z + k₂·[sign(z)|z|^α − μ_α]: лінійний член + ОДИН знако-парний дробовий
член з ОПТИМАЛЬНИМ α. У GSA-формі це базис {z, sign(z)|z|^α}, де коефіцієнти вже
оптимальні (F·K=Y), а вільний параметр — лише експонента α. Тут α* обирається за
ДЕТЕКЦІЙНИМ критерієм (max J = дефлекція H0/H1) — доречний аналог oPMM-вибору α
(той мінімізує дисперсію ОЦІНКИ; нам потрібна дефлекція ДЕТЕКЦІЇ).

Контрольна гіпотеза: і PATP, і oPMM мають МОНОТОННИЙ базис sign(z)|z|^α→∞, а
оптимальний LLR важкого хвоста REDESCENDING (score t₅ → 0). Тож додаємо
redescending-член (Welsch z·exp(−(z/c)²)) — він НЕ з родини oPMM/PATP — щоб перевірити,
чи саме redescending-форма, а не вибір α, є справжнім важелем.

Детектори (усі двосторонні CUSUM, калібровані до спільної ARL₀=370):
  page          Page-CUSUM (k=δ/2)                          — лінійний baseline
  winsor        Winsorized CUSUM (c=1.5)                    — конкурент, що тримає нічию
  gsa_frac      GSA-frac, фікс. [0.5,1,1.5]                 — поточний рукопис
  gsa_opmm      GSA {z, sign|z|^α*}, α* за max J            — oPMM-концепція
  gsa_redesc    GSA {z, Welsch(z)}                          — redescending-контроль
  gsa_full      GSA {z, sign|z|^α*, Welsch(z)}              — комбінований
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import distributions as D
import detectors as det
import gsa
import robust_cusum as RC
import realdata as R

DELTAS = [0.5, 1.0, 1.5]
DELTA_DESIGN = 1.0
TARGET = 370.0
N_CAL_DET = 200_000     # вибірка для побудови детектора (моменти)
N_CAL_THR = 20_000      # калібрування порога
N_EVAL = 40_000
MAX_STEPS = 6000
ALPHA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.3, 1.5]  # без 1.0 (=лінійний член)


def welsch(c: float = 2.5):
    """Redescending Welsch-score z·exp(−(z/c)²): лінійний у центрі, →0 на хвостах."""
    return lambda z: z * np.exp(-((z / c) ** 2)), f"Welsch(c={c:g})"


def build_custom(dist, delta, funcs, labels, n_cal, rng, ridge=1e-6):
    """GSA-детектор на ДОВІЛЬНОМУ базисі funcs (емпіричні моменти H0/H1)."""
    x0 = dist.sample(rng, n_cal, delta=0.0)
    x1 = dist.sample(rng, n_cal, delta=delta)
    P0 = np.column_stack([f(x0) for f in funcs])
    P1 = np.column_stack([f(x1) for f in funcs])
    E0, E1 = P0.mean(axis=0), P1.mean(axis=0)
    M0 = (P0.T @ P0) / len(x0)
    M1 = (P1.T @ P1) / len(x1)
    K, r, J, cond_F = gsa._design_from_moments(E0, E1, M0, M1, ridge)
    return gsa.GSADetector(K=K, basis_funcs=funcs, basis_labels=labels, r=r, J=J,
                           E0=E0, E1=E1, cond_F=cond_F, meta={"labels": labels})


def select_alpha(dist, delta, n_cal, seed=4040):
    """oPMM-вибір експоненти: α* = argmax_α J на базисі {z, sign(z)|z|^α} (дефлекція)."""
    best = None
    for a in ALPHA_GRID:
        funcs = [(lambda z: z), (lambda z, p=a: np.sign(z) * np.abs(z) ** p)]
        d = build_custom(dist, delta, funcs, ["z", f"sgn|z|^{a:g}"], n_cal,
                         np.random.default_rng(seed + int(a * 100)))
        if np.isfinite(d.J) and (best is None or d.J > best[1]):
            best = (a, d.J)
    return best[0], best[1]


def make_builders(dist):
    """Повертає {key: (label, builder)}; builder(delta)->GSADetector для виявлення зсуву delta."""
    alpha_star, J_star = select_alpha(dist, DELTA_DESIGN, N_CAL_DET)
    wfun, wlabel = welsch(2.5)

    def frac_fixed(delta):
        return gsa.build_empirical(dist, delta, "frac", 3, exponents=[0.5, 1.0, 1.5],
                                   n_cal=N_CAL_DET, rng=np.random.default_rng(11))

    def opmm(delta):
        funcs = [(lambda z: z), (lambda z, p=alpha_star: np.sign(z) * np.abs(z) ** p)]
        return build_custom(dist, delta, funcs, ["z", f"sgn|z|^{alpha_star:g}"],
                            N_CAL_DET, np.random.default_rng(12))

    def redesc(delta):
        funcs = [(lambda z: z), wfun]
        return build_custom(dist, delta, funcs, ["z", wlabel], N_CAL_DET,
                           np.random.default_rng(13))

    def full(delta):
        funcs = [(lambda z: z), (lambda z, p=alpha_star: np.sign(z) * np.abs(z) ** p), wfun]
        return build_custom(dist, delta, funcs, ["z", f"sgn|z|^{alpha_star:g}", wlabel],
                           N_CAL_DET, np.random.default_rng(14))

    return alpha_star, J_star, {
        "gsa_frac": ("GSA-frac [0.5,1,1.5]", frac_fixed),
        "gsa_opmm": (f"GSA-oPMM {{z, sgn|z|^{alpha_star:g}}}", opmm),
        "gsa_redesc": ("GSA-redesc {z, Welsch}", redesc),
        "gsa_full": (f"GSA-full {{z, sgn|z|^{alpha_star:g}, Welsch}}", full),
    }


def arl1_curve(dist, det_up, det_lo, h, deltas, seed=707):
    out = {}
    step, ns = det.generic_cusum_step(det_up.increment, det_lo.increment, h)
    for d in deltas:
        rng = np.random.default_rng(seed + int(d * 1000))
        rl = det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL, MAX_STEPS, rng)
        out[d] = float(rl.mean())
    return out


def eval_gsa_like(dist, builder, deltas):
    """Калібрування + ARL₁ для GSA-подібного детектора (up=+δ, lo=−δ)."""
    det_up = builder(+DELTA_DESIGN)
    det_lo = builder(-DELTA_DESIGN)
    h, a0 = det.calibrate_gsa(dist, det_up, det_lo, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
    return a0, det_up.J, det_up.cond_F, arl1_curve(dist, det_up, det_lo, h, deltas)


def run_case(name, dist):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    rows = []

    # Page
    k = DELTA_DESIGN / 2.0
    h, a0 = det.calibrate_page(dist, k, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
    su, sl = (lambda z: z - k), (lambda z: -z - k)
    step, ns = det.page_cusum_step(k, h)
    arl = {d: float(det.simulate_run_lengths(det.make_sampler(dist, d), step, ns,
            N_EVAL, MAX_STEPS, np.random.default_rng(707 + int(d * 1000))).mean()) for d in DELTAS}
    rows.append(dict(method="page", label="Page-CUSUM", ARL0=round(a0, 1), J=np.nan, condF=np.nan,
                     **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))

    # Winsorized
    up, lo = RC.robust_builder(RC.huber_score(1.5), n_cal=N_CAL_DET)(dist, DELTA_DESIGN)
    h, a0 = det.calibrate_gsa(dist, up, lo, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
    arl = arl1_curve(dist, up, lo, h, DELTAS)
    rows.append(dict(method="winsor", label="Winsorized CUSUM (c=1.5)", ARL0=round(a0, 1),
                     J=round(up.Jdef, 4), condF=1.0,
                     **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))

    # GSA-подібні
    alpha_star, J_star, builders = make_builders(dist)
    print(f"  oPMM α* (max J на {{z, sgn|z|^α}}) = {alpha_star:g}   (J={J_star:.4f})")
    for key, (label, builder) in builders.items():
        a0, J, condF, arl = eval_gsa_like(dist, builder, DELTAS)
        rows.append(dict(method=key, label=label, ARL0=round(a0, 1), J=round(J, 4),
                         condF=round(condF, 1), **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    return df


def main():
    cases = [
        ("Student t5 (синтетика, важкі хвости)", D.StudentT(5.0)),
        ("Well-log (реальні дані, γ4≈11.5)", R.CASES["welllog"]()["dist"]),
    ]
    alld = []
    for name, dist in cases:
        df = run_case(name, dist)
        df.insert(0, "case", name.split(" (")[0])
        alld.append(df)
    out = pd.concat(alld, ignore_index=True)
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "results", "opmm_probe.csv")
    out.to_csv(path, index=False)
    print(f"\nЗбережено: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
