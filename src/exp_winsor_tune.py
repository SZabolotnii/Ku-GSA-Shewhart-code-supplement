"""
exp_winsor_tune.py — STEELMAN Winsorized CUSUM: чи виживає перевага GSA на
асиметрії, якщо дати конкуренту ЙОГО НАЙКРАЩИЙ рівень кліпу c?

Зауваження рецензента (розрив A, раунд 2): фіксований c=1.5 «гандикапить» Winsor;
гострий рецензент скаже «підлаштуйте c під кожен розподіл — і перевага зникне».
Структурна гіпотеза захисту: Winsor ψ_c(z)=clip(z,-c,c) СИМЕТРИЧНИЙ, тож жоден c не
схопить скошеність → розрив на асиметрії має вціліти. Тут це перевіряємо емпірично.

Дві частини:
  (1) Скан дефлекції J(c) по сітці c для усіх 4 розподілів → c*=argmax J (best-case Winsor).
  (2) Повний ARL₁ на сітці c для АСИМЕТРИЧНИХ (skew-normal, TPN): беремо НАЙКРАЩИЙ
      Winsor по всіх c і порівнюємо з GSA-адапт. Розрив після тюнінгу = головне число.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import distributions as D
import detectors as det
import gsa
import robust_cusum as RC
import realdata as R

DELTA_DESIGN = 1.0
TARGET = 370.0
N_CAL_DET = 300_000
N_CAL_THR = 20_000
N_EVAL = 40_000
MAX_STEPS = 6000
C_GRID = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
C_ARL = [1.0, 1.5, 2.0, 3.0]   # підмножина для дорогого повного ARL₁


def deflection_J(dist, c, delta, n_cal, seed):
    """J(c)=(E1[ψ_c]−E0[ψ_c])²/(Var0+Var1) для Winsor-score (детекційна дефлекція)."""
    rng = np.random.default_rng(seed)
    x0 = dist.sample(rng, n_cal, delta=0.0)
    x1 = dist.sample(rng, n_cal, delta=delta)
    s0 = np.clip(x0, -c, c)
    s1 = np.clip(x1, -c, c)
    var = s0.var() + s1.var()
    return float((s1.mean() - s0.mean()) ** 2 / var) if var > 0 else 0.0


def arl1_at(dist, det_up, det_lo, deltas, seed=707):
    h, a0 = det.calibrate_gsa(dist, det_up, det_lo, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
    step, ns = det.generic_cusum_step(det_up.increment, det_lo.increment, h)
    arl = {}
    for d in deltas:
        rl = det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL, MAX_STEPS,
                                      np.random.default_rng(seed + int(d * 1000)))
        arl[d] = (float(rl.mean()), float(rl.std(ddof=1) / np.sqrt(len(rl))))
    return a0, arl


def winsor_pair(dist, c):
    return RC.robust_builder(RC.huber_score(c), n_cal=N_CAL_DET)(dist, DELTA_DESIGN)


CASES = {
    "skewnormal": dict(dist=D.SkewNormal(4.0), title="skew-normal (γ3=0.78)",
                       gsa=("poly", 3), asym=True),
    "tpn": dict(dist=D.TwoPieceNormal(2.0), title="two-piece normal (γ3=0.50)",
                gsa=("poly", 2), asym=True),
    "student_t5": dict(dist=D.StudentT(5.0), title="Student t5 (heavy sym)",
                       gsa=("frac", 3), asym=False),
    "welllog": dict(dist=R.CASES["welllog"]()["dist"], title="well-log (real heavy sym)",
                    gsa=("frac", 3), asym=False),
}
DELTAS = [0.5, 1.0]


def gsa_pair(dist, basis, s):
    exps = [0.5, 1.0, 1.5] if basis == "frac" else None
    up = gsa.build_empirical(dist, +DELTA_DESIGN, basis, s, exponents=exps,
                             n_cal=N_CAL_DET, rng=np.random.default_rng(41))
    lo = gsa.build_empirical(dist, -DELTA_DESIGN, basis, s, exponents=exps,
                             n_cal=N_CAL_DET, rng=np.random.default_rng(42))
    return up, lo


def main():
    print("\n=== (1) Скан дефлекції J(c) ===")
    jrows = []
    for key, cfg in CASES.items():
        dist = cfg["dist"]
        Js = {c: deflection_J(dist, c, DELTA_DESIGN, N_CAL_DET, seed=50 + i)
              for i, c in enumerate(C_GRID)}
        c_star = max(Js, key=Js.get)
        jrows.append(dict(case=key, asym=cfg["asym"], c_star=c_star,
                          J_cstar=round(Js[c_star], 4), J_c15=round(Js[1.5], 4),
                          J_lin=round(deflection_J(dist, 1e6, DELTA_DESIGN, N_CAL_DET, 99), 4)))
        prof = "  ".join(f"c={c:g}:{Js[c]:.3f}" for c in C_GRID)
        print(f"  {key:12s} c*={c_star:g}  J(c*)={Js[c_star]:.4f}  J(1.5)={Js[1.5]:.4f}\n    {prof}")
    print(pd.DataFrame(jrows).to_string(index=False))

    print("\n=== (2) Повний ARL₁: НАЙКРАЩИЙ Winsor (по всіх c) проти GSA-адапт ===")
    for key, cfg in CASES.items():
        if not cfg["asym"]:
            continue
        dist = cfg["dist"]
        print(f"\n--- {cfg['title']} ---")
        rows = []
        # Page
        kk = DELTA_DESIGN / 2.0
        h, a0 = det.calibrate_page(dist, kk, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
        step, ns = det.page_cusum_step(kk, h)
        arl = {d: (float(det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL,
                MAX_STEPS, np.random.default_rng(707 + int(d * 1000))).mean())) for d in DELTAS}
        rows.append(dict(method="Page", **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))
        # Winsor по сітці c
        for c in C_ARL:
            up, lo = winsor_pair(dist, c)
            a0, arl = arl1_at(dist, up, lo, DELTAS)
            rows.append(dict(method=f"Winsor c={c:g}",
                             **{f"ARL1@{d}": round(arl[d][0], 2) for d in DELTAS}))
        # GSA-адапт
        up, lo = gsa_pair(dist, *cfg["gsa"])
        a0, arl = arl1_at(dist, up, lo, DELTAS)
        rows.append(dict(method=f"GSA {cfg['gsa'][0]} s{cfg['gsa'][1]}",
                         **{f"ARL1@{d}": round(arl[d][0], 2) for d in DELTAS}))
        # Oracle
        h, a0 = det.calibrate_oracle(dist, DELTA_DESIGN, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
        inc_up = lambda z: dist.exact_llr(z, DELTA_DESIGN)
        inc_lo = lambda z: dist.exact_llr(z, -DELTA_DESIGN)
        step, ns = det.generic_cusum_step(inc_up, inc_lo, h)
        arl = {d: float(det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL,
                MAX_STEPS, np.random.default_rng(909 + int(d * 1000))).mean()) for d in DELTAS}
        rows.append(dict(method="Oracle", **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        # розрив GSA проти НАЙКРАЩОГО Winsor
        gsa_row = df[df.method.str.startswith("GSA")].iloc[0]
        win_rows = df[df.method.str.startswith("Winsor")]
        for d in DELTAS:
            best_win = win_rows[f"ARL1@{d}"].min()
            best_win_c = win_rows.loc[win_rows[f"ARL1@{d}"].idxmin(), "method"]
            gap = (best_win - gsa_row[f"ARL1@{d}"]) / best_win * 100
            print(f"  δ={d}: GSA={gsa_row[f'ARL1@{d}']:.2f}  best Winsor={best_win:.2f} "
                  f"({best_win_c})  → розрив GSA vs best-Winsor = {gap:+.1f}%")


if __name__ == "__main__":
    main()
