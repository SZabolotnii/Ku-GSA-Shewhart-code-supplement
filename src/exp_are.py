"""
exp_are.py — ARE (асимптотична відносна ефективність) обмежених score-функцій
відносно ОПТИМАЛЬНОГО location-score, що робить «плато майже-оптимальності» строгим.

Класичний результат (пітменівська ефективність): для виявлення location-зсуву δ→0
ефективність детектора зі score ψ відносно оптимального дорівнює квадрату кореляції
між ψ і оптимальним score s(x)=−d/dx·log f(x):

    ARE(ψ) = ρ²(ψ, s),    s(x) = −f'(x)/f(x).

ARE=1 ⇔ асимптотично оптимальний (оракул); ARE<1 кількісно показує втрату. Дешево:
кореляції на одній великій H₀-вибірці, без CUSUM-симуляції. Це й перетворює
евристичне «плато» на стандартну метрику, яку рецензент Q1 впізнає.

Очікування: на симетричних важких хвостах усі обмежені score ≈ оптимальні (ARE→1,
плато); на асиметрії симетричні Winsor/sign структурно нижчі за GSA (ρ² менший).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import distributions as D
import gsa
from exp_cf_probe import SymmetricStable, cf_gsa_builder

N = 1_000_000
H = 0.01  # крок числового диференціювання log f


def opt_score(dist, x):
    """Оптимальний location-score s(x)=−d/dx log f(x), числово з .logpdf."""
    if isinstance(dist, SymmetricStable):
        s_grid = -np.gradient(dist._logp, dist._zgrid)
        return np.interp(np.clip(x, -dist._zmax, dist._zmax), dist._zgrid, s_grid)
    return -(dist.logpdf(x + H) - dist.logpdf(x - H)) / (2 * H)


def are(psi, s):
    return float(np.corrcoef(psi, s)[0, 1] ** 2)


def winsor_best_are(x, s):
    best = (-1.0, None)
    for c in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
        a = are(np.clip(x, -c, c), s)
        if a > best[0]:
            best = (a, c)
    return best


CASES = [
    ("skew-normal (γ3=0.78)", D.SkewNormal(4.0), ("poly", 3)),
    ("two-piece normal (γ3=0.50)", D.TwoPieceNormal(2.0), ("poly", 2)),
    ("Student t5 (heavy sym)", D.StudentT(5.0), ("frac", 3)),
    ("α-stable α=1.5 (inf. var)", SymmetricStable(1.5), ("cf", 8)),
]


def gsa_lambda(dist, basis, s_or_M, x):
    if basis == "cf":
        build, _ = cf_gsa_builder(dist, M=s_or_M)
        gdet = build(1.0)
    else:
        exps = [0.5, 1.0, 1.5] if basis == "frac" else None
        gdet = gsa.build_empirical(dist, 1.0, basis, s_or_M, exponents=exps,
                                   n_cal=300_000, rng=np.random.default_rng(3))
    return gdet.Lambda(x)


def main():
    rows = []
    for name, dist, (basis, s_or_M) in CASES:
        x = dist.sample(np.random.default_rng(7), N, delta=0.0)
        s = opt_score(dist, x)
        adapted = "CF {sin}" if basis == "cf" else f"{basis} s{s_or_M}"
        row = dict(case=name,
                   ARE_self=round(are(s, s), 3),              # sanity → 1.0
                   ARE_linear=round(are(x, s), 3),            # Page/raw-z score
                   ARE_sign=round(are(np.sign(x), s), 3))
        a_win, c_win = winsor_best_are(x, s)
        row["ARE_winsor"] = round(a_win, 3)
        row["winsor_c*"] = c_win
        row["ARE_GSA"] = round(are(gsa_lambda(dist, basis, s_or_M, x), s), 3)
        row["GSA_basis"] = adapted
        # для α-stable додатково: ARE моментного frac-базису (має бути нижчою — розпад)
        if basis == "cf":
            try:
                row["ARE_moment_frac"] = round(are(gsa_lambda(dist, "frac", 3, x), s), 3)
            except Exception:
                row["ARE_moment_frac"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "results", "are_table.csv")
    df.to_csv(path, index=False)
    print(f"\nЗбережено: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
