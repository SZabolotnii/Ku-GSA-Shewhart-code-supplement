"""
pmm.py — метод максимізації полінома (PMM) Кунченка для оцінювання параметра
положення за негаусівськими залишками + коефіцієнти зменшення дисперсії.

PMM2 (степінь 2, асиметричні залишки):
    оцінна функція h(u)=u + t·(u²−σ²),  t = −γ₃/((2+γ₄)·σ),  u=x−μ
    розв'язок Σ_i h(x_i−μ̂)=0 (квадратне рівняння відносно μ̂);
    асимптотична дисперсія Var(μ̂)=(σ²/n)·g₂,  g₂ = 1 − γ₃²/(2+γ₄).

PMM3 (степінь 3, симетричні платикуртичні залишки, γ₃=0):
    g₃ = 1 − γ₄²/(6 + 9γ₄ + γ₆).

Тут γ₃ — асиметрія, γ₄ — надлишковий ексцес, γ₆ — нормований 6-й кумулянт.
Для важкохвостових із нескінченними вищими моментами (напр., t₅: E[X⁶]=∞) момент-
орієнтований PMM деградує — це коректно фіксується і обговорюється (потрібен CF-PMM).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def g2_coefficient(gamma3: float, gamma4: float) -> float:
    """Коефіцієнт зменшення дисперсії PMM2 (відносно вибіркового середнього)."""
    return 1.0 - gamma3 ** 2 / (2.0 + gamma4)


def g3_coefficient(gamma4: float, gamma6: float) -> float:
    """Коефіцієнт зменшення дисперсії PMM3 (симетричний випадок)."""
    denom = 6.0 + 9.0 * gamma4 + gamma6
    if not np.isfinite(denom) or denom == 0:
        return 1.0
    return 1.0 - gamma4 ** 2 / denom


def pmm2_location(x: np.ndarray, sigma=None, gamma3=None, gamma4=None) -> float:
    """PMM2-оцінка положення. Якщо моменти не задані — оцінюються з вибірки (plug-in)."""
    x = np.asarray(x, dtype=float)
    if sigma is None:
        sigma = x.std(ddof=1)
    if gamma3 is None:
        gamma3 = stats.skew(x)
    if gamma4 is None:
        gamma4 = stats.kurtosis(x)  # надлишковий
    t = -gamma3 / ((2.0 + gamma4) * sigma)
    # Σ[(x-μ) + t((x-μ)²-σ²)] = 0  -> квадратне за μ:
    #   t·n·μ² + (-1·n - 2t·Σx)·μ + (Σx + t·Σx² - t·n·σ²) = 0
    n = len(x)
    Sx = x.sum()
    Sx2 = (x ** 2).sum()
    A = t * n
    B = -n - 2.0 * t * Sx
    C = Sx + t * Sx2 - t * n * sigma ** 2
    if abs(A) < 1e-12:
        return Sx / n  # вироджується у середнє
    disc = B * B - 4 * A * C
    disc = max(disc, 0.0)
    r1 = (-B + np.sqrt(disc)) / (2 * A)
    r2 = (-B - np.sqrt(disc)) / (2 * A)
    mean = Sx / n
    # обираємо корінь, ближчий до вибіркового середнього (фізичний)
    return r1 if abs(r1 - mean) <= abs(r2 - mean) else r2


def theoretical_g_for(dist, n_mc=4_000_000, rng=None):
    """Теоретичні γ₃, γ₄, g₂ для розподілу (за великою вибіркою)."""
    if rng is None:
        rng = np.random.default_rng(7)
    x = dist.sample(rng, n_mc)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x)
    return dict(gamma3=g3, gamma4=g4, g2=g2_coefficient(g3, g4))


if __name__ == "__main__":
    import distributions as D
    rng = np.random.default_rng(1)
    print("Перевірка PMM2: дисперсія оцінки положення відносно середнього\n")
    for dname, dist in [("gaussian", D.Gaussian()), ("skewnormal", D.SkewNormal(4.0)),
                        ("tpn", D.TwoPieceNormal(2.0)), ("laplace", D.Laplace())]:
        info = theoretical_g_for(dist)
        n = 50
        reps = 20000
        est_mean = np.empty(reps)
        est_pmm = np.empty(reps)
        for r in range(reps):
            x = dist.sample(rng, n, delta=0.0)
            est_mean[r] = x.mean()
            est_pmm[r] = pmm2_location(x, sigma=1.0, gamma3=info["gamma3"], gamma4=info["gamma4"])
        ratio = est_pmm.var() / est_mean.var()
        print(f"{dname:12s} γ₃={info['gamma3']:+.3f} γ₄={info['gamma4']:+.3f}  "
              f"g₂(теор)={info['g2']:.4f}  Var(PMM2)/Var(mean)={ratio:.4f}")
