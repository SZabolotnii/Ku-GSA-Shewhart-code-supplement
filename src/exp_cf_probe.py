"""
exp_cf_probe.py — CF-шлях на НЕСКІНЧЕННІЙ дисперсії: чи дає GSA-Shewhart унікальну
територію там, де моментні методи (Page/GSA-frac/oPMM) ламаються?

Завершує трикутник PATP / oPMM / CF. На СКІНЧЕННО-дисперсійних важких хвостах (t₅,
well-log) усі score-функції збіглися до майже-оракула (плато). Тут перевіряємо
α-stable (α=1.5, β=0) — дисперсія НЕСКІНЧЕННА, E[|z|^p]=∞ при p≥α. Там:
  - Page-CUSUM (raw z)      — інкремент z−k; один stable-стрибок вибиває поріг → деградує.
  - GSA-frac / oPMM         — F-матриця з E[|z|^{2α}]=∞ → конструкція розходиться.
  - Winsorized / Sign       — обмежені, ПРАЦЮЮТЬ (робастні), але не підлаштовані під форму.
  - CF-GSA (sine-базис)     — Λ(z)=Σ K_m sin(u_m z): обмежений + здатний бути REDESCENDING
                              (Cauchy-bridge: віконний sine-score ∝ location-score Коші).
  - Oracle (точний LLR)     — межа Лордена через scipy.stats.levy_stable.logpdf.

Стандартизація: для α<2 дисперсії нема, тож масштаб = γ=1, зсув δ у одиницях γ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import detectors as det
import gsa
import robust_cusum as RC
from distributions import Distribution, StudentT
from exp_opmm_probe import build_custom

DELTAS = [0.5, 1.0, 1.5]
DELTA_DESIGN = 1.0
TARGET = 370.0
N_CAL_DET = 200_000
N_CAL_THR = 20_000
N_EVAL = 40_000
MAX_STEPS = 6000


# ============================================================
#  Симетричний α-stable розподіл (CMS-sampler + оракул-logpdf)
# ============================================================

class SymmetricStable(Distribution):
    """Симетричний α-stable, scale γ=1 (CF exp(−|u|^α)). Дисперсія ∞ при α<2.

    Семплінг — Chambers–Mallows–Stuck (швидкий, векторизований). logpdf — попередньо
    табульований через scipy.stats.levy_stable на сітці + лінійна інтерполяція; поза
    сіткою logpdf затискається (хвости важкі, LLR location-зсуву там →0, redescending)."""

    def __init__(self, alpha: float = 1.5, grid_max: float = 200.0, n_grid: int = 1600):
        if not (0 < alpha < 2):
            raise ValueError("0<alpha<2")
        self.alpha = alpha
        self.name = f"stable_a{alpha:g}"
        self.latex = rf"$S_{{{alpha:g}}}$"
        zs = np.linspace(-grid_max, grid_max, n_grid)
        self._zgrid = zs
        self._logp = stats.levy_stable.logpdf(zs, alpha, 0.0, loc=0.0, scale=1.0)
        self._zmax = grid_max

    def _standard_sample(self, rng, size):
        U = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
        W = rng.exponential(1.0, size=size)
        a = self.alpha
        if abs(a - 1.0) < 1e-9:
            return np.tan(U)
        c = np.cos(U)
        t1 = np.sin(a * U) / np.power(c, 1.0 / a)
        t2 = np.power(np.cos(U - a * U) / W, (1.0 - a) / a)
        return t1 * t2

    def logpdf(self, z):
        z = np.clip(np.asarray(z, dtype=float), -self._zmax, self._zmax)
        return np.interp(z, self._zgrid, self._logp)


# ============================================================
#  CF-GSA: GSA на тригонометричному (sine) базисі {sin(u_m z)}
# ============================================================

def cf_frequencies(dist, M=8, seed=303):
    """Сітка частот u_max=π/MAD, M точок (як у Ku_CF_SP grid_mad)."""
    x = dist.sample(np.random.default_rng(seed), 50_000, delta=0.0)
    mad = float(np.median(np.abs(x - np.median(x))))
    u_max = np.pi / max(mad, 1e-6)
    return np.linspace(u_max / M, u_max, M)


def cf_gsa_builder(dist, M=8):
    """builder(delta)->GSADetector на базисі sin(u_m z) (обмежений, redescending-здатний)."""
    freqs = cf_frequencies(dist, M)
    funcs = [(lambda z, u=u: np.sin(u * z)) for u in freqs]
    labels = [f"sin({u:.2f}z)" for u in freqs]

    def build(delta):
        return build_custom(dist, delta, funcs, labels, N_CAL_DET, np.random.default_rng(21))
    return build, freqs


# ============================================================
#  оцінка одного детектора
# ============================================================

def arl1_curve(dist, inc_up, inc_lo, h, seed=707):
    step, ns = det.generic_cusum_step(inc_up, inc_lo, h)
    out = {}
    for d in DELTAS:
        rl = det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL, MAX_STEPS,
                                      np.random.default_rng(seed + int(d * 1000)))
        out[d] = float(rl.mean())
    return out


def eval_pair(dist, det_up, det_lo, label, method):
    try:
        h, a0 = det.calibrate_gsa(dist, det_up, det_lo, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
        arl = arl1_curve(dist, det_up.increment, det_lo.increment, h)
        J = getattr(det_up, "J", np.nan)
        return dict(method=method, label=label, ARL0=round(a0, 1), J=round(float(J), 4),
                    **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS})
    except Exception as e:  # моментні методи можуть розійтися на α<2
        return dict(method=method, label=label, ARL0=np.nan, J=np.nan,
                    **{f"ARL1@{d}": "BREAK" for d in DELTAS}, note=str(e)[:40])


def run_case(name, dist, has_oracle):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    rows = []

    # Page (raw z) — еталон деградації на важких хвостах
    k = DELTA_DESIGN / 2.0
    try:
        h, a0 = det.calibrate_page(dist, k, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
        step, ns = det.page_cusum_step(k, h)
        arl = {d: float(det.simulate_run_lengths(det.make_sampler(dist, d), step, ns, N_EVAL,
                MAX_STEPS, np.random.default_rng(707 + int(d * 1000))).mean()) for d in DELTAS}
        rows.append(dict(method="page", label="Page-CUSUM (raw z)", ARL0=round(a0, 1), J=np.nan,
                         **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))
    except Exception as e:
        rows.append(dict(method="page", label="Page-CUSUM (raw z)", ARL0=np.nan, J=np.nan,
                         **{f"ARL1@{d}": "BREAK" for d in DELTAS}, note=str(e)[:40]))

    # GSA-frac (moment-based) — має розійтися при α<2
    try:
        up = gsa.build_empirical(dist, +DELTA_DESIGN, "frac", 3, exponents=[0.5, 1.0, 1.5],
                                 n_cal=N_CAL_DET, rng=np.random.default_rng(31))
        lo = gsa.build_empirical(dist, -DELTA_DESIGN, "frac", 3, exponents=[0.5, 1.0, 1.5],
                                 n_cal=N_CAL_DET, rng=np.random.default_rng(32))
        rows.append(eval_pair(dist, up, lo, "GSA-frac [0.5,1,1.5]", "gsa_frac"))
    except Exception as e:
        rows.append(dict(method="gsa_frac", label="GSA-frac [0.5,1,1.5]", ARL0=np.nan, J=np.nan,
                         **{f"ARL1@{d}": "BREAK" for d in DELTAS}, note=str(e)[:40]))

    # Winsorized / Sign — обмежені робастні конкуренти
    up, lo = RC.robust_builder(RC.huber_score(1.5), n_cal=N_CAL_DET)(dist, DELTA_DESIGN)
    rows.append(eval_pair(dist, up, lo, "Winsorized CUSUM (c=1.5)", "winsor"))
    up, lo = RC.robust_builder(RC.sign_score(), n_cal=N_CAL_DET)(dist, DELTA_DESIGN)
    rows.append(eval_pair(dist, up, lo, "Sign CUSUM", "sign"))

    # CF-GSA (sine-базис) — метод під тестом
    build, freqs = cf_gsa_builder(dist, M=8)
    print(f"  CF-GSA частоти u∈[{freqs[0]:.2f},{freqs[-1]:.2f}], M={len(freqs)}")
    rows.append(eval_pair(dist, build(+DELTA_DESIGN), build(-DELTA_DESIGN),
                          "CF-GSA {sin(u_m z)}", "cf_gsa"))

    # Oracle (точний LLR) — межа Лордена
    if has_oracle:
        inc_up = lambda z: dist.exact_llr(z, DELTA_DESIGN)
        inc_lo = lambda z: dist.exact_llr(z, -DELTA_DESIGN)
        try:
            h, a0 = det.calibrate_oracle(dist, DELTA_DESIGN, TARGET, N=N_CAL_THR, max_steps=MAX_STEPS)
            arl = arl1_curve(dist, inc_up, inc_lo, h)
            rows.append(dict(method="oracle", label="Oracle (точний LLR)", ARL0=round(a0, 1),
                             J=np.nan, **{f"ARL1@{d}": round(arl[d], 2) for d in DELTAS}))
        except Exception as e:
            rows.append(dict(method="oracle", label="Oracle (точний LLR)", ARL0=np.nan, J=np.nan,
                             **{f"ARL1@{d}": "BREAK" for d in DELTAS}, note=str(e)[:40]))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    cols = ["method", "label", "ARL0", "J"] + [f"ARL1@{d}" for d in DELTAS]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    return df


def main():
    import os
    cases = [
        ("Student t5 (скінченна дисперсія — контроль плато)", StudentT(5.0), True),
        ("Symmetric α-stable α=1.5 (НЕСКІНЧЕННА дисперсія)", SymmetricStable(1.5), True),
    ]
    alld = []
    for name, dist, has_oracle in cases:
        df = run_case(name, dist, has_oracle)
        df.insert(0, "case", name.split(" (")[0])
        alld.append(df)
    out = pd.concat(alld, ignore_index=True)
    path = os.path.join(os.path.dirname(__file__), "..", "results", "cf_probe.csv")
    out.to_csv(path, index=False)
    print(f"\nЗбережено: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
