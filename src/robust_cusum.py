"""
robust_cusum.py — РОБАСТНІ/НЕПАРАМЕТРИЧНІ CUSUM-конкуренти GSA-CUSUM.

Закриває зауваження рецензента (розрив A): head-to-head бенчмарк проти СУЧАСНИХ
робастних CUSUM, спроєктованих саме під негаусівські/важкохвостові дані, а не лише
проти лінійного Page-CUSUM (1954). Дві канонічні обмежені (bounded) score-функції:

  - Winsorized / Huber-score CUSUM:  ψ_c(z)=clip(z,-c,c)
      Обмежує вплив хвостових викидів — робастний аналог Page для важких хвостів.
      [She2025] (вінзоризований CUSUM, JBES); [Durre2019] (обмежені перетворення, robcp).
  - Sign CUSUM:                      ψ(z)=sign(z)
      Граничний (повністю непараметричний) випадок — розподіло-вільний.
      [Gordon1994] (ефективна послідовна непараметрична схема).

Кожен детектор — двосторонній CUSUM на score-статистиці з ОПОРНИМ значенням
k=½(E₀[ψ]+E_δ[ψ]) (стандартна опора CUSUM на перетвореній статистиці — середина між
in-/out-control середніми score; для ψ(z)=z це точно k=δ/2 класичного Page). Опора
оцінюється емпірично з калібрувальних вибірок H₀/H₁ — той самий бюджет, що й у GSA,
отже порівняння чесне. Поріг h калібрується бісекцією під спільну ARL₀ (як усі карти).

Нижній акумулятор виявляє зсув ВНИЗ так само, як у Page (інкремент на −ψ зі своєю
опорою), що дає додатний знос за H₁=−δ і від'ємний за H₀.

Зауваження про масштаб: ARL CUSUM інваріантний до додатного множника інкремента (він
поглинається порогом h при калібруванні), тож довільна нормалізація ψ не впливає на
результат — важлива лише ФОРМА score-функції та опора k.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------- обмежені score-функції: (psi, label) ----------

def huber_score(c: float = 1.5):
    """Winsorized / Huber ψ_c(z)=clip(z,-c,c). c=1.5 ≈ 95% ефективності за Гаусса."""
    return (lambda z: np.clip(z, -c, c)), f"Winsorized CUSUM (c={c:g})"


def sign_score():
    """Знаковий (непараметричний) score ψ(z)=sign(z)."""
    return (lambda z: np.sign(z)), "Sign CUSUM"


@dataclass
class RobustDetector:
    """Робастний односторонній CUSUM-детектор: інкремент u(z)=s·ψ(z)−k.

    s=+1 — виявлення зсуву ВГОРУ; s=−1 — ВНИЗ (нижній акумулятор, як у Page)."""

    psi: object
    sign: float
    k: float
    Jdef: float                 # коефіцієнт дефлекції (E1−E0)²/(Var0+Var1) — аналог J(s)
    E0: float
    E1: float
    # поля сумісності з gsa_specs-петлею benchmark.run_benchmark (логування J/cond/K):
    J: float = 0.0
    cond_F: float = 1.0
    K: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    def increment(self, z) -> np.ndarray:
        return self.sign * self.psi(np.asarray(z, dtype=float)) - self.k


def build_robust(dist, delta_out: float, score_fn, sign: float = 1.0,
                 n_cal: int = 400_000, rng: np.random.Generator | None = None) -> RobustDetector:
    """Будує односторонній робастний CUSUM з емпіричною опорою.

    H₀: delta=0; H₁: delta=delta_out (для верхньої гілки delta_out=+δ, для нижньої −δ).
    Score статистики гілки — s·ψ(z); опора k=½(E₀[s·ψ]+E₁[s·ψ])."""
    if rng is None:
        rng = np.random.default_rng(0)
    psi, _label = score_fn
    x0 = dist.sample(rng, n_cal, delta=0.0)
    x1 = dist.sample(rng, n_cal, delta=delta_out)
    s0 = sign * psi(x0)
    s1 = sign * psi(x1)
    E0, E1 = float(s0.mean()), float(s1.mean())
    k = 0.5 * (E0 + E1)
    var = float(s0.var() + s1.var())
    Jdef = float((E1 - E0) ** 2 / var) if var > 0 else 0.0
    return RobustDetector(psi=psi, sign=sign, k=k, Jdef=Jdef, E0=E0, E1=E1,
                          J=Jdef, K=np.array([E1 - E0]))


def robust_builder(score_fn, n_cal: int = 400_000, seed: int = 99):
    """Builder для benchmark.run_benchmark: build(dist, delta_design)->(det_up, det_lo).

    Двосторонній CUSUM: верхня гілка ловить +δ, нижня — −δ (дзеркало Page)."""
    def build(dist, delta_design):
        up = build_robust(dist, +delta_design, score_fn, sign=+1.0,
                          n_cal=n_cal, rng=np.random.default_rng(seed))
        lo = build_robust(dist, -delta_design, score_fn, sign=-1.0,
                          n_cal=n_cal, rng=np.random.default_rng(seed + 1))
        return up, lo
    return build


if __name__ == "__main__":
    import distributions as D
    import detectors as det

    # Sanity: для ψ(z)=clip(z,±∞) (тобто ψ=z) опора k має дорівнювати δ/2 (= Page).
    rng = np.random.default_rng(1)
    g = D.Gaussian()
    raw = build_robust(g, 1.0, (lambda z: z, "identity"), sign=+1.0, n_cal=2_000_000,
                       rng=np.random.default_rng(7))
    print(f"[identity score] k={raw.k:.4f} (очік. δ/2=0.5), Jdef={raw.Jdef:.4f} (очік. ≈0.5)")

    # Sanity: Winsorized/Sign калібруються до ARL0~370 на Гауссі й дають скінченне ARL1.
    for name, sf in [("Winsorized c=1.5", huber_score(1.5)), ("Sign", sign_score())]:
        up, lo = robust_builder(sf, n_cal=200_000)(g, 1.0)
        h, a0 = det.calibrate_gsa(g, up, lo, target_arl0=370.0, N=20_000, max_steps=4000)
        sampler = det.make_sampler(g, 1.0)
        step, ns = det.generic_cusum_step(up.increment, lo.increment, h)
        rl = det.simulate_run_lengths(sampler, step, ns, 40_000, 4000, np.random.default_rng(3))
        print(f"[{name:16s}] h={h:.3f} ARL0={a0:.0f} ARL1(δ=1)={rl.mean():.2f} "
              f"Jdef={up.Jdef:.4f} k={up.k:.4f}")
