"""
gsa.py — GSA-апроксимація логарифма відношення правдоподібності (LLR)
через стохастичні поліноми Кунченка.

Пайплайн (5 кроків школи Кунченка):
  1. Вибір базису {phi_i}: степеневий (poly) phi_i(z)=z^i; знакозбережний
     дробовий (frac) phi_i(z)=sign(z)|z|^{p_i}.
  2. Оцінка моментів E0[phi_i], E1[phi_i] та коваріацій Cov0, Cov1
     (теоретичні для Гаусса+poly, інакше — за калібрувальними вибірками H0/H1).
  3. Система F·K = Y, де
        Y_i = E1[phi_i] − E0[phi_i],
        F_ij = Cov0(phi_i,phi_j) + Cov1(phi_i,phi_j).
     Розв'язок K — коефіцієнти оптимальної (у сенсі Кунченка) апроксимації LLR.
  4. Статистика Λ(z) = Σ_i K_i phi_i(z); опорне значення
        r = ½ (E0[Λ] + E1[Λ]).
     Інкремент u(z)=Λ(z)−r має знос −J(s)/2 за H0 та +J(s)/2 за H1.
  5. Інформаційний функціонал J(s) = Kᵀ Y = Kᵀ F K (≥0) — міра роздільної
     здатності апроксимації; J(s) монотонно зростає зі степенем s.

Sanity: для N(0,1)→N(δ,1), poly s=1 дає K1=δ/2, тобто u(z)=½(z−δ/2) —
з точністю до сталого множника це інкремент класичного Page-CUSUM (k=δ/2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------- базиси ----------

def poly_basis(s: int):
    """Степеневий базис phi_i(z)=z^i, i=1..s."""
    funcs = [(lambda z, i=i: np.power(z, i)) for i in range(1, s + 1)]
    labels = [f"z^{i}" for i in range(1, s + 1)]
    return funcs, labels


def frac_basis(exponents):
    """Знакозбережний (Form-B) базис phi(z)=sign(z)|z|^p для p у exponents."""
    funcs = [(lambda z, p=p: np.sign(z) * np.power(np.abs(z), p)) for p in exponents]
    labels = [f"sgn·|z|^{p:g}" for p in exponents]
    return funcs, labels


# ---------- теоретичні моменти N(δ,1) (для точного poly-кейсу) ----------

def _normal_raw_moments(delta: float, kmax: int) -> np.ndarray:
    """Сирі моменти E[Z^k], k=0..kmax, для Z~N(delta,1) (рекурсія Ермітового типу)."""
    m = np.zeros(kmax + 1)
    m[0] = 1.0
    if kmax >= 1:
        m[1] = delta
    for k in range(2, kmax + 1):
        m[k] = delta * m[k - 1] + (k - 1) * m[k - 2]
    return m


@dataclass
class GSADetector:
    """Збудований GSA-детектор: статистика Λ та опора r для CUSUM-інкремента."""

    K: np.ndarray
    basis_funcs: list
    basis_labels: list
    r: float
    J: float                      # інформаційний функціонал J(s)
    E0: np.ndarray
    E1: np.ndarray
    cond_F: float
    meta: dict = field(default_factory=dict)

    def Lambda(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.zeros_like(z, dtype=float)
        for k, f in zip(self.K, self.basis_funcs):
            out = out + k * f(z)
        return out

    def increment(self, z: np.ndarray) -> np.ndarray:
        """CUSUM-інкремент u(z)=Λ(z)−r (для виявлення зсуву ВГОРУ)."""
        return self.Lambda(z) - self.r


def _design_from_moments(E0, E1, M0, M1, ridge: float):
    """Будує K, r, J з векторів моментів і матриць других моментів.
    M0=E0[phi_i phi_j], M1=E1[phi_i phi_j]."""
    Y = E1 - E0
    Cov0 = M0 - np.outer(E0, E0)
    Cov1 = M1 - np.outer(E1, E1)
    F = Cov0 + Cov1
    s = len(Y)
    F = F + ridge * np.eye(s)
    cond_F = np.linalg.cond(F)
    K = np.linalg.solve(F, Y)
    r = 0.5 * (K @ E0 + K @ E1)
    J = float(K @ Y)
    return K, r, J, cond_F


def build_gaussian_poly(delta: float, s: int, ridge: float = 1e-9) -> GSADetector:
    """Точний GSA-детектор для N(0,1)→N(δ,1) на степеневому базисі (теор. моменти)."""
    funcs, labels = poly_basis(s)
    mom0 = _normal_raw_moments(0.0, 2 * s)
    mom1 = _normal_raw_moments(delta, 2 * s)
    E0 = mom0[1:s + 1]
    E1 = mom1[1:s + 1]
    M0 = np.empty((s, s))
    M1 = np.empty((s, s))
    for i in range(1, s + 1):
        for j in range(1, s + 1):
            M0[i - 1, j - 1] = mom0[i + j]
            M1[i - 1, j - 1] = mom1[i + j]
    K, r, J, cond_F = _design_from_moments(E0, E1, M0, M1, ridge)
    return GSADetector(K=K, basis_funcs=funcs, basis_labels=labels, r=r, J=J,
                       E0=E0, E1=E1, cond_F=cond_F,
                       meta={"mode": "gaussian_poly", "delta": delta, "s": s})


def build_empirical(dist, delta: float, basis: str = "poly", s: int = 2,
                    exponents=None, n_cal: int = 400_000, ridge: float = 1e-6,
                    rng: np.random.Generator | None = None,
                    winsor: float = 0.0) -> GSADetector:
    """GSA-детектор за калібрувальними вибірками H0 (delta=0) і H1 (delta) розподілу dist.
    Підходить для будь-якого розподілу та базису; моменти оцінюються емпірично."""
    if rng is None:
        rng = np.random.default_rng(0)
    if basis == "poly":
        funcs, labels = poly_basis(s)
    elif basis == "frac":
        if exponents is None:
            exponents = [0.5, 1.0, 1.5][:s]
        funcs, labels = frac_basis(exponents)
    else:
        raise ValueError(basis)

    x0 = dist.sample(rng, n_cal, delta=0.0)
    x1 = dist.sample(rng, n_cal, delta=delta)
    if winsor > 0:
        lo, hi = np.quantile(x0, [winsor, 1 - winsor])
        x0 = np.clip(x0, lo, hi)
        x1 = np.clip(x1, lo, hi)

    P0 = np.column_stack([f(x0) for f in funcs])  # (n_cal, s)
    P1 = np.column_stack([f(x1) for f in funcs])
    E0 = P0.mean(axis=0)
    E1 = P1.mean(axis=0)
    M0 = (P0.T @ P0) / len(x0)
    M1 = (P1.T @ P1) / len(x1)
    K, r, J, cond_F = _design_from_moments(E0, E1, M0, M1, ridge)
    return GSADetector(K=K, basis_funcs=funcs, basis_labels=labels, r=r, J=J,
                       E0=E0, E1=E1, cond_F=cond_F,
                       meta={"mode": "empirical", "basis": basis, "delta": delta,
                             "s": s, "exponents": exponents, "winsor": winsor})


if __name__ == "__main__":
    # Sanity 1: Гаусс, poly s=1 => K1=δ/2
    for delta in (0.5, 1.0, 2.0):
        det = build_gaussian_poly(delta, s=1)
        print(f"[poly s=1] δ={delta}: K={det.K.round(4)} (очік. δ/2={delta/2}), "
              f"r={det.r:.4f}, J(1)={det.J:.4f}")
    # Sanity 2: J(s) монотонність для Гаусса
    print("J(s) для δ=1:", [round(build_gaussian_poly(1.0, s).J, 4) for s in (1, 2, 3, 4)])
    # Sanity 3: емпіричний vs теоретичний для Гаусса
    import distributions as D
    det_e = build_empirical(D.Gaussian(), 1.0, "poly", 1, n_cal=2_000_000,
                            rng=np.random.default_rng(1))
    print(f"[empirical Gauss s=1] K={det_e.K.round(4)} J={det_e.J:.4f} cond(F)={det_e.cond_F:.2f}")
