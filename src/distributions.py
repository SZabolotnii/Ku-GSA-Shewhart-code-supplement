"""
distributions.py — генератори стандартизованих розподілів для SPC-експериментів.

Усі розподіли стандартизовані до E[X]=0, Var[X]=1 у стані статистичного контролю
(in-control, H0). Зсув середнього величиною ``delta`` (у одиницях σ) моделюється
як адитивний зсув: вибірка H1 = стандартизована вибірка H0 + delta. Форма розподілу
при цьому зберігається — змінюється лише положення (location shift), що точно
відповідає сценарію базової статті (зсув μ при незмінній дисперсії та формі).

Кожен розподіл надає:
  - sample(rng, size, delta=0.0): вибірка з можливим зсувом середнього;
  - logpdf(z): лог-щільність стандартизованого in-control розподілу (для оракульного LLR);
  - name, latex: ідентифікатори.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class Distribution:
    """Базовий клас стандартизованого розподілу із зсувом середнього."""

    name = "base"
    latex = "base"

    def _standard_sample(self, rng: np.random.Generator, size) -> np.ndarray:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, size, delta: float = 0.0) -> np.ndarray:
        return self._standard_sample(rng, size) + delta

    def logpdf(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def exact_llr(self, z: np.ndarray, delta: float) -> np.ndarray:
        """Точний логарифм відношення правдоподібності для зсуву location на delta:
        ell(z) = log p(z - delta) - log p(z)."""
        return self.logpdf(z - delta) - self.logpdf(z)


class Gaussian(Distribution):
    name = "gaussian"
    latex = r"$\mathcal{N}(0,1)$"

    def _standard_sample(self, rng, size):
        return rng.standard_normal(size)

    def logpdf(self, z):
        return stats.norm.logpdf(z)


class StudentT(Distribution):
    """Стандартизований Стьюдент: t_df, масштабований до одиничної дисперсії (df>2)."""

    def __init__(self, df: float = 5.0):
        if df <= 2:
            raise ValueError("df>2 потрібно для скінченної дисперсії")
        self.df = df
        self._scale = np.sqrt(df / (df - 2.0))  # std сирого t_df
        self.name = f"student_t{int(df)}"
        self.latex = rf"$t_{{{int(df)}}}$"

    def _standard_sample(self, rng, size):
        return rng.standard_t(self.df, size=size) / self._scale

    def logpdf(self, z):
        # z = raw_t / scale  => raw_t = z*scale; logpdf_std(z) = logpdf_t(z*scale) + log(scale)
        return stats.t.logpdf(z * self._scale, df=self.df) + np.log(self._scale)


class SkewNormal(Distribution):
    """Стандартизований skew-normal із параметром форми a (асиметрія)."""

    def __init__(self, a: float = 4.0):
        self.a = a
        delta = a / np.sqrt(1.0 + a * a)
        self._mu = np.sqrt(2.0 / np.pi) * delta            # середнє сирого skewnorm(a)
        self._sd = np.sqrt(1.0 - 2.0 * delta * delta / np.pi)  # std сирого
        self.name = f"skewnormal_a{a:g}"
        self.latex = rf"$\mathrm{{SN}}(\alpha={a:g})$"

    def _standard_sample(self, rng, size):
        raw = stats.skewnorm.rvs(self.a, size=size, random_state=rng)
        return (raw - self._mu) / self._sd

    def logpdf(self, z):
        raw = z * self._sd + self._mu
        return stats.skewnorm.logpdf(raw, self.a) + np.log(self._sd)

    @property
    def skewness(self):
        d = self.a / np.sqrt(1.0 + self.a * self.a)
        g1 = (4 - np.pi) / 2 * (d * np.sqrt(2 / np.pi)) ** 3 / (1 - 2 * d * d / np.pi) ** 1.5
        return g1


class TwoPieceNormal(Distribution):
    """Двосегментний (split) нормальний розподіл: різні σ ліворуч/праворуч від моди.
    Платикуртично/асиметричний; стандартизований до mean0 var1."""

    def __init__(self, sigma_ratio: float = 2.0):
        self.r = sigma_ratio  # sigma_right / sigma_left
        # для split-normal зі σ1 (ліво) і σ2 (право), σ2=r*σ1:
        s1, s2 = 1.0, sigma_ratio
        A = np.sqrt(2.0 / np.pi) / (s1 + s2)  # нормуюча для unscaled
        mean_raw = np.sqrt(2.0 / np.pi) * (s2 - s1)  # E[X] сирого (мода в 0)
        var_raw = (1.0 - 2.0 / np.pi) * (s2 - s1) ** 2 + s1 * s2
        self._mu = mean_raw
        self._sd = np.sqrt(var_raw)
        self._s1, self._s2 = s1, s2
        self._A = A
        self.name = f"tpn_r{sigma_ratio:g}"
        self.latex = rf"$\mathrm{{TPN}}(r={sigma_ratio:g})$"

    def _raw_sample(self, rng, size):
        n = int(np.prod(size)) if np.ndim(size) else int(size)
        s1, s2 = self._s1, self._s2
        p_left = s1 / (s1 + s2)
        u = rng.random(n)
        left = u < p_left
        out = np.empty(n)
        z = np.abs(rng.standard_normal(n))
        out[left] = -z[left] * s1
        out[~left] = z[~left] * s2
        return out.reshape(size) if np.ndim(size) else out

    def _standard_sample(self, rng, size):
        return (self._raw_sample(rng, size) - self._mu) / self._sd

    def logpdf(self, z):
        raw = z * self._sd + self._mu
        s1, s2 = self._s1, self._s2
        out = np.where(
            raw < 0,
            self._A * np.exp(-0.5 * (raw / s1) ** 2),
            self._A * np.exp(-0.5 * (raw / s2) ** 2),
        )
        return np.log(out) + np.log(self._sd)


class Laplace(Distribution):
    name = "laplace"
    latex = r"$\mathrm{Laplace}$"

    def __init__(self):
        self._b = 1.0 / np.sqrt(2.0)  # масштаб для одиничної дисперсії

    def _standard_sample(self, rng, size):
        return rng.laplace(0.0, self._b, size=size)

    def logpdf(self, z):
        return stats.laplace.logpdf(z, scale=self._b)


def make(name: str) -> Distribution:
    table = {
        "gaussian": Gaussian,
        "student_t5": lambda: StudentT(5),
        "student_t3": lambda: StudentT(3),
        "skewnormal": lambda: SkewNormal(4.0),
        "tpn": lambda: TwoPieceNormal(2.0),
        "laplace": Laplace,
    }
    return table[name]()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for d in [Gaussian(), StudentT(5), SkewNormal(4.0), TwoPieceNormal(2.0), Laplace()]:
        x = d.sample(rng, 2_000_000)
        print(f"{d.name:18s} mean={x.mean():+.4f} var={x.var():.4f} "
              f"skew={stats.skew(x):+.3f} kurt={stats.kurtosis(x):+.3f}")
