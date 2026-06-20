"""
realdata.py — реальні набори даних для валідації GSA-Shewhart (milestone M6-real).

Закриває обмеження (iii) §6.5 статті: усі попередні дані синтетичні. Тут — два
РЕАЛЬНІ ряди з відомими точками розладки:

  • well-log — 4050 вимірювань ядерно-магнітного відгуку зонда під час буріння
               свердловини (Ó Ruanaidh & Fitzgerald, 1996). Кусково-стала середня
               (межі порід) + забруднення викидами → ВАЖКІ ХВОСТИ. Прямий тест
               frac-базису (M3A-t5) та дефляції ARL₀ під наївними ±3σ (M3B).
  • Nile     — річний стік Нілу біля Асуана 1871–1970 (n=100); різкий спад середнього
               ~1899 (будівництво Асуанської греблі). Чистий single change-point,
               майже-гаусів — еталонний тест швидкості виявлення проти Шухарта (M2).

Ключова конструкція — EmpiricalDistribution: обгортка над реальною in-control
вибіркою, що бутстрапує спостереження (i.i.d. resampling) і додає зсув location на
delta. Це підставляє ЕМПІРИЧНИЙ marginal у наявний синтетичний пайплайн
(detectors/benchmark/gsa/pmm) без жодних змін у ньому:
    H0 = bootstrap(real in-control),   H1 = bootstrap(real in-control) + delta.
Так ми ізолюємо зсув середнього (який вивчає стаття) на РЕАЛЬНІЙ негаусівській формі.

Застереження (перетин з обмеженням (ii) статті): реальні ряди автокорельовані, а
бутстрап припускає i.i.d. Тому це ІЛЮСТРАТИВНА валідація на реальній формі розподілу,
а не строгий ARL-доказ; межі сегментів використано як орієнтири точок розладки.

Джерела (кешуються локально у data/ при першому запуску):
  well_log.txt — https://raw.githubusercontent.com/doccstat/fastcpd/main/data-raw/well_log.txt
  Nile.csv     — https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Nile.csv
"""
from __future__ import annotations

import csv
import os
import urllib.request

import numpy as np
from scipy import stats

from distributions import Distribution

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
WELLLOG_URL = "https://raw.githubusercontent.com/doccstat/fastcpd/main/data-raw/well_log.txt"
NILE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Nile.csv"


# ---------- завантаження з кешуванням ----------

def _ensure(path: str, url: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"  завантаження {os.path.basename(path)} ← {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_welllog() -> np.ndarray:
    """4050 вимірювань ядерно-магнітного відгуку (одна колонка)."""
    return np.loadtxt(_ensure(os.path.join(DATA, "well_log.txt"), WELLLOG_URL))


def load_nile():
    """Повертає (роки, стік). 100 річних значень 1871–1970."""
    p = _ensure(os.path.join(DATA, "Nile.csv"), NILE_URL)
    years, vals = [], []
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            years.append(int(float(row["time"])))
            vals.append(float(row["value"]))
    return np.asarray(years), np.asarray(vals, dtype=float)


# ---------- стандартизація ----------

def robust_standardize(x, center=None, scale=None):
    """Робастна Phase-I стандартизація: (x − медіана) / MAD-σ.

    Робастні оцінки центру й масштабу — стандартна метрологічна практика Phase-I:
    викиди не «роздувають» контрольні межі, тож важкі хвости лишаються проявленими.
    """
    x = np.asarray(x, dtype=float)
    if center is None:
        center = float(np.median(x))
    if scale is None:
        scale = float(stats.median_abs_deviation(x, scale="normal"))
    return (x - center) / scale, center, scale


# ---------- емпіричний marginal для наявного пайплайна ----------

class EmpiricalDistribution(Distribution):
    """Згладжений bootstrap-marginal зі стандартизованої реальної in-control вибірки + зсув.

    Реалізує інтерфейс Distribution (sample/_standard_sample), тож працює напряму з
    detectors.py, benchmark.py, gsa.build_empirical і pmm. Метод logpdf свідомо НЕ
    визначено: для реальних даних істинна щільність невідома, тому оракульний LLR
    недоступний — у бенчмарку виключаємо include="oracle".

    Застосовано ЗГЛАДЖЕНИЙ (ядровий) bootstrap із збереженням дисперсії (Silverman,
    1986): x* = x̄ + (x_i − x̄ + h·ε) / √(1 + h²/s²),  ε~N(0,1). Це дає НЕПЕРЕРВНУ
    носійну множину замість дискретної (важливо для калібрування карти Шухарта за
    індивідуальними значеннями на короткому in-control сегменті, де дискретний bootstrap
    обмежений максимумом |x_i| і не калібрується до ARL₀=370). Ширину h беремо за
    правилом Сільвермана h = 0.9·min(s, IQR/1.349)·n^{-1/5}; вона мала для великих n,
    тож форма (асиметрія, ексцес) практично зберігається. h=0 вимикає згладжування.
    """

    def __init__(self, incontrol_std, name="empirical", latex=None, smooth=True, bw=None):
        self.base = np.asarray(incontrol_std, dtype=float)
        self.name = name
        self.latex = latex or name
        self.skew = float(stats.skew(self.base))
        self.exkurt = float(stats.kurtosis(self.base))  # надлишковий ексцес
        self.n = int(self.base.size)
        self._mean = float(self.base.mean())
        self._var = float(self.base.var())
        if bw is not None:
            self.bw = float(bw)
        elif smooth:
            iqr = float(np.subtract(*np.percentile(self.base, [75, 25])))
            scale = min(np.sqrt(self._var), iqr / 1.349) if iqr > 0 else np.sqrt(self._var)
            self.bw = 0.9 * scale * self.n ** (-0.2)
        else:
            self.bw = 0.0
        self._shrink = np.sqrt(1.0 + self.bw ** 2 / self._var) if self._var > 0 else 1.0

    def _standard_sample(self, rng, size):
        x = rng.choice(self.base, size=size, replace=True)
        if self.bw > 0:
            x = self._mean + (x - self._mean + self.bw * rng.standard_normal(size)) / self._shrink
        return x


# ---------- описи двох кейсів (in-control сегмент + точка розладки) ----------
#
# Сегменти вибрано за розвідувальним аналізом (window-means + robust-z outliers):
#   well-log: перший стабільний пласт [0:1030) має надлишковий ексцес ≈ 11.5 (25 викидів)
#             — кусково-стала середня; великий висхідний зсув середнього ~індекс 1036.
#   Nile:     in-control = до 1899 (28 років); різкий спад ~1.84σ з 1899-го.

def welllog_case():
    w = load_welllog()
    incontrol = w[:1030]                       # перший пласт (heavy-tailed)
    base, c, s = robust_standardize(incontrol)
    dist = EmpiricalDistribution(base, name="welllog",
                                 latex=r"well-log (важкі хвости)")
    return dict(
        key="welllog", title="Well-log: ядерно-магнітний відгук (важкі хвости)",
        series_full=(w - c) / s, x_index=np.arange(w.size), x_label="індекс виміру",
        incontrol_slice=slice(0, 1030), change_point=1036,
        view=slice(0, 1300),                   # вікно для фігури виявлення
        detection_figure=False,                # ряд має викидний сплеск на старті — фігуру виявлення опускаємо
        dist=dist, center=c, scale=s,
        gsa_basis="frac", gsa_label="GSA-CUSUM (frac |z|^{0.5,1,1.5})",
        gsa_plot_key="gsa_frac", gsa_exponents=[0.5, 1.0, 1.5], gsa_s=3,
    )


def nile_case():
    yr, val = load_nile()
    cp_year = 1899
    incontrol = val[yr < cp_year]
    # Phase-I: для майже-гаусового Nile беремо класичні mean/std in-control сегмента
    c = float(incontrol.mean())
    s = float(incontrol.std(ddof=1))
    base = (incontrol - c) / s
    dist = EmpiricalDistribution(base, name="nile", latex=r"Nile (стік річки)")
    cp_idx = int(np.argmax(yr >= cp_year))
    return dict(
        key="nile", title="Nile: річний стік 1871–1970 (зсув ~1899)",
        series_full=(val - c) / s, x_index=yr, x_label="рік",
        incontrol_slice=slice(0, cp_idx), change_point=cp_idx,
        change_year=cp_year, view=slice(0, len(yr)),
        detection_figure=True,
        dist=dist, center=c, scale=s,
        real_shift=abs(val[yr < cp_year].mean() - val[yr >= cp_year].mean()) / s,
        gsa_basis="poly", gsa_label="GSA-CUSUM (poly s=2)",
        gsa_plot_key="gsa2", gsa_exponents=None, gsa_s=2,
    )


CASES = {"welllog": welllog_case, "nile": nile_case}


if __name__ == "__main__":
    for k, fn in CASES.items():
        c = fn()
        d = c["dist"]
        print(f"{k:8s} n_incontrol={d.n:5d}  skew={d.skew:+.2f}  "
              f"exkurt={d.exkurt:+.2f}  cp@{c['change_point']}")
