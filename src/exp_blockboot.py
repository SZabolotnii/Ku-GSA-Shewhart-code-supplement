"""
exp_blockboot.py — Block-bootstrap real-data robustness check (well-log).

Зауваження рецензента: реальні ряди (well-log, Nile) автокорельовані, але M6-валідація
використовує i.i.d. згладжений bootstrap → ARL під залежністю може бути спотвореним.

Тут:
  (1) Кількісно оцінюємо автокореляцію стандартизованого in-control сегмента well-log
      (той самий [0:1030), що в M6): ACF lag 1..5.
  (2) Будуємо MOVING-BLOCK-BOOTSTRAP генератор in-control потоку: замість i.i.d.-вибірки
      тягнемо суміжні блоки довжини L з base і конкатенуємо — серійна залежність H0
      зберігається в межах блоку. H1 = H0 + δ (той самий location-shift, що скрізь).
  (3) Для кожної довжини блоку L∈{10,25,50}:
        • SCORE-функції детекторів проєктуються на МАРГІНАЛ (Shewhart L; Page k=0.5;
          Winsor c=1.5; GSA-frac |z|^{0.5,1,1.5}) — маргінал block-bootstrap ідентичний
          i.i.d.-марґіналу, тож оптимальний score не змінюється; змінюється лише поріг.
        • ARL₀-ДРЕЙФ: беремо i.i.d.-калібровані пороги й МІРЯЄМО ARL₀ під блок-bootstrap.
        • RECALIBRATION: перекалібровуємо ПОРІГ кожної карти до ARL₀=370 ПІД залежністю,
          потім міряємо ARL₁ при δ∈{0.5,1.0,2.0}.

Порівняння з i.i.d. M6 (results/M6_arl_welllog.csv, Table 7 @ δ=1.0:
  Page=22.1, Winsor=11.6, GSA-frac=11.6, Shewhart=206.6).

Підхід = ПОВНА перекалібровка порога під залежністю (score лишається маргінальним).
Артефакт: results/blockboot_welllog_table.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import detectors as det
import gsa
import realdata as R
import robust_cusum as RC

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

TARGET = 370.0
N_CAL = 15_000
N_EVAL = 30_000
MAX_STEPS = 6000
BLOCK_LENS = [10, 25, 50]
DELTAS = [0.5, 1.0, 2.0]
N_CAL_DET = 300_000   # бюджет дизайну score-функцій (на марґіналі)
DELTA_DESIGN = 1.0


# ============================================================
#  Moving-block-bootstrap симулятор довжин серій
# ============================================================

def simulate_blockboot(base, delta, step_fn, n_state, N, L, max_steps, rng):
    """N паралельних послідовностей; кожна — конкатенація суміжних блоків довжини L
    з base (moving-block bootstrap, старт блоку рівномірно з [0, n-L]). H1=H0+delta.
    Серійна залежність зберігається в межах блоку. Інтерфейс step_fn — як в i.i.d.-
    симуляторі; стан компактизуємо (завершені послідовності вилучаємо)."""
    n = len(base)
    active = np.arange(N)
    state = np.zeros((N, n_state)) if n_state > 0 else np.zeros((N, 0))
    run_len = np.full(N, max_steps, dtype=np.int64)
    src_start = rng.integers(0, n - L + 1, size=N)   # позиція початку поточного блоку
    offset = np.zeros(N, dtype=np.int64)             # зсув усередині блоку
    for t in range(1, max_steps + 1):
        a = active
        m = a.size
        if m == 0:
            break
        need_new = offset[a] >= L
        if need_new.any():
            idx = a[need_new]
            src_start[idx] = rng.integers(0, n - L + 1, size=idx.size)
            offset[idx] = 0
        z = base[src_start[a] + offset[a]] + delta
        offset[a] += 1
        state, crossed = step_fn(state, z, t)
        if crossed.any():
            run_len[a[crossed]] = t
            keep = ~crossed
            active = a[keep]
            state = state[keep]
    return run_len


def arl0_blockboot(base, make_step_h, h, n_state, N, L, max_steps, seed):
    rng = np.random.default_rng(seed)
    step, ns = make_step_h(h)
    rl = simulate_blockboot(base, 0.0, step, ns, N, L, max_steps, rng)
    return float(rl.mean())


def calibrate_blockboot(base, make_step_h, n_state, L, target=TARGET,
                        N=N_CAL, max_steps=MAX_STEPS, lo=0.2, hi=12.0,
                        tol=0.02, max_iter=34, seed=771):
    """Бісекція порога h до target-ARL₀ ПІД блок-bootstrap H0."""
    def a0(h):
        return arl0_blockboot(base, make_step_h, h, n_state, N, L, max_steps, seed)

    a0_hi = a0(hi)
    it = 0
    while a0_hi < target and hi < 200 and it < 12:
        hi *= 1.6
        a0_hi = a0(hi)
        it += 1
    mid, am = hi, a0_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        am = a0(mid)
        if abs(am - target) / target < tol:
            return mid, am
        if am < target:
            lo = mid
        else:
            hi = mid
    return mid, am


def arl1_blockboot(base, step, ns, delta, L, seed=991):
    rng = np.random.default_rng(seed + int(delta * 1000))
    rl = simulate_blockboot(base, delta, step, ns, N_EVAL, L, MAX_STEPS, rng)
    return float(rl.mean()), float(rl.std(ddof=1) / np.sqrt(len(rl)))


# ============================================================
#  Детектори: score-функції з марґіналу + калібратор за параметром
# ============================================================

def build_detectors(dist):
    """Повертає dict label -> (kind, payload). Score/опору проєктуємо на МАРГІНАЛ dist.
    kind='L' → Shewhart (параметр L); kind='h' → CUSUM (параметр h, фікс. інкременти)."""
    out = {}
    # Shewhart: параметр — межа L
    out["Shewhart"] = ("L", dict(make=lambda L: det.shewhart_step(L),
                                 n_state=0, lo=1.5, hi=8.0))
    # Page-CUSUM k=0.5 (δ_design/2)
    k = DELTA_DESIGN / 2.0
    out["Page"] = ("h", dict(make=lambda h: det.page_cusum_step(k, h),
                             n_state=2, lo=0.5, hi=14.0))
    # Winsorized CUSUM c=1.5 — опора k з марґіналу (i.i.d. cal)
    wu, wl = RC.robust_builder(RC.huber_score(1.5), n_cal=N_CAL_DET)(dist, DELTA_DESIGN)
    out["Winsor c=1.5"] = ("h", dict(
        make=lambda h: det.generic_cusum_step(wu.increment, wl.increment, h),
        n_state=2, lo=0.2, hi=14.0))
    # GSA-frac |z|^{0.5,1,1.5} — score з марґіналу (i.i.d. cal)
    gu = gsa.build_empirical(dist, +DELTA_DESIGN, "frac", 3, exponents=[0.5, 1.0, 1.5],
                             n_cal=N_CAL_DET, rng=np.random.default_rng(41))
    gl = gsa.build_empirical(dist, -DELTA_DESIGN, "frac", 3, exponents=[0.5, 1.0, 1.5],
                             n_cal=N_CAL_DET, rng=np.random.default_rng(42))
    out["GSA-frac"] = ("h", dict(
        make=lambda h: det.generic_cusum_step(gu.increment, gl.increment, h),
        n_state=2, lo=0.2, hi=14.0))
    return out


def iid_threshold(dist, label, kind, cfg):
    """i.i.d.-калібрований поріг/межа (на згладженому марґіналі — як у M6)."""
    if label == "Shewhart":
        L, a0 = det.calibrate_shewhart(dist, TARGET, N=N_CAL, max_steps=MAX_STEPS)
        return L, a0
    if label == "Page":
        h, a0 = det.calibrate_page(dist, DELTA_DESIGN / 2.0, TARGET,
                                   N=N_CAL, max_steps=MAX_STEPS)
        return h, a0
    # Winsor / GSA: загальний CUSUM-калібратор по порогу (i.i.d.)
    rng = np.random.default_rng(303)
    s_h0 = det.make_sampler(dist, 0.0)
    h, a0 = det.calibrate_threshold(s_h0, cfg["make"], cfg["n_state"],
                                    target_arl0=TARGET, N=N_CAL, max_steps=MAX_STEPS,
                                    lo=cfg["lo"], hi=cfg["hi"])
    return h, a0


def main():
    case = R.welllog_case()
    dist = case["dist"]
    base = np.asarray(dist.base, dtype=float)   # стандартизований in-control [0:1030)

    # ---- (1) автокореляція in-control сегмента ----
    x = base - base.mean()
    denom = float(np.dot(x, x))
    acf = {lag: float(np.dot(x[:-lag], x[lag:]) / denom) for lag in range(1, 6)}
    print(f"=== (1) Well-log in-control ACF (n={base.size}) ===")
    print("  " + "  ".join(f"lag{lag}={acf[lag]:+.3f}" for lag in range(1, 6)))

    # i.i.d. M6 reference (Table 7) для контексту
    iidref = pd.read_csv(os.path.join(OUT, "M6_arl_welllog.csv"))
    iidmap = {"Shewhart": "shewhart", "Page": "page",
              "Winsor c=1.5": "winsor", "GSA-frac": "welllog_frac"}

    dets = build_detectors(dist)

    # i.i.d.-калібровані пороги (одноразово; не залежать від L)
    iid_thr = {}
    print("\n=== i.i.d.-калібровані пороги (для ARL₀-дрейфу) ===")
    for label, (kind, cfg) in dets.items():
        thr, a0 = iid_threshold(dist, label, kind, cfg)
        iid_thr[label] = (kind, thr)
        print(f"  {label:14s} thr={thr:8.4f}  ARL0_iid={a0:6.1f}")

    rows = []
    for L in BLOCK_LENS:
        print(f"\n{'='*72}\nBLOCK LENGTH L={L}\n{'='*72}")
        for label, (kind, cfg) in dets.items():
            make = cfg["make"] if kind == "h" else (lambda v: det.shewhart_step(v))
            n_state = cfg["n_state"]

            # (a) ARL₀-дрейф: i.i.d.-поріг, оцінений під блок-bootstrap
            _, thr_iid = iid_thr[label]
            arl0_drift = arl0_blockboot(base, make, thr_iid, n_state,
                                        N=N_EVAL, L=L, max_steps=MAX_STEPS, seed=555 + L)

            # (b) перекалібровка порога ПІД залежністю до ARL₀=370
            thr_bb, a0_bb = calibrate_blockboot(base, make, n_state, L,
                                                lo=cfg["lo"], hi=cfg["hi"], seed=771 + L)
            step, ns = make(thr_bb)

            # (c) ARL₁ під блок-bootstrap з перекаліброваним порогом
            arl1 = {}
            for d in DELTAS:
                m, se = arl1_blockboot(base, step, ns, d, L)
                arl1[d] = m

            iid_arl1_d1 = float(iidref[(iidref.method == iidmap[label]) &
                                       (iidref.delta == 1.0)].arl.iloc[0])
            row = dict(L=L, method=label,
                       thr_iid=round(thr_iid, 4), arl0_iiThr_blockboot=round(arl0_drift, 1),
                       thr_blockboot=round(thr_bb, 4), arl0_blockboot=round(a0_bb, 1),
                       arl1_d05=round(arl1[0.5], 2), arl1_d10=round(arl1[1.0], 2),
                       arl1_d20=round(arl1[2.0], 2),
                       iid_M6_arl1_d10=round(iid_arl1_d1, 2))
            rows.append(row)
            print(f"  {label:14s} iidThr={thr_iid:7.3f} → ARL0_bb={arl0_drift:6.1f} (drift {arl0_drift/TARGET:.2f}×) | "
                  f"recal thr={thr_bb:7.3f} ARL0={a0_bb:5.1f} | "
                  f"ARL1[0.5/1/2]={arl1[0.5]:6.2f}/{arl1[1.0]:6.2f}/{arl1[2.0]:6.2f} "
                  f"(iid M6 @1.0={iid_arl1_d1:.2f})")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "blockboot_welllog_table.csv"), index=False)

    # ---- підсумок: чи виживає parity GSA≈Winsor і перевага над Page? ----
    print(f"\n{'='*72}\nПІДСУМОК: parity + перевага над Page під залежністю\n{'='*72}")
    for L in BLOCK_LENS:
        sub = table[table.L == L].set_index("method")
        for d, col in [(1.0, "arl1_d10")]:
            page = sub.loc["Page", col]
            win = sub.loc["Winsor c=1.5", col]
            gsa_ = sub.loc["GSA-frac", col]
            shew = sub.loc["Shewhart", col]
            print(f"  L={L:2d} δ=1.0: Shewhart={shew:6.2f}  Page={page:6.2f}  "
                  f"Winsor={win:6.2f}  GSA-frac={gsa_:6.2f}  | "
                  f"GSA/Winsor={gsa_/win:.2f}  Page/GSA={page/gsa_:.2f}")

    print(f"\nЗбережено: {os.path.abspath(os.path.join(OUT, 'blockboot_welllog_table.csv'))}")


if __name__ == "__main__":
    main()
