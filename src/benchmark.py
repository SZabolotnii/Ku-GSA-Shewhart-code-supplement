"""
benchmark.py — уніфікований ARL-бенчмаркер для набору детекторів.

Калібрує кожен детектор до спільного ARL0 на заданому розподілі, потім оцінює
ARL1(δ) на сітці справжніх зсувів (zero-state). Повертає tidy-таблицю результатів
і словник калібрування. Спільний для M2 (Гаусс) і M3 (негаусівські).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import detectors as det
import gsa


def _arl_curve(dist, deltas, step, ns, N, max_steps, seed):
    out = []
    for d in deltas:
        rng = np.random.default_rng(seed + int(round(d * 1000)))
        sampler = det.make_sampler(dist, float(d))
        rl = det.simulate_run_lengths(sampler, step, ns, N, max_steps, rng)
        se = rl.std(ddof=1) / np.sqrt(len(rl))
        out.append((float(d), float(rl.mean()), float(se)))
    return out


def gaussian_gsa_builder(s):
    def build(dist, delta_design):
        return (gsa.build_gaussian_poly(delta_design, s),
                gsa.build_gaussian_poly(-delta_design, s))
    return build


def empirical_gsa_builder(basis, s, exponents=None, n_cal=400_000, winsor=0.0, seed=99):
    def build(dist, delta_design):
        rng = np.random.default_rng(seed)
        up = gsa.build_empirical(dist, delta_design, basis, s, exponents,
                                 n_cal=n_cal, rng=rng, winsor=winsor)
        rng2 = np.random.default_rng(seed + 1)
        lo = gsa.build_empirical(dist, -delta_design, basis, s, exponents,
                                 n_cal=n_cal, rng=rng2, winsor=winsor)
        return up, lo
    return build


def run_benchmark(dist, deltas, delta_design=1.0, gsa_specs=(),
                  include=("shewhart", "ewma", "page", "oracle"),
                  ewma_lambda=0.2, target_arl0=370.0,
                  N_cal=40_000, N_eval=60_000, max_steps=8000, seed=2024,
                  verbose=True):
    """gsa_specs: список (key, label, builder) де builder(dist, delta_design)->(det_up,det_lo)."""
    rows = []
    calib = {}

    def log(*a):
        if verbose:
            print(*a)

    # ---- Shewhart ----
    if "shewhart" in include:
        L, a0 = det.calibrate_shewhart(dist, target_arl0, N=N_cal, max_steps=max_steps)
        step, ns = det.shewhart_step(L)
        calib["shewhart"] = dict(label="Шухарт ±Lσ", thr=L, arl0=a0, plot_key="shewhart")
        log(f"  Shewhart:  L={L:.3f}  ARL0={a0:.1f}")
        for d, arl, se in _arl_curve(dist, deltas, step, ns, N_eval, max_steps, seed):
            rows.append(dict(method="shewhart", label="Шухарт ±Lσ", delta=d, arl=arl, se=se))

    # ---- EWMA ----
    if "ewma" in include:
        Le, a0 = det.calibrate_ewma(dist, ewma_lambda, target_arl0, N=N_cal, max_steps=max_steps)
        step, ns = det.ewma_step(ewma_lambda, Le)
        calib["ewma"] = dict(label=f"EWMA (λ={ewma_lambda})", thr=Le, arl0=a0, plot_key="ewma")
        log(f"  EWMA:      L={Le:.3f}  ARL0={a0:.1f}")
        for d, arl, se in _arl_curve(dist, deltas, step, ns, N_eval, max_steps, seed):
            rows.append(dict(method="ewma", label=f"EWMA (λ={ewma_lambda})", delta=d, arl=arl, se=se))

    # ---- Page-CUSUM (k=δ_design/2) ----
    if "page" in include:
        k = delta_design / 2.0
        h, a0 = det.calibrate_page(dist, k, target_arl0, N=N_cal, max_steps=max_steps)
        step, ns = det.page_cusum_step(k, h)
        calib["page"] = dict(label=f"Page-CUSUM (k={k:g})", thr=h, arl0=a0, plot_key="page")
        log(f"  Page:      k={k:g} h={h:.3f}  ARL0={a0:.1f}")
        for d, arl, se in _arl_curve(dist, deltas, step, ns, N_eval, max_steps, seed):
            rows.append(dict(method="page", label=f"Page-CUSUM (k={k:g})", delta=d, arl=arl, se=se))

    # ---- Oracle-CUSUM (точний LLR) ----
    if "oracle" in include:
        h, a0 = det.calibrate_oracle(dist, delta_design, target_arl0, N=N_cal, max_steps=max_steps)
        inc_up = lambda z: dist.exact_llr(z, delta_design)
        inc_lo = lambda z: dist.exact_llr(z, -delta_design)
        step, ns = det.generic_cusum_step(inc_up, inc_lo, h)
        calib["oracle"] = dict(label="Оракул (точний LLR)", thr=h, arl0=a0, plot_key="oracle")
        log(f"  Oracle:    h={h:.3f}  ARL0={a0:.1f}")
        for d, arl, se in _arl_curve(dist, deltas, step, ns, N_eval, max_steps, seed):
            rows.append(dict(method="oracle", label="Оракул (точний LLR)", delta=d, arl=arl, se=se))

    # ---- GSA-CUSUM ----
    for key, label, builder, plot_key in gsa_specs:
        det_up, det_lo = builder(dist, delta_design)
        h, a0 = det.calibrate_gsa(dist, det_up, det_lo, target_arl0, N=N_cal, max_steps=max_steps)
        step, ns = det.generic_cusum_step(det_up.increment, det_lo.increment, h)
        calib[key] = dict(label=label, thr=h, arl0=a0, plot_key=plot_key,
                          J=det_up.J, condF=det_up.cond_F, K=det_up.K.tolist())
        log(f"  {key:9s} h={h:.3f}  ARL0={a0:.1f}  J(s)={det_up.J:.4f}  cond(F)={det_up.cond_F:.1f}")
        for d, arl, se in _arl_curve(dist, deltas, step, ns, N_eval, max_steps, seed):
            rows.append(dict(method=key, label=label, delta=d, arl=arl, se=se))

    return pd.DataFrame(rows), calib
