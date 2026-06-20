"""
detectors.py — послідовні детектори розладки та векторизована оцінка ARL.

Реалізовані карти (усі двосторонні, каліброві до спільного ARL0):
  - Shewhart(L):          сигнал коли |z_t| > L                       (карта базової статті)
  - EWMA(lambda, L):      W_t = λz_t+(1-λ)W_{t-1}; |W_t| > L·σ_W(t)
  - PageCUSUM(k, h):      табличний CUSUM, k=δ/2 (= гаусів оракул)
  - GSACUSUM(det±, h):    інкремент u(z)=Λ(z)-r з GSA-апроксимації LLR
  - OracleCUSUM(h):       точний LLR ell(z)=log p(z∓δ)-log p(z)  (межа Лордена)

Оцінка ARL — Монте-Карло по N паралельних послідовностях із компактизацією
(завершені послідовності вилучаються), що робить «хвіст» дешевим.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ============================================================
#  Загальний симулятор довжин серій (run lengths)
# ============================================================

def simulate_run_lengths(sampler, step_fn, n_state: int, N: int, max_steps: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Симулює N послідовностей. Повертає масив довжин серій (run length до сигналу).
    Цензуровані (без сигналу до max_steps) повертаються як max_steps.

    sampler(rng, size) -> ndarray розміру size зі спостереженнями monitored-статистики.
    step_fn(state, z, t) -> (new_state, crossed_bool) — state має форму (n_active, n_state).
    """
    active_idx = np.arange(N)
    state = np.zeros((N, n_state)) if n_state > 0 else np.zeros((N, 0))
    run_len = np.full(N, max_steps, dtype=np.int64)

    for t in range(1, max_steps + 1):
        m = active_idx.size
        if m == 0:
            break
        z = sampler(rng, m)
        state, crossed = step_fn(state, z, t)
        if crossed.any():
            signaled = active_idx[crossed]
            run_len[signaled] = t
            keep = ~crossed
            active_idx = active_idx[keep]
            state = state[keep]
    return run_len


# ============================================================
#  Детектори: кожен повертає (step_fn, n_state)
# ============================================================

def shewhart_step(L: float):
    def step(state, z, t):
        crossed = np.abs(z) > L
        return state, crossed
    return step, 0


def ewma_step(lam: float, L: float):
    ss = lam / (2.0 - lam)

    def step(state, z, t):
        W = state[:, 0]
        W = lam * z + (1.0 - lam) * W
        var_t = ss * (1.0 - (1.0 - lam) ** (2 * t))
        crossed = np.abs(W) > L * np.sqrt(var_t)
        state = W[:, None]
        return state, crossed
    return step, 1


def page_cusum_step(k: float, h: float):
    def step(state, z, t):
        cp = np.maximum(0.0, state[:, 0] + (z - k))
        cm = np.maximum(0.0, state[:, 1] + (-z - k))
        crossed = (cp > h) | (cm > h)
        return np.column_stack([cp, cm]), crossed
    return step, 2


def generic_cusum_step(inc_up, inc_lo, h: float):
    """Двосторонній CUSUM з довільними інкрементами (GSA або оракул)."""
    def step(state, z, t):
        cp = np.maximum(0.0, state[:, 0] + inc_up(z))
        cm = np.maximum(0.0, state[:, 1] + inc_lo(z))
        crossed = (cp > h) | (cm > h)
        return np.column_stack([cp, cm]), crossed
    return step, 2


# ============================================================
#  Калібрування порога під цільовий ARL0
# ============================================================

def estimate_arl(sampler, make_step, param, n_state, N, max_steps, rng):
    step, ns = make_step(param)
    rl = simulate_run_lengths(sampler, step, ns, N, max_steps, rng)
    return rl.mean(), rl


def calibrate_threshold(sampler_h0, make_step_from_h, n_state, target_arl0=370.0,
                        N=40_000, max_steps=8000, lo=0.1, hi=30.0, tol=0.01,
                        max_iter=40, seed=12345, verbose=False):
    """Бісекція по порогу h (монотонне зростання ARL0 з h) до target_arl0.
    make_step_from_h(h) -> (step_fn, n_state)."""
    def arl0(h, seed_):
        rng = np.random.default_rng(seed_)
        step, ns = make_step_from_h(h)
        rl = simulate_run_lengths(sampler_h0, step, ns, N, max_steps, rng)
        return rl.mean()

    # розширення верхньої межі за потреби
    a0_hi = arl0(hi, seed)
    it = 0
    while a0_hi < target_arl0 and hi < 200 and it < 12:
        hi *= 1.6
        a0_hi = arl0(hi, seed)
        it += 1

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        a0 = arl0(mid, seed)
        if verbose:
            print(f"   h={mid:.4f} -> ARL0={a0:.1f}")
        if abs(a0 - target_arl0) / target_arl0 < tol:
            return mid, a0
        if a0 < target_arl0:
            lo = mid
        else:
            hi = mid
    return mid, a0


# ============================================================
#  Зручні обгортки під конкретні детектори
# ============================================================

def make_sampler(dist, delta):
    def sampler(rng, size):
        return dist.sample(rng, size, delta=delta)
    return sampler


def calibrate_shewhart(dist, target_arl0=370.0, **kw):
    s_h0 = make_sampler(dist, 0.0)
    L, a0 = calibrate_threshold(s_h0, lambda L: shewhart_step(L), 0,
                                target_arl0=target_arl0, lo=1.5, hi=6.0, **kw)
    return L, a0


def calibrate_ewma(dist, lam=0.2, target_arl0=370.0, **kw):
    s_h0 = make_sampler(dist, 0.0)
    L, a0 = calibrate_threshold(s_h0, lambda L: ewma_step(lam, L), 1,
                                target_arl0=target_arl0, lo=1.5, hi=4.5, **kw)
    return L, a0


def calibrate_page(dist, k, target_arl0=370.0, **kw):
    s_h0 = make_sampler(dist, 0.0)
    h, a0 = calibrate_threshold(s_h0, lambda h: page_cusum_step(k, h), 2,
                                target_arl0=target_arl0, lo=0.5, hi=12.0, **kw)
    return h, a0


def calibrate_gsa(dist, det_up, det_lo, target_arl0=370.0, **kw):
    s_h0 = make_sampler(dist, 0.0)
    iu, il = det_up.increment, det_lo.increment
    h, a0 = calibrate_threshold(s_h0, lambda h: generic_cusum_step(iu, il, h), 2,
                                target_arl0=target_arl0, lo=0.2, hi=12.0, **kw)
    return h, a0


def calibrate_oracle(dist, delta, target_arl0=370.0, **kw):
    s_h0 = make_sampler(dist, 0.0)
    inc_up = lambda z: dist.exact_llr(z, delta)
    inc_lo = lambda z: dist.exact_llr(z, -delta)
    h, a0 = calibrate_threshold(s_h0, lambda h: generic_cusum_step(inc_up, inc_lo, h), 2,
                                target_arl0=target_arl0, lo=0.2, hi=20.0, **kw)
    return h, a0


def arl_curve(dist, deltas, make_step, n_state, N=40_000, max_steps=8000, seed=777):
    """ARL(δ) для набору справжніх зсувів deltas (zero-state: зсув з t=1)."""
    out = []
    for d in deltas:
        rng = np.random.default_rng(seed + int(round(d * 1000)))
        sampler = make_sampler(dist, d)
        step, ns = make_step
        rl = simulate_run_lengths(sampler, step, ns, N, max_steps, rng)
        se = rl.std(ddof=1) / np.sqrt(len(rl))
        out.append((d, rl.mean(), se))
    return out


if __name__ == "__main__":
    import distributions as D
    rng = np.random.default_rng(42)
    g = D.Gaussian()

    # Shewhart ARL0 має бути ~370 при L=3
    s0 = make_sampler(g, 0.0)
    step, ns = shewhart_step(3.0)
    rl = simulate_run_lengths(s0, step, ns, 100_000, 8000, np.random.default_rng(1))
    print(f"Shewhart L=3: ARL0={rl.mean():.1f} (теор. 370.4)")

    # Shewhart ARL1 для δ=1,2,3 -> 1/P
    for d in (1, 2, 3):
        sd = make_sampler(g, float(d))
        rl = simulate_run_lengths(sd, *shewhart_step(3.0)[:1], 0, 100_000, 8000,
                                  np.random.default_rng(d))
        from scipy import stats as st
        p = 1 - (st.norm.cdf(3 - d) - st.norm.cdf(-3 - d))
        print(f"Shewhart δ={d}: ARL1={rl.mean():.2f} (теор. 1/P={1/p:.2f}, P={p:.4f})")
