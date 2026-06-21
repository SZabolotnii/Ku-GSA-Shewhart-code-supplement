# Ku-GSA-Shewhart — code supplement

Reproducible code for the paper

> **Shape-Adaptive Control Charts for Non-Gaussian Processes via Kunchenko Stochastic Polynomials**

This supplement reproduces every numerical result, table, and figure in the paper.

## What the method does

The classical Shewhart control chart is insensitive to small sustained mean shifts and its
nominal false-alarm rate (`ARL₀`) is distorted under non-Gaussian data. This work approximates
the optimal sequential change-point statistic — the log-likelihood ratio (LLR) of a mean shift —
by a finite **Kunchenko stochastic polynomial** `Λ(z) = Σ Kᵢ φᵢ(z)`, with coefficients from the
normal system `F·K = Y` and a basis that adapts to the distribution shape (power `zⁱ` for
asymmetry, fractional sign-preserving `sign(z)|z|^p` for heavy tails). All detectors are calibrated
by Monte Carlo to a common `ARL₀ = 370` and compared by detection speed (`ARL₁`).

The GSA-LLR apparatus itself (the `F·K=Y` system, the information functional `J(s)=KᵀY`, the
threshold theory) is developed in the companion change-point-detection paper
[Zabolotnii, *Generalized Stochastic Approximation of the Log-Likelihood Ratio for Robust Sequential
Change-Point Detection*, arXiv:2605.23419]. **This** paper and supplement are the **statistical
process control** application: the Shewhart-chart instantiation, PMM-based control-limit
calibration, a cumulant shape-monitoring chart, validation on real measurement series, and a
control-chart-native benchmark against robust CUSUMs, run rules, steady-state and Phase-I ARL.

## Requirements

```
python >= 3.10
numpy, scipy, pandas, matplotlib
```

```bash
pip install -r requirements.txt
```

## Quick start

All experiments are run from the `src/` directory; each writes its CSV tables and PNG figures to
`../results/` (created on first run). The real datasets are downloaded and cached automatically on
the first `exp_m6_realdata.py` run.

```bash
cd src
python3 exp_m1.py            # M1: Shewhart baseline
python3 exp_m2.py            # M2: Gaussian unification of cumulative charts
python3 exp_m3_detection.py  # M3A: non-Gaussian gain + robust competitors
python3 exp_m3_pmm.py        # M3B: limit calibration + PMM2 center line
python3 exp_m5_cumulant.py   # M5: cumulant shape-monitoring chart
python3 exp_m6_realdata.py   # M6: real data (well-log, Nile) + competitors
python3 exp_runsrules.py     # Western Electric run-rules comparison
python3 exp_steadystate.py   # steady-state ARL
python3 exp_phase1.py        # estimated Phase-I parameters
python3 exp_winsor_tune.py   # best-c Winsorized robustness check
python3 exp_are.py           # asymptotic relative efficiency (ARE)
python3 exp_opmm_probe.py    # oPMM-style optimized-exponent probe
python3 exp_cf_probe.py      # moment-free CF basis on alpha-stable
python3 exp_ewma_tune.py     # EWMA lambda-tuning robustness check
python3 exp_blockboot.py     # block-bootstrap dependence check (well-log)
python3 exp_ewma_gsa_probe.py # GSA-EWMA hybrid: GSA score in an EWMA accumulator (§5.3)
```

All runs are **deterministic** (fixed `numpy.random.default_rng` seeds), so the printed numbers
reproduce the paper to Monte-Carlo precision.

## Experiment → paper map

| Script | Paper section | Tables / Figures |
|---|---|---|
| `exp_m1.py` | M1 (§5.1) | Table 1; Figs 1–2 |
| `exp_m2.py` | M2 (§5.2) | Table 2; Figs 3–4 |
| `exp_m3_detection.py` | M3A (§5.3) | Table 3; Figs 5–7 |
| `exp_m3_pmm.py` | M3B (§5.4) | Table 4; Figs 8–9 |
| `exp_m5_cumulant.py` | M5 (§5.5) | Table 5; Fig 10 |
| `exp_m6_realdata.py` | M6 (§5.6) | Tables 6–8; Figs 11–15 |
| `exp_runsrules.py` | §5.7 | Table 10 |
| `exp_steadystate.py` | §5.8 | Table 11 |
| `exp_phase1.py` | §5.9 | Table 12 |
| `exp_winsor_tune.py` | §5.3 (robustness) | best-`c` Winsorized check |
| `exp_are.py` | §6.3 | Table 9 (ARE) |
| `exp_opmm_probe.py` | §6.3 | oPMM-style basis probe |
| `exp_cf_probe.py` | §6.3, §6.6 | CF basis on α-stable |
| `exp_ewma_tune.py` | §5.2 (robustness) | EWMA λ-tuning check |
| `exp_blockboot.py` | §5.6 (robustness) | block-bootstrap dependence check |
| `exp_ewma_gsa_probe.py` | §5.3 (follow-up) | GSA-EWMA hybrid: ZS/SS vs linear EWMA & GSA-CUSUM |

## Module layout (`src/`)

| Module | Purpose |
|---|---|
| `distributions.py` | Standardized non-Gaussian generators (Gaussian, Laplace, Student-t, skew-normal, two-piece normal); additive mean shift; exact log-densities for the oracle |
| `gsa.py` | Kunchenko GSA approximation of the LLR: power/fractional bases, normal system `F·K=Y`, statistic `Λ`, information functional `J(s)` |
| `detectors.py` | Sequential detectors (Shewhart, EWMA, Page-CUSUM, GSA-CUSUM, oracle) + vectorized Monte-Carlo ARL with compaction + threshold calibration by bisection |
| `robust_cusum.py` | Robust competitors: Winsorized/Huber-clip CUSUM and sign CUSUM |
| `benchmark.py` | Unified ARL benchmarker (calibrate to common `ARL₀`, then `ARL₁(δ)` curve) |
| `pmm.py` | PMM2 location estimator; variance-reduction coefficients `g₂`, `g₃` |
| `realdata.py` | Loaders for the real series (well-log, Nile) + smoothed-bootstrap empirical distribution |
| `plotting.py` | Unified publication figure style |
| `exp_*.py` | Individual experiment drivers (see map above) |

## Data availability

- **well-log** — 4050 nuclear-magnetic-resonance probe responses during drilling, from the
  `fastcpd` R package data; downloaded and cached automatically.
- **Nile** — annual flow of the Nile at Aswan, 1871–1970 (`datasets::Nile`), via the Rdatasets
  mirror; downloaded and cached automatically.

Both are standard public change-point benchmarks. No data files are bundled with this repository.

## Citation

If you use this code, please cite the paper (and the companion method paper for the GSA-LLR
apparatus):

```bibtex
@article{ZabolotniiGSAShewhart,
  author  = {Zabolotnii, Serhii},
  title   = {Shape-Adaptive Control Charts for Non-Gaussian Processes via Kunchenko Stochastic Polynomials},
  year    = {2026},
  note    = {Manuscript}
}

@article{ZabolotniiGSALLR,
  author  = {Zabolotnii, Serhii},
  title   = {Generalized Stochastic Approximation of the Log-Likelihood Ratio
             for Robust Sequential Change-Point Detection},
  year    = {2026},
  journal = {arXiv preprint arXiv:2605.23419}
}
```

## License

MIT — see [LICENSE](LICENSE).
