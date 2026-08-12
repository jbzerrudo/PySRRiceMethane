# PySRRiceMethane

Analysis code, derived data and results for:

> Zerrudo et al. (in preparation). *Symbolic Regression Reveals Site-specific
> Functional Forms of Methane Flux Across Three Asian Rice Paddies Under
> Contrasting Water Management.* Target journal: Agricultural and Forest
> Meteorology.

Half-hourly eddy-covariance CH4 flux at three rice paddies spanning a
water-management gradient (Mase, continuous flooding; Cheorwon, intermittent
flooding; IRRI, alternate wetting and drying), plus a pooled arm on intensive
water variables. Symbolic regression (PySR) under twelve seeds, two selection
criteria declared in advance, day-grouped cross-validation with a
coefficient-stability screen, RF/NN benchmarks, Sobol sensitivity, and
leave-one-site-out validation of the pooled equation.

## Full Repository Layout

The project is organized into the following directories:

**`code/`** — The complete processing pipeline, numbered 0 through 11: feature selection, VIF filtering, Celsius conversion, PySR search, form recurrence, LOSO, RF/NN benchmark, Sobol sensitivity, and figures. Also contains the site-specific flux quality-control scripts (`clean_KORCRK_flux_papale_hampel.py`, `clean_PHL_flux_light.py`) described in §2.1 of the paper.

**`data/`** — Derived analysis tables for all three sites (Mase, Cheorwon, IRRI) and the pooled arm. The IRRI tables are published with the co-authors' agreement.

**`results/gamrf/`** — GAM–random-forest union outputs of the feature-selection cascade, including the per-predictor permutation importances behind Table 3 and Table 4.

**`results/collincheck/`** — Collinearity screening outputs, including the per-predictor variance inflation factors used for the VIF ≤ 5 filter.

**`results/vif10_companion/`** — Companion PySR runs at the more permissive VIF ≤ 10 threshold, for Mase and the pooled arm. These are the runs behind the choice of VIF ≤ 5 in §2.3 of the paper.

**`results/pysr_runs/`** — Per-seed outputs: Pareto fronts, equation reports, and the cross-validation and coefficient-stability outputs (txt/csv) behind the selection rules.

**`results/nnrf/`** — Random-forest and neural-network ceiling and residual benchmarks (Table 9).

**`results/sobol/`** — Variance-based sensitivity analysis results (Table 10).

**`results/loso/`** — Leave-one-site-out validation outcomes for the pooled equation (Table 8).

**`figures/`** — Generated figures for the paper.

## Reproducing

Python 3.11+, PySR, scikit-learn, pyGAM, SciPy, numpy, pandas, matplotlib.
Scripts carry their own usage headers; the order is the numbering. PySR runs
are not bit-reproducible under multithreading (Tonda, 2025): robustness is
established across the twelve seeds, not by replaying one.

## Data and licensing

- **Code**: MIT (see LICENSE).
- **data/JPN_retvars_pass2_C.csv**, **data/KOR_retvars_pass2_C.csv**: derived
  from the FLUXNET-CH4 community product, CC-BY-4.0. Original records:
  Mase (JP-MSE) doi:10.18140/FLX/1669647, Cheorwon (SK-CRK)
  doi:10.18140/FLX/1669649. Cite the originals when using these tables.
- **IRRI 2016 record and all tables derived from it (PHL, POOLED)**: the
  authors' own measurements, available on reasonable request; not part of this
  repository. Contact the corresponding author.
 - Water depth and depth-derived predictors are in cm; the paper's fitted coefficients assume this convention.

## Citation

If you use this code or data, cite the paper and this repository. Full author list in CITATION.cff.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21893634.svg)](https://doi.org/10.5281/zenodo.21893634)

> Zerrudo, J., et al. (2026). PySRRiceMethane: symbolic regression pipeline and results
> for rice-paddy CH4 flux equations across three Asian sites (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.21893634
