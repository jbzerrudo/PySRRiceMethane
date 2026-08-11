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

## Layout

    code/               the pipeline as run, numbered 0..11
                        (feature selection, VIF, Celsius conversion, PySR,
                        recurrence, LOSO, RF/NN benchmark, Sobol, figures)
    data/               derived analysis tables for Mase and Cheorwon
    results/pysr_runs/  per-seed Pareto fronts, equation reports, stage-8
                        CV and coefficient-stability outputs (txt/csv)
    results/nnrf/       RF and NN ceiling and residual benchmarks
    results/sobol/      variance-based sensitivity results
    results/loso/       leave-one-site-out validation of the pooled equation
    figures/            the paper's generated figures

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
  authors' own measurement, available on reasonable request; not part of this
  repository. Contact the corresponding author.

## Citation

See CITATION.cff. A Zenodo DOI is minted per tagged release.
