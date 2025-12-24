# VCA SPARC Rotation Curve Analysis - README

## Overview
This repository contains the complete analysis pipeline for testing the "Velocity-Coupled Acceleration" (VCA) phenomenological model against SPARC galaxy rotation curves.

**Citation**: If you use this code or data, please cite:
- Lelli et al. (2016) for SPARC data: [AJ 152, 157](https://doi.org/10.3847/0004-6256/152/6/157)
- [Your paper citation - to be added upon publication]

**Zenodo DOI**: [To be assigned upon archival]

## Reproduction (Submission Grade)

To reproduce the full analysis presented in the paper, run the following commands in order:

### 1. Baseline Fits & Sensitivity Analysis
Run the least-squares fitting pipeline for multiple error floors ($\sigma_0 \in \{0, 3, 5, 8\}$ km/s):
```bash
python3 -m src.analysis
```
This generates `fit_results_all_models.csv`.

### 2. Sensitivity Artifacts
Generate the sensitivity summary table and histograms:
```bash
python3 src/generate_sensitivity_analysis.py
```
Outputs: `results_comparison/sensitivity_table.tex`, `results_comparison/sensitivity_histograms.pdf`.

### 3. MCMC Analysis (Parallel)
Run the full Bayesian inference pipeline (WARNING: computationally intensive):
```bash
python3 -m src.analysis_mcmc_refined_parallel
```
This generates `results_mcmc_refined/mcmc_summary_all_galaxies.csv` and diagnostic CSVs.

### 4. Post-Processing & Plots
Generate all final manuscript figures and tables:
```bash
# Identifiability Analysis (3-Tier classification)
python3 src/analysis_identifiability.py

# Model Comparison (AIC)
python3 src/analysis_model_comparison.py

# RAR Scatter Analysis
python3 -m src.analysis_rar

# Final Diagnostic & Coverage Plots
python3 src/generate_final_plots.py
```

### 5. Compile Manuscript
```bash
pdflatex paper.tex
# content of refs.bib is required for full bibliography
pdflatex paper.tex
```

### Outputs
- `fit_results_all_models.csv`: Full sample sensitivity results
- `results_mcmc_refined/`: MCMC posteriors and diagnostics
- `figures/`, `results_comparison/`, `results_rar/`: Publication figures
- `paper.pdf`: Compiled manuscript

## Repository Structure
```
Rotmod_LTG/
├── run_pipeline.py          # Master reproducibility script
├── requirements.txt         # Pinned dependencies
├── README.md               # This file
├── data/                   # SPARC rotation curve files
│   └── *_rotmod.dat
├── src/
│   ├── data_loader.py      # SPARC file parser
│   ├── models_extended.py  # All halo models + VCA
│   ├── mcmc_fitting.py     # Bayesian inference with emcee
│   ├── cross_validation.py # Predictive testing
│   ├── analysis.py         # Least-squares baseline
│   └── analysis_mcmc.py    # MCMC driver
├── results_mcmc/           # MCMC outputs
├── figures/                # All plots
└── paper/
    ├── paper.tex
    └── refs.bib
```

## Models Implemented
1. **VCA (Proposed)**: Velocity-coupled acceleration with saturating coupling
2. **NFW**: Navarro-Frenk-White cuspy halo
3. **Burkert**: Cored halo profile
4. **Einasto**: 3-parameter generalized profile
5. **Pseudo-Isothermal**: Simple cored sphere
6. **MOND/RAR**: Empirical radial acceleration relation
7. **Baryons-only**: No dark component (baseline)

## Methodology
- **Least-squares**: Fast optimization for full sample screening
- **MCMC**: Bayesian inference with `emcee` for uncertainty quantification
- **Cross-validation**: Fit inner 70% radii, predict outer 30%
- **Error model**: Systematic error floor σ₀ = 5 km/s (tested 0, 3, 5, 8)

## System Requirements
- Python 3.8+
- ~4 GB RAM
- ~2 hours compute time for full MCMC (parallelizable)

## Data Availability
SPARC rotation curves are publicly available at:
http://astroweb.case.edu/SPARC/

Place `*_rotmod.dat` files in the `data/` directory.

## License
[To be specified - suggest MIT or GPL-3.0]

## Contact
[Author contact information]

## Acknowledgments
This research uses data from the SPARC database (Lelli et al. 2016).
MCMC inference uses the `emcee` package (Foreman-Mackey et al. 2013).
