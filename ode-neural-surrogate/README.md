# ODE Neural Surrogate for Biochemical Reaction Dynamics

Neural network surrogate for ODE-based biochemical simulations with integrated mass-balance and toxicity analysis.

## Overview

This project combines mechanistic ODE modeling with deep learning to approximate biochemical reaction dynamics. An ODE system is used to generate time-course simulation data across different `Keq2` and `f1` values. A PyTorch neural network is then trained as a fast surrogate model to predict the system state.

## Key Features

- ODE simulation of a biochemical reaction network
- Parameter sweep over `Keq2` and HAME feed rate `f1`
- Automatic dataset generation from numerical simulations
- PyTorch MLP surrogate model for fast prediction
- Mass-balance checks for HAME and alanine
- Toxicity threshold flagging for HAME
- ODE-vs-NN error analysis and heatmap visualization

## Repository Structure

```text
src/
  config.py       # constants, parameters, input/output columns
  ode_model.py    # ODE system and simulation function
  dataset.py      # dataset generation and final-time extraction
  nn_model.py     # PyTorch MLP model
  train.py        # full training pipeline
  evaluate.py     # metrics, prediction, comparison tables
  plots.py        # diagnostic and comparison plots

notebooks/
  LYN_29.APRIL.ipynb  # original notebook
outputs/              # generated CSVs and plots, ignored by Git
models/               # trained weights/scalers, ignored by Git
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Full Pipeline

```bash
python -m src.train --output-dir outputs --epochs 80
```

For a quick test without plots:

```bash
python -m src.train --output-dir outputs --epochs 5 --skip-plots
```

## Outputs

The pipeline generates:

- `kinetic_dataset_Keq2_f1.csv`
- `final_ODE_results.csv`
- `final_ODE_results_with_toxicity.csv`
- `training_history.csv`
- `test_metrics.csv`
- `full_ODE_NN_error_all_timepoints.csv`
- `final_comparison_ODE_NN_errors.csv`
- ODE-vs-NN trajectory plots
- heatmaps for ODE predictions, NN predictions, and errors

## Method Summary

1. Simulate ODE trajectories for a grid of `Keq2` and `f1` values.
2. Compute mass-balance diagnostics and final-state summaries.
3. Train a neural network surrogate using simulation inputs and ODE states.
4. Evaluate the surrogate using R² and RMSE.
5. Compare ODE and NN predictions across all time points and final states.

## Git Notes

Generated datasets, model files, and plots are intentionally ignored by Git. Commit only the source code, notebook, README, and configuration files.
