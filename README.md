# Sparse Bayesian Deep Learning for Stochastic Frontier Analysis

This repository contains the code and data analysis for the project **Sparse Bayesian Deep Learning for Stochastic Frontier Analysis**.

## Overview

This project develops Bayesian deep learning methods for stochastic frontier analysis (SFA), with a focus on flexible production frontier estimation, shape constraints, sparsity, and technical efficiency (TE) estimation.

The proposed methods are compared with traditional stochastic frontier and Bayesian models using simulation studies and real-world agricultural data.

## Methods

The repository includes implementations of:

- MLE Stochastic Frontier Model (MLE-SFM)
- Bayesian Linear Model (BLM)
- Bayesian Neural Network (BNN)
- Shape-constrained Bayesian Neural Network
- CReLU-based models
- Half-Normal prior
- Horseshoe prior

## Repository Structure

- `translog/` — Translog simulation experiments
- `ces/` — CES simulation experiments
- `real_data/` — Real-data analysis

## Evaluation

The analysis evaluates model performance using measures including:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Continuous Ranked Probability Score (CRPS)
- Posterior credible interval coverage
- Technical efficiency (TE) estimates
