# Implied Volatility Surface Reconstruction — NIFTY50 Options

This repository contains a machine learning pipeline to reconstruct implied volatility (IV) surfaces for NIFTY50 options. The dataset contains option strikes observed every 5 minutes over 13 trading days, with approximately 19% missingness concentrated in illiquid deep-OTM contracts and early-session bars.

## Approach

The approach utilizes a robust two-stage pipeline:

1. **Smile Interpolation + EWMA Smoothing:** 
   At each timestamp, the observed IVs across strikes form a volatility smile. Missing strikes are interpolated using a blend of cubic and linear splines. An exponentially-weighted moving average (EWMA) of recent values is applied to smooth out microstructure noise.

2. **Gradient Boosting Residual Correction:** 
   An ensemble of LightGBM and CatBoost models is trained on the residuals (observed IV minus spline prediction) to learn systematic errors made by the spline. Calls, puts, and expiry-day options are modelled separately, as their error distributions differ fundamentally.

The final predicted IV is a weighted blend of the spline baseline and the ML ensemble correction, tuned by moneyness zones.

## Project Structure

- `dataset.csv` - The original dataset with missing IV values.
- `iv_surface.ipynb` - The primary notebook containing data processing, feature engineering, and the modeling pipeline. Generates `submission.csv`.
- `submission-converter.ipynb` - A helper notebook that takes the filled dataset and converts it into the final submission format (`id`, `value`).

## Setup and Usage

1. Create a virtual environment and install the required packages:
   ```bash
   pip install pandas numpy scipy lightgbm catboost matplotlib seaborn
   ```
2. Run `iv_surface.ipynb` to train the models and generate the imputed volatility surface (`filled_dataset.csv`).
3. Run `submission-converter.ipynb` to format the outputs into `submission.csv` for evaluation.
