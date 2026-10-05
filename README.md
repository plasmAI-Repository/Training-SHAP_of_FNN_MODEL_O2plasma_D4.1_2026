# Training and SHAP Analysis of FNN Model for O2 Plasma

This repository contains the training codes, SHAP analysis gradients, and the full ETH O2 dataset for the v31 Feedforward Neural Network (FNN/MLP) model.

## Overview
This repository is designed for automated and agentic AI systems to run model training, testing, and SHAP-based feature importance analysis for an O2 plasma dataset. The code is modular, with separate scripts for data loading, model definition, training, and interpretation.

## Repository Structure

- `main_code_training.py`: The entry point for single-seed training.
- `train.py`: The core training loop and evaluation code.
- `run_multiple_seeds.py`: Script to run experiments across multiple random seeds for statistical validation.
- `data_loader.py` & `data_split.py`: Utilities for loading and splitting the dataset.
- `model.py` & `loss.py`: Definition of the MLP model and the custom loss functions.
- `config.yaml`: Configuration file for hyperparameters.
- `v3_3_1_data/`: Contains the full ETH O2 dataset (`v3_3_1_O2_eth_dataset_full.txt`) and pre-split train/test files.
- `shap_analysis/`: Contains scripts for SHAP analysis, specifically `run_shap_analysis_gradient.py` for gradient-based SHAP and `plot_mlp_importance.py` for visualization.

## Guidelines for Agentic AI

If you are an AI agent instructed to run the training or SHAP analysis, follow these steps:

### 1. Model Training
To train the model, you can use the `main_code_training.py` script. The configuration is managed in `config.yaml`.
**Command:**
```powershell
python main_code_training.py
```
This will read the hyperparameters from `config.yaml`, load the dataset from `v3_3_1_data/v3_3_1_O2_eth_dataset_full.txt`, and train the MLP. 

To evaluate stability across multiple initializations, run:
```powershell
python run_multiple_seeds.py
```
This will train the model using different random seeds and aggregate the results.

### 2. Dataset Setup
Ensure that the `config.yaml` points to the correct dataset path. By default, it uses the dataset located at `v3_3_1_data/v3_3_1_O2_eth_dataset_full.txt`. The full ETH O2 dataset is provided in this repository.

### 3. SHAP Analysis (Gradient SHAP)
After training the model (or if a trained model checkpoint exists), you can perform feature importance analysis using Gradient SHAP.

**Navigate to the SHAP directory:**
```powershell
cd shap_analysis
```

**Run Gradient SHAP:**
```powershell
python run_shap_analysis_gradient.py
```
This script will compute the SHAP values using the `GradientExplainer` from the `shap` library.

**Plotting the results:**
```powershell
python plot_mlp_importance.py
```
This will generate the importance plots based on the computed SHAP values, saving them in the `results_gradient` directory.

### 4. Dependencies
Ensure the Python environment has the following packages installed:
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `matplotlib`
- `pyyaml`

You can use `pip install` to install any missing dependencies before executing the scripts.

---
**Note to AI Agents:** When modifying the architecture or data paths, update `config.yaml` and verify that the input dimensions in `model.py` match the feature count of the loaded dataset.
