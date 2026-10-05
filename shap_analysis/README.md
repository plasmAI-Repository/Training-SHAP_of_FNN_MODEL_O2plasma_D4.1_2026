# SHAP Analysis with Gradient Explainer (Integrated Gradients)

This directory contains scripts for calculating and visualizing feature attributions (SHAP values) for the Multi-Layer Perceptron (MLP) model predicting the 10-point radial profile of Etch Rate.

This folder specifically utilizes **`shap.GradientExplainer`** to provide mathematically rigorous explanations with strict additivity guarantees.

---

## 1. How to Run the Code

### Prerequisites
Ensure you have the required Python environment installed. The following libraries are required:
```bash
pip install torch numpy pandas shap matplotlib
```

### Steps to Run
1. **Prepare the Model & Stats**:
   Ensure that the trained model weights (`trained_model_base.pth`) and dataset statistics (`stats_random_split.json`) are present in this folder.
   
2. **Execute the Script**:
   Run the following command in your terminal:
   ```bash
   python run_shap_analysis_gradient.py
   ```
   
3. **View the Results**:
   All outputs are saved to the **`results_gradient/`** subfolder:
   * **`shap_values_detailed.csv`**: A detailed, human-readable table containing inputs, predictions, ground truths, and SHAP attributions for each sample and radial point.
   * **`radial_feature_importance.png`**: Plot showing global feature importance across all 10 radial positions.
   * **`summary_beeswarm_point1_center.png` / `summary_beeswarm_point10_edge.png`**: Native SHAP summary plots highlighting the impact direction (positive/negative) of features at the center vs. edge.
   * **`dependence_point5_Pressure.png`**: Scatter plot showing non-linear relationships and interactions at the mid-radius.

---

## 2. Details

To explain why the model made a specific prediction, we look at the contribution of each of the 4 inputs: **Power, Pressure, Feed, and Vbias**.

### The Core Concept: Shapley Values
SHAP values explain how much each feature contributed to pushing the prediction away from a baseline "expected value" (average prediction):
$$\text{Prediction} = \text{Expected Value (Baseline)} + \sum_{i=1}^{4} \text{SHAP Value}_i$$

### What is GradientExplainer?
`shap.GradientExplainer` uses the **Integrated Gradients** method. It is designed specifically for differentiable models (like neural networks) and works by looking at the gradients of the outputs with respect to the inputs.

Instead of evaluating gradients at just one point (which fails to capture non-linear behaviors), it integrates the gradients along a path from a reference baseline to the actual sample.

### Step-by-Step Explanation

#### Step A: Establishing the Baseline (Background Set)
A neural network needs a baseline to compare against. The explainer is initialized with a **background dataset** (e.g., 100 random samples from the training set):
```python
explainer = shap.GradientExplainer(model, X_background)
```
The model's average prediction over these background samples defines the **Expected Value** (the starting baseline).

#### Step B: Path Integration (Riemann/Monte Carlo Sums)
For any given test sample $x$ and a background sample $x_{bg}$:
1. The explainer constructs a straight line path in the 4D input space from $x_{bg}$ to $x$.
2. It samples points along this path using the parameter `nsamples` (which defaults to `200`).
3. For each sampled point, it calculates the gradient of the model's output with respect to the inputs ($\frac{\partial f}{\partial x}$).
4. It averages these gradients along the path and multiplies them by the overall difference $(x - x_{bg})$:
   $$\text{Attribution} = (x - x_{bg}) \times \int_{0}^{1} \frac{\partial f(x_{bg} + \alpha(x - x_{bg}))}{\partial x} d\alpha$$
5. This process is repeated across all background samples and averaged to produce the final **SHAP values**.

---

## 3. Why use `GradientExplainer` over `DeepExplainer`?

* **Handling Non-Linear Activations:** Our MLP model uses non-linear activations (**SiLU** in hidden layers, **ELU** in the output layer). `DeepExplainer` (DeepSHAP) uses heuristic rules to approximate gradients through layers, which introduces numerical errors on non-linear shapes. This results in an **additivity discrepancy** (the SHAP values do not sum up exactly to the prediction change).
* **Strict Mathematical Guarantee:** Because `GradientExplainer` integrates actual computed gradients, it is mathematically guaranteed to satisfy the **Completeness (Additivity) Axiom**. The sum of the attributions will match the prediction change within a small integration tolerance (e.g. $<5\%$).
* **Controlling Accuracy:** You can reduce the numerical approximation error further by increasing `nsamples` (e.g. `nsamples=500` or `1000`). This draws more points along the path, reducing sampling noise at the expense of longer computation time.
