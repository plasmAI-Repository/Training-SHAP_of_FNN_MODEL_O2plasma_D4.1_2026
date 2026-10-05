import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap

# Add the main workspace to the path to import Model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
from model import Model

def main():
    print("Initializing SHAP analysis...")
    
    # 1. Directory and file paths
    data_dir = os.path.join(base_dir, "v3_3_1_data")
    shap_dir = os.path.join(base_dir, "shap_analysis")
    
    model_path = os.path.join(base_dir, "O2", "baseline_seed_3123", "trained_model_base.pth")
    stats_path = os.path.join(base_dir, "dataset_stats.json")

    train_dataset_path = os.path.join(base_dir, "v3_3_1_data", "train_data_v3_3_1_O2.txt")
    test_dataset_path = os.path.join(base_dir, "v3_3_1_data", "test_data_v3_3_1_O2.txt")
    
    # 2. Load Train and Test Sets directly
    df_train = pd.read_csv(train_dataset_path, sep="\t")
    df_test = pd.read_csv(test_dataset_path, sep="\t")
    
    input_cols = ["Power", "Pressure", "Feed", "Vbias"]
    output_cols = [f"Point{i}_EtchRate" for i in range(1, 11)]
    
    print(f"Loaded datasets: {len(df_train)} train, {len(df_test)} test.")
    
    # 3. Load scaling parameters
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # 4. Standardize inputs
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    for col in input_cols + output_cols:
        col_min = stats[col]["min"]
        col_max = stats[col]["max"]
        df_train_scaled[col] = (df_train[col] - col_min) / (col_max - col_min)
        df_test_scaled[col] = (df_test[col] - col_min) / (col_max - col_min)
        
    X_train = df_train_scaled[input_cols].to_numpy(dtype=np.float32)
    X_test = df_test_scaled[input_cols].to_numpy(dtype=np.float32)
    
    # 5. Load the trained MLP Model
    model = Model(h1=20, num_layers=2, freeze_layers=[], output_size=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    
    # 6. Select Background/Reference Dataset (100 random samples from training set)
    bg_rng = np.random.RandomState(42)
    bg_idx = bg_rng.choice(len(X_train), size=100, replace=False)
    X_background = torch.tensor(X_train[bg_idx])
    X_test_tensor = torch.tensor(X_test)
    
    # 7. Initialize shap.GradientExplainer
    print("Calculating SHAP values using GradientExplainer...")
    explainer = shap.GradientExplainer(model, X_background)
    
    # Compute SHAP values for all test samples
    # For a multi-output PyTorch model, shap_values returns a list of length 10
    # each element of list is a numpy array of shape (num_samples, num_features)
    # 'nsamples' parameter controls the number of steps in the path integration
    shap_values_list = explainer.shap_values(X_test_tensor, nsamples=400) 
    
    # Debug print to inspect shapes and types
    print("--- Debug Information ---")
    print("Type of shap_values_list:", type(shap_values_list))
    if isinstance(shap_values_list, list):
        print("Length of list:", len(shap_values_list))
        print("Shape of first element:", shap_values_list[0].shape)
    elif hasattr(shap_values_list, "shape"):
        print("Shape of array:", shap_values_list.shape)
    
    # Calculate expected value as mean of background predictions (shape (10,))
    with torch.no_grad():
        expected_val_np = model(X_background).numpy().mean(axis=0)
    
    print("Baseline (mean background prediction) shape:", expected_val_np.shape)
    expected_values = expected_val_np
    
    print("Shape of explainer.expected_value:", expected_val_np.shape)
    print("-------------------------")
    
    # Convert list of arrays to a single numpy array of shape (num_samples, num_features, output_size)
    shap_values_arr = shap_values_list
    
    # --- Custom Additivity Check for Printing ---
    with torch.no_grad():
        model_predictions = model(X_test_tensor).numpy() # shape: (num_samples, 10)
    

    
    # Sum of SHAP values per sample and output: shape (num_samples, 10)
    shap_sum = np.sum(shap_values_arr, axis=1)
    
    # Expected output change: (num_samples, 10)
    expected_change = model_predictions - expected_values
    
    # Absolute difference
    additivity_diffs = np.abs(expected_change - shap_sum)
    max_diff = np.max(additivity_diffs)
    mean_diff = np.mean(additivity_diffs)
    
    # Print the comparison metrics - evaluate correctness
    print("\n--- Additivity Check (Discrepancy Analysis) ---")
    print(f"  Maximum discrepancy: {max_diff:.5f}")
    print(f"  Mean discrepancy:    {mean_diff:.5f}")
    print(f"  Standardized output scale (min/max prediction): {model_predictions.min():.2f} / {model_predictions.max():.2f}")
    print("------------------------------------------------\n")
    
    # Create results folder
    results_dir = os.path.join(shap_dir, "results_gradient")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save SHAP values and scaled test inputs for future use
    np.save(os.path.join(results_dir, "shap_values.npy"), shap_values_arr)
    np.save(os.path.join(results_dir, "X_test_scaled.npy"), X_test)
    print(f"SHAP values binary files saved in: {results_dir}")
    
    # --- Save human-readable CSV of SHAP values ---
    print("Generating human-readable SHAP values CSV...")
    flat_rows = []
    df_test_orig = df_test.copy()
    
    # Denormalize predictions
    pred_full_denorm = np.zeros_like(model_predictions)
    for i, col in enumerate(output_cols):
        col_min = stats[col]["min"]
        col_max = stats[col]["max"]
        pred_full_denorm[:, i] = model_predictions[:, i] * (col_max - col_min) + col_min
        
    for j in range(len(df_test_orig)):
        power_val = df_test_orig.loc[j, "Power"]
        pressure_val = df_test_orig.loc[j, "Pressure"]
        feed_val = df_test_orig.loc[j, "Feed"]
        vbias_val = df_test_orig.loc[j, "Vbias"]
        
        for pt_idx in range(10):
            pt_name = f"Point{pt_idx+1}"
            
            # Attributions
            power_shap = shap_values_arr[j, 0, pt_idx]
            pressure_shap = shap_values_arr[j, 1, pt_idx]
            feed_shap = shap_values_arr[j, 2, pt_idx]
            vbias_shap = shap_values_arr[j, 3, pt_idx]
            
            pred_val = pred_full_denorm[j, pt_idx]
            gt_val = df_test_orig.loc[j, output_cols[pt_idx]]
            
            flat_rows.append({
                "Sample_Index": j,
                "Radial_Point": pt_name,
                "Power_W": power_val,
                "Pressure_mTorr": pressure_val,
                "Feed_sccm": feed_val,
                "Vbias_V": vbias_val,
                "Power_SHAP": power_shap,
                "Pressure_SHAP": pressure_shap,
                "Feed_SHAP": feed_shap,
                "Vbias_SHAP": vbias_shap,
                "Predicted_EtchRate": pred_val,
                "Actual_EtchRate": gt_val
            })
            
    df_shap_flat = pd.DataFrame(flat_rows)
    df_shap_flat.to_csv(os.path.join(results_dir, "shap_values_detailed.csv"), sep="\t", index=False)
    print(f"Saved human-readable table: {os.path.join(results_dir, 'shap_values_detailed.csv')}")
    
    # Plotting code has been relocated to plot_mlp_importance.py
    print(f"\nSHAP analysis completed successfully. All artifacts saved in {results_dir} folder.")

    ##-neuron attribution analysis- 
    print("\nRunning neuron-level attribution analysis...")
    neuron_dir = os.path.join(results_dir, "neuron_analysis")
    os.makedirs(neuron_dir, exist_ok=True)

    # 1. Define sub-models that calculate final outputs starting from the intermediate layers
    class SubModelFromLayer(torch.nn.Module):
        def __init__(self, full_model, start_layer_idx):
            super().__init__()
            self.full_model = full_model
            self.start_layer_idx = start_layer_idx

        def forward(self, hidden_activations):
            x = hidden_activations
            # Pass activations through remaining layers
            for i in range(self.start_layer_idx, self.full_model.num_layers):
                x = torch.nn.functional.relu(self.full_model.layers[i](x))
            x = self.full_model.out(x)
            x = torch.nn.functional.elu(x)
            return x

    # 2. Define the target layers and their index in the model.layers list
    # We explain the final outputs using the activations of these layers as features.
    layer_targets = [
        ("hidden_layer_0", 1), # Layers after layers[0] (indices 1, 2)
        ("hidden_layer_1", 2), # Layers after layers[1] (index 2)
    ]

    for label, start_idx in layer_targets:
        print(f"  Explaining final outputs using neurons of {label} as features...")
        
        # Extract the hidden activations for background and test sets
        with torch.no_grad():
            H_background = X_background
            H_test = X_test_tensor
            for i in range(start_idx):
                H_background = torch.nn.functional.relu(model.layers[i](H_background))
                H_test = torch.nn.functional.relu(model.layers[i](H_test))
        
        # Instantiate the second half of the model starting from this layer
        sub_model = SubModelFromLayer(model, start_idx)
        
        # Initialize GradientExplainer using hidden activations as features
        neuron_explainer = shap.GradientExplainer(sub_model, H_background)
        
        # shap_values returns a list of length 10 (one for each radial output point)
        # each element is a numpy array of shape (num_test_samples, num_neurons)
        neuron_shap_list = neuron_explainer.shap_values(H_test)
        
        # SHAP may return a list of arrays or a single array
        if isinstance(neuron_shap_list, list):
            neuron_shap = np.stack(neuron_shap_list, axis=-1)
        else:
            neuron_shap = np.array(neuron_shap_list)
            # If shape is (samples, outputs, neurons), swap to (samples, neurons, outputs)
            if neuron_shap.shape[-1] != 10 and neuron_shap.shape[1] == 10:
                neuron_shap = np.swapaxes(neuron_shap, 1, 2)

        np.save(os.path.join(neuron_dir, f"neuron_shap_{label}.npy"), neuron_shap)

        mean_abs_neuron_shap = np.mean(np.abs(neuron_shap), axis=0)  # Shape: (num_neurons, 10)

        # Save a per-neuron, per-radial-point CSV for this layer with tab delimitation
        neuron_rows = []
        for n_idx in range(mean_abs_neuron_shap.shape[0]):
            row = {"Neuron_Index": n_idx}
            for pt_idx in range(10):
                row[f"Point{pt_idx+1}_MeanAbsSHAP"] = mean_abs_neuron_shap[n_idx, pt_idx]
            neuron_rows.append(row)
        pd.DataFrame(neuron_rows).to_csv(
            os.path.join(neuron_dir, f"neuron_importance_{label}.csv"), sep="\t", index=False
        )

        print(f"    Saved: neuron_importance_{label}.csv (tab-delimited), neuron_shap_{label}.npy")

        # Quick console summary: top-3 neurons for the center and edge points
        for pt_idx, pt_label in [(0, "Point1 (center)"), (9, "Point10 (edge)")]:
            top_neurons = np.argsort(mean_abs_neuron_shap[:, pt_idx])[::-1][:3]
            print(f"    Top neurons for {pt_label}: {list(top_neurons)}")

    print(f"Neuron-level attribution analysis completed. Artifacts saved in {neuron_dir} folder.")
    
    print(f"\nSHAP analysis completed successfully. All artifacts saved in {results_dir} folder.")

if __name__ == "__main__":
    main()

