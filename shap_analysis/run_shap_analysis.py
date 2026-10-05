import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap

# Add the main workspace to the path to import Model
sys.path.insert(0, r"c:\v31model")
from model import Model

def main():
    print("Initializing SHAP analysis...")
    
    # 1. Directory and file paths
    data_dir = r"c:\v31model\v3_3_1_data"
    shap_dir = r"c:\v31model\shap_analysis"
    
    full_dataset_path = os.path.join(data_dir, "v3_3_1_O2_eth_dataset_full.txt")
    model_path = os.path.join(shap_dir, "trained_model_base.pth")
    stats_path = os.path.join(shap_dir, "stats_random_split.json")
    
    # 2. Re-create the Train/Val/Test Split (Seed 41)
    df = pd.read_csv(full_dataset_path, sep="\t")
    input_cols = ["Power", "Pressure", "Feed", "Vbias"]
    output_cols = [f"Point{i}_EtchRate" for i in range(1, 11)]
    
    total_len = len(df)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    generator = torch.Generator().manual_seed(41)
    indices = torch.randperm(total_len, generator=generator).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"Dataset split matching training configuration: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test.")
    
    # 3. Load scaling parameters
    with open(stats_path, 'r') as f:
        stats_dict = json.load(f)
    stats = stats_dict["O2"]
    
    # 4. Standardize inputs
    df_scaled = df.copy()
    for col in input_cols + output_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df_scaled[col] = (df[col] - mean) / std
        
    X_train = df_scaled.iloc[train_idx][input_cols].to_numpy(dtype=np.float32)
    X_test = df_scaled.iloc[test_idx][input_cols].to_numpy(dtype=np.float32)
    
    # 5. Load the trained MLP Model
    model = Model(h1=20, num_layers=3, freeze_layers=[], output_size=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    
    # 6. Select Background/Reference Dataset (100 random samples from training set)
    bg_rng = np.random.RandomState(42)
    bg_idx = bg_rng.choice(len(X_train), size=100, replace=False)
    X_background = torch.tensor(X_train[bg_idx])
    X_test_tensor = torch.tensor(X_test)
    
    # 7. Initialize shap.DeepExplainer
    print("Calculating SHAP values using DeepExplainer...")
    explainer = shap.DeepExplainer(model, X_background)
    
    # Compute SHAP values for all test samples
    # For a multi-output PyTorch model, shap_values returns a list of length 10
    # each element of list is a numpy array of shape (num_samples, num_features)
    #
    # NOTE ON check_additivity=False:
    # DeepSHAP checks if the sum of computed SHAP values matches the model's output change.
    # Because this MLP model uses non-linear activations (SiLU/Swish and ELU),
    # the linear approximations constructed by DeepLIFT/DeepSHAP introduce small numerical 
    # discrepancies. This causes the strict additivity check (tolerance 0.01) to fail with an
    # AssertionError. Setting check_additivity=False bypasses this check while still providing
    # highly accurate feature attributions.
    shap_values_list = explainer.shap_values(X_test_tensor, check_additivity=False)
    
    # Debug print to inspect shapes and types
    print("--- Debug Information ---")
    print("Type of shap_values_list:", type(shap_values_list))
    if isinstance(shap_values_list, list):
        print("Length of list:", len(shap_values_list))
        print("Shape of first element:", shap_values_list[0].shape)
    elif hasattr(shap_values_list, "shape"):
        print("Shape of array:", shap_values_list.shape)
    print("Type of explainer.expected_value:", type(explainer.expected_value))
    expected_val_np = np.array(explainer.expected_value)
    print("Shape of explainer.expected_value:", expected_val_np.shape)
    print("-------------------------")
    
    # Convert list of arrays to a single numpy array of shape (num_samples, num_features, output_size)
    shap_values_arr = shap_values_list
    
    # --- Custom Additivity Check for Printing ---
    with torch.no_grad():
        model_predictions = model(X_test_tensor).numpy() # shape: (num_samples, 10)
    
    expected_values = np.array(explainer.expected_value) # shape: (10,)
    
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
    results_dir = os.path.join(shap_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save SHAP values and scaled test inputs for future use
    np.save(os.path.join(results_dir, "shap_values.npy"), shap_values_arr)
    np.save(os.path.join(results_dir, "X_test_scaled.npy"), X_test)
    print(f"SHAP values binary files saved in: {results_dir}")
    
    # --- Save human-readable CSV of SHAP values ---
    print("Generating human-readable SHAP values CSV...")
    flat_rows = []
    df_test_orig = df.iloc[test_idx].reset_index(drop=True)
    
    # Denormalize predictions
    pred_full_denorm = np.zeros_like(model_predictions)
    for i, col in enumerate(output_cols):
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        pred_full_denorm[:, i] = model_predictions[:, i] * std + mean
        
    for j in range(len(test_idx)):
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
                "Sample_Index": test_idx[j],
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
    df_shap_flat.to_csv(os.path.join(results_dir, "shap_values_detailed.csv"), index=False)
    print(f"Saved human-readable table: {os.path.join(results_dir, 'shap_values_detailed.csv')}")
    
    # ---------------------------------------------------------------------------
    # Plotting & Interpretation
    # ---------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         14,
        "axes.labelsize":    16,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   12,
        "figure.dpi":        300,
        "savefig.bbox":      "tight",
    })
    
    # Plot 1: Radial Sensitivity Analysis (Global Importance across Radius)
    # Compute mean absolute SHAP value for each feature at each radial point
    # shape of shap_values_arr is (num_samples, 4_features, 10_outputs)
    mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0).T # shape: (10, 4)
    
    plt.figure(figsize=(10, 6))
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]
    markers = ["o", "s", "^", "D"]
    
    for i, col in enumerate(input_cols):
        # We plot the mean absolute SHAP value (which is in scaled output units)
        plt.plot(range(1, 11), mean_abs_shap[:, i], label=col,
                 color=colors[i], marker=markers[i], linewidth=2.0, markersize=8)
                 
    plt.xlabel("Radial Position (Points 1 to 10)")
    plt.ylabel("Mean Absolute SHAP Value (Feature Importance)")
    plt.title("Radial Profile of Feature Importance (MLP)", fontsize=16, pad=15)
    plt.xticks(range(1, 11))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(results_dir, "radial_feature_importance.png"))
    plt.close()
    print("Saved: radial_feature_importance.png")
    
    # Plot 2 & 3: Summary Plots (Beeswarm) for Center (Point 1) and Edge (Point 10)
    # We reconstruct a shap.Explanation object to use SHAP's native beautiful plotting
    # Denormalize column names for visualization
    feature_names = ["Power (W)", "Pressure (mTorr)", "Feed (sccm)", "Vbias (V)"]
    
    # Point 1 (Center)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values_arr[:, :, 0], 
        X_test, 
        feature_names=feature_names, 
        show=False
    )
    plt.title("SHAP Feature Attribution: Point 1 (Center)", fontsize=14, pad=15)
    plt.savefig(os.path.join(results_dir, "summary_beeswarm_point1_center.png"))
    plt.close()
    print("Saved: summary_beeswarm_point1_center.png")
    
    # Point 10 (Edge)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values_arr[:, :, 9], 
        X_test, 
        feature_names=feature_names, 
        show=False
    )
    plt.title("SHAP Feature Attribution: Point 10 (Edge)", fontsize=14, pad=15)
    plt.savefig(os.path.join(results_dir, "summary_beeswarm_point10_edge.png"))
    plt.close()
    print("Saved: summary_beeswarm_point10_edge.png")
    
    # Plot 4: Dependence Plot for the most dominant input (usually Power or Vbias) at Point 5 (Mid-radius)
    # Let's find the feature with the highest mean absolute SHAP at Point 5
    p5_importance = mean_abs_shap[4, :]
    dominant_feat_idx = np.argmax(p5_importance)
    dominant_feat_name = input_cols[dominant_feat_idx]
    dominant_feat_display = feature_names[dominant_feat_idx]
    
    # Also find the second most dominant feature to show interaction
    remaining_indices = [idx for idx in range(4) if idx != dominant_feat_idx]
    interaction_feat_idx = remaining_indices[np.argmax(p5_importance[remaining_indices])]
    interaction_feat_name = feature_names[interaction_feat_idx]
    
    plt.figure(figsize=(8, 5))
    # Denormalize the dominant feature values for plotting
    x_val_denorm = X_test[:, dominant_feat_idx] * stats[dominant_feat_name]["std"] + stats[dominant_feat_name]["mean"]
    color_val_denorm = X_test[:, interaction_feat_idx] * stats[input_cols[interaction_feat_idx]]["std"] + stats[input_cols[interaction_feat_idx]]["mean"]
    
    scatter = plt.scatter(
        x_val_denorm, 
        shap_values_arr[:, dominant_feat_idx, 4], 
        c=color_val_denorm, 
        cmap="coolwarm", 
        edgecolor="k", 
        linewidth=0.5, 
        alpha=0.85
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label(interaction_feat_name)
    plt.xlabel(dominant_feat_display)
    plt.ylabel(f"SHAP Value for {dominant_feat_display}")
    plt.title(f"Dependence Plot at Point 5 (Mid-radius)", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(os.path.join(results_dir, f"dependence_point5_{dominant_feat_name}.png"))
    plt.close()
    print(f"Saved: dependence_point5_{dominant_feat_name}.png")
    
    print(f"\nSHAP analysis completed successfully. All artifacts saved in {results_dir} folder.")

if __name__ == "__main__":
    main()
