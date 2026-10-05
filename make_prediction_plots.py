import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# Configuration
base_dir = r'c:\v31model\O2'
folders = [
    "baseline_with_previous_hyperparams_dec25",
    "remove_relu_from_output_dec25",
    "remove_relu_from_output_and_add_L1_at_loss_dec25_lamda_1e-2",
    "remove_relu_from_output_and_add_L1_at_loss_dec25_lamda_1e-4",
    "softplus_output_no_l1_dec25",
    "softplus_output_l1_lamda_1e-4_dec25",
    "elu_output_no_l1_dec25",
    "elu_output_l1_lamda_1e-4_dec25"
]

input_cols_count = 4
target_cols_count = 10
pred_cols_count = 10

# Hardcoded indices to match the report/previous selection
selected_indices = [327, 57, 12, 379]

print(f"Selected sample indices: {selected_indices}")

# Process each folder
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    file_path = os.path.join(folder_path, "comparison_base.txt")
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        continue
        
    print(f"Processing folder: {folder}")
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # Iterate through the selected indices
        for idx in selected_indices:
            # Check if index is valid for this specific file, just in case
            if idx >= len(df):
                print(f"  Warning: Index {idx} is out of bounds for {folder} (len={len(df)}). Skipping.")
                continue

            row = df.iloc[idx]
            
            # Extract data
            # Structure: [Inputs (4)] [Targets (10)] [Predictions (10)]
            inputs = row.iloc[0:input_cols_count]
            targets = row.iloc[input_cols_count : input_cols_count + target_cols_count]
            preds = row.iloc[input_cols_count + target_cols_count : input_cols_count + target_cols_count + pred_cols_count]
            
            # Create input string for title
            input_str = ", ".join([f"{col}={val:.4f}" for col, val in inputs.items()])
            
            # Plot
            plt.figure(figsize=(8, 6))
            
            x_points = range(1, target_cols_count + 1)
            
            plt.plot(x_points, targets, 'o-', label='Target (Physics)', color='black', linewidth=2)
            plt.plot(x_points, preds, 'x--', label='Neural Network', color='red', linewidth=2)
            
            plt.title(f"Sample {idx}\n{input_str}", fontsize=12)
            plt.xlabel("Etching Rate Point", fontsize=12)
            plt.ylabel("Etching Rate", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.5)
            
            # Save plot
            save_name = f"comparison_plot_sample_{idx}.png"
            save_path = os.path.join(folder_path, save_name)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"  Saved plot for sample {idx} to {save_path}")

    except Exception as e:
        print(f"Error processing {folder}: {e}")

print("Done.")
