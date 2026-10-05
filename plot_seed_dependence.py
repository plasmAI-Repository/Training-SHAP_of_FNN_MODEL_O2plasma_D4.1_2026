import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to match where your multiple seed runs are stored.
# For example, if you ran the MLP on v3_3_1_data with different seeds, 
# you would list the folders containing their comparison_base.txt (or comparison_geometry.txt) outputs here.
base_dir = r'c:\Users\enisb\Documents\GitHub\v31model\O2'
seed_folders = [
    "baseline_seed_3434", 
    "baseline_seed_4112", 
    "baseline_seed_6052",
    "baseline_seed_8178",
    "baseline_seed_8526"
]

# The file name generated during testing
comparison_file_name = "comparison_base.txt"

# Indices of the samples you want to plot (e.g., specific operating conditions)
selected_indices = [27, 56, 12, 79, 45, 5]

input_cols_count = 4
target_cols_count = 10
pred_cols_count = 10

# ==========================================
# PLOTTING AESTHETICS (Publication Quality)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (10, 7)
})

def main():
    print(f"Selected sample indices: {selected_indices}")
    
    # Store data for all seeds
    # Format: { index: {'targets': [], 'preds': [], 'inputs': []} }
    sample_data = {idx: {'preds': []} for idx in selected_indices}
    
    for folder in seed_folders:
        file_path = os.path.join(base_dir, folder, comparison_file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        print(f"Loading data from: {folder}")
        df = pd.read_csv(file_path, sep='\t')
        
        for idx in selected_indices:
            if idx >= len(df):
                print(f"Warning: Index {idx} out of bounds for {folder}")
                continue
                
            row = df.iloc[idx]
            
            inputs = row.iloc[0:input_cols_count]
            targets = row.iloc[input_cols_count : input_cols_count + target_cols_count]
            preds = row.iloc[input_cols_count + target_cols_count : input_cols_count + target_cols_count + pred_cols_count]
            
            # Targets and inputs are the same across seeds, just save them once
            if 'targets' not in sample_data[idx]:
                sample_data[idx]['targets'] = targets.values.astype(float)
                sample_data[idx]['inputs'] = inputs
                
            sample_data[idx]['preds'].append(preds.values.astype(float))

    # Generate plots
    for idx, data in sample_data.items():
        if not data['preds']:
            print(f"No prediction data found for index {idx}. Skipping.")
            continue
            
        targets = data['targets']
        preds_array = np.array(data['preds'])  # Shape: (num_seeds, 10)
        
        # Calculate mean and standard deviation across seeds
        preds_mean = np.mean(preds_array, axis=0)
        preds_std = np.std(preds_array, axis=0)
        
        inputs = data['inputs']
        
        units_map = {'Power': 'W', 'Pressure': 'Torr', 'Feed': 'sccm', 'Vbias': 'Volt'}
        formatted_inputs = []
        for col, val in inputs.items():
            unit = units_map.get(col, '')
            if col == 'Pressure':
                formatted_inputs.append(f"{val:.3f} {unit}")
            else:
                formatted_inputs.append(f"{val:.1f} {unit}")
                
        input_str = ", ".join(formatted_inputs)
        
        x_points = np.arange(1, target_cols_count + 1)
        
        plt.figure()
        
        # Plot ground truth (PBM)
        plt.plot(x_points, targets, 'o-', color='black', linewidth=2.5, markersize=8, label='PBM (Ground Truth)')
        
        # Plot FNN Mean prediction
        plt.plot(x_points, preds_mean, 's--', color='#d62728', linewidth=2.5, markersize=8, label='FNN (Mean)')
        
        # Plot shaded region for seed dependence (Standard Deviation)
        plt.fill_between(x_points, preds_mean - preds_std, preds_mean + preds_std, 
                         color='#d62728', alpha=0.25, label='Seed Dependence ($\pm 1\sigma$)')
        
        plt.title(f"Radial Etching Rate - Sample {idx}\n{input_str}", pad=15)
        plt.xlabel("Etching Rate Point")
        plt.ylabel("Etching Rate")
        plt.xticks(x_points)
        
        # Add legend and grid
        plt.legend(loc='best', framealpha=0.9, edgecolor='black')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save publication-ready figure
        save_name = f"publication_seed_dependence_sample_{idx}.png"
        save_path = os.path.join(base_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved publication plot: {save_path}")

    # Calculate and print the overall average seed dependence (Coefficient of Variation)
    all_preds_full = []
    for folder in seed_folders:
        file_path = os.path.join(base_dir, folder, comparison_file_name)
        if os.path.exists(file_path):
            df_full = pd.read_csv(file_path, sep='\t')
            preds_full = df_full.iloc[:, input_cols_count + target_cols_count : input_cols_count + target_cols_count + pred_cols_count].values
            all_preds_full.append(preds_full)
            
    if all_preds_full:
        all_preds_arr = np.array(all_preds_full)
        mean_preds = np.mean(all_preds_arr, axis=0)
        std_preds = np.std(all_preds_arr, axis=0)
        
        mask = mean_preds > 1e-6
        cv = (std_preds[mask] / mean_preds[mask]) * 100
        avg_cv = np.mean(cv)
        print(f"\n==========================================")
        print(f"Overall Average Seed Dependence (CV): {avg_cv:.2f}%")
        print(f"==========================================\n")

if __name__ == "__main__":
    main()
