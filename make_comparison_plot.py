import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Base directory usually relative to where script is run, but using absolute path for safety based on user environment
base_dir = r'c:\v31model\O2'

# List of folders to process
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

# Readable labels for the plot
labels = [
    "Baseline",
    "No ReLU",
    "No ReLU + L1 (1e-2)",
    "No ReLU + L1 (1e-4)",
    "Softplus no L1",
    "Softplus (L1)", 
    "ELU (No L1)", 
    "ELU (L1)"
]

plt.figure(figsize=(12, 8))

colors = sns.color_palette("husl", len(folders))

for i, (folder, label) in enumerate(zip(folders, labels)):
    file_path = os.path.join(base_dir, folder, "residuals_data_base.txt")
    
    if os.path.exists(file_path):
        print(f"Processing: {label} ({folder})")
        try:
            # Read the data
            df = pd.read_csv(file_path, sep='\t')
            
            # Flatten all columns into a single array of residuals and take absolute value
            flat_data = df.values.flatten()
            flat_data = pd.Series(flat_data).abs()
            
            # Plot Histogram with outline (poly) to look like a curve
            sns.histplot(flat_data, label=label, element="poly", fill=False, stat="count", bins=1000, linewidth=2, color=colors[i])
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")
    else:
        print(f"Warning: File not found: {file_path}")

plt.title('Comparison of Residual Distributions (All Outputs)', fontsize=16)
plt.xlabel('Absolute Percentage Error in Prediction', fontsize=14)
plt.ylabel('Number of Samples', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 200)  
plt.tight_layout()

output_path = os.path.join(base_dir, 'residuals_comparison_plot_dec25.png')
plt.savefig(output_path, dpi=300)
print(f"Comparison plot saved to: {output_path}")
