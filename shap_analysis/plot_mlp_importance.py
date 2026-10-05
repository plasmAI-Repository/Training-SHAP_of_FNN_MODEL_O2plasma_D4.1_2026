import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import shap

# Add the main workspace to the path to import Model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
from model import Model

def plot_percentage_waterfall(percentages, feature_names, title, save_path):
    plt.figure(figsize=(8, 6))
    
    # Sort by percentage descending
    sorted_idx = np.argsort(percentages)[::-1]
    sorted_pcts = percentages[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Calculate cumulative sums for bottoms
    cumulative = np.zeros(len(sorted_pcts))
    for i in range(1, len(sorted_pcts)):
        cumulative[i] = cumulative[i-1] + sorted_pcts[i-1]
        
    # Plot bars
    bars = plt.bar(sorted_names, sorted_pcts, bottom=cumulative, color='#4C72B0', edgecolor='black', alpha=0.8)
    
    # Add a total bar
    total_bar = plt.bar(['Total'], [100.0], color='#55A868', edgecolor='black', alpha=0.8)
    
    # Add text annotations
    for i, bar in enumerate(bars):
        yval = bar.get_y() + bar.get_height() / 2
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{sorted_pcts[i]:.1f}%", 
                 ha='center', va='center', fontweight='bold', color='white')
                 
    plt.text(len(sorted_names), 50, "100%", ha='center', va='center', fontweight='bold', color='white')
    
    plt.ylabel("Percentage Contribution (%)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Draw step lines
    for i in range(len(sorted_pcts)-1):
        plt.plot([i+0.4, i+1.4], [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5)
    plt.plot([len(sorted_pcts)-1+0.4, len(sorted_pcts)+0.4], [100, 100], 'k--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    shap_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(shap_dir)
    results_dir = os.path.join(shap_dir, "results_gradient")
    neuron_dir = os.path.join(results_dir, "neuron_analysis")
    model_path = os.path.join(base_dir, "O2", "baseline_seed_3123", "trained_model_base.pth")
    stats_path = os.path.join(base_dir, "dataset_stats.json")
    
    # ---------------------------------------------------------------------------
    # 0. Load Data and Scaling stats
    # ---------------------------------------------------------------------------
    print("Loading data files...")
    shap_values_arr = np.load(os.path.join(results_dir, "shap_values.npy"))
    X_test = np.load(os.path.join(results_dir, "X_test_scaled.npy"))
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    input_cols = ["Power", "Pressure", "Feed", "Vbias"]
    feature_names = ["Power (W)", "Pressure (mTorr)", "Feed (sccm)", "Vbias (V)"]
    
    # ---------------------------------------------------------------------------
    # 1. Plot 1: Radial Sensitivity Analysis (Global Importance across Radius)
    # ---------------------------------------------------------------------------
    print("Generating radial feature importance plot...")
    mean_abs_shap = np.mean(np.abs(shap_values_arr), axis=0).T # shape: (10, 4)
    
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
    
    plt.figure(figsize=(10, 6))
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]
    markers = ["o", "s", "^", "D"]
    
    for i, col in enumerate(input_cols):
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
    print("  Saved: radial_feature_importance.png")
    
    print("Generating global percentage waterfall plot...")
    # Calculate global mean across all 10 points
    global_mean_abs_shap = np.mean(mean_abs_shap, axis=0)
    percentages = global_mean_abs_shap / np.sum(global_mean_abs_shap) * 100
    waterfall_path = os.path.join(results_dir, "global_percentage_waterfall.png")
    plot_percentage_waterfall(percentages, feature_names, "Global Feature Contribution to Etch Rate (MLP)", waterfall_path)
    print("  Saved: global_percentage_waterfall.png")
    
    # ---------------------------------------------------------------------------
    # 2. Plot 2 & 3: Summary Plots (Beeswarm) for Center (Point 1) and Edge (Point 10)
    # ---------------------------------------------------------------------------
    print("Generating summary beeswarm plots...")
    
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
    print("  Saved: summary_beeswarm_point1_center.png")
    
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
    print("  Saved: summary_beeswarm_point10_edge.png")
    
    # ---------------------------------------------------------------------------
    # 3. Plot 4: Dependence Plot for Point 5 (Mid-radius)
    # ---------------------------------------------------------------------------
    print("Generating dependence plot...")
    p5_importance = mean_abs_shap[4, :]
    dominant_feat_idx = np.argmax(p5_importance)
    dominant_feat_name = input_cols[dominant_feat_idx]
    dominant_feat_display = feature_names[dominant_feat_idx]
    
    remaining_indices = [idx for idx in range(4) if idx != dominant_feat_idx]
    interaction_feat_idx = remaining_indices[np.argmax(p5_importance[remaining_indices])]
    interaction_feat_name = feature_names[interaction_feat_idx]
    
    plt.figure(figsize=(8, 5))
    x_val_denorm = X_test[:, dominant_feat_idx] * (stats[dominant_feat_name]["max"] - stats[dominant_feat_name]["min"]) + stats[dominant_feat_name]["min"]
    color_val_denorm = X_test[:, interaction_feat_idx] * (stats[input_cols[interaction_feat_idx]]["max"] - stats[input_cols[interaction_feat_idx]]["min"]) + stats[input_cols[interaction_feat_idx]]["min"]
    
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
    print(f"  Saved: dependence_point5_{dominant_feat_name}.png")
    
    # ---------------------------------------------------------------------------
    # 4. Plot 5: Neuron Attribution Heatmaps
    # ---------------------------------------------------------------------------
    print("Generating neuron-level attribution heatmaps...")
    layer_targets = [
        ("hidden_layer_0", "hidden_layer_0"),
        ("hidden_layer_1", "hidden_layer_1"),
        ("hidden_layer_2", "hidden_layer_2"),
    ]
    
    for label, file_name in layer_targets:
        npy_path = os.path.join(neuron_dir, f"neuron_shap_{file_name}.npy")
        if not os.path.exists(npy_path):
            continue
            
        neuron_shap = np.load(npy_path)
        
        # Determine the correct axis to average over to get (num_neurons, 10)
        if len(neuron_shap.shape) == 3:
            if neuron_shap.shape[2] == 10:
                mean_abs_neuron_shap = np.mean(np.abs(neuron_shap), axis=0)
            elif neuron_shap.shape[1] == 10:
                mean_abs_neuron_shap = np.mean(np.abs(neuron_shap), axis=2)
            else:
                continue
        else:
            continue
            
        plt.figure(figsize=(8, 6))
        im = plt.imshow(mean_abs_neuron_shap, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Mean |SHAP value|")
        plt.xlabel("Radial Position (Point 1 to 10)")
        plt.ylabel("Neuron Index")
        plt.xticks(range(10), range(1, 11))
        plt.yticks(range(mean_abs_neuron_shap.shape[0]))
        plt.title(f"Neuron Attribution Heatmap: {label}", fontsize=14, pad=15)
        
        fig_path = os.path.join(neuron_dir, f"neuron_heatmap_{label}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: neuron_heatmap_{label}.png")

    # ---------------------------------------------------------------------------
    # 5. Plot 6: MLP Architecture Feature Dominance & Node Importance map
    # ---------------------------------------------------------------------------
    print("Generating MLP architecture importance map...")
    model = Model(h1=20, num_layers=2, freeze_layers=[], output_size=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    
    W0 = model.layers[0].weight.data.numpy()      # Shape: (20, 4)
    W1 = model.layers[1].weight.data.numpy()      # Shape: (20, 20)
    W_out = model.out.weight.data.numpy()         # Shape: (10, 20)
    
    layer0_dominant_input = np.argmax(np.abs(W0), axis=1) # (20,)
    layer1_eff_weights = np.abs(W1) @ np.abs(W0)          # (20, 4)
    layer1_dominant_input = np.argmax(layer1_eff_weights, axis=1) # (20,)
    
    csv0_path = os.path.join(neuron_dir, "neuron_importance_hidden_layer_0.csv")
    csv1_path = os.path.join(neuron_dir, "neuron_importance_hidden_layer_1.csv")
    
    df0 = pd.read_csv(csv0_path, sep="\t")
    df1 = pd.read_csv(csv1_path, sep="\t")
    
    importance_cols = [f"Point{i}_MeanAbsSHAP" for i in range(1, 11)]
    imp0 = df0[importance_cols].mean(axis=1).to_numpy() # (20,)
    imp1 = df1[importance_cols].mean(axis=1).to_numpy() # (20,)
    
    def normalize_alpha(imp):
        if imp.max() == imp.min():
            return np.ones_like(imp)
        return 0.15 + 0.85 * (imp - imp.min()) / (imp.max() - imp.min())
        
    alpha0 = normalize_alpha(imp0)
    alpha1 = normalize_alpha(imp1)
    
    x_coords = [0, 1.2, 2.4, 3.6]
    y_input = np.linspace(-1.5, 1.5, 4)
    y_hidden0 = np.linspace(-4.5, 4.5, 20)
    y_hidden1 = np.linspace(-4.5, 4.5, 20)
    y_output = np.linspace(-3.0, 3.0, 10)
    
    input_colors = {
        0: "#1F77B4",  # Power (Blue)
        1: "#FF7F0E",  # Pressure (Orange)
        2: "#2CA02C",  # Feed (Green)
        3: "#D62728"   # Vbias (Red)
    }
    input_names = ["Power", "Pressure", "Feed", "Vbias"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    
    max_w0 = np.abs(W0).max()
    for i in range(4):
        for j in range(20):
            w = np.abs(W0[j, i]) / max_w0
            ax.plot([x_coords[0], x_coords[1]], [y_input[i], y_hidden0[j]], 
                    color=input_colors[i], alpha=0.03 + 0.12 * w, linewidth=0.3 + 1.2 * w)
                    
    max_w1 = np.abs(W1).max()
    for i in range(20):
        for j in range(20):
            w = np.abs(W1[j, i]) / max_w1
            color = input_colors[layer0_dominant_input[i]]
            ax.plot([x_coords[1], x_coords[2]], [y_hidden0[i], y_hidden1[j]], 
                    color=color, alpha=0.02 + 0.08 * w, linewidth=0.2 + 0.8 * w)

    max_w_out = np.abs(W_out).max()
    for i in range(20):
        for j in range(10):
            w = np.abs(W_out[j, i]) / max_w_out
            color = input_colors[layer1_dominant_input[i]]
            ax.plot([x_coords[2], x_coords[3]], [y_hidden1[i], y_output[j]], 
                    color=color, alpha=0.02 + 0.08 * w, linewidth=0.2 + 0.8 * w)

    node_radius = 0.07
    for i in range(4):
        color = input_colors[i]
        circle = Circle((x_coords[0], y_input[i]), radius=node_radius, 
                        facecolor=color, edgecolor="#333333", linewidth=1.0, zorder=5)
        ax.add_patch(circle)
        ax.text(x_coords[0] - 0.15, y_input[i], input_names[i], 
                horizontalalignment="right", verticalalignment="center", fontweight="bold", color=color)

    for j in range(20):
        dom_in = layer0_dominant_input[j]
        color = input_colors[dom_in]
        alpha = alpha0[j]
        circle = Circle((x_coords[1], y_hidden0[j]), radius=node_radius, 
                        facecolor=color, edgecolor="#222222", linewidth=0.8, alpha=alpha, zorder=5)
        ax.add_patch(circle)
        outline = Circle((x_coords[1], y_hidden0[j]), radius=node_radius, 
                         facecolor="none", edgecolor="#555555", linewidth=0.5, alpha=0.3, zorder=6)
        ax.add_patch(outline)

    for j in range(20):
        dom_in = layer1_dominant_input[j]
        color = input_colors[dom_in]
        alpha = alpha1[j]
        circle = Circle((x_coords[2], y_hidden1[j]), radius=node_radius, 
                        facecolor=color, edgecolor="#222222", linewidth=0.8, alpha=alpha, zorder=5)
        ax.add_patch(circle)
        outline = Circle((x_coords[2], y_hidden1[j]), radius=node_radius, 
                         facecolor="none", edgecolor="#555555", linewidth=0.5, alpha=0.3, zorder=6)
        ax.add_patch(outline)

    for j in range(10):
        circle = Circle((x_coords[3], y_output[j]), radius=node_radius, 
                        facecolor="#444444", edgecolor="#111111", linewidth=1.0, zorder=5)
        ax.add_patch(circle)
        ax.text(x_coords[3] + 0.15, y_output[j], f"Point {j+1}", 
                horizontalalignment="left", verticalalignment="center", fontweight="bold")

    ax.text(x_coords[0], 4.8, "Input\nLayer", ha="center", va="bottom", fontweight="bold")
    ax.text(x_coords[1], 4.8, "Hidden\nLayer 1", ha="center", va="bottom", fontweight="bold")
    ax.text(x_coords[2], 4.8, "Hidden\nLayer 2", ha="center", va="bottom", fontweight="bold")
    ax.text(x_coords[3], 4.8, "Output\nLayer", ha="center", va="bottom", fontweight="bold")
    
    legend_handles = []
    for i, name in enumerate(input_names):
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", label=name,
                                         markerfacecolor=input_colors[i], markersize=10))
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=4, frameon=True, facecolor="#F8F9FA", edgecolor="#E5E5E5")
              
    plt.title("MLP Feature Dominance and Neuron Importance Map", fontsize=16, fontweight="bold", pad=20)
    
    out_path = os.path.join(results_dir, "mlp_importance_architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: mlp_importance_architecture.png")
    
    # ---------------------------------------------------------------------------
    # 6. Plot 7: Weight Distribution of MLP Layers
    # ---------------------------------------------------------------------------
    print("Generating weight distribution plot...")
    all_weights = []
    labels = []
    
    for i, layer in enumerate(model.layers):
        all_weights.append(layer.weight.data.numpy().flatten())
        labels.append(f"Hidden {i+1}")
        
    all_weights.append(model.out.weight.data.numpy().flatten())
    labels.append("Output")
    
    plt.figure(figsize=(10, 6))
    
    # Plot individual dots with random x-jitter for better visibility (like a strip plot)
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']
    for i, w in enumerate(all_weights):
        # Generate random jitter around the x-coordinate (i+1)
        x_jitter = np.random.normal(i + 1, 0.05, size=len(w))
        plt.scatter(x_jitter, w, alpha=0.7, s=40, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        # Add a small line indicating the mean weight for the layer
        mean_w = np.mean(w)
        plt.hlines(mean_w, i + 0.8, i + 1.2, color='black', linewidth=2, zorder=3)
        
    plt.xticks(range(1, len(labels) + 1), labels)
    
    plt.yscale('symlog', linthresh=0.01)
    plt.ylabel("Weight Value", fontsize=12)
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.title("Weight Distribution across MLP Layers", fontsize=15, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    dist_path = os.path.join(results_dir, "mlp_weight_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: mlp_weight_distribution.png")

    print(f"\nAll plots generated successfully. Artifacts saved in {results_dir} folder.")

if __name__ == "__main__":
    main()
