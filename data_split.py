import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl

def outer_window_split(df, train_ratio, outer_window_ratio):
    """
    Split data with outer windows for testing
    - Middle section for training
    - Outer sections for testing
    """
    total_len = len(df)

    # Calculate sizes
    outer_size = int(total_len * outer_window_ratio)  # 10% from each end
    train_size = int(total_len * train_ratio)

    # Create test sets from outer windows
    test_start = df.iloc[:outer_size]  # First 10%
    test_end = df.iloc[-outer_size:]  # Last 10%
    test_df = pd.concat([test_start, test_end])

    # Training set from middle section
    train_df = df.iloc[outer_size:outer_size + train_size]

    return train_df, test_df

def ood_split(df, train_ratio=0.8, ood_ratio=0.2, input_cols=None):
    """
    Split dataset into training and OOD test sets based on distance from the mean in input space.
    """
    if input_cols is None:
        input_cols = df.columns[:4]  # Default: first 4 columns are inputs

    # Normalize inputs
    X = df[input_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute distance from dataset center
    center = np.mean(X_scaled, axis=0)
    distances = np.linalg.norm(X_scaled - center, axis=1)

    # Add distances and sort
    df = df.copy()
    df["distance"] = distances
    df_sorted = df.sort_values(by="distance", ascending=False)

    # Split
    test_size = int(len(df) * ood_ratio)
    test_df = df_sorted.iloc[:test_size]
    train_df = df_sorted.iloc[test_size:]

    # Clean up
    train_df = train_df.drop(columns=["distance"])
    test_df = test_df.drop(columns=["distance"])

    return train_df, test_df, X_scaled, distances


df = pd.read_csv('v3_1_data/v3_1_O2_eth_dataset_full.txt', sep='\t')
# Apply in of distribution split
# train_df, test_df = outer_window_split(df, train_ratio=0.8, outer_window_ratio=0.1)

# Apply out of distribution split
train_df, test_df, X_scaled, distances = ood_split(df, ood_ratio=0.2)

# Save to individual TXT files with headers
train_df.to_csv('v3_1_data/train_data_v3_1_O2.txt', sep='\t', index=False)
test_df.to_csv('v3_1_data/test_data_v3_1_O2.txt', sep='\t', index=False)

print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
print(f"Testing samples: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
print(f"Training data saved to: v3_1_data/train_data.txt")
print(f"Testing data saved to: v3_1_data/test_data.txt")

# --- Visualization ---
def plot_ood_split(X_scaled, distances, test_ratio=0.2):
    """
    Visualize OOD vs training points in 2D (using PCA projection).
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # Sort by distance to find cutoff
    sorted_indices = np.argsort(distances)[::-1]
    test_cutoff = int(len(X_scaled) * test_ratio)
    test_idx = sorted_indices[:test_cutoff]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], color="skyblue", label="Training Data", alpha=0.5)
    plt.scatter(X_2d[test_idx, 0], X_2d[test_idx, 1], color="red", label="OOD Test Data", alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Out-of-Distribution (OOD) Split Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

# visualization:
plot_ood_split(X_scaled, distances)