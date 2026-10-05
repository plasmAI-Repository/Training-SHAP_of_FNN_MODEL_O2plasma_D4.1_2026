import pandas as pd
import json

df = pd.read_csv('v3_1_data/v3_1_O2_eth_dataset_full.txt', sep='\t')

stats = {}
for column in df.columns:
    # Only calculate for numeric columns
    if pd.api.types.is_numeric_dtype(df[column]):
        stats[column] = {
            'min': float(df[column].min()),
            'max': float(df[column].max())
        }
    else:
        stats[column] = {
            'min': 'N/A',
            'max': 'N/A',
            'dtype': str(df[column].dtype)
        }

with open('dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)