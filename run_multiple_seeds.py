import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import set_seed, setup_device
from train import train
from test import test

def run_multiple_seeds():
    device = setup_device()

    # Open .yaml file to get configuration
    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    gas = 'O2'
    config_O2 = config[gas]
    config_arch = config['nn_arch']
    config_train = config['training']
    outputs_points = [i+1 for i in range(0, 10)]

    # 1. PREPARE THE DATASET ONCE FOR ALL SEEDS
    # We create a single Train/Test split for all seeds. If we created a new test set for every seed,
    # we wouldn't be able to plot the seed dependence on specific hold-out conditions!
    full_data_path = f'./v3_3_1_data/v3_3_1_{gas}_eth_dataset_full.txt'
    if os.path.exists(full_data_path):
        print("Creating fixed Train/Test split from v3_3_1_data for all seeds...")
        df = pd.read_csv(full_data_path, sep='\t')
        # Use a fixed random state here so the test set remains the same across different runs of this script
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_df.to_csv(f'./v3_3_1_data/train_data_v3_3_1_{gas}.txt', sep='\t', index=False)
        test_df.to_csv(f'./v3_3_1_data/test_data_v3_3_1_{gas}.txt', sep='\t', index=False)
    else:
        print(f"Warning: Could not find {full_data_path}")

    # Generate 5 random seeds between 1 and 10000
    import random
    seeds = [random.randint(1, 10000) for _ in range(5)]
    print(f"\nGenerated random seeds for this experiment: {seeds}")
    
    experiment_base_name = 'baseline_seed'

    for seed_num in seeds:
        print(f"\n{'='*50}")
        print(f"Starting training for seed {seed_num}")
        print(f"{'='*50}")
        
        # Set the seed for reproducibility across the run
        # This will now ALSO shuffle the Train/Validation split inside train.py!
        set_seed(seed_num)
        
        training_type = f'{experiment_base_name}_{seed_num}'
        dir_path = f'./{gas}/{training_type}'
        os.makedirs(dir_path, exist_ok=True)
        
        print(f"Training will be saved to: {dir_path}")
        
        saved_model_path, _ = train(gas=gas, 
                                    outputs_points=outputs_points, 
                                    freeze_layers=[], 
                                    geometry_layer=False,
                                    model_pth=False, 
                                    config_gas=config_O2, 
                                    config_arch=config_arch,
                                    config_train=config_train, 
                                    device=device, 
                                    dir_path=dir_path, 
                                    unfreeze_layers=False,
                                    layers_to_unfreeze=None,
                                    verbose=False)
                                    
        print(f"Testing for seed {seed_num}...")
        test(gas=gas, 
             config_arch=config_arch, 
             outputs_points=outputs_points, 
             freeze_layers=[], 
             geometry_layer=False,
             dir_path=dir_path, 
             trained_pth=saved_model_path, 
             device=device)
             
        print(f"Finished seed {seed_num}. Results saved to {dir_path}\n")

if __name__ == "__main__":
    run_multiple_seeds()
