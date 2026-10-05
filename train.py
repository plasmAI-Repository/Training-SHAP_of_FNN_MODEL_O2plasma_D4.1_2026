import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
import torch.nn as nn
import time
import matplotlib.pyplot as plt

import pandas as pd
import json
import numpy as np
from data_loader import MergedDataset, unscale_min_max
from model import Model, CombinedModel
from loss import WeightedMSE
from utils import train_regression_model


def train(gas, config_gas, config_arch, config_train, outputs_points, freeze_layers, dir_path, model_pth,
          device, geometry_layer, unfreeze_layers, layers_to_unfreeze, verbose=False):

    csv_file_path_train = f'./v3_3_1_data/train_data_v3_3_1_{gas}.txt'
    dataset = MergedDataset(csv_file_path_train, stats_file="dataset_stats.json", columns_idx=outputs_points)

    batch = config_gas['batch_size']
    lr = config_gas['lr']
    weight_decay = config_gas['weight_decay']

    neurons_per_layer = config_arch['neurons_per_layer']
    layers = config_arch['layers']

    epochs = config_train['epochs']
    patience = config_train['patience']

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    if geometry_layer:
        pretrained_models = []

        for model_idx in outputs_points:
            model_i = Model(h1=neurons_per_layer, num_layers=layers, freeze_layers=[], output_size=1)
            model_i.load_state_dict(torch.load(f'{dir_path}/trained_model_{model_idx}.pth'))
            pretrained_models.append(model_i)

        model = CombinedModel(h1=neurons_per_layer, num_layers=layers, pretrained_models=pretrained_models)
    else:
        model = Model(h1=neurons_per_layer,
                      num_layers=layers,
                      freeze_layers=freeze_layers,
                      output_size=len(outputs_points))

    if model_pth and not geometry_layer:
        model.load_state_dict(torch.load(model_pth))
        print("Loaded pre-trained model.")
        lr /= 20
        print("Adjusted learning rate.")

    l1_lambda = float(config_train.get('l1_lambda', 0.0))
    print(f"L1 regularization lambda: {l1_lambda}")

    criterion = WeightedMSE(reduction='mean', outputs_points=outputs_points, device=device, model=model, l1_lambda=l1_lambda)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    #outputs_idx_str = '_'.join(map(str, outputs_points))
    if geometry_layer:
        outputs_idx_str = 'geometry'
    else:
        outputs_idx_str = 'base'

    start_time = time.time()
    trained_model, losses, val_losses, mean_r2_scores = train_regression_model(model=model,
                                                                               train_loader=train_loader,
                                                                               val_loader=val_loader,
                                                                               criterion=criterion,
                                                                               optimizer=optimizer,
                                                                               num_epochs=epochs,
                                                                               device=device,
                                                                               patience=patience,
                                                                               scheduler=scheduler,
                                                                               dir_path=dir_path,
                                                                               outputs_idx_str=outputs_idx_str,
                                                                               unfreeze_layers=unfreeze_layers,
                                                                               layers_to_unfreeze=layers_to_unfreeze,
                                                                               geometry_layer=geometry_layer)
    end_time = time.time()

    model_save_path = f'{dir_path}/trained_model_{outputs_idx_str}.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Time for training: {(end_time-start_time):.4f}")

    print(f"Time for training: {(end_time-start_time):.4f}")

    # --------------------------------------------------------------------------
    # Generate predictions on the Training Set (comparison_train.txt)
    # --------------------------------------------------------------------------
    print("Generating predictions on the training set...")
    
    # Use a non-shuffled loader for consistent ordering/evaluation
    train_loader_pred = DataLoader(train_dataset, batch_size=batch, shuffle=False)
    
    trained_model.eval()
    all_inputs = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in train_loader_pred:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = trained_model(inputs)
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Unscale data
    with open("dataset_stats.json", 'r') as f:
        scaling_info = json.load(f)

    # Prepare column names
    # Inputs: Power, Pressure, Feed, Vbias
    input_cols = ['Power', 'Pressure', 'Feed', 'Vbias']
    
    # Output columns (Targets & Preds)
    # Based on outputs_points which are indices 1..10
    output_cols_names = [f"Point{i}_EtchRate" for i in outputs_points]

    # Unscale
    # Note: dimensions of all_inputs is (N, 4), all_targets is (N, 10), all_preds is (N, 10)
    unscaled_inputs = unscale_min_max(all_inputs.copy(), input_cols, scaling_info)
    unscaled_targets = unscale_min_max(all_targets.copy(), output_cols_names, scaling_info)
    unscaled_preds = unscale_min_max(all_preds.copy(), output_cols_names, scaling_info)

    # Create DataFrame
    # Structure: Inputs | Targets | Predictions
    # Column naming convention matches existing files or simple readable format
    
    df_inputs = pd.DataFrame(unscaled_inputs, columns=input_cols)
    df_targets = pd.DataFrame(unscaled_targets, columns=output_cols_names)
    df_preds = pd.DataFrame(unscaled_preds, columns=[f"{col}_pred" for col in output_cols_names])
    
    comparison_train_df = pd.concat([df_inputs, df_targets, df_preds], axis=1)
    
    comp_train_path = f'{dir_path}/comparison_train.txt'
    comparison_train_df.to_csv(comp_train_path, index=False, sep='\t')
    print(f"Training set predictions saved to {comp_train_path}")
    # --------------------------------------------------------------------------

    if verbose:
        plt.plot(list(range(len(losses))), losses, label="Training Loss")
        plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
        plt.ylabel("Total Loss")
        plt.xlabel("Epoch")
        plt.title("Training & Validation Loss Progression")
        plt.legend()
        plt.tick_params(axis='both', which='major')
        plt.show()

    return model_save_path, max(mean_r2_scores)
