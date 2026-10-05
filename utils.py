import torch
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import time


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}.')
    return device


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience,
                           scheduler, dir_path, outputs_idx_str, unfreeze_layers, layers_to_unfreeze, geometry_layer,
                           verbose=False):
    model.to(device)

    train_losses, val_losses = [], []
    mse_values = []

    best_val_loss = float("inf")
    early_counter = 0

    best_model_state = None

    for epoch in range(num_epochs):

        if epoch == 100:
            if unfreeze_layers:

                if not geometry_layer:
                    for layer_idx in layers_to_unfreeze:
                        for param in model.layers[layer_idx].parameters():
                            param.requires_grad = True
                else:
                    for layer in model.pretrained_layers:
                        for param in layer.parameters():
                            param.requires_grad = True

                    for param in model.layers[-1].parameters():
                        param.requires_grad = True

                # Reduce learning rate when unfreezing
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5

                print(f"Layers unfrozen at epoch {epoch} and learning rate adjusted.")

        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            train_loss += loss.item() * inputs.size(0)

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation step
        val_loss = 0
        r2_values, mean_r2_scores = [], []
        all_predictions, all_targets = [], []

        model.eval()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_outputs = model(inputs)
                loss_val = criterion(val_outputs, targets)
                val_loss += loss_val.item() * inputs.size(0)

                all_predictions.append(val_outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                for col in range(val_outputs.shape[1]):
                    pred_col = val_outputs[:, col].cpu().numpy()
                    target_col = targets[:, col].cpu().numpy()
                    r2 = r2_score(target_col, pred_col)
                    r2_values.append(r2)
                    # Calculate MSE for the current column
                    mse = mean_squared_error(target_col, pred_col)
                    mse_values.append(mse)

        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        # Create a DataFrame with meaningful column names
        num_targets = all_targets.shape[1]
        pred_columns = [f'Pred_{i + 1}' for i in range(num_targets)]
        target_columns = [f'Target_{i + 1}' for i in range(num_targets)]
        df_val = pd.DataFrame(np.hstack((all_predictions, all_targets)), columns=pred_columns + target_columns)

        # Save CSV file
        df_val.to_csv(f'{dir_path}/val_set_results_{outputs_idx_str}.csv', index=False, sep=';')

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        mean_r2 = np.mean(r2_values)
        mean_mse = np.mean(mse_values)  # Calculate mean MSE for all columns
        mean_r2_scores.append(mean_r2)

        if scheduler:
            scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == num_epochs-1:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

            if verbose:
                for col in range(outputs.shape[1]):
                    print(f"  Validation R² for column {col + 1}: {r2_values[col]:.4f}")
                    print(f"  Validation MSE for column {col + 1}: {mse_values[col]:.4f}")

            print(f"  Validation mean R² : {mean_r2:.4f}")
            print(f"  Validation mean MSE: {mean_mse:.4f}")

        # Early stopping implementation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0  # reset the counter if validation loss improves
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_counter += 1
            if early_counter >= patience:
                print(f"Early stopping triggered at {epoch + 1} epoch")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, mean_r2_scores


def test_model(model, test_loader, criterion, device):

    model.to(device)
    model.eval()

    test_loss = 0
    r2_values, mse_values = [], []
    all_predictions, all_targets = [], []

    with torch.no_grad():
        start_time = time.time()
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            for col in range(outputs.shape[1]):
                pred_col = outputs[:, col].cpu().numpy()
                target_col = targets[:, col].cpu().numpy()
                r2 = r2_score(target_col, pred_col)
                r2_values.append(r2)
                mse = mean_squared_error(target_col, pred_col)
                mse_values.append(mse)

        end_time = time.time()

    elapsed_time = end_time - start_time
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute overall MSE, MAE, and R²
    overall_mse = mean_squared_error(all_targets, all_predictions)
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_r2 = r2_score(all_targets, all_predictions)

    print(f"\nOverall Performance Metrics:")
    print(f"  Mean MSE: {overall_mse:.7f}")
    print(f"  Mean MAE: {overall_mae:.7f}")
    print(f"  Mean R²: {overall_r2:.5f}")

    test_loss /= len(test_loader.dataset)

    return test_loss, all_predictions, all_targets, r2_values, elapsed_time


def unscale(data, column_names, scaling_info):
    unscaled_data = data
    num_columns = data.shape[1]
    for i, column in enumerate(column_names):
        if i >= num_columns:  # Check if index exceeds the number of columns
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        mean = scaling_info[column]['mean']
        std = scaling_info[column]['std']
        unscaled_data[:, i] = (data[:, i] * std) + mean  # Reverse normalization
    return unscaled_data
