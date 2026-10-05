import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, h1, num_layers, freeze_layers, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()  # List to hold layers

        # First layer
        self.layers.append(nn.Linear(4, h1))

        # Add hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))  # Each hidden layer has h1 neurons

        # Output layer
        self.out = nn.Linear(h1, output_size)

        if len(freeze_layers) > 0:
            print('Freezing layers.')

            # Freeze selected layers
            for layer_idx in freeze_layers:
                if layer_idx < len(self.layers):  # Ensure valid layer index
                    for param in self.layers[layer_idx].parameters():
                        param.requires_grad = False

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.layers[i](x))
        x = self.out(x) # Output layer
        # x = F.softplus(x)
        x = F.elu(x)
        #x = F.relu(x)
        # x = self.output_activation(x)
        return x


class CombinedModel(nn.Module):

    def __init__(self, h1, num_layers, pretrained_models):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(2, h1))

        # Add hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))

        self.layers[2] = pretrained_models[6].layers[2]

        #self.layers = pretrained_models[-1].layers

        for param in self.layers[2].parameters():
            param.requires_grad = False

        # Freeze the last layer from each pre-trained model
        self.pretrained_layers = nn.ModuleList(
            [model.out for model in pretrained_models])

        # Freeze the last layers (these layers will not be trainable)
        for layer in self.pretrained_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.elu(self.layers[i](x))

        # Collect the outputs of the last layer from each pre-trained model
        last_layer_outputs = [layer(x) for layer in self.pretrained_layers]

        # Concatenate the outputs of the last layers to form (h, 10)
        combined_output = torch.cat(last_layer_outputs, dim=-1)

        return combined_output

