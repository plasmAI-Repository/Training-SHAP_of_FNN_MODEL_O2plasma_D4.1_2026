import os, sys
base_dir = r"c:\Users\enisb\Documents\GitHub\v31model"
sys.path.insert(0, base_dir)
from model import Model
import torch
import numpy as np
import shap
import pandas as pd

model = Model(h1=20, num_layers=2, freeze_layers=[], output_size=10)
model_path = os.path.join(base_dir, "O2", "baseline_seed_3123", "trained_model_base.pth")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

class SubModel(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.full_model = full_model
    def forward(self, x):
        x = torch.nn.functional.relu(self.full_model.layers[1](x))
        x = self.full_model.out(x)
        x = torch.nn.functional.elu(x)
        return x

bg = torch.randn(10, 20)
test = torch.randn(5, 20)
sub = SubModel(model)

explainer = shap.GradientExplainer(sub, bg)
shap_vals = explainer.shap_values(test)

print("Type:", type(shap_vals))
if isinstance(shap_vals, list):
    print("List len:", len(shap_vals))
    print("First item shape:", shap_vals[0].shape)
else:
    print("Shape:", shap_vals.shape)

try:
    stacked = np.stack(shap_vals, axis=-1)
    print("Stacked shape:", stacked.shape)
except Exception as e:
    print("Error stacking:", e)
