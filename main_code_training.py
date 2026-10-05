import yaml
import os

from utils import set_seed, setup_device
from train import train
from test import test

device = setup_device()

seed_num = 41
set_seed(seed_num)
# Open .yaml file to get configuration
with open('./config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

training_type = 'elu_output_no_l1_dec25'
gas = 'O2'

dir_path = f'./{gas}/{training_type}'
os.makedirs(dir_path, exist_ok=True)
#saved_model_path = f"{dir_path}/trained_model_O2_base.pth"

config_O2 = config[gas]
config_arch = config['nn_arch']
config_train = config['training']
outputs_points = [i+1 for i in range(0, 10)]

saved_model_path, _ = train(gas=gas, outputs_points=outputs_points, freeze_layers=[], geometry_layer=False,
                            model_pth=False, config_gas=config_O2, config_arch=config_arch,
                            config_train=config_train, device=device, dir_path=dir_path, unfreeze_layers=False,
                            layers_to_unfreeze=None,verbose=True)

# saved_model_path = f"{dir_path}/trained_model_base.pth" #uncomment this line in order to run the test only. Make sure to write the correct directory
test(gas=gas, config_arch=config_arch, outputs_points=outputs_points, freeze_layers=[], geometry_layer=False,
     dir_path=dir_path, trained_pth=saved_model_path, device=device)

