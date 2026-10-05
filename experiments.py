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

gas = 'Ar'
training_type = 'geometry'

dir_path = f'./{gas}/{training_type}'
os.makedirs(dir_path, exist_ok=True)

config_Ar = config[gas]
config_arch = config['nn_arch']
config_train = config['training']

outputs_points = [i+1 for i in range(0, 10)]

saved_model_path = None

max_r2_list = []

# Train for each output individually to obtain the 'geometry layer'
'''for i, outputs_point in enumerate(outputs_points):

    print(f'Training output point {outputs_point}')

    if i == 0:
        freeze_layers = []
        unfreeze_layers = False
    else:
        freeze_layers = [0, 1]
        unfreeze_layers = False #True

    saved_model_path, max_r2 = train(gas=gas, outputs_points=[outputs_point], freeze_layers=freeze_layers,
                                     geometry_layer=False, model_pth=saved_model_path, config_gas=config_Ar,
                                     config_arch=config_arch, config_train=config_train, device=device,
                                     dir_path=dir_path, unfreeze_layers=unfreeze_layers,
                                     layers_to_unfreeze=freeze_layers)

    max_r2_list.append(max_r2)

print(max(range(len(max_r2_list)), key=max_r2_list.__getitem__))

print(max_r2_list)'''

saved_model_path = f"{dir_path}/trained_model_10.pth"
print(saved_model_path)


gas = 'O2'
config_O2 = config[gas]

saved_model_path, _ = train(gas=gas, outputs_points=outputs_points, freeze_layers=[], geometry_layer=True,
                            model_pth=saved_model_path, config_gas=config_O2, config_arch=config_arch,
                            config_train=config_train, device=device, dir_path=dir_path, unfreeze_layers=True,
                            layers_to_unfreeze=None)

saved_model_path = f"{dir_path}/trained_model_1_2_3_4_5_6_7_8_9_10.pth"

test(gas=gas, config_arch=config_arch, outputs_points=outputs_points, freeze_layers=[], geometry_layer=True,
     dir_path=dir_path, trained_pth=saved_model_path, device=device)


train_baseline = False

if train_baseline:
    saved_model_path = None
    gas = 'O2'

    training_type = 'baseline'

    dir_path = f'./{gas}/{training_type}'
    os.makedirs(dir_path, exist_ok=True)

    config_O2 = config['O2_init']

    saved_model_path, _ = train(gas=gas, outputs_points=outputs_points, freeze_layers=[], geometry_layer=False,
                             model_pth=saved_model_path, config_gas=config_O2, config_arch=config_arch,
                             config_train=config_train, device=device, dir_path=dir_path, unfreeze_layers=False,
                             layers_to_unfreeze=[])

    saved_model_path = f"{dir_path}/trained_model_1_2_3_4_5_6_7_8_9_10.pth"

    test(gas=gas, config_arch=config_arch, outputs_points=outputs_points, freeze_layers=[], geometry_layer=False,
         dir_path=dir_path, trained_pth=saved_model_path, device=device)