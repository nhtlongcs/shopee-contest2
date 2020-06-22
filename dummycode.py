import yaml
import torch
from utils.getter import get_instance

config_path = 'configs/train/baseline.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
train_dataset = get_instance(config['dataset']['train'])