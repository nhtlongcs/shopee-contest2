import argparse 
import yaml
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from torchnet import meter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from workers.trainer import Trainer 
from metrics.metric import Accuracy
from losses.losses import * 
from datasets.products import *
from models.model import * 
# from utils.getter import *

import argparse 

def get_instance(config, **kwargs):
    assert 'name' in config 
    config.setdefault('args', {})
    if config['args'] is None: config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)

import time
import os
import csv

@torch.no_grad()
def evaluate(config):
    dev_id = 'cuda:{}'.format(config['gpus']) \
            if torch.cuda.is_available() and config.get('gpus', None) is not None \
            else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config["pretrained"]
    output_dir = config['output']
    try:
        os.makedirs(output_dir)
    except:
        pass
    assert os.path.exists(pretrained_path)
    pretrained = torch.load(pretrained_path, map_location=dev_id)
    for item in ["model"]:
        config[item] = pretrained["config"][item]
    print(config)

    # 1: Load datasets
    dataset = get_instance(config['dataset']['test'])
    dataloader = DataLoader(dataset,
                            **config['dataset']['test']['loader']
    )
    # dataset = Product(root_dir='data/SHREC20_test_ext', csv_path='test.csv', 
    #                   type='render', ring_id=[4,0,1], is_train=False,
    #                   return_name=True)
    # dataloader = DataLoader(dataset, batch_size=128)


    # 2: Define network
    model = get_instance(config['model']).to(device)
    state_dict = pretrained['model_state_dict']
    # state_dict = { k.replace('resnet', 'cnn'): v for k, v in state_dict.items() }
    model.load_state_dict(state_dict)

    # print(model)
    # 5: Define metrics
    metric = Accuracy()#ConfusionMatrix(nclasses=40)

    # exit(0)
    lines = [['id', 'predict', *[f'score_{i}' for i in range(model.nclasses)]]]
    print(lines)
    tbar = tqdm(dataloader)
    for idx, (inps, lbls) in enumerate(tbar):
        # Get network output
        model.eval()
        # start = time.time()
        outs = model(inps.to(device))
        outs = F.softmax(outs, dim=1)
        
        # print("Prediction time: %f" % (time.time()-start))
        for out, lbl in zip(outs, lbls):
            lbl2 = torch.argmax(out)
            lines.append([int(lbl), str(int(lbl2)).zfill(2), *out.detach().cpu().numpy().tolist()])
        # Post-process output for true prediction
        # score = metric.calculate(out, lbl.to(device))
        # metric.update(score)
    csv.writer(open('test_product.csv', 'w')).writerows(lines)
    # metric.display()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default = None)

    args = parser.parse_args()

    config_path = args.config
    print(config_path)
    config = yaml.load(open(config_path, 'r'), Loader = yaml.Loader)
    config['gpus'] = args.gpus 
    evaluate(config)