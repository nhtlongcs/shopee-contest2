import yaml 
import torch 
import torch.nn
from torch.utils.data import DataLoader 
from torch.utils import data
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm 
from torchnet import meter

from workers.trainer import Trainer 
from metrics.metric import Accuracy
from losses.losses import * 
from datasets.products import *
from models.model import * 

import argparse 

def get_instance(config, **kwargs):
    assert 'name' in config 
    config.setdefault('args', {})
    if config['args'] is None: config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)

def train(config):
    assert config is not None, "Do not have config file!"
    # print(config)
    dev_id = 'cuda:{}'.format(config['gpus']) \
        if torch.cuda.is_available() and config.get('gpus', None) is not None \
            else 'cpu'
    device = torch.device(dev_id)

    pretrained_path = config['pretrained']
    pretrained = None 
    # Load datasets
    train_dataset = get_instance(config['dataset']['train'])
    train_dataloader = DataLoader(train_dataset,
                            **config['dataset']['train']['loader']
    )
    val_dataset = get_instance(config['dataset']['val'])
    val_dataloader = DataLoader(val_dataset,
                            **config['dataset']['val']['loader']
    )

    # Define network 
    model = get_instance(config['model']).to(device)
    # print(model)

    # Define loss
    criterion = get_instance(config['loss']).to(device) 

    # Define optimizer
    optimizer = get_instance(config['optimizer'], 
                        params = model.parameters())
    print(optimizer)
    # Define scheduler
    scheduler = get_instance(config['scheduler'],
                        optimizer = optimizer)

    # Define metrics 
    metric = get_instance(config['metric'][0])

    # Create trainer 
    trainer = Trainer(device=device,
                        config = config,
                        model = model,
                        criterion = criterion,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        metric = metric)
    
    # Train
    trainer.train(train_dataloader = train_dataloader,
                    val_dataloader = val_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default = None)

    args = parser.parse_args()

    config_path = args.config
    print(config_path)
    config = yaml.load(open(config_path, 'r'), Loader = yaml.Loader)
    config['gpus'] = args.gpus 
    train(config)