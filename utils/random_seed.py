import torch
import numpy as np
import random

SEED = 18120134


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
