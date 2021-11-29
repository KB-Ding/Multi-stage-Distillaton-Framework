import random
import os
import numpy as np
import torch

def set_seed(seed: int, for_multi_gpu: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if for_multi_gpu:
        torch.cuda.manual_seed_all(seed)