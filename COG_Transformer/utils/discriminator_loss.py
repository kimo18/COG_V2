import numpy as np
import torch
import torch_dct as dct #https://github.com/zh217/torch-dct

import os

def disc_l2_loss(disc_value):
    
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def adv_disc_l2_loss(fake_disc_value, real_disc_value):
    kb = fake_disc_value.shape[0]
    ka = real_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la + lb
