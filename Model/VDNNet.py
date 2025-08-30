import numpy as np
import torch
from torch import nn


class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=1, keepdim=True)

