import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_1 = nn.Linear(in_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, out_dim)
    
    def forward(self, x):
        """
        x: the current observation
        returns: mean action (actor) or value function (critic)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x