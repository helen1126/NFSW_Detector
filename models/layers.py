from math import sqrt
from torch import FloatTensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=False, residual=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if residual:
            if in_features != out_features:
                self.residual = nn.Conv1d(in_features, out_features, kernel_size=5, padding=2)
            else:
                self.residual = lambda x: x
        else:
            self.residual = lambda x: 0

    def forward(self, input, adj):
        support = input @ self.weight
        output = adj @ support
        if self.in_features != self.out_features and not isinstance(self.residual, type(lambda x: x)):
            res = input.permute(0, 2, 1)
            res = self.residual(res)
            res = res.permute(0, 2, 1)
            output = output + res
        else:
            output = output + self.residual(input)
        if self.bias is not None:
            output = output + self.bias
        return output


class DistanceAdj(Module):
    def __init__(self):
        super().__init__()
        self.sigma = Parameter(FloatTensor(1))
        nn.init.constant_(self.sigma, 0.1)

    def forward(self, batch_size, max_seqlen):
        idx = np.arange(max_seqlen)
        dist = squareform(pdist(idx.reshape(-1, 1), 'cityblock'))
        dist = torch.from_numpy(dist).float().to(self.sigma.device)
        adj = torch.exp(-dist / torch.exp(torch.tensor(1.)))
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        return adj
