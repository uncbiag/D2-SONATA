import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage import gaussian_filter


###############################################################################
#################################  Archieved  #################################
###############################################################################


'''class GaussianLayer(nn.Module):
    def __init__(self, kernel_size = 3):
        super(GaussianLayer, self).__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size, stride = 1, padding = int(kernel_size / 2), bias = None)
        self.weights_init(kernel_size)

    def forward(self, X):
        # X: (slc, row, col)
        return self.conv((X.unsqueeze(0)).unsqueeze(0))[0, 0]
    
    def weights_init(self, kernel_size):
        w = np.zeros((kernel_size, kernel_size, kernel_size))
        center = int(kernel_size / 2)
        w[center, center, center] = 1
        k = gaussian_filter(w, sigma = 1)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))'''


class GaussianLayer(nn.Module):
    '''Fixed Gaussian Layer'''
    def __init__(self, kernel_size = 3, sigma = 1):
        super(GaussianLayer, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.conv = nn.Conv3d(1, 1, self.kernel_size, stride = 1, padding = int(kernel_size / 2), bias = None)
        self.weights_init()

    def forward(self, X):
        # X: (slc, row, col)
        return self.conv((X.unsqueeze(0)).unsqueeze(0))[0, 0]
    
    def weights_init(self):
        w = np.zeros((self.kernel_size, self.kernel_size, self.kernel_size))
        center = int(self.kernel_size / 2)
        w[center, center, center] = 1
        k = gaussian_filter(w, sigma = self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            #f.requires_grad = False

def smoothing_params(func_params, smoother, smooth_param_type = 'diff'):
    for full_name, p in func_params:
        p_name = full_name.split('.')[-1]
        if 'V' in p_name and 'adv'in smooth_param_type:
            p.data = smoother(p.data)
        if 'D' in p_name and 'diff' in smooth_param_type:
            p.data = smoother(p.data)
    return



################################################################



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def zero_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        init.uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)