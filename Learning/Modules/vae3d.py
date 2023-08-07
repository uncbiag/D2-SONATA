import os
import abc
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

#######################################################
######################## Utils ########################
#######################################################

from utils import *
from ODE.FD import FD_torch
from Learning.Modules.utils import *


'''
Conv3d:

out_size = (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

kernel_size := 2 * k + 1 
padding := k
dilation := 1 (default)
stride := 1 (default)
'''



def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

##################################################################################


class VAE3D_a(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, in_channels, out_channels, SepActivate = None, z_dim = 1024, ndf = 32): #, nc=3
        super(VAE3D_a, self).__init__()
        print('=========== Register VAE3D_a ===========')
        
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SepActivate = SepActivate

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, ndf, 4, 2, 1),          # B,  32, 16, 16, 16
            nn.BatchNorm3d(ndf),
            nn.ReLU(True),
            nn.Conv3d(32, ndf*2, 4, 2, 1),          # B,  64,  8,  8,  8
            nn.BatchNorm3d(ndf*2),
            nn.ReLU(True),
            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1),          # B,  128,  4,  4,  4
            nn.BatchNorm3d(ndf*4),
            nn.ReLU(True),
            nn.Conv3d(ndf*4, ndf*8, 4, 2, 1),          # B,  256,  2,  2,  2
            nn.BatchNorm3d(ndf*8),
            nn.ReLU(True),
            #nn.Conv3d(ndf*8, ndf*8, 4, 2, 1),         # B, 256,  1,  1,  1
            #nn.BatchNorm3d(ndf*8),
            #nn.ReLU(True),
            View((-1, ndf*8*2*2*2)),                     # B, 256
            nn.Linear(ndf*8*2*2*2, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, ndf*8*2*2*2),                   # B, 256
            View((-1, ndf*8, 2, 2, 2)),                 # B, 256,  2,  2,  2
            nn.ReLU(True),
            #nn.ConvTranspose3d(ndf*8, ndf*8, 4, 2, 1), # B,  256,  2,  2,  2
            #nn.BatchNorm3d(ndf*8),
            #nn.ReLU(True),
            nn.ConvTranspose3d(ndf*8, ndf*4, 4, 2, 1), # B,  128,  4,  4,  4
            nn.BatchNorm3d(ndf*4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ndf*4, ndf*2, 4, 2, 1), # B,  64,  8,  8,  8
            nn.BatchNorm3d(ndf*2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ndf*2, ndf, 4, 2, 1),   # B,  32, 16, 16, 16
            nn.BatchNorm3d(ndf),
            nn.ReLU(True),
            nn.ConvTranspose3d(ndf, ndf, 4, 2, 1),   # B,  32, 32, 32, 32
            nn.BatchNorm3d(ndf),
            nn.ReLU(True),
            #nn.ConvTranspose3d(32, 32, 4, 2, 1),  # B, nc, 64, 64, 64
        )
        if self.SepActivate:
            self.outcV = nn.Sequential(OutConv(32, self.SepActivate[0]))
            self.outcD = nn.Sequential(OutConv(32, self.SepActivate[1]))
        else:
            self.outc = nn.Sequential(OutConv(32, out_channels))

        #self.weight_init()

    '''def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
'''
    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        y = self.decoder(z)
        if self.SepActivate:
            y = torch.cat([self.outcV(y), self.outcD(y)], dim = 1)
        else:
            y = self.outc(y)

        return [[y, mu, logvar]]



# TODO
class VAE3D_b(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, in_channels, out_channels, SepActivate = None, z_dim = 10): #, nc=1
        super(VAE3D_b, self).__init__()
        print('=========== Register VAE3D_b ===========')

        self.z_dim = z_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SepActivate = SepActivate

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv3d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv3d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv3d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4*4)),                  # B, 512
            nn.Linear(32*4*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose3d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 32, 4, 2, 1), # B,  32, 32, 32
            #nn.ReLU(True),
            #nn.ConvTranspose3d(32, 32, 4, 2, 1), # B,  nc, 64, 64
        )
        if self.SepActivate:
            self.outcV = OutConv(32, self.SepActivate[0])
            self.outcD = OutConv(32, self.SepActivate[1])
        else:
            self.outc = OutConv(32, out_channels)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        #x_recon = self.decode(z).view(x.size())
        y = self.decoder(z)
        if self.SepActivate:
            y = torch.cat([self.outcV(y), self.outcD(y)], dim = 1)
        else:
            y = self.outc(y)

        return y, mu, logvar



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv3d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv3d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()



