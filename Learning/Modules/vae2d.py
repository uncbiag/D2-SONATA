import os
import abc
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

#######################################################
# Utils ###############################################
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

k = 1
global_padding = k
global_kz = 2 * k + 1



##########################################################################################################
##########################################################################################################
############################################ Utility Modules #############################################
##########################################################################################################
##########################################################################################################



class Encoder3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, 2, 1),  # x.size() / 2
            nn.BatchNorm3d(out_channels), 
            nn.LeakyReLU(0.2) 
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, mode = 'trilinear', align_corners = True):
        super(Decoder3D, self).__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = mode, align_corners = align_corners), # x.size() * 2
            nn.ReplicationPad3d(1), #  same-padding on the boundaries -> x.size() * 2 + 2
            nn.Conv3d(in_channels, out_channels, 3, 1), # Deconvolution: x.size() * 2
            nn.BatchNorm3d(out_channels, 1.e-3), 
            nn.LeakyReLU(0.2) 
        )
    def forward(self, x):
        return self.decoder(x)

class SameSizeConv2D(nn.Module):
    def __init__(self, channels):
        super(SameSizeConv2D, self).__init__()
        self.net = nn.Sequential(
            nn.ReplicationPad2d(1),  # x.size() + 2
            nn.Conv2d(channels, channels, 3, 1) # x.size() 
        )
    def forward(self, x):
        return self.net(x)

class Encoder2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder2D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),  # x.size() / 2
            nn.BatchNorm2d(out_channels), 
            nn.LeakyReLU(0.2) 
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, mode = 'bilinear', align_corners = True):
        super(Decoder2D, self).__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = mode, align_corners = align_corners), # x.size() * 2
            nn.ReplicationPad2d(1), #  same-padding on the boundaries -> x.size() * 2 + 2
            nn.Conv2d(in_channels, out_channels, 3, 1), # Deconvolution: x.size() * 2
            nn.BatchNorm2d(out_channels, 1.e-3), 
            nn.LeakyReLU(0.2) 
        )
    def forward(self, x):
        return self.decoder(x)


############################################################################
############################# Base Modules: 3D #############################
############################################################################


class VAE3D(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3, img_size=128, latent_variable_size=5000):
        super(VAE3D, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size 
        self.n_layers = int(math.log(self.img_size, 2) - 1) # deep as log_2(img_size)-1 -> smallest img_size := 2
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(self.img_size * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 3), latent_variable_size) # For mu
        self.fc2 = nn.Linear(int(self.img_size * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 3), latent_variable_size) # For sigma
        self.d1  = nn.Linear(latent_variable_size, int(2 * self.img_size * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 3)) # For decoder reshape
        #self.final_activation = nn.Sigmoid() # n_batch, self.nc, (x.size())
        #self.final_activation =  nn.Conv3d(self.nc, self.nc, 1) # n_batch, self.nc, (x.size())
        self.final_activation =  nn.Linear(int(self.out_channels * self.img_size ** 3), int(self.out_channels * self.img_size ** 3)) # n_batch, self.nc, (x.size())
        
        # Define encoders
        self.encoders = [Encoder3D(self.in_channels, self.img_size)]
        for i_layer in range(self.n_layers-2): # x.size() / (2 * 2 ** i_layer)
            self.encoders.append(Encoder3D(self.img_size * 2 ** i_layer, 2 * self.img_size * 2 ** i_layer))
        self.encoders.append(Encoder3D(self.img_size * 2 ** (self.n_layers - 2), self.img_size * 2 ** (self.n_layers - 2))) # x.size() / (2 ** self.n_layers)
        self.encoders = nn.Sequential(*self.encoders)
        # Define decoders
        self.decoders = []
        for i_layer in range(self.n_layers-2, -1, -1): # x.size() / (2 ** self.n_layers) * (2 ** (self.n_layers - i_layer))
            self.decoders.append(Decoder2D(2 * self.img_size * 2 ** i_layer, self.img_size * 2 ** i_layer))
        self.decoders.extend([
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True), # x.size() / (2 * 2)
            nn.ReplicationPad2d(1), nn.Conv2d(self.img_size, self.out_channels, 3, 1)
        ])
        self.decoders = nn.Sequential(*self.decoders)

    def encode(self, x):
        h = self.encoders(x)
        h = h.view(-1, int(self.img_size * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 3))
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.device is not 'cpu':
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        h = self.relu(self.d1(z))
        h = h.view(-1, int(2 * self.img_size * 2 ** (self.n_layers - 2)), int(self.img_size / (2 ** self.n_layers)), int(self.img_size / (2 ** self.n_layers)), int(self.img_size / (2 ** self.n_layers)))
        h = self.decoders(h)
        #return self.final_activation(h) # For Conv as final_activation
        return self.final_activation(h.view(-1, int(self.out_channels * self.img_size ** 3))).view(-1, self.out_channels, self.img_size, self.img_size, self.img_size) # For Linear as final_activation

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.img_size, self.img_size, self.img_size))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.img_size, self.img_size, self.img_size))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

############################################################################
############################# Base Modules: 2D #############################
############################################################################

class VAE2D(nn.Module):
    def __init__(self, args, in_channels=2, out_channels=2, img_size = 128, n_filter=128, latent_variable_size=5000):
        super(VAE2D, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.n_filter = n_filter
        self.n_layers = int(math.log(self.img_size, 2) - 1) # deep as log_2(img_size)-1 -> smallest img_size := 2
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(self.n_filter * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 2), latent_variable_size) # For mu
        self.fc2 = nn.Linear(int(self.n_filter * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 2), latent_variable_size) # For sigma
        self.d1  = nn.Linear(latent_variable_size, int(2 * self.n_filter * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 2)) # For decoder reshape
        #self.final_activation = nn.Sigmoid() # n_batch, self.out_channels, (x.size())
        #self.final_activation =  nn.Conv2d(self.out_channels, self.out_channels, 1) # n_batch, self.out_channels, (x.size())
        self.final_activation =  nn.Linear(int(self.out_channels * self.img_size ** 2), int(self.out_channels * self.img_size ** 2)) # n_batch, self.nc, (x.size())
        
        # Define encoders
        self.encoders = [Encoder2D(self.in_channels, self.n_filter)]
        for i_layer in range(self.n_layers-2): # x.size() / (2 * 2 ** i_layer)
            self.encoders.append(Encoder2D(self.n_filter * 2 ** i_layer, 2 * self.n_filter * 2 ** i_layer))
            self.encoders.append(SameSizeConv2D(2 * self.n_filter * 2 ** i_layer))
        self.encoders.append(Encoder2D(self.n_filter * 2 ** (self.n_layers - 2), self.n_filter * 2 ** (self.n_layers - 2))) # x.size() / (2 ** self.n_layers)
        self.encoders = nn.Sequential(*self.encoders)
        # Define decoders
        self.decoders = []
        for i_layer in range(self.n_layers-2, -1, -1): # x.size() / (2 ** self.n_layers) * (2 ** (self.n_layers - i_layer))
            self.decoders.append(Decoder2D(2 * self.n_filter * 2 ** i_layer, self.n_filter * 2 ** i_layer))
            self.decoders.append(SameSizeConv2D(self.n_filter * 2 ** i_layer))
        self.decoders.extend([
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True), # x.size() / (2 * 2)
            nn.ReplicationPad2d(1), nn.Conv2d(self.n_filter, self.out_channels, 3, 1)
        ])
        self.decoders = nn.Sequential(*self.decoders)

    def encode(self, x):
        h = self.encoders(x)
        h = h.view(-1, int(self.n_filter * 2 ** (self.n_layers - 2) * (self.img_size / (2 ** self.n_layers)) ** 2))
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.device is not 'cpu':
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        h = self.relu(self.d1(z))
        h = h.view(-1, int(2 * self.n_filter * 2 ** (self.n_layers - 2)), int(self.img_size / (2 ** self.n_layers)), int(self.img_size / (2 ** self.n_layers)))
        h = self.decoders(h)
        return self.final_activation(h.view(-1, int(self.out_channels * self.img_size ** 2))).view(-1, self.out_channels, self.img_size, self.img_size)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.img_size, self.img_size, self.img_size))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.img_size, self.img_size))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z) # (n_batch, out_channels, img_size, img_size)
        return res, mu, logvar




##################################################################################

class VAE2D_32(nn.Module):
    def __init__(self, nc, ngf, latent_variable_size):
        super(VAE2D, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.e2 = nn.Conv2d(32, 32*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(32*2)

        self.e3 = nn.Conv2d(32*2, 32*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(32*4)

        self.e4 = nn.Conv2d(32*4, 32*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(32*8)

        self.e5 = nn.Conv2d(32*8, 32*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(32*8)

        self.fc1 = nn.Linear(32*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(32*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, 32*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, 32, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, 32, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


##################################################################################


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAE2D_a(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, in_channels, out_channels, SepActivate = None, z_dim=10): #, nc=3
        super(VAE2D_a, self).__init__()
        print('=========== Register VAE2D_a ===========')
        
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.SepActivate = SepActivate

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            #nn.ReLU(True),
            #nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B, nc, 64, 64
        )
        if self.SepActivate:
            self.outcV = OutConv(32, self.SepActivate[0], 4, 2, 1)
            self.outcD = OutConv(32, self.SepActivate[1], 4, 2, 1)
        else:
            self.outc = OutConv(32, out_channels, 4, 2, 1)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        y = self.decode(z)
        if self.SepActivate:
            y = torch.cat([self.outcV(y), self.outcD(y)], dim = 1)
        else:
            y = self.outc(y)

        return x_recon, mu, logvar



class VAE2D_b(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, in_channels, out_channels, SepActivate = None, z_dim=10): #, nc=3
        super(VAE2D_b, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SepActivate = SepActivate
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
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
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            #nn.ReLU(True),
            #nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  nc, 64, 64
        )
        if self.SepActivate:
            self.outcV = OutConv(32, self.SepActivate[0], 4, 2, 1)
            self.outcD = OutConv(32, self.SepActivate[1], 4, 2, 1)
        else:
            self.outc = OutConv(32, out_channels, 4, 2, 1)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        #x_recon = self.decode(z).view(x.size())
        y = self.decode(z)
        if self.SepActivate:
            y = torch.cat([self.outcV(y), self.outcD(y)], dim = 1)
        else:
            y = self.outc(y)

        return y, mu, logvar





