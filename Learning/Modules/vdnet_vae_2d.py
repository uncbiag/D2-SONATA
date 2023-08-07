
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn

from utils import *
from Learning.Modules.unet2d import *

'''https://github.com/milesial/Pytorch-UNet'''

#################################################
##################### V + D #####################
#################################################

class UNet2D_streamVscalarD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamVscalarD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 2)
        else:
            #self.unet = UNet2D_64(args, in_channels, 2)
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 2)
            else:
                self.unet = UNet2D_64(args, in_channels, 2)
        self.actv_D = nn.ReLU()
    def forward(self, x):
        outs = self.unet(x)
        VDs = []
        for out in outs: # (n_batch, 2, r, c) channels order: Phi, D
            VD = []
            Vx, Vy = stream_2D(out[:, 0], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            VD.append(torch.stack([Vx, Vy], dim = 1))
            VD.append(self.actv_D(out[:, 1]))
            VDs.append(VD)
        return VDs # list of list: (level, [V, D])
    def get_Phis(self, x):
        outs = self.unet(x)
        Phis = []
        for out in outs:
            Phis.append(out[:, 0]) # (n_batch, r, c)
        return Phis 
 
class UNet2D_streamVspectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamVspectralD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 4) # (4 channels: Phi, S, L1, L2) 
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 4)
            else:
                self.unet = UNet2D_64(args, in_channels, 4)
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        #S, L1, L2 = raw_D[:, 0], abs(raw_D[:, 1]), abs(raw_D[:, 2]) # (n_batch, r, c)
        S, L = raw_D[:, 0], self.relu(raw_D[:, 1:])
        L1 = L[:, 0]
        L2 = L[:, 1] # (n_batch, r, c)
        #L2 = L1 + L[:, 1] # (n_batch, r, c)
        U = cayley_map(S)
        Dxx = U[..., 0, 0] ** 2 * L1 + U[..., 0, 1] ** 2 * L2
        Dxy = U[..., 0, 0] * U[..., 1, 0] * L1 + U[..., 0, 1] * U[..., 1, 1] * L2
        Dyy = U[..., 1, 0] ** 2 * L1 + U[..., 1, 1] ** 2 * L2
        return torch.stack([Dxx, Dxy, Dyy], dim = 1) # (batch, 3, r, c): (Dxx, Dxy, Dyy)
    def forward(self, x):
        outs = self.unet(x)
        VDs = []
        for out in outs: # (n_batch, 4, r, c) channels order: Phi, S, L1, L2
            VD = []
            Vx, Vy = stream_2D(out[:, 0], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            VD.append(torch.stack([Vx, Vy], dim = 1))
            VD.append(self.actv_D(out[:, 1:]))
            VDs.append(VD)
        return VDs # list of list: (level, [V, D])
    def get_Ss(self, x):
        outs = self.unet(x)
        Ss = []
        for out in outs:
            Ss.append(out[:, 1])
        return Ss
    def get_Us(self, x):
        outs = self.unet(x)
        Us = []
        for out in outs:
            U = cayley_map(out[:, 1])
            Us.append(torch.stack([U[..., 0, 0], U[..., 0, 1], U[..., 1, 0], U[..., 1, 1]], dim = 1)) # (n_batch, r, c, 2, 2) -> (n_batch, 4, r, c): Uxx, Uxy, Uyx, Uyy
        return Us 
    def get_Ls(self, x): # L: eigen-values
        outs = self.unet(x)
        Ls = []
        for out in outs:
            L = self.relu(out[:, 2:]) # (n_batch, 2, r, c): L1, L2
            #L[:, 1] += L[:, 0]
            Ls.append(L) # (n_batch, 2, r, c): L1, L2
        return Ls
    def get_Phis(self, x):
        outs = self.unet(x)
        Phis = []
        for out in outs:
            Phis.append(out[:, 0]) # (n_batch, r, c)
        return Phis 


class UNet2D_streamVcholeskyD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamVcholeskyD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 4)
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 4)
            else:
                self.unet = UNet2D_64(args, in_channels, 4) # (4 channels: Phi, Lxx, Lxy, Lyy) # NOTE: should output 4 channels (2: unstable)
        # NOTE: network not efficient for tensor D learning #
        #self.actv_Phi = nn.Conv2d(1, 1, kernel_size = 1)
        #self.actv_L = nn.Conv2d(3, 3, kernel_size = 1)
        self.relu = nn.ReLU()
    def actv_D(self, raw_L):
        #Lxx, Lxy, Lyy = raw_L[:, 0], raw_L[:, 1], raw_L[:, 2]
        #Lxx, Lxy, Lyy = abs(raw_L[:, 0]), raw_L[:, 1], abs(raw_L[:, 2])
        Lxx, Lxy, Lyy = self.relu(raw_L[:, 0]), raw_L[:, 1], self.relu(raw_L[:, 2])
        Dxx = Lxx ** 2
        Dxy = Lxx * Lxy
        Dyy = Lxy ** 2 + Lyy ** 2
        return torch.stack([Dxx, Dxy, Dyy], dim = 1) # (batch, 3, r, c): (Dxx, Dxy, Dyy)
    def forward(self, x):
        outs = self.unet(x)# (n_batch, 4, r, c) channels order: Phi, Lxx, Lxy, Lyy
        VDs = []
        for out in outs:
            VD = []
            #Vx, Vy = stream_2D(self.actv_Phi(out[:, 0].unsqueeze(1))[:, 0], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            Vx, Vy = stream_2D(out[:, 0], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            VD.append(torch.stack([Vx, Vy], dim = 1))
            #VD.append(self.actv_D(self.actv_L(out[:, 1:]))) # cholesky construction: D = LL^T (L w. non-negative diagonal entries)
            VD.append(self.actv_D(out[:, 1:])) # cholesky construction: D = LL^T (L w. non-negative diagonal entries)
            VDs.append(VD)
        return VDs
    def get_Ls(self, x):
        outs = self.unet(x)
        Ls = []
        for out in outs:
            #raw_L = self.actv_L(out[:, 1:]) #self.unet(x)[:, 1:]
            raw_L = out[:, 1:] #self.unet(x)[:, 1:]
            #Lxx, Lxy, Lyy = abs(raw_L[:, 0]), raw_L[:, 1], abs(raw_L[:, 2])
            Lxx, Lxy, Lyy = self.relu(raw_L[:, 0]), raw_L[:, 1], self.relu(raw_L[:, 2])
            Ls.append(torch.stack([Lxx, Lxy, Lyy], dim = 1)) # (n_batch, 3, r, c): Lxx, Lxy, Lyy
        return Ls
    def get_Phis(self, x):
        outs = self.unet(x)
        Phis = []
        for out in outs:
            #Phis.append(self.actv_Phi(out[:, 0].unsqueeze(1))[:, 0]) # (n_batch, r, c)
            Phis.append(out[:, 0]) # (n_batch, r, c)
        return Phis 

class UNet2D_streamVsymmetricD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamVsymmetricD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 4) # (4 channels: Phi, Lxx, Lxy, Lyy) # NOTE: should output 4 channels (2: unstable)
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 4)
            else:
                self.unet = UNet2D_64(args, in_channels, 4)
        # NOTE: network not efficient for tensor D learning #
        self.actv_Phi = nn.Conv2d(1, 1, kernel_size = 1)
        self.actv_L = nn.Conv2d(3, 3, kernel_size = 1)
    def actv_D(self, L):
        Lxx, Lxy, Lyy = L[:, 0], L[:, 1], L[:, 2]
        Dxx = Lxx ** 2 + Lxy ** 2
        Dxy = (Lxx + Lyy) * Lxy
        Dyy = Lxy ** 2 + Lyy ** 2
        return torch.stack([Dxx, Dxy, Dyy], dim = 1) # (batch, 3, r, c): (Dxx, Dxy, Dyy)
    def forward(self, x):
        outs = self.unet(x) # (n_batch, 4, r, c) channels order: Phi, Lxx, Lxy, Lyy
        VDs = []
        for out in outs: 
            VD = []
            Vx, Vy = stream_2D(self.actv_Phi(out[:, 0].unsqueeze(1))[:, 0], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            VD.append(torch.stack([Vx, Vy], dim = 1))  
            VD.append(self.actv_D(self.actv_L(out[:, 1:]))) # cholesky construction: D = LL^T (L w. non-negative diagonal entries)
            VDs.append(VD)
        return VDs
    def get_Ls(self, x):
        outs = self.unet(x) # (n_batch, 4, r, c) channels order: Phi, Lxx, Lxy, Lyy
        Ls = []
        for out in outs: 
            Ls.append(self.actv_L(out[:, 1:])) # (n_batch, 3, r, c): Lxx, Lxy, Lyy
        return Ls
    def get_Phis(self, x):
        outs = self.unet(x) # (n_batch, 4, r, c) channels order: Phi, Lxx, Lxy, Lyy
        Phis = []
        for out in outs: 
            Phis.append(self.actv_Phi(out[:, 0].unsqueeze(1))[:, 0]) # (n_batch, r, c)
        return Phis

#################################################
##################### For D #####################
#################################################


class UNet2D_scalarD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_scalarD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 1)
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 1)
            else:
                self.unet = UNet2D_64(args, in_channels, 1)
        self.actv = nn.ReLU()
    def forward(self, x):
        outs = self.unet(x)
        Ds = []
        for out in outs:
            Ds.append(self.actv(out[:, 0])) # (n_batch, 1, r, c) -> (n_batch, r, c) (NOTE: BEST) # activate as PSD
        return Ds
        #return D ** 2 # take square as PSD constraint # NOTE: square is not stable -> easy to explode
        #return abs(D) # take absolute value as PSD constraint
        #return D # non-PSD constraint

class UNet2D_diagD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_diagD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 2) # (2 channels: Dxx, Dyy)
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 2)
            else:
                self.unet = UNet2D_64(args, in_channels, 2)
        self.actv = nn.ReLU()
    def forward(self, x):
        outs = self.unet(x)
        Ds = []
        for out in outs:
            Ds.append(self.actv(out)) # (n_batch, 2, r, c) # take positive part as PSD
        return Ds 

class UNet2D_choleskyD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet2D_choleskyD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 3) # (3 channels: Lxx, Lxy, Lyy)
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 3)
            else:
                self.unet = UNet2D_64(args, in_channels, 3)
        self.relu = nn.ReLU()
    def actv(self, raw_L):
        Lxx, Lxy, Lyy = abs(raw_L[:, 0]), raw_L[:, 1], abs(raw_L[:, 2])
        Dxx = Lxx ** 2
        Dxy = Lxx * Lxy
        Dyy = Lxy ** 2 + Lyy ** 2
        return torch.stack([Dxx, Dxy, Dyy], dim = 1) # (batch, 3, r, c): (Dxx, Dxy, Dyy)
    def forward(self, x):
        outs = self.unet(x)
        Ds = []
        for out in outs: # (n_batch, 3, r, c) channels order: Lxx, Lxy, Lyy
            Ds.append(self.actv(out)) # cholesky construction: D = LL^T (L w. non-negative diagonal entries)
        return Ds 
    def get_Ls(self, x):
        outs = self.unet(x)
        Ls = []
        for out in outs:
            Lxx, Lxy, Lyy = abs(out[:, 0]), out[:, 1], abs(out[:, 2])
            Ls.append(torch.stack([Lxx, Lxy, Lyy], dim = 1)) # (n_batch, 3, r, c): Lxx, Lxy, Lyy
        return Ls

class UNet2D_symmetricD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet2D_symmetricD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 3) # (3 channels: Lxx, Lxy, Lyy)
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 3)
            else:
                self.unet = UNet2D_64(args, in_channels, 3)
        self.actv_L = nn.Conv2d(3, 3, kernel_size = 1)
    def actv_D(self, L):
        Lxx, Lxy, Lyy = L[:, 0], L[:, 1], L[:, 2]
        Dxx = Lxx ** 2 + Lxy ** 2
        Dxy = (Lxx + Lyy) * Lxy
        Dyy = Lxy ** 2 + Lyy ** 2
        return torch.stack([Dxx, Dxy, Dyy], dim = 1) # (batch, 3, r, c): (Dxx, Dxy, Dyy)
    def forward(self, x):
        outs = self.unet(x)
        Ds = []
        for out in outs:
            Ds.append(self.actv_D(self.actv_L(out))) # cholesky construction: D = LL^T (L w. non-negative diagonal entries)
        return Ds
    def get_Ls(self, x):
        outs = self.unet(x)
        Ls = []
        for out in outs:
            Ls.append(self.actv_L(out)) # (n_batch, 3, r, c): Lxx, Lxy, Lyy
        return Ls

class UNet2D_spectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(UNet2D_spectralD, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 3) # (4 channels: S, L1, L2) 
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 3)
            else:
                self.unet = UNet2D_64(args, in_channels, 3)
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        #S, L1, L2 = raw_D[:, 0], abs(raw_D[:, 1]), abs(raw_D[:, 2]) # (n_batch, r, c)
        S, L = raw_D[:, 0], self.relu(raw_D[:, 1:]) # (n_batch, r, c)
        L1, L2 = L[:, 0], L[:, 1]
        U = cayley_map(S)
        Dxx = U[..., 0, 0] ** 2 * L1 + U[..., 0, 1] ** 2 * L2
        Dxy = U[..., 0, 0] * U[..., 1, 0] * L1 + U[..., 0, 1] * U[..., 1, 1] * L2
        Dyy = U[..., 1, 0] ** 2 * L1 + U[..., 1, 1] ** 2 * L2
        return torch.stack([Dxx, Dxy, Dyy], dim = 1) # (batch, 3, r, c): (Dxx, Dxy, Dyy)
    def forward(self, x):
        outs = self.unet(x)
        Ds = []
        for out in outs: # (n_batch, 3, r, c) channels order: S, L1, L2
            Ds.append(self.actv_D(out))
        return Ds # list of list: (level, [D])
    def get_Ss(self, x):
        outs = self.unet(x)
        Ss = []
        for out in outs:
            Ss.append(out[:, 0])
        return Ss
    def get_Us(self, x):
        outs = self.unet(x)
        Us = []
        for out in outs:
            U = cayley_map(out[:, 0])
            Us.append(torch.stack([U[..., 0, 0], U[..., 0, 1], U[..., 1, 0], U[..., 1, 1]], dim  = 1)) # (n_batch, 4, r, c): Uxx, Uxy, Uyx, Uyy
        return Us 
    def get_Ls(self, x): # L: eigen-values
        outs = self.unet(x)
        Ls = []
        for out in outs:
            Ls.append(self.relu(out[:, 1:])) # (n_batch, 2, r, c): L1, L2
        return Ls

#################################################
##################### For V #####################
#################################################

class UNet2D_HHDV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_HHDV, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 2) 
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 2)
            else:
                self.unet = UNet2D_64(args, in_channels, 2)
    def forward(self, x):
        outs = self.unet(x)
        Vs = []
        for out in outs:
            Vx, Vy = HHD_2D(out[:, 0], out[:, 1], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            Vs.append(torch.stack([Vx, Vy]).permute(1, 0, 2, 3)) # (2, n_batch, r, c) -> (n_batch, 2, r, c)
        return Vs
    def get_Phis(self, x):
        outs = self.unet(x)
        Phis = []
        for out in outs:
            Phis.append(out[:, 0]) # (n_batch, 1, r, c) -> (n_batch, r, c)
        return Phis
    def get_H(self, x):
        outs = self.unet(x)
        Hs = []
        for out in outs:
            Hs.append(out[:, 0]) # (n_batch, 1, r, c) -> (n_batch, r, c)
        return Hs

class UNet2D_streamV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamV, self).__init__()
        self.args = args
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 1) 
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 1)
            else:
                self.unet = UNet2D_64(args, in_channels, 1)
    def forward(self, x):
        outs = self.unet(x)
        Vs = []
        for out in outs:
            Vx, Vy = stream_2D(out[:, 0], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            Vs.append(torch.stack([Vx, Vy], dim = 1)) # (n_batch, 2, r, c)
        return Vs
    def get_Phis(self, x):
        outs = self.unet(x)
        Phis = []
        for out in outs:
            Phis.append(out[:, 0]) # (n_batch, 1, r, c) -> (n_batch, r, c)
        return  Phis

class UNet2D_vectorV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_vectorV, self).__init__()
        if args.max_down_scales > 0:
            self.unet = UNet2D_2Levels(args, in_channels, 2) 
        else:
            if args.data_dim[0] < 64:
                self.unet = UNet2D_32(args, in_channels, 2)
            else:
                self.unet = UNet2D_64(args, in_channels, 2)
    def forward(self, x):
        outs = self.unet(x)
        Vs = []
        for out in outs:
            Vs.append(out) # (n_batch, 2, r, c)
        return Vs 
