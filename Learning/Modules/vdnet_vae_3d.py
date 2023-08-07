
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from Learning.Modules.utils import *
from Learning.Modules.vae3d import *

'''https://github.com/milesial/Pytorch-UNet'''



def get_net(args, in_channels, out_channels, SepActivate = None):
    if args.max_down_scales > 0:
        return VAE3D_2Levels(in_channels, out_channels, SepActivate = SepActivate)
    elif args.data_dim[-1] < 32: # TODO
        return VAE3D_32_Shallow(in_channels, out_channels, SepActivate = SepActivate)
    else:
        if args.VesselAttention:
            return VAE3D_VesselAttent_32(in_channels, out_channels, SepActivate = SepActivate)
        else:
            return VAE3D_a(in_channels, out_channels, SepActivate = SepActivate) # TODO

#################################################
##################### V + D #####################
#################################################

class VAE3D_streamVscalarD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(VAE3D_streamVscalarD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 4, SepActivate = [3, 1]) # (4 channels: Phi_a, Phi_b, Phi_c, D)
        self.actv_D = nn.ReLU()
    def forward(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        VDs =[]
        for outs_i in outs:
            out, _, _ = outs_i
            VD = []
            Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            VD.append(torch.stack([Vx, Vy, Vz], dim = 1))
            VD.append(self.actv_D(out[:, 3])) # activate as PSD: (n_batch, s, r, c)
            VDs.append(VD)
        return VDs
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_vessel_attention(self, x):
        outs = self.net(x)
        VesAttents = []
        for i in range(len(outs[1])):
            VesAttents.append(outs[1][i][:, 0]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return VesAttents
    def get_Phis(self, x):
        outs = self.net(x)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i
            Phis.append(out[:, :3]) # (n_batch, 3, s, r, c)
        return Phis

class VAE3D_clebschVscalarD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(VAE3D_clebschVscalarD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 3, SepActivate = [2, 1]) # (3 channels: Phi_a, Phi_b, D) 
        self.relu = nn.ReLU()
    def forward(self, x):
        outs = self.net(x)
        VDs = []
        for outs_i in outs: # (n_batch, 3, s, r, c) channels order: Phi_a, Phi_b, L
            VD = []
            out, _, _ = outs_i
            Vx, Vy, Vz = clebsch_3D(out[:, 0], out[:, 1], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
            VD.append(torch.stack([Vx, Vy, Vz], dim = 1))
            VD.append(self.relu(out[:, 2]))
            VDs.append(VD)
        return VDs # list of list: (level, [V, D])
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_vessel_attention(self, x):
        outs = self.net(x)
        VesAttents = []
        for i in range(len(outs[1])):
            VesAttents.append(outs[1][i][:, 0]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return VesAttents
    def get_Phis(self, x):
        outs = self.net(x)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i
            Phis.append(out[:, :2]) # (n_batch, 2, s, r, c)
        return Phis


class VAE3D_streamVcholeskyD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(VAE3D_streamVcholeskyD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 9, SepActivate = [3, 6]) # (9 channels: Phi_a, Phi_b, Phi_c, Lxx, Lxy, Lxz, Lyy, Lyz, Lzz)
        self.relu = nn.ReLU()
    def actv_D(self, raw_L):
        Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = abs(raw_L[:, 0]), raw_L[:, 1], raw_L[:, 2], abs(raw_L[:, 3]), raw_L[:, 4], abs(raw_L[:, 5])
        #Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(raw_L[:, 0].unsqueeze(1)).squeeze(1), raw_L[:, 1], raw_L[:, 2], \
        #    self.relu(raw_L[:, 3].unsqueeze(1)).squeeze(1), raw_L[:, 4], self.relu(raw_L[:, 5].unsqueeze(1)).squeeze(1) # NOTE: to test
        Dxx = Lxx ** 2
        Dxy = Lxx * Lxy
        Dxz = Lxx * Lxz
        Dyy = Lxy ** 2 + Lyy ** 2
        Dyz = Lxy * Lxz + Lyy * Lyz
        Dzz = Lxz ** 2 + Lyz ** 2 + Lzz ** 2
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x) # (n_batch, 9, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
        VDs = []
        for outs_i in outs:
            VD = []
            out, _, _ = outs_i
            Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            VD.append(torch.stack([Vx, Vy, Vz], dim = 1))
            VD.append(self.actv_D(out[:, 3:])) # cholesky construction: D = LL^T (L w. non-negative diagonal entries)
            VDs.append(VD)
        if len(outs) > 1:
            #print('Add vessel attention')
            assert len(outs[1]) == len(VDs)
            for i in range(len(outs[1])):
                #VDs[i][0] = VDs[i][0] * outs[1][i] # V * vessel_attent_map
                VDs[i][1] = VDs[i][1] * (1 - outs[1][i]) # D * (1 - vessel_attent_map)
        return VDs
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_vessel_attention(self, x):
        outs = self.net(x)
        VesAttents = []
        for i in range(len(outs[1])):
            VesAttents.append(outs[1][i][:, 0]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return VesAttents
    def get_Phis(self, x):
        outs = self.net(x)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i
            Phis.append(out[:, :3]) # (n_batch, 3, s, r, c)
        return Phis
    def get_Ls(self, x):
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            out, _, _ = outs_i
            raw_L = out[:, 3:]
            #Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(raw_L[:, 0].unsqueeze(1)).squeeze(1), raw_L[:, 1], raw_L[:, 2], \
            #    self.relu(raw_L[:, 3].unsqueeze(1)).squeeze(1), raw_L[:, 4], self.relu(raw_L[:, 5].unsqueeze(1)).squeeze(1)
            Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = abs(raw_L[:, 0]), raw_L[:, 1], raw_L[:, 2], abs(raw_L[:, 3]), raw_L[:, 4], abs(raw_L[:, 5])
            Ls.append(torch.stack([Lxx, Lxy, Lxz, Lyy, Lyz, Lzz], dim = 1)) # (n_batch, 6, r, c): Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
        return Ls
    

class VAE3D_clebschVspectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(VAE3D_clebschVspectralD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 8, SepActivate = [2, 6]) # (8 channels: Phi_a, Phi_b, S1, S2, S3, L1, L2, L3) 
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        #S, L1, L2, L3 = raw_D[:, :3], abs(raw_D[:, 3]), abs(raw_D[:, 4]), abs(raw_D[:, 5]) # (n_batch, s, r, c)
        U, L = cayley_map(raw_D[:, :3]), self.relu(raw_D[:, 3:]) # S1, S2, S3, L1, L2, L3
        L1 = L[:, 0]
        L2 = L[:, 1]
        L3 = L[:, 2]
        Dxx = U[..., 0, 0] ** 2 * L1 + U[..., 0, 1] ** 2 * L2 + U[..., 0, 2] ** 2 * L3
        Dxy = U[..., 0, 0] * U[..., 1, 0] * L1 + U[..., 0, 1] * U[..., 1, 1] * L2 + U[..., 0, 2] * U[..., 1, 2] * L3
        Dxz = U[..., 0, 0] * U[..., 2, 0] * L1 + U[..., 0, 1] * U[..., 2, 1] * L3 + U[..., 0, 2] * U[..., 2, 2] * L3
        Dyy = U[..., 1, 0] ** 2 * L1 + U[..., 1, 1] ** 2 * L2 + U[..., 1, 2] ** 2 * L3
        Dyz = U[..., 1, 0] * U[..., 2, 0] * L1 + U[..., 1, 1] * U[..., 2, 1] * L2 + U[..., 1, 2] * U[..., 2, 2] * L3
        Dzz = U[..., 2, 0] ** 2 * L1 + U[..., 2, 1] ** 2 * L2 + U[..., 2, 2] ** 2 * L3
        D = torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1)
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, s, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x)
        VDs = []
        for outs_i in outs: # (n_batch, 8, s, r, c) channels order: Phi_a, Phi_b, S1, S2, S3, L1, L2, L3
            VD = []
            out, _, _ = outs_i
            Vx, Vy, Vz = clebsch_3D(out[:, 0], out[:, 1], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
            VD.append(torch.stack([Vx, Vy, Vz], dim = 1)) # (n_batch, 3, s, r, c)
            VD.append(self.actv_D(out[:, 2:])) # (n_batch, 6, s, r, c)
            VDs.append(VD)
        if len(outs) > 1:
            #print('Add vessel attention')
            assert len(outs[1]) == len(VDs)
            for i in range(len(outs[1])):
                #VDs[i][0] = VDs[i][0] * outs[1][i] # V * vessel_attent_map
                VDs[i][1] = VDs[i][1] * (1 - outs[1][i]) # D * (1 - vessel_attent_map)
        return VDs # list of list: (level, [V, D])
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_vessel_attention(self, x):
        outs = self.net(x)
        VesAttents = []
        for i in range(len(outs[1])):
            VesAttents.append(outs[1][i][:, 0]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return VesAttents
    def get_Phis(self, x):
        outs = self.net(x)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i
            Phis.append(out[:, :2]) # (n_batch, 2, s, r, c)
        return Phis
    def get_Us(self, x):
        outs = self.net(x)
        Us = []
        for outs_i in outs:
            out, _, _ = outs_i
            U = cayley_map(out[:, 2:5])
            Us.append(torch.stack([U[..., 0, 0], U[..., 0, 1], U[..., 0, 2], \
                U[..., 1, 0], U[..., 1, 1], U[..., 1, 2], \
                    U[..., 2, 0], U[..., 2, 1], U[..., 2, 2]], dim = 1)) # (n_batch, 9, r, c): Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz
        return Us 
    def get_Ss(self, x): # L: eigen-values
        outs = self.net(x)
        Ss = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ss.append(out[:, 2:5]) # (n_batch, 3, s, r, c): S1, S2, S3
        return Ss
    def get_Ls(self, x): # L: eigen-values
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            out, _, _ = outs_i
            L = self.relu(out[:, 5:])
            L1 = L[:, 0]
            L2 = L[:, 1]
            L3 = L[:, 2]
            Ls.append(torch.stack([L1, L2, L3], dim = 1)) # (n_batch, 3, s, r, c): L1 < L2 < L3
        return Ls

class VAE3D_streamVspectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 (L1 <= L2 <= L3) '''
    def __init__(self, args, in_channels=2):
        super(VAE3D_streamVspectralD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 9, SepActivate = [3, 6]) # (9 channels: Phi_a, Phi_b, Phi_c, S1, S2, S3, L1, L2, L3) 
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        #S, L1, L2, L3 = raw_D[:, :3], abs(raw_D[:, 3]), abs(raw_D[:, 4]), abs(raw_D[:, 5]) # (n_batch, s, r, c)
        U, L = cayley_map(raw_D[:, :3]), self.relu(raw_D[:, 3:]) # S1, S2, S3, L1, L2, L3
        L1 = L[:, 0]
        L2 = L[:, 1]
        L3 = L[:, 2]
        Dxx = U[..., 0, 0] ** 2 * L1 + U[..., 0, 1] ** 2 * L2 + U[..., 0, 2] ** 2 * L3
        Dxy = U[..., 0, 0] * U[..., 1, 0] * L1 + U[..., 0, 1] * U[..., 1, 1] * L2 + U[..., 0, 2] * U[..., 1, 2] * L3
        Dxz = U[..., 0, 0] * U[..., 2, 0] * L1 + U[..., 0, 1] * U[..., 2, 1] * L3 + U[..., 0, 2] * U[..., 2, 2] * L3
        Dyy = U[..., 1, 0] ** 2 * L1 + U[..., 1, 1] ** 2 * L2 + U[..., 1, 2] ** 2 * L3
        Dyz = U[..., 1, 0] * U[..., 2, 0] * L1 + U[..., 1, 1] * U[..., 2, 1] * L2 + U[..., 1, 2] * U[..., 2, 2] * L3
        Dzz = U[..., 2, 0] ** 2 * L1 + U[..., 2, 1] ** 2 * L2 + U[..., 2, 2] ** 2 * L3
        D = torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1)
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, s, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x)
        VDs = []
        for outs_i in outs: # (n_batch, 9, s, r, c) channels order: Phi_a, Phi_b, Phi_c, S1, S2, S3, L1, L2, L3
            VD = []
            out, _, _ = outs_i
            Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
            VD.append(torch.stack([Vx, Vy, Vz], dim = 1))
            VD.append(self.actv_D(out[:, 3:]))
            VDs.append(VD)
        if len(outs) > 1:
            #print('Add vessel attention')
            assert len(outs[1]) == len(VDs)
            for i in range(len(outs[1])):
                #VDs[i][0] = VDs[i][0] * outs[1][i] # V * vessel_attent_map
                VDs[i][1] = VDs[i][1] * (1 - outs[1][i]) # D * (1 - vessel_attent_map)
        return VDs # list of list: (level, [V, D])
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_vessel_attention(self, x):
        outs = self.net(x)
        VesAttents = []
        for i in range(len(outs[1])):
            VesAttents.append(outs[1][i][:, 0]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return VesAttents
    def get_Phis(self, x):
        outs = self.net(x)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i
            Phis.append(out[:, :3]) # (n_batch, 3, s, r, c)
        return Phis
    def get_Us(self, x):
        outs = self.net(x)
        Us = []
        for outs_i in outs:
            out, _, _ = outs_i
            U = cayley_map(out[:, 3:6])
            Us.append(torch.stack([U[..., 0, 0], U[..., 0, 1], U[..., 0, 2], \
                U[..., 1, 0], U[..., 1, 1], U[..., 1, 2], \
                    U[..., 2, 0], U[..., 2, 1], U[..., 2, 2]], dim = 1)) # (n_batch, 9, r, c): Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz
        return Us 
    def get_Ss(self, x): # L: eigen-values
        outs = self.net(x)
        Ss = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ss.append(out[:, 3:6]) # (n_batch, 3, s, r, c): S1, S2, S3
        return Ss
    def get_Ls(self, x): # L: eigen-values
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            out, _, _ = outs_i
            L = self.relu(out[:, 6:])
            L1 = L[:, 0]
            L2 = L[:, 1]
            L3 = L[:, 2]
            Ls.append(torch.stack([L1, L2, L3], dim = 1)) # (n_batch, 3, s, r, c): L1 < L2 < L3
        return Ls

class VAE3D_streamVsemispectralD(nn.Module):
    ''' Construct PSD via Semi-Spectral decomposition: w. eigen-values >= 0 (L1 = L2 <= L3) '''
    def __init__(self, args, in_channels=2):
        super(VAE3D_streamVsemispectralD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 8, SepActivate = [3, 5]) # (8 channels: Phi_a, Phi_b, Phi_c, S1, S2, S3, L1, L2) get_net
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        U, L = cayley_map(raw_D[:, :3]), self.relu(raw_D[:, 3:]) # S1, S2, S3, L1, L2
        L1 = L[:, 0]
        L2 = L1
        L3 = L[:, 1]
        Dxx = U[..., 0, 0] ** 2 * L1 + U[..., 0, 1] ** 2 * L2 + U[..., 0, 2] ** 2 * L3
        Dxy = U[..., 0, 0] * U[..., 1, 0] * L1 + U[..., 0, 1] * U[..., 1, 1] * L2 + U[..., 0, 2] * U[..., 1, 2] * L3
        Dxz = U[..., 0, 0] * U[..., 2, 0] * L1 + U[..., 0, 1] * U[..., 2, 1] * L3 + U[..., 0, 2] * U[..., 2, 2] * L3
        Dyy = U[..., 1, 0] ** 2 * L1 + U[..., 1, 1] ** 2 * L2 + U[..., 1, 2] ** 2 * L3
        Dyz = U[..., 1, 0] * U[..., 2, 0] * L1 + U[..., 1, 1] * U[..., 2, 1] * L2 + U[..., 1, 2] * U[..., 2, 2] * L3
        Dzz = U[..., 2, 0] ** 2 * L1 + U[..., 2, 1] ** 2 * L2 + U[..., 2, 2] ** 2 * L3
        D = torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1)
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, s, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x)
        VDs = []
        for outs_i in outs: # (n_batch, 8, s, r, c) channels order: Phi_a, Phi_b, Phi_c, S1, S2, S3, L1, L2
            VD = []
            out, _, _ = outs_i
            Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
            VD.append(torch.stack([Vx, Vy, Vz], dim = 1))
            VD.append(self.actv_D(out[:, 3:]))
            VDs.append(VD)
        if len(outs) > 1:
            #print('Add vessel attention')
            assert len(outs[1]) == len(VDs)
            for i in range(len(outs[1])):
                #VDs[i][0] = VDs[i][0] * outs[1][i] # V * vessel_attent_map
                VDs[i][1] = VDs[i][1] * (1 - outs[1][i]) # D * (1 - vessel_attent_map)
        return VDs # list of list: (level, [V, D])
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_vessel_attention(self, x):
        outs = self.net(x)
        VesAttents = []
        for i in range(len(outs[1])):
            VesAttents.append(outs[1][i][:, 0]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return VesAttents
    def get_Phis(self, x):
        outs = self.net(x)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i
            Phis.append(out[:, :3]) # (n_batch, 3, s, r, c)
        return Phis
    def get_Us(self, x):
        outs = self.net(x)
        Us = []
        for outs_i in outs:
            out, _, _ = outs_i
            U = cayley_map(out[:, 3:6])
            Us.append(torch.stack([U[..., 0, 0], U[..., 0, 1], U[..., 0, 2], \
                U[..., 1, 0], U[..., 1, 1], U[..., 1, 2], \
                    U[..., 2, 0], U[..., 2, 1], U[..., 2, 2]], dim = 1)) # (n_batch, 9, r, c): Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz
        return Us 
    def get_Ss(self, x): # L: eigen-values
        outs = self.net(x)
        Ss = []
        for out in outs:
            Ss.append(out[:, 3:6]) # (n_batch, 3, s, r, c): S1, S2, S3
        return Ss
    def get_Ls(self, x): # L: eigen-values
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            out, _, _ = outs_i
            L = self.relu(out[:, 6:])
            L1 = L[:, 0]
            L2 = L1
            L3 = L[:, 1]
            Ls.append(torch.stack([L1, L2, L3], dim = 1)) # (n_batch, 3, s, r, c): L1 < L2 < L3
        return Ls

class VAE3D_streamVdualD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(VAE3D_streamVdualD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 9, SepActivate = [3, 6]) # (9 channels: Phi_a, Phi_b, Phi_c, L1, L2, L3, L4, L5, L6)
        self.relu = nn.ReLU()
    def get_vessel_attention(self, x):
        outs = self.net(x)
        VesAttents = []
        for i in range(len(outs[1])):
            VesAttents.append(outs[1][i][:, 0]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return VesAttents
    def actv_D(self, L):
        ''' Symmetric diffusion tensor based on Dual basis (https://www.sciencedirect.com/science/article/pii/S1361841502000531) ''' 
        L1, L2, L3, L4, L5, L6 = L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4], L[:, 5]
        Dxx = L1 - L2 + L3 - L4 + L5 + L6
        Dxy = L1 - L5
        Dxz = L3 - L6
        Dyy = L1 + L2 - L3 + L4 + L5 - L6
        Dyz = L2 - L4
        Dzz = - L1 + L2 + L3 + L4 - L5 + L6
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x) # (n_batch, 9, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, L1, L2, L3, L4, L5, L6
        VDs = []
        for outs_i in outs:
            VD = []
            out, _, _ = outs_i
            Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
            VD.append(torch.stack([Vx, Vy, Vz], dim = 1))
            VD.append(self.actv_D(out[:, 3:])) # cholesky construction: D = LL^T (L w. non-negative diagonal entries)
            VDs.append(VD)
        return VDs
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_Phis(self, x):
        outs = self.net(x)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i 
            Phis.append(out[:, :3]) # (n_batch, 3, s, r, c)
        return Phis
    def get_Ls(self, x):
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ls.append(out[:, 3:]) # (n_batch, 6, r, c): L1, L2, L3, L4, L5, L6
        return Ls

#################################################
##################### For D #####################
################################################# 


class VAE3D_scalarD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(VAE3D_scalarD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 1)
        self.actv = nn.ReLU()
    def forward(self, x):
        outs = self.net(x)
        Ds = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ds.append(self.actv(out[:, 0])) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c) (NOTE: BEST) # activate as PSD
        return Ds 
        #return D ** 2 # take square as PSD constraint # NOTE: square is not stable -> easy to explode
        #return abs(D) # take absolute value as PSD constraint
        #return D # non-PSD constraint
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars

class VAE3D_diagD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(VAE3D_diagD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 3) # (3 channels: Dxx, Dyy, Dzz)
        self.actv = nn.ReLU()
    def forward(self, x):
        outs = self.net(x)
        Ds = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ds.append(self.actv(out)) # (n_batch, 3, s, r, c) # take positive part as PSD
        return Ds
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars

class VAE3D_choleskyD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(VAE3D_choleskyD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 6) # (6 channels: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def actv_D(self, raw_L):
        Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = abs(raw_L[:, 0]), raw_L[:, 1], raw_L[:, 2], abs(raw_L[:, 3]), raw_L[:, 4], abs(raw_L[:, 5])
        Dxx = Lxx ** 2
        Dxy = Lxx * Lxy
        Dxz = Lxx * Lxz
        Dyy = Lxy ** 2 + Lyy ** 2
        Dyz = Lxy * Lxz + Lyy * Lyz
        Dzz = Lxz ** 2 + Lyz ** 2 + Lzz ** 2
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x)
        Ds = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ds.append(self.actv_D(out)) # (n_batch, 6, s, r, c) channels order: Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
        return Ds 
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_Ls(self, x):
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            raw_L, _, _ = outs_i
            Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = abs(raw_L[:, 0]), raw_L[:, 1], raw_L[:, 2], abs(raw_L[:, 3]), raw_L[:, 4], abs(raw_L[:, 5])
            Ls.append(torch.stack([Lxx, Lxy, Lxz, Lyy, Lyz, Lzz], dim = 1))  # (n_batch, 6, s, r, c)
        return Ls
    

class VAE3D_dualD(nn.Module):
    ''' Symmetric diffusion tensor based on Dual basis (https://www.sciencedirect.com/science/article/pii/S1361841502000531) ''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(VAE3D_dualD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 6) # (6 channels: L1, L2, L3, L4, L5, L6)
    def actv_D(self, L):
        L1, L2, L3, L4, L5, L6 = L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4], L[:, 5]
        Dxx = L1 - L2 + L3 - L4 + L5 + L6
        Dxy = L1 - L5
        Dxz = L3 - L6
        Dyy = L1 + L2 - L3 + L4 + L5 - L6
        Dyz = L2 - L4
        Dzz = - L1 + L2 + L3 + L4 - L5 + L6
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x)
        Ds = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ds.append(self.actv_D(out)) # (n_batch, 6, s, r, c) channels order: Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
        return Ds 
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_Ls(self, x):
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            L, _, _ = outs_i # (n_batch, 6, s, r, c)
            Ls.append(L)
        return Ls
    
class VAE3D_spectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(VAE3D_spectralD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 6) # (6 channels: S1, S2, S3, L1, L2, L3) 
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        U, L = cayley_map(raw_D[:, :3]), raw_D[:, 3:]
        L1, L2, L3 = L[:, 0], L[:, 1], L[:, 2] # (n_batch, s, r, c)
        Dxx = U[..., 0, 0] ** 2 * L1 + U[..., 0, 1] ** 2 * L2 + U[..., 0, 2] ** 2 * L3
        Dxy = U[..., 0, 0] * U[..., 1, 0] * L1 + U[..., 0, 1] * U[..., 1, 1] * L2 + U[..., 0, 2] * U[..., 1, 2] * L3
        Dxz = U[..., 0, 0] * U[..., 2, 0] * L1 + U[..., 0, 1] * U[..., 2, 1] * L3 + U[..., 0, 2] * U[..., 2, 2] * L3
        Dyy = U[..., 1, 0] ** 2 * L1 + U[..., 1, 1] ** 2 * L2 + U[..., 1, 2] ** 2 * L3
        Dyz = U[..., 1, 0] * U[..., 2, 0] * L1 + U[..., 1, 1] * U[..., 2, 1] * L2 + U[..., 1, 2] * U[..., 2, 2] * L3
        Dzz = U[..., 2, 0] ** 2 * L1 + U[..., 2, 1] ** 2 * L2 + U[..., 2, 2] ** 2 * L3
        return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = 1) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        outs = self.net(x)
        Ds = []
        for outs_i in outs:
            out, _, _ = outs_i # (n_batch, 6, r, c) channels order: S1, S2, S3, L1, L2, L3
            Ds.append(self.actv_D(out))
        return Ds # list of list: (level, [D])
    def get_vars(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        Vars = []
        for outs_i in outs:
            _, mu, logvar = outs_i
            Vars.append([mu, logvar])
        return Vars
    def get_Us(self, x):
        outs = self.net(x)
        Us = []
        for outs_i in outs:
            out, _, _ = outs_i
            U = cayley_map(out[:, :3])
            Us.append(torch.stack([U[..., 0, 0], U[..., 0, 1], U[..., 0, 2], \
                U[..., 1, 0], U[..., 1, 1], U[..., 1, 2], \
                    U[..., 2, 0], U[..., 2, 1], U[..., 2, 2]], dim = 1)) # (n_batch, 9, r, c): Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz
        return Us 
    def get_Ss(self, x): # L: eigen-values
        outs = self.net(x)
        Ss = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ss.append(out[:, :3]) # (n_batch, 3, s, r, c): S1, S2, S3
        return Ss
    def get_Ls(self, x): # L: eigen-values
        outs = self.net(x)
        Ls = []
        for outs_i in outs:
            out, _, _ = outs_i
            Ls.append(self.relu(out[:, 3:]))  # (n_batch, 3, s, r, c): L1, L2, L3
        return Ls

#################################################
##################### For V #####################
#################################################

class VAE3D_HHDV(nn.Module): # TODO
    def __init__(self, args, in_channels=2):
        super(VAE3D_HHDV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 4)
    def forward(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c)
        Vs = []
        for outs_i in outs:
            out, _, _ = outs_i
            Vx, Vy, Vz = HHD_3D(out[:, 0], out[:, 1], out[:, 2], out[:, 3], batched = True, delta_lst = self.args.data_spacing)
            Vs.append(torch.stack([Vx, Vy, Vz], dim = 1)) # (n_batch, 3, s, r, c)
        return Vs
    def get_Phis(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c)
        Phis = []
        for outs_i in outs:
            out, _, _ = outs_i
            Phis.append(out[:, :3]) # (n_batch, 3, s, r, c)
        return Phis
    def get_Hs(self, x):
        outs = self.net(x) # (n_batch, 4, s, r, c)
        Hs = []
        for out in outs:
            Hs.append(out[:, 3]) # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
        return Hs

class VAE3D_streamV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(VAE3D_streamV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 3)
    def forward(self, x):
        outs = self.net(x) # (n_batch, 3, s, r, c)
        Vs = []
        for outs_i in outs:
            Phi, _, _ = outs_i
            Vx, Vy, Vz = stream_3D(Phi[:, 0], Phi[:, 1], Phi[:, 2], batched = True, delta_lst = self.args.data_spacing)
            Vs.append(torch.stack([Vx, Vy, Vz], dim = 1)) # (n_batch, 3, s, r, c)
        return Vs
    def get_Phis(self, x):
        outs = self.net(x) # (n_batch, 3, s, r, c)
        Phis = []
        for outs_i in outs:
            Phi, _, _ = outs_i
            Phis.append(Phi) # (n_batch, 3, s, r, c)
        return Phis

class VAE3D_clebschV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(VAE3D_clebschV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 2)
    def forward(self, x):
        outs = self.net(x)
        Vs = []
        for outs_i in outs:
            Phi, _, _ = outs_i
            Vx, Vy, Vz = clebsch_3D(Phi[:, 0], Phi[:, 1], batched = True, delta_lst = self.args.data_spacing)
            Vs.append(torch.stack([Vx, Vy, Vz], dim = 1))
        return Vs # (n_batch, 2, s, r, c)
    def get_Phis(self, x):
        outs = self.net(x) # (n_batch, 2, s, r, c)
        Phis = []
        for outs_i in outs:
            Phi, _, _ = outs_i
            Phis.append(Phi) # (n_batch, 2, s, r, c)
        return Phis

