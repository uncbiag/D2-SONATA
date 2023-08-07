
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from Learning.Modules.utils import *
from Learning.Modules.unet3d import *

'''https://github.com/milesial/Pytorch-UNet'''


def get_net(args, in_channels, out_channels, trilinear = False, SepActivate = None, stochastic = False):
    if not args.joint_predict and args.perf_pattern == 'adv_diff': 
        return UNet3D_32_SplitDecoder(args, in_channels, out_channels, trilinear, SepActivate, stochastic)
    else:
        return UNet3D_32_JointDecoder(args, in_channels, out_channels, trilinear, SepActivate, stochastic)


def actv_L(raw_L, actv_func = None):
    if actv_func is not None:
        raw_L = actv_func(raw_L)
    L1 = raw_L[:, 0] + raw_L[:, 1] + raw_L[:, 2] 
    L2 = raw_L[:, 0] + raw_L[:, 1] 
    L3 = raw_L[:, 0]
    return torch.stack([L1, L2, L3], dim = 1) # NOTE: L in descending order #
     

def actv_spectral_param_3D(raw_D, L_actv_func): # S: S1, S2, S3 - (3, s, r, c); L: L1, L2, L3 - (3, s, r, c)
    U = cayley_map(raw_D[:, :3]) # (3, s, r, c)
    L = actv_L(raw_L = raw_D[:, 3:6], actv_func = L_actv_func) # NOTE: L in descending order: (n_batch, 3, s, r, c)
    return U, L



#################################################
##################### Digit #####################
#################################################
 

class UNet3D_Segment(nn.Module): 
    def __init__(self, args, in_channels=2):
        super(UNet3D_Segment, self).__init__()
        self.args = args
        self.net = UNet3D_32(args, in_channels) # out: (n_batch, 1, s, r, c)
    def forward(self, x, threshold = None):
        prob = self.net(x) # (n_batch, 1, r, c) 
        if threshold is not None:
            return (prob > threshold).float()
        return prob


#################################################
##################### V + D #####################
#################################################


class UNet3D_streamVscalarD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet3D_streamVscalarD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 4, trilinear = False, SepActivate = [3, 1]) # (4 channels: Phi_a, Phi_b, Phi_c, D)
        self.actv_D = nn.ReLU()
    def forward(self, x):
        out = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, D
        
        Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
        V = torch.stack([Vx, Vy, Vz], dim = 1)
        D = self.actv_D(out[:, 3]) # activate as PSD: (n_batch, s, r, c)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.) 
            return V, D, Sigma
        return V, D, 0
    def get_Phi(self, x):
        out = self.net(x)
        return out[:, :2] # (n_batch, 2, s, r, c)
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet3D_clebschVscalarD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(UNet3D_clebschVscalarD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 3, trilinear = False, SepActivate = [2, 1]) # (3 channels: Phi_a, Phi_b, D) 
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x) # (n_batch, 3, s, r, c) channels order: Phi_a, Phi_b, L
        Vx, Vy, Vz = clebsch_3D(out[:, 0], out[:, 1], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
        V = torch.stack([Vx, Vy, Vz], dim = 1)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return V, self.relu(out[:, 2]), Sigma
        return V, self.relu(out[:, 2]), 0 
    def get_Phi(self, x):
        out = self.net(x)
        return out[:, :2] # (n_batch, 2, s, r, c)
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet3D_streamVcholeskyD(nn.Module): 
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet3D_streamVcholeskyD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 9, trilinear = False, SepActivate = [3, 6]) # (9 channels: Phi_a, Phi_b, Phi_c, Lxx, Lxy, Lxz, Lyy, Lyz, Lzz)
        self.relu = nn.ReLU()
    def actv_D(self, raw_L):
        Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(raw_L[:, 0]), raw_L[:, 1], raw_L[:, 2], self.relu(raw_L[:, 3]), raw_L[:, 4], self.relu(raw_L[:, 5])
        #Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(raw_L[:, 0].unsqueeze(1)).squeeze(1), raw_L[:, 1], raw_L[:, 2], \
        #    self.relu(raw_L[:, 3].unsqueeze(1)).squeeze(1), raw_L[:, 4], self.relu(raw_L[:, 5].unsqueeze(1)).squeeze(1) # NOTE: to test
        return construct_choleskyD_3D(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        out = self.net(x) # (n_batch, 9, s, r, c) channels order:  Phi_a, Phi_b, Phi_c, Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
        Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
        V = torch.stack([Vx, Vy, Vz], dim = 1)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return V, self.actv_D(out[:, 3:9]), Sigma
        return V, self.actv_D(out[:, 3:9]), 0
    def get_Phi(self, x):
        return self.net(x)[:, :3]  # (n_batch, 3, s, r, c)
    def get_L(self, x):
        out = self.net(x)
        raw_L = out[:, 3:9]
        #Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(raw_L[:, 0].unsqueeze(1)).squeeze(1), raw_L[:, 1], raw_L[:, 2], \
        #    self.relu(raw_L[:, 3].unsqueeze(1)).squeeze(1), raw_L[:, 4], self.relu(raw_L[:, 5].unsqueeze(1)).squeeze(1)
        Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(raw_L[:, 0]), raw_L[:, 1], raw_L[:, 2], self.relu(raw_L[:, 3]), raw_L[:, 4], self.relu(raw_L[:, 5])
        return torch.stack([Lxx, Lxy, Lxz, Lyy, Lyz, Lzz], dim = 1) # (n_batch, 6, r, c): Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)
    

class UNet3D_clebschVspectralD(nn.Module): 
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(UNet3D_clebschVspectralD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 8, trilinear = False, SepActivate = [2, 6]) # (8 channels: Phi_a, Phi_b, S1, S2, S3, L1, L2, L3) 
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        U, L = actv_spectral_param_3D(raw_D, L_actv_func = self.relu) 
        return construct_spectralD_3D(U, L) # (batch, 6, s, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        out = self.net(x)
        # (n_batch, 8, s, r, c) channels order: Phi_a, Phi_b, S1, S2, S3, L1, L2, L3
        Vx, Vy, Vz = clebsch_3D(out[:, 0], out[:, 1], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
        V = torch.stack([Vx, Vy, Vz], dim = 1) # (n_batch, 3, s, r, c)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return V, self.actv_D(out[:, 2:8]), Sigma
        return V, self.actv_D(out[:, 2:8]), 0
    def get_Phi(self, x):
        return self.net(x)[:, :2] # (n_batch, 2, s, r, c)
    def get_U(self, x):
        U = cayley_map(self.net(x)[:, 2:5])
        return flatten_U_3D(U) # (n_batch, 9, r, c): Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz
    def get_S(self, x): # L: eigen-values
        return self.net(x)[:, 2:5] # (n_batch, 3, s, r, c): S1, S2, S3
    def get_L(self, x): # L: eigen-values
        out = self.net(x)
        L = self.relu(out[:, 5:8])
        L1 = L[:, 0] + L[:, 1] + L[:, 2]
        L2 = L[:, 0] + L[:, 1]
        L3 = L[:, 0]
        return torch.stack([L1, L2, L3], dim = 1) # (n_batch, 3, s, r, c) 
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)

####################
# TODO: up-to-date #
class UNet3D_streamVspectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 (L1 <= L2 <= L3) '''
    def __init__(self, args, in_channels=2):
        super(UNet3D_streamVspectralD, self).__init__()
        print('______________________UNet3D_streamVspectralD______________________')
        self.args = args
        # NOTE: (3 + 6 channels: Phi_a, Phi_b, Phi_c, S1, S2, S3, L1, L2, L3); delta: (3 channels: delta_L1, delta_L2, delta_L3) # NOTE: U is same as orig
        if args.predict_deviation and args.deviation_separate_net:
            self.net = get_net(args, in_channels, 9, trilinear = False, SepActivate = [3, 6], stochastic = args.stochastic)
            self.delta_net = get_net(args, in_channels, 6, trilinear = False, SepActivate = [3, 3], stochastic = False)
        else:
            self.net = get_net(args, in_channels, 12, trilinear = False, SepActivate = [3, 6, 3], stochastic = args.stochastic) 
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        U, L = actv_spectral_param_3D(raw_D, L_actv_func = self.relu) 
        return construct_spectralD_3D(U, L) # (batch, 6, s, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def actv_V(self, raw_V):
        Vx, Vy, Vz = stream_3D(raw_V[:, 0], raw_V[:, 1], raw_V[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c) 
        V = torch.stack([Vx, Vy, Vz], dim = 1)
        return V
    def forward(self, x): 
        if self.args.deviation_separate_net:
            self.out, _, _ = self.net(x)
            self.delta_out, _, _ = self.delta_net(x)
        else:
            self.out, self.delta_out, self.value_mask, sigma = self.net(x) # (n_batch, 9, s, r, c) channels order: Phi_a, Phi_b, Phi_c, S1, S2, S3, L1, L2, L3
        base_V, base_D = self.actv_V(self.out[:, :3]), self.actv_D(self.out[:, 3:9]) # (n_batch, 2, r, c)  
        self.sigma, delta_V, delta_D = 0., None, None
        if self.args.stochastic:
            self.sigma = sigma
        if self.delta_out is not None: # NOTE: default output delta_L > 0, actual delta_L should be the opposite negtiva one 
            delta_V, delta_D = self.actv_V(self.delta_out[:, :3]), self.actv_D(torch.cat([self.out[:, 3:6], - self.delta_out[:, 3:6]], dim = 1)) # delta_D <-- orig_U & delta_L
        #print('sigma size:', self.sigma.size())
        return base_V, base_D, delta_V, delta_D, self.sigma # NOTE: delta_D <= 0
    def get_Phi(self):
        return self.out[:, :3]
    def get_U(self):
        U = cayley_map(self.out[:, 3:6]) 
        return flatten_U_3D(U) # (n_batch, 9, r, c): Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz
    def get_S(self): # L: eigen-values
        return self.out[:, 3:6] # (n_batch, 3, s, r, c): S1, S2, S3
    def get_L(self): # L: eigen-values  
        base_L = actv_L(raw_L = self.out[:, 6:9], actv_func = self.relu) 
        delta_L = None
        if self.args.predict_deviation: # NOTE: default output delta_L > 0, actual delta_L should be the opposite negtiva one
            delta_L = - actv_L(raw_L = self.delta_out[:, 3:6], actv_func = self.relu)  # (n_batch, 3, s, r, c): L1, L2
        return base_L, delta_L # delta_L <= 0
    def get_Sigma(self):
        # Hold true if self.stochasitc #  
        return self.sigma # (n_batch, 1/2, r, c)

####################
####################

class UNet3D_streamVdualD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet3D_streamVdualD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 9, trilinear = False, SepActivate = [3, 6])
        self.relu = nn.ReLU()
    def actv_D(self, L):
        ''' Symmetric diffusion tensor based on Dual basis (https://www.sciencedirect.com/science/article/pii/S1361841502000531) ''' 
        L1, L2, L3, L4, L5, L6 = L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4], L[:, 5]
        return construct_dualD_3D(L1, L2, L3, L4, L5, L6) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        out = self.net(x) # (n_batch, 9, s, r, c) channels order: Phi_a, Phi_b, Phi_c, L1, L2, L3, L4, L5, L6
        Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing) # (n_batch, s, r, c)
        V = torch.stack([Vx, Vy, Vz], dim = 1)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return V, self.actv_D(out[:, 3:9]), Sigma
        return V, self.actv_D(out[:, 3:9]), 0
    def get_Phi(self, x):
        return self.net(x)[:, :3]  # (n_batch, 3, s, r, c)
    def get_L(self, x):
        return self.net(x)[:, 3:9] # (n_batch, 6, r, c): L1, L2, L3, L4, L5, L6
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


#################################################
##################### For D #####################
#################################################


class UNet3D_scalarD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet3D_scalarD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 1)
        self.actv = nn.ReLU()
    def forward(self, x): # (n_batch, 1, s, r, c) -> (n_batch, s, r, c) (NOTE: BEST) # activate as PSD
        out = self.net(x)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return self.actv(out[:, 0]), Sigma
        return self.actv(out[:, 0]), 0
        #return D ** 2 # take square as PSD constraint # NOTE: square is not stable -> easy to explode
        #return abs(D) # take absolute value as PSD constraint
        #return D # non-PSD constraint
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet3D_diagD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet3D_diagD, self).__init__()
        self.args = args
        get_net(args, in_channels, 3) # (3 channels: Dxx, Dyy, Dzz)
        self.actv = nn.ReLU()
    def forward(self, x): # (n_batch, 3, s, r, c) # take positive part as PSD
        out = self.net(x)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return self.actv(out[:, 0]), Sigma
        return self.actv(out[:, :3]), 0
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet3D_choleskyD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet3D_choleskyD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 6) # (6 channels: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
        self.relu = nn.ReLU()
    def actv_D(self, raw_L):
        Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(raw_L[:, 0]), raw_L[:, 1], raw_L[:, 2], self.relu(raw_L[:, 3]), raw_L[:, 4], self.relu(raw_L[:, 5])
        return construct_choleskyD_3D(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):  # (n_batch, 6, s, r, c) channels order: Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
        out = self.net(x)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return self.actv_D(out[:, :6]), Sigma
        return self.actv_D(out[:, :6]), 0
    def get_L(self, x):
        out = self.net(x)
        Lxx, Lxy, Lxz, Lyy, Lyz, Lzz = self.relu(out[:, 0]), out[:, 1], out[:, 2], self.relu(out[:, 3]), out[:, 4], self.relu(out[:, 5])
        return torch.stack([Lxx, Lxy, Lxz, Lyy, Lyz, Lzz], dim = 1) # (n_batch, 6, s, r, c)
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)
    

class UNet3D_dualD(nn.Module):
    ''' Symmetric diffusion tensor based on Dual basis (https://www.sciencedirect.com/science/article/pii/S1361841502000531) ''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet3D_dualD, self).__init__()
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
    def forward(self, x): # (n_batch, 6, s, r, c) channels order: Lxx, Lxy, Lxz, Lyy, Lyz, Lzz
        out = self.net(x)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return self.actv_D(out[:, :6]), Sigma
        return self.actv_D(out[:, :6]), 0
    def get_L(self, x):
        return self.net(x)# (n_batch, 6, s, r, c)
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet3D_spectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(UNet3D_spectralD, self).__init__()
        print('______________________UNet3D_spectralD______________________')
        self.args = args
        self.net = get_net(args, in_channels, 6) # (6 channels: S1, S2, S3, L1, L2, L3) 
        self.relu = nn.ReLU()
    def actv_D(self, raw_D):
        U, L = actv_spectral_param_3D(raw_D, L_actv_func = self.relu) 
        return construct_spectralD_3D(U, L) # (batch, 6, s, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
    def forward(self, x):
        out = self.net(x)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return self.actv_D(out[:, :6]), Sigma
        return self.actv_D(out[:, :6]), 0 # list of list: (level, [D])
    def get_U(self, x):
        U = cayley_map(self.net(x)[:, :3]) 
        return flatten_U_3D(U) # (n_batch, 9, r, c): Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz
    def get_S(self, x): # L: eigen-values # (n_batch, 3, s, r, c): S1, S2, S3
        return self.net(x)[:, :3]
    def get_L(self, x): # L: eigen-values
        out = self.net(x) 
        return self.relu(out[:, 3:6])  # (n_batch, 3, s, r, c): L1, L2, L3
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


#################################################
##################### For V #####################
#################################################


class UNet3D_HHDV(nn.Module): # TODO
    def __init__(self, args, in_channels=2):
        super(UNet3D_HHDV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 4)
    def forward(self, x):
        out = self.net(x) # (n_batch, 4, s, r, c)
        Vx, Vy, Vz = HHD_3D(out[:, 0], out[:, 1], out[:, 2], out[:, 3], batched = True, delta_lst = self.args.data_spacing)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return torch.stack([Vx, Vy, Vz], dim = 1), Sigma
        return torch.stack([Vx, Vy, Vz], dim = 1), 0 # (n_batch, 3, s, r, c) 
    def get_Phi(self, x): # (n_batch, 4, s, r, c)
        return self.net(x)[:, :3]# (n_batch, 3, s, r, c)
    def get_Hs(self, x):  # (n_batch, 4, s, r, c)
        return self.net(x)[:, 3] # (n_batch, 1, s, r, c) -> (n_batch, s, r, c)
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)

class UNet3D_streamV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet3D_streamV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 3) 
    def forward(self, x):
        out = self.net(x) # (n_batch, 3, s, r, c)
        Vx, Vy, Vz = stream_3D(out[:, 0], out[:, 1], out[:, 2], batched = True, delta_lst = self.args.data_spacing)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return torch.stack([Vx, Vy, Vz], dim = 1), Sigma
        return torch.stack([Vx, Vy, Vz], dim = 1), 0 # (n_batch, 3, s, r, c)
    def get_Phi(self, x): # (n_batch, 3, s, r, c)
        return self.net(x)[:, :3]
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)

class UNet3D_clebschV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet3D_clebschV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 2)
    def forward(self, x):
        out = self.net(x)
        Vx, Vy, Vz = clebsch_3D(out[:, 0], out[:, 1], batched = True, delta_lst = self.args.data_spacing)
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
            return torch.stack([Vx, Vy, Vz], dim = 1), Sigma
        return torch.stack([Vx, Vy, Vz], dim = 1), 0 # (n_batch, 2, s, r, c)
    def get_Phi(self, x): # (n_batch, 2, s, r, c)
        return self.net(x)[:, :2]
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)
