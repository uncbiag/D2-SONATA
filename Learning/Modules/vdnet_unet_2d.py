
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn

from utils import *
from Learning.Modules.unet2d import *

'''https://github.com/milesial/Pytorch-UNet'''


def get_net(args, in_channels, out_channels, trilinear = False, SepActivate = None, stochastic = False):
    if not args.joint_predict and args.perf_pattern == 'adv_diff': 
        return UNet2D_64_SplitDecoder(args, in_channels, out_channels, trilinear, SepActivate, stochastic = stochastic)
    else:
        return UNet2D_64_JointDecoder(args, in_channels, out_channels, trilinear, SepActivate, stochastic = stochastic)

def actv_L(raw_L, actv_func = None):
    if actv_func is not None:
        raw_L = actv_func(raw_L)
    L1 = raw_L[:, 0] + raw_L[:, 1]
    L2 = raw_L[:, 0]
    return torch.stack([L1, L2], dim = 1) # NOTE: L in descending order #

def actv_constant_spectral_param_2D(raw_D, L_actv_func): # [S, L1, L2]: (n_batch, 3, r, c)
    U = cayley_map(raw_D[:, 0]) # (n_batch, r, c, 2, 2)
    L = actv_L(raw_L = raw_D[:, 1:], actv_func = L_actv_func)
    return U, L

def actv_spectral_param_2D(raw_D, L_actv_func): # [S, L1, L2]: (n_batch, 3, r, c)
    U = cayley_map(raw_D[:, 0]) # (n_batch, r, c, 2, 2)
    L = actv_L(raw_L = raw_D[:, 1:], actv_func = L_actv_func) # NOTE: L in descending order #
    return U, L


#################################################
##################### Seg. ######################
#################################################


class UNet2D_Segment(nn.Module): 
    def __init__(self, args, in_channels=2):
        super(UNet2D_Segment, self).__init__()
        self.args = args
        self.net = UNet2D_64(args, in_channels) # out: (n_batch, 1, r, c)
    def forward(self, x, threshold = None):
        prob = self.net(x) # (n_batch, 1, r, c) 
        if threshold is not None:
            return (prob > threshold).float()
        return prob


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
        if args.deviation_separate_net:
            self.net = get_net(args, in_channels, 2, trilinear = False, SepActivate = [1, 1], stochastic = args.stochastic) # (2 channels: Phi, D) 
            self.delta_net = get_net(args, in_channels, 2, trilinear = False, SepActivate = [1, 1], stochastic = False)
        else:
            self.net = get_net(args, in_channels, 2, trilinear = False, SepActivate = [1, 1, 1], stochastic = args.stochastic) # (2 channels: Phi, D) 
        self.actv_D = nn.ReLU()
    def actv_V(self, raw_V):
        Vx, Vy = stream_2D(raw_V, batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c))
        V = torch.stack([Vx, Vy], dim = 1)
        return V
    def forward(self, x):
        if self.args.deviation_separate_net:
            self.out, _, _, _ = self.net(x)
            self.delta_out, _, _, _ = self.delta_net(x) 
        else:
            self.out, self.delta_out, self.value_mask, sigma = self.net(x) # (n_batch, 4, r, c) channels order:  Phi, S, L1, L2
        base_V, base_D = self.actv_V(self.out[:, 0]), self.actv_D(self.out[:, 1]) # (n_batch, 2, r, c)  
        self.sigma, delta_V, delta_D = 0., None, None
        
        if self.args.stochastic:
            self.sigma = sigma # (n_batch, 1/2, r, c) 
        if self.delta_out is not None: # NOTE: default output delta_L > 0, actual delta_L should be the opposite negative one
            delta_V, delta_D = self.actv_V(self.delta_out[:, 0]), - self.actv_D(self.delta_out[:, 1])
        return base_V, base_D, delta_V, delta_D, self.sigma # NOTE: delta_D <= 0 
    def get_Phi(self):
        return self.out[:, 0] # (n_batch, r, c)
    def get_Sigma(self):
        # Hold true if self.stochasitc #
        return self.sigma # (n_batch, 1, r, c)


# TODO: up-to-date #
class UNet2D_streamVspectralD(nn.Module): 
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamVspectralD, self).__init__()
        self.args = args
        if args.deviation_separate_net:
            self.net = get_net(args, in_channels, 4, trilinear = False, SepActivate = [1, 3], stochastic = args.stochastic)
            self.delta_net = get_net(args, in_channels, 3, trilinear = False, SepActivate = [1, 2], stochastic = False)
        else:
            self.net = get_net(args, in_channels, 6, trilinear = False, SepActivate = [1, 3, 2], stochastic = args.stochastic) # (4 channels: Phi, S, L1, L2), delta: (2 channels: delta_L1, delta_L2) # NOTE: S is same as orig
        self.relu = nn.ReLU() 
    def actv_D(self, raw_D):
        U, L = actv_spectral_param_2D(raw_D, L_actv_func = self.relu)
        return construct_spectralD_2D(U, L, batched = True) # (batch, 3, r, c): (Dxx, Dxy, Dyy)
    def actv_V(self, raw_V):
        Vx, Vy = stream_2D(raw_V, batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
        V = torch.stack([Vx, Vy], dim = 1)
        return V
    def forward(self, x):
        if self.args.deviation_separate_net:
            self.out, _, _, _ = self.net(x)
            self.delta_out, _, _, _ = self.delta_net(x) 
        else:
            self.out, self.delta_out, self.value_mask, sigma = self.net(x) # (n_batch, 4, r, c) channels order:  Phi, S, L1, L2
        base_V, base_D = self.actv_V(self.out[:, 0]), self.actv_D(self.out[:, 1:4]) # (n_batch, 2, r, c)  
        self.sigma, delta_V, delta_D = 0., None, None
        if self.args.stochastic:
            self.sigma = sigma # (n_batch, 1/2, r, c)
        if self.delta_out is not None: # NOTE: default output delta_L > 0, actual delta_L should be the opposite negative one
            delta_V, delta_D = self.actv_V(self.delta_out[:, 0]), - self.actv_D(torch.cat([self.out[:, 1][:, None], self.delta_out[:, 1:3]], dim = 1))
        #print('sigma size:', self.sigma.size())
        return base_V, base_D, delta_V, delta_D, self.sigma # NOTE: delta_D <= 0
    def get_Phi(self): 
        return self.out[:, 0]
    def get_U(self): 
        U = cayley_map(self.out[:, 1]) 
        return flatten_U_2D(U) # (n_batch, 4, r, c): Uxx, Uxy, Uyx, Uyy
    def get_S(self): # -> U: eigen-values 
        return self.out[:, 1] # (n_batch, r, c): S 
    def get_L(self): # L: eigen-values  
        base_L = actv_L(raw_L = self.out[:, 2:4], actv_func = self.relu) 
        delta_L = None
        if self.args.predict_deviation: # NOTE: default output delta_L > 0, actual delta_L should be the opposite negtiva one
            base_L = - actv_L(raw_L = self.delta_out[:, 1:3], actv_func = self.relu)  # (n_batch, 2, r, c): L1, L2
        return base_L, delta_L # NOTE: delta_L <= 0
    def get_Sigma(self):  
        # Hold true if self.stochasitc # 
        return self.sigma # (n_batch, 1/2, r, c)



class UNet2D_streamVcholeskyD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamVcholeskyD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 4, trilinear = False, SepActivate = [1, 3]) # (4 channels: Phi, Lxx, Lxy, Lyy) 
        self.relu = nn.ReLU()
    def actv_D(self, raw_L):
        Lxx, Lxy, Lyy = self.relu(raw_L[:, 0]), raw_L[:, 1], self.relu(raw_L[:, 2])
        return construct_choleskyD_2D(Lxx, Lxy, Lyy)
    def actv_V(self, raw_V):
        Vx, Vy = stream_2D(raw_V, batched = True, delta_lst = self.args.data_spacing) # (n_batch, r, c)
        V = torch.stack([Vx, Vy], dim = 1)
    def forward(self, x):
        out, delta_out = self.net(x) # (n_batch, 4, s, r, c) channels order:  Phi, Lxx, Lxy, Lyy
        base_V, base_D = self.actv_V(out[:, 0]), self.actv_D(out[:, 1:4]) # (n_batch, 2, r, c)  
        Sigma, delta_V, delta_D = 0., None, None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.) 
        if delta_out is not None:
            delta_V, delta_D = self.actv_V(delta_out[:, 0]), self.actv_D(delta_out[:, 1:4])
        return base_V, base_D, delta_V, delta_D, Sigma   
    def get_Phi(self, x):
        return self.net(x)[:, 0]  # (n_batch, r, c)
    def get_L(self, x):
        out, delta_out = self.net(x)
        raw_L = out[:, 1:4]
        Lxx, Lxy, Lyy = self.relu(raw_L[:, 0]), raw_L[:, 1], self.relu(raw_L[:, 2])
        return torch.stack([Lxx, Lxy, Lyy], dim = 1) # (n_batch, 3, r, c): Lxx, Lxy, Lyy
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)



#################################################
##################### For D #####################
#################################################



class UNet2D_scalarD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_scalarD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 1)
        self.actv_D = nn.ReLU() 
    def forward(self, x):
        out, delta_out = self.net(x)  
        D = self.actv_D(out[:, 0])
        Sigma, delta_D = 0., None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
        if delta_out is not None:
            delta_D = self.actv_D(delta_out[:, 0])
        return D, delta_D, Sigma    
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet2D_diagD(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_diagD, self).__init__()
        self.args = args
        get_net(args, in_channels, 2) # (2 channels: Dxx, Dyy)
        self.actv_D = nn.ReLU()
    def forward(self, x):
        out, delta_out = self.net(x) 
        D = self.actv_D(out[:, :2])
        Sigma, delta_D = 0., None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
        if delta_out is not None:
            delta_D = self.actv_D(delta_out[:, :2])
        return D, delta_D, Sigma     
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet2D_choleskyD(nn.Module):
    ''' Non-negative diagonal entries L version: abs''' 
    # sqr not work: underflow in nan # 
    # ReLU not work well: cannot back-propagate well: Lxx, Lyy -> 0 #
    def __init__(self, args, in_channels=2):
        super(UNet2D_choleskyD, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 3) # (3 channels: Dxx, Dxy, Dyy)
        self.relu = nn.ReLU()
    def actv_D(self, raw_L):
        Lxx, Lxy, Lyy = self.relu(raw_L[:, 0]), raw_L[:, 1], self.relu(raw_L[:, 2])
        return construct_choleskyD_2D(Lxx, Lxy, Lyy)  
    def forward(self, x):
        out, delta_out = self.net(x) 
        D = self.actv_D(out[:, :3])
        Sigma, delta_D = 0., None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
        if delta_out is not None:
            delta_D = self.actv_D(delta_out[:, :3])
        return D, delta_D, Sigma     
    def get_L(self, x):
        raw_L = self.net(x)[:3]
        Lxx, Lxy, Lyy = self.relu(raw_L[:, 0]), raw_L[:, 1], self.relu(raw_L[:, 2])
        return torch.stack([Lxx, Lxy, Lyy], dim = 1) # (n_batch, 3, r, c)
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet2D_spectralD(nn.Module):
    ''' Construct PSD via Spectral decomposition: w. eigen-values >= 0 '''
    def __init__(self, args, in_channels=2):
        super(UNet2D_spectralD, self).__init__()
        print('______________________UNet3D_spectralD______________________')
        self.args = args
        self.net = get_net(args, in_channels, 3) # (6 channels: S, L1, L2) 
        self.relu = nn.ReLU()
        self.L_base = None # NOTE: just initialization, assign value during implementing if needed
    def actv_D(self, raw_D):
        U, L = actv_spectral_param_2D(raw_D, L_actv_func = self.relu) 
        return construct_spectralD_3D(U, L) 
    def forward(self, x):
        out, delta_out = self.net(x) 
        D = self.actv_D(out[:, :3])
        Sigma, delta_D = 0., None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
        if delta_out is not None:
            delta_D = self.actv_D(delta_out[:, :3])
        return D, delta_D, Sigma    
    def get_U(self, x):
        U = cayley_map(self.net(x)[:, 0]) 
        return flatten_U_3D(U) # (n_batch, 4, r, c): Uxx, Uxy, Uyx, Uyy
    def get_S(self, x): 
        return self.net(x)[:, 0]
    def get_L(self, x): # L: eigen-values
        out, delta_out = self.net(x)[:3] 
        raw_L = self.relu(out[:, 1:])
        return torch.stack([raw_L[:, 0] + raw_L[:, 1], raw_L[:, 0]], dim = 1)  # (n_batch, 2, r, c): L1, L2
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


#################################################
##################### For V #####################
#################################################


class UNet2D_HHDV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_HHDV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 2)
    def actv_V(self, raw_Vx, raw_Vy):
        Vx, Vy = HHD_2D(raw_Vx, raw_Vy, batched = True, delta_lst = self.args.data_spacing)
        V = torch.stack([Vx, Vy], dim = 1)
        return V
    def forward(self, x):
        out, delta_out = self.net(x) 
        V = self.actv_V(out[:, 0], out[:, 1])
        Sigma, delta_V = 0., None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
        if delta_out is not None:
            delta_V = self.actv_V(delta_out[:, 0], delta_out[:, 1])
        return V, delta_V, Sigma   
    def get_Phi(self, x):  
        return self.net(x)[:, 0] 
    def get_Hs(self, x): 
        return self.net(x)[:, 1]  
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet2D_streamV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 1)
    def actv_V(self, raw_V):
        Vx, Vy = stream_2D(raw_V, batched = True, delta_lst = self.args.data_spacing)
        V = torch.stack([Vx, Vy], dim = 1)
        return V
    def forward(self, x):
        out, delta_out = self.net(x) 
        V = self.actv_V(out[:, 0])
        Sigma, delta_V = 0., None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
        if delta_out is not None:
            delta_V = self.actv_V(delta_out[:, 0])
        return V, delta_V, Sigma   
    def get_Phi(self, x): # (n_batch, r, c)
        return self.net(x)[:, 0]
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)


class UNet2D_vectorV(nn.Module):
    def __init__(self, args, in_channels=2):
        super(UNet2D_streamV, self).__init__()
        self.args = args
        self.net = get_net(args, in_channels, 2)
    def forward(self, x):
        out, delta_out = self.net(x) 
        V = out[:, :2]
        Sigma, delta_V = 0., None
        if self.args.stochastic:
            Sigma = torch.clamp(out[:, -1], min = 0.)
        if delta_out is not None:
            delta_V = delta_out[:, :2]
        return V, delta_V, Sigma
    def get_Sigma(self, x):
        # Hold true if self.stochasitc #
        out, delta_out = self.net(x)
        return torch.clamp(out[:, -1], min = 0.)
