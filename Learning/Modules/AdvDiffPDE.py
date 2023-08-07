from ast import Del, Return
from functools import partial
import os, sys
from re import S
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import abc
import math
import torch
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter

from utils import *
from ODE.FD import FD_torch
from ODE.adjoint import odeint_adjoint as odeint
from Learning.Modules.vdnet_vae_2d import * 
from Learning.Modules.vdnet_vae_3d import * 
from Learning.Modules.vdnet_unet_2d import *
from Learning.Modules.vdnet_unet_3d import *


'''
Advectin-Diffusion PDE Base Module: 
:input: batched C
:return: dC/dt := \\div(D * \\nabla C) - V * \\nabla C
''' 

##############################################################################

class PIANOOneTime(nn.Module):
    '''
    Advection-Diffusion PDE Conservation Loss: 
    ConsLoss = dC/dt - (\\div(D * \\nabla C) - V * \\nabla C)
            \\approx (C ^ {k+1} - C ^ k) - 0.5 * {\
            (\\div(D * \\nabla C ^ {k}) - V * \\nabla C ^ {k}) + \
            (\\div(D * \\nabla C ^ {k + 1}) - V * \\nabla C ^ {k + 1})}
    '''          
    def __init__(self, args, GT_du, perf_pattern, data_dim, data_spacing, device, in_channels=2, out_channels=1, contour = None):
        super(PIANOOneTime, self).__init__()
        self.PDEFunc = PIANO(args, GT_du, perf_pattern, data_dim, data_spacing, device, in_channels, out_channels, contour = None)
        self.args = args
    def conserv_loss(self, t, batch_Ct):
        batch_C0, batch_C1 = batch_Ct[:, 0], batch_Ct[:, 1]
        dC = self.PDEFunc(t, batch_C0)
        return (abs((batch_C1 - batch_C0) / self.args.dt - dC)).mean()
    def predict(self, t, batch_Ct):
        batch_C0, batch_C1 = batch_Ct[:, 0], batch_Ct[:, 1]
        dC = self.PDEFunc(t, batch_C0) # (n_batch - 1, r, c)
        return batch_Ct[:, 0] + dC * self.args.dt

##############################################################################
 
class VD_DeepBase(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self, args, data_dim, perf_pattern, in_channels, device):
        super(VD_DeepBase, self).__init__()
        self.args = args
        self.device = device
        self.data_dim = data_dim
        self.dimension = len(data_dim)
        self.in_channels = in_channels
        self.perf_pattern = perf_pattern
        self.stochastic = self.args.stochastic
        self.predict_segment = self.args.predict_segment
        self.predict_deviation = self.args.predict_deviation
        self.predict_value_mask = self.args.predict_value_mask
        assert not (self.predict_segment and (self.predict_value_mask or self.predict_deviation)) 
        self.D_type, self.V_type = self.args.PD_D_type, self.args.PD_V_type
        self.vessel_mask, self.vessel_mirror_mask = 1., 1. # TODO (n_batch, s, r, c) NOTE: Just initialization, assign value during training if needed
        self.anomaly_mask, self.anomaly_D_mask, self.anomaly_V_mask = 1., 1., 1. # TODO (n_batch, s, r, c) NOTE: Just initialization, assign value during training if needed
        self.sigma_mask, self.sigma_D_mask, self.sigma_V_mask = 1., 1., 1. # TODO (n_batch, s, r, c) NOTE: Just initialization, assign value during training if needed
        self.diffusion_mask = 1. # TODO (n_batch, s, r, c) NOTE: Just initialization, assign value during training if needed
        self.D_magnitude, self.V_magnitude = args.D_magnitude, args.V_magnitude #  NOTE: Just initialization, assign value during training if needed
        self.input_features = None # NOTE: Just initialization, assign value during training

        if 'adv_only' in self.perf_pattern:
            self.VNet = self.register_VNet()
        elif 'diff_only' in self.perf_pattern:
            self.DNet = self.register_DNet()  
        else:
            self.VDNet = self.register_VDNet() 

        if self.predict_segment: 
            if self.args.segment_net_type == 'conc':
                if self.dimension == 2:
                    self.SegNet = UNet2D_Segment(args, in_channels = self.in_channels).to(self.device) 
                elif self.dimension == 3:
                    self.SegNet = UNet3D_Segment(args, in_channels = self.in_channels).to(self.device) 
            elif self.args.segment_net_type == 'dev': # TODO
                assert self.perf_pattern == 'adv_diff' and self.D_type == 'full_spectral' and self.V_type == 'vector_div_free_stream' 
                if self.dimension == 2:
                    self.SegNet = UNet2D_Segment(args, in_channels = 4).to(self.device) # V1, V2, L1, L2
                elif self.dimension == 3:
                    self.SegNet = UNet3D_Segment(args, in_channels = 6).to(self.device) # V1, V2, V3, L1, L2, L3 
            else:
                raise NotImplementedError('Unsupported segment_net_type: %s!' % self.args.segment_net_type)

        if self.dimension == 1:
            self.neumann_BC = torch.nn.ReplicationPad1d(1)
        elif self.dimension == 2:
            self.neumann_BC = torch.nn.ReplicationPad2d(1)
        elif self.dimension == 3:
            self.neumann_BC = torch.nn.ReplicationPad3d(1)
        else:
            raise NotImplementedError('Unsupported dimension: %s!' % self.dimension)

    @property
    # NOTE For bondary condition of divergence-free V #
    def div_free_v_cond(self):
        if self.dimension == 1:
            return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(1))[1] # V: (n_batch, c) #
        elif self.dimension == 2:
            return lambda X: self.neumann_BC(X[:, :, 1:-1, 1:-1]) # V: (n_batch, 2, r, c) #
        elif self.dimension == 3:
            return lambda X: self.neumann_BC(X[:, :, 1:-1, 1:-1, 1:-1]) # V: (n_batch, 3, s, r, c) #
        else:
            raise NotImplementedError('Unsupported dimension!')
                 
    @property
    def set_BC(self):
    # NOTE For bondary condition of mass concentration #
        '''X: (n_batch, spatial_shape)''' 
        if self.BC == 'neumann' or self.BC == 'cauchy':
            if self.dimension == 1:
                return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3: 
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        elif self.BC == 'dirichlet_neumann' or self.BC == 'source_neumann':
            ctrl_wdth = self.args.dirichlet_width if self.BC == 'dirichlet_neumann' else self.args.source_width
            if self.dimension == 1:
                self.dirichlet_BC = torch.nn.ReplicationPad1d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                self.dirichlet_BC = torch.nn.ReplicationPad2d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                self.dirichlet_BC = torch.nn.ReplicationPad3d(ctrl_wdth)
                return lambda X: self.neumann_dirichlet_BCBC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        else:
            return lambda X: X    

    def register_VDNet(self):
        if self.args.joint_predict:
            print('-------------- Joint Decoder: Adv_Diff --------------')
        else:
            print('-------------- Split Decoders: Adv_Diff --------------')
        return self.get_VDNet[self.args.model_type][self.V_type][self.D_type](self.args, self.in_channels).to(self.device) 
    def register_VNet(self):
        return self.get_VNet[self.args.model_type][self.V_type](self.args, self.in_channels).to(self.device) 
    def register_DNet(self):
        return self.get_DNet[self.args.model_type][self.D_type](self.args, self.in_channels).to(self.device) 

    @property
    def get_VNet(self):
        print('-------------- Get V Net --------------')
        if self.dimension == 2:
            return {
                'unet': {
                    'vector': UNet2D_vectorV,
                    'vector_div_free_clebsch': UNet2D_streamV,
                    'vector_div_free_stream': UNet2D_streamV,
                },
                'vae': {
                    'vector': VAE2D_vectorV,
                    'vector_div_free_stream': VAE2D_streamV,
                }
            }
        elif self.dimension == 3:
            return {
                'unet': {
                    'vector_div_free_clebsch': UNet3D_clebschV,
                    'vector_div_free_stream': UNet3D_streamV,
                    'vector_div_free_stream_gauge': UNet3D_streamV,  
                    'vector_HHD': UNet3D_HHDV, 
                },
                'vae': {
                    'vector_div_free_clebsch': VAE3D_clebschV,
                    'vector_div_free_stream': VAE3D_streamV,
                    'vector_div_free_stream_gauge': VAE3D_streamV,  
                    'vector_HHD': VAE3D_HHDV,  
                }
            }
        else:
            raise NotImplementedError('Not supported dimension !')
    @property
    def get_DNet(self):
        print('-------------- Get D Net --------------')
        if self.dimension == 2:
            return {
                'unet': {
                    'scalar': UNet2D_scalarD,
                    'diag': UNet2D_diagD,
                    'full_cholesky': UNet2D_choleskyD, 
                    'full_symmetric': UNet2D_symmetricD, 
                    'full_spectral': UNet2D_spectralD, 
                },
                'vae': {
                    'scalar': VAE2D_scalarD,
                    'diag': VAE2D_diagD,
                    'full_cholesky': VAE2D_choleskyD, 
                    'full_symmetric': VAE2D_symmetricD, 
                    'full_spectral': VAE2D_spectralD, 
                }
            }
        elif self.dimension == 3:
            return {
                'unet': {
                    'scalar': UNet3D_scalarD,
                    'diag': UNet3D_diagD,
                    'full_cholesky': UNet3D_choleskyD, 
                    'full_dual': UNet3D_dualD, 
                    'full_spectral': UNet3D_spectralD, 
                },
                'vae': {
                    'scalar': VAE3D_scalarD,
                    'diag': VAE3D_diagD,
                    'full_cholesky': VAE3D_choleskyD, 
                    'full_dual': VAE3D_dualD, 
                    'full_spectral': VAE3D_spectralD, 
                },
            }
        else:
            raise NotImplementedError('Not supported dimension !')
    @property
    def get_VDNet(self):
        print('-------------- Get V & D Net --------------')
        if self.dimension == 2:
            return {
                'unet': {
                    'vector_div_free_stream': {
                        'scalar': UNet2D_streamVscalarD,
                        'full_cholesky': UNet2D_streamVcholeskyD,
                        'full_symmetric': UNet2D_streamVsymmetricD,
                        'full_spectral': UNet2D_streamVspectralD,
                    }},
                #'vae': {
                #    'vector_div_free_stream': {
                #        'scalar': VAE2D_streamVscalarD,
                #        'full_cholesky': VAE2D_streamVcholeskyD,
                #        'full_symmetric': VAE2D_streamVsymmetricD,
                #        'full_spectral': VAE2D_streamVspectralD,
                #    }},
            }
        elif self.dimension == 3:
            return {
                'unet': {
                    'vector_div_free_stream': {
                        'scalar': UNet3D_streamVscalarD,
                        'full_cholesky': UNet3D_streamVcholeskyD,
                        'full_dual': UNet3D_streamVdualD,
                        'full_spectral': UNet3D_streamVspectralD,
                        #'full_symmetric': UNet3D_streamVsymmetricD,
                    },
                    'vector_div_free_clebsch': {
                        'scalar': UNet3D_clebschVscalarD,
                        'full_spectral': UNet3D_clebschVspectralD,
                    },
                },
                'vae': {
                    'vector_div_free_stream': {
                        'scalar': VAE3D_streamVscalarD,
                        'full_cholesky': VAE3D_streamVcholeskyD,
                        'full_dual': VAE3D_streamVdualD,
                        'full_spectral': VAE3D_streamVspectralD,
                    },
                    'vector_div_free_clebsch': {
                        'scalar': VAE3D_clebschVscalarD,
                        'full_spectral': VAE3D_clebschVspectralD,
                    },
                },
            }
        else:
            raise NotImplementedError('Not supported dimension !')

    def get_H(self):
        ''' 
        Only support for PD_V_type as HHD 
        partialH: curl-free by definition (http://www.sci.utah.edu/~hbhatia/pubs/2013_TVCG_survey.pdf)
        partialH is the curl-free component in HHD
        ''' 
        return self.VNet.get_H() # (n_batch, (s), r, c)
    def get_partialH(self):
        ''' 
        Only support for PD_V_type as HHD 
        partialH: curl-free by definition (http://www.sci.utah.edu/~hbhatia/pubs/2013_TVCG_survey.pdf)
        partialH is the curl-free component in HHD
        ''' 
        H = self.get_H() # (n_batch, r, c)  
        partialH = gradient_c(H, batched = True, delta_lst = self.data_spacing)
        if self.dimension == 2:
            return partialH.permute(0, 3, 1, 2) # (n_batch, r, c, 2) -> (n_batch, 2, r, c)
        elif self.dimension == 3:
            return partialH.permute(0, 4, 1, 2, 3) # (n_batch, s, r, c, 3) -> (n_batch, 3, s, r, c)
        else:
            raise NotImplementedError('Not supported dimension !')
    def get_Phi(self): 
        ''' 
        Only support for PD_V_type as stream or HHD 
        curlPhi: divergence-free be definition (http://www.sci.utah.edu/~hbhatia/pubs/2013_TVCG_survey.pdf)
        curlPhi is the output divergence-free vector field computed by stream function (or the divergence-free component in HHD)
        ''' 
        if self.perf_pattern == 'adv_diff':
            return self.VDNet.get_Phi() # (n_batch, 3, s, r, c) - 3D or (n_batch, r, c) - 2D
        else:
            return self.VNet.get_Phi()
    def get_curlPhi(self):
        ''' 
        Only support for PD_V_type as stream or HHD 
        curlPhi: divergence-free be definition (http://www.sci.utah.edu/~hbhatia/pubs/2013_TVCG_survey.pdf)
        curlPhi is the output divergence-free vector field computed by stream function (or the divergence-free component in HHD)
        ''' 
        Phi = self.get_Phi() # (n_batch, (3, s), r, c)
        if self.dimension == 2:
            curlPhi = torch.stack(stream_2D(Phi, batched = True, delta_lst = self.data_spacing))
            return curlPhi.permute(1, 0, 2, 3) # (2, n_batch, r, c) -> (n_batch, 2, r, c)
        elif self.dimension == 3:
            curlPhi = torch.stack(stream_3D(Phi[0], Phi[1], Phi[2], batched = True, delta_lst = self.data_spacing))
            return curlPhi.permute(1, 0, 2, 3, 4) # (3, n_batch, s, r, c) -> (n_batch, 3, s, r, c)
        else:
            raise NotImplementedError('Not supported dimension !')
    def get_L(self):
        ''' 
        Only support for PD_D_type as full_cholesky / full_spectral
        '''
        if self.perf_pattern == 'adv_diff':
            # (n_batch, 3, r, c): Lxx, Lxy, Lyy or (n_batch, 6, s, r, c)
            # (n_batch, 2, r, c): L1, L2  or (n_batch, 3, s, r, c): L1, L2, L3
            base_L, delta_L = self.VDNet.get_L() 
            return base_L, delta_L
        else:
            return self.DNet.get_L()
    def get_S(self):
        ''' 
        Only support for PD_D_type as full_spectral 
        '''
        if self.perf_pattern == 'adv_diff':
            return self.VDNet.get_S() # (n_batch, r, c): S or (n_batch, 3, s, r, c): S1, S2, S3
        else:
            return self.DNet.get_S() 
    def get_Sigma(self):
        ''' 
        Only support for PD_D_type as full_spectral 
        '''
        if self.perf_pattern == 'adv_diff':
            # TODO: Testing shared Sigma #
            '''if self.args.separate_DV_value_mask:
                Sigma = self.VDNet.get_Sigma()
                Sigma_D = Sigma[:, 0].unsqueeze(1) * self.sigma_D_mask
                Sigma_V = Sigma[:, 1].unsqueeze(1) * self.sigma_V_mask
                Sigma = torch.cat([Sigma_D, Sigma_V], dim = 1) # (n_batch, 2, r, c)
                return Sigma
            else:'''
            return self.VDNet.sigma * self.sigma_mask # (n_batch, r, c) or (n_batch, s, r, c)
        elif 'adv' not in self.perf_pattern:
            return self.DNet.get_Sigma() * self.sigma_mask
        elif 'diff' not in self.perf_pattern:
            return self.VNet.get_Sigma() * self.sigma_mask
    
    def get_U(self):
        ''' 
        Only support for stochastic as True 
        '''
        if self.perf_pattern == 'adv_diff':
            return self.VDNet.get_U() # (n_batch, 4, r, c): Uxx, Uxy, Uyx, Uyy or (n_batch, 9, s, r, c)
        else:
            return self.DNet.get_U() 
    def get_vars(self):
        ''' Only support for model_type as vae '''
        assert self.args.model_type == 'vae'
        mu, logvar = [], []
        if 'adv' not in self.perf_pattern:
            mu, logvar = self.DNet.get_vars()
        elif 'diff' not in self.perf_pattern:
            mu, logvar = self.VNet.get_vars()
        else:
            mu, logvar = self.VDNet.get_vars()
        return mu, logvar
    def get_value_mask(self):
        assert self.predict_value_mask
        if self.args.separate_DV_value_mask:
            D_value_mask, V_value_mask = self.VDNet.value_mask[:, 0].unsqueeze(1), self.VDNet.value_mask[:, 1].unsqueeze(1) # (n_batch, 1, (s,) r, c)
            return D_value_mask * self.anomaly_D_mask, V_value_mask * self.anomaly_V_mask
        else:
            return self.VDNet.value_mask * self.anomaly_mask # (n_batch, 1, (s,) r, c)
    def get_segment(self, threshold = 0.5, physics_deviation = None):
        assert self.predict_segment
        input_features = self.input_features if self.args.segment_net_type == 'conc' else physics_deviation # dV: (n_batch, dim, (s,) r, c); dL: (n_batch, dim, (s,) r, c)
        seg_mask = self.SegNet(input_features, threshold)
        return seg_mask # (n_batch, 1, s, r, c)
    def get_V(self):
        if self.predict_segment and self.args.segment_condition_on_physics:
            assert self.args.segment_net_type == 'conc'
            seg_mask = self.get_segment(threshold = 0.5) # (n_batch, 1, s, r, c)
        if self.perf_pattern == 'adv_diff':
            base_V, base_D, delta_V, delta_D, Sigma = self.VDNet(self.input_features) 
        else:
            if self.args.model_type == 'unet':
                base_V, delta_V, Sigma = self.VNet(self.input_features) 
            elif self.args.model_type == 'vae': # TODO: not updated to the latest features
                base_V, _, _ = self.VNet(self.input_features)  # (n_batch, 2/3, (s), r, c) # TODO: SDE version 
        base_V = self.div_free_v_cond(base_V) * self.vessel_mask * self.V_magnitude
        # TODO: Testing shared Sigma #
        self.Sigma = Sigma * self.sigma_mask
        '''if self.args.separate_DV_value_mask:
            Sigma = Sigma[:, 1].unsqueeze(1) * self.sigma_V_mask
        else:
            Sigma = Sigma * self.sigma_mask'''
        if self.predict_deviation and delta_V is not None:
            delta_V = self.div_free_v_cond(delta_V) * self.vessel_mask * self.V_magnitude
            if self.predict_segment and self.args.segment_condition_on_physics: 
                delta_V = seg_mask * delta_V  
        self.delta_V = delta_V
        return base_V, self.delta_V, self.Sigma # (n_batch, 3, s, r, c)
    
    ###########################################
    
    def get_Vt(self, it = 0):
        if it > 0: # FlowV version: V as time-varying field
            if self.dimension == 2:
                curr_Vx, curr_Vy = self.Vx_series[it], self.Vy_series[it] # (n_batch, r, c)
                return torch.stack([curr_Vx, curr_Vy], dim = 1), self.delta_V, self.Sigma # (n_batch, 2, r, c)
            elif self.dimension == 3:
                curr_Vx, curr_Vy, curr_Vz = self.Vx_series[it], self.Vy_series[it], self.Vz_series[it] # (r, c)
                return torch.stack([curr_Vx, curr_Vy, curr_Vz], dim = 1), self.delta_V, self.Sigma # (n_batch, 3, s, r, c)
            else:
                raise NotImplementedError('Dimension not supported!') 
        else: # PIANO version: V as time-independent field
            return self.get_V()
    
    def get_VDt(self, it = 0):
        if it > 0: # FlowV version: V as time-varying field
            _, base_D, self.delta_V, self.delta_D, self.Sigma = self.get_VD()
            if self.dimension == 2:
                return torch.stack([self.Vx_series[it], self.Vy_series[it]], dim = 1), base_D, self.delta_V, self.delta_D, self.Sigma # (n_batch, 2, r, c)
            elif self.dimension == 3:
                return torch.stack([self.Vx_series[it], self.Vy_series[it], self.Vz_series[it]], dim = 1), base_D, self.delta_V, self.delta_D, self.Sigma # (n_batch, 3, s, r, c)
            else:
                raise NotImplementedError('Dimension not supported!') 
        else: # PIANO version: V as time-independent field
            return self.get_VD()
        
    def get_V_series(self, t_tensor): 
        """
        Call once for generating V_series. 
        Then use the V_series to call self.forward() with one-time interval
        """
        # NOTE: should re-call for each training sample
        # assign actual Vlst and Dlst to PIANO_FlowV
        V, D, _, _, _ = self.get_VD()
        Vlst, Dlst = self.get_VDlst(V, D)
        self.flowV_PDE.update_DVlst(Dlst, Vlst) # (n_batch=1, r, c)
        
        self.flowV_PDE.axis = "x"
        self.Vx_series = odeint(self.flowV_PDE, Vlst['Vx'], t_tensor, method = 'dopri5', options = self.args)[:-1] # (nT, n_batch, r, c)
        self.flowV_PDE.axis = "y"
        self.Vy_series = odeint(self.flowV_PDE, Vlst['Vy'], t_tensor, method = 'dopri5', options = self.args)[:-1] # (nT, n_batch, r, c)
        if self.dimension == 3:
            self.flowV_PDE.axis = "z" 
            self.Vz_series = odeint(self.flowV_PDE, Vlst['Vz'], t_tensor, method = 'dopri5', options = self.args)[:-1] # (nT, n_batch, r, c)
        
        if self.dimension == 2:
            return self.Vx_series, self.Vy_series # (nT, n_batch, r, c)
        elif self.dimension == 3:
            return self.Vx_series, self.Vy_series, self.Vz_series # (nT, n_batch, r, c)
        else:
            raise NotImplementedError('Dimension not supported!')
        
    ###########################################
    
    def get_D(self):
        if self.predict_segment and self.args.segment_condition_on_physics:
            assert self.args.segment_net_type == 'conc'
            seg_mask = self.get_segment(threshold = 0.5) # (n_batch, 1, s, r, c)
        if self.perf_pattern == 'adv_diff':
            base_V, base_D, delta_V, delta_D, Sigma = self.VDNet(self.input_features)
        else:
            if self.args.model_type == 'unet': 
                base_D, delta_D, Sigma = self.DNet(self.input_features)
            elif self.args.model_type == 'vae':
                base_D, _, _ = self.DNet(self.input_features) # TODO: SDE version
        base_D = base_D * self.diffusion_mask * self.D_magnitude # (n_batch, s, r, c) for scalar or (n_batch, 3, s, r, c) for diag or (n_batch, 6, s, r, c) for tensor
        # TODO: Testing shared Sigma #
        self.Sigma = Sigma * self.sigma_mask
        '''if self.args.separate_DV_value_mask:
            Sigma = Sigma[:, 0].unsqueeze(1) * self.sigma_D_mask
        else:
            Sigma = Sigma * self.sigma_mask'''
        if self.predict_deviation and delta_D is not None:
            delta_D = delta_D * self.diffusion_mask * self.D_magnitude
            if self.predict_segment and self.args.segment_condition_on_physics: 
                delta_D = seg_mask * delta_D 
        self.delta_D = delta_D
        return base_D, self.delta_D, self.Sigma
    def get_VD(self):
        if self.perf_pattern == 'adv_diff':
            base_V, base_D, delta_V, delta_D, Sigma = self.VDNet(self.input_features) 
            if self.args.predict_value_mask:
                base_V = self.div_free_v_cond(base_V) * self.V_magnitude
                base_D = base_D * self.D_magnitude
            else:
                base_V = self.div_free_v_cond(base_V) * self.vessel_mask * self.V_magnitude
                base_D = base_D * self.diffusion_mask * self.D_magnitude 
        elif 'adv' not in self.perf_pattern: 
            base_D, delta_D, Sigma = self.DNet(self.input_features)  
            if self.args.predict_value_mask:
                base_D = base_D * self.D_magnitude
            else:
                base_D = base_D * self.diffusion_mask * self.D_magnitude 
        elif 'diff' not in self.perf_pattern:
            base_V, delta_V, Sigma = self.VNet(self.input_features)  
            if self.args.predict_value_mask:
                base_V = self.div_free_v_cond(base_V) * self.V_magnitude
            else:
                base_V = self.div_free_v_cond(base_V) * self.vessel_mask * self.V_magnitude 
        if self.predict_deviation and delta_V is not None and delta_D is not None: # NOTE: archived -- not work well
            delta_V = self.div_free_v_cond(delta_V) * self.vessel_mask * self.V_magnitude
            delta_D = delta_D * self.diffusion_mask * self.D_magnitude
            if self.predict_segment and self.args.segment_condition_on_physics:
                L, delta_L = self.get_L() 
                seg_mask = self.get_segment(threshold = 0.5, physics_deviation = torch.cat([delta_V, delta_L], dim = 1)) # (n_batch, 1, s, r, c) 
                delta_V = seg_mask * delta_V
                delta_D = seg_mask * delta_D  
        if self.args.stochastic:
            # TODO: Testing shared Sigma #
            self.Sigma = Sigma * self.sigma_mask
            '''if self.args.separate_DV_value_mask:
                Sigma_D = Sigma[:, 0].unsqueeze(1) * self.sigma_D_mask
                Sigma_V = Sigma[:, 1].unsqueeze(1) * self.sigma_V_mask
                Sigma = torch.cat([Sigma_D, Sigma_V], dim = 1)
            else:
                Sigma = Sigma * self.sigma_mask'''
        else:
            self.Sigma = None
        self.delta_V, self.delta_D = delta_V, delta_D
        return base_V, base_D, self.delta_V, self.delta_D, self.Sigma
    def get_Vlst(self, V):
        if self.V_type == 'constant' or self.V_type == 'scalar': 
            return {'V': V}
        elif self.dimension == 2:
            return {'Vx': V[:, 0], 'Vy': V[:, 1]}
        elif self.dimension == 3:
            return {'Vx': V[:, 0], 'Vy': V[:, 1], 'Vz': V[:, 2]}
    def get_Dlst(self, D):
        if self.D_type == 'constant' or self.D_type == 'scalar':
            return {'D': D}
        elif 'diag' in self.D_type:
            if self.dimension == 2:
                return {'Dxx': D[:, 0], 'Dyy': D[:, 1]}
            elif self.dimension == 3:
                return {'Dxx': D[:, 0], 'Dyy': D[:, 1], 'Dzz': D[:, 2]}
            else:
                raise NotImplementedError('Not supported dimension !')
        elif 'full' in self.D_type:
            if self.dimension == 2:
                return {'Dxx': D[:, 0], 'Dxy': D[:, 1], 'Dyy': D[:, 2]}
            elif self.dimension == 3:
                return {'Dxx': D[:, 0], 'Dxy': D[:, 1], 'Dxz': D[:, 2], 'Dyy': D[:, 3], 'Dyz': D[:, 4], 'Dzz': D[:, 5]}
            else:
                raise NotImplementedError('Not supported dimension !')
    def get_VDlst(self, V, D):
        if 'vector' in self.V_type and 'full' in self.D_type:
            if self.dimension == 2:
                return {'Vx': V[:, 0], 'Vy': V[:, 1]}, {'Dxx': D[:, 0], 'Dxy': D[:, 1], 'Dyy': D[:, 2]}
            elif self.dimension == 3:
                return {'Vx': V[:, 0], 'Vy': V[:, 1], 'Vz': V[:, 2]}, \
                        {'Dxx': D[:, 0], 'Dxy': D[:, 1], 'Dxz': D[:, 2], 'Dyy': D[:, 3], 'Dyz': D[:, 4], 'Dzz': D[:, 5]}
        elif 'vector' in self.V_type and 'scalar' in self.D_type:
            if self.dimension == 2:
                return {'Vx': V[:, 0], 'Vy': V[:, 1]}, {'D': D}
            elif self.dimension == 3:
                return {'Vx': V[:, 0], 'Vy': V[:, 1], 'Vz': V[:, 2]}, {'D': D}
        else:
            raise NotImplementedError('Not supported V_type + D_type combination !')
            

#############################################################################################################


class PIANOinD(VD_DeepBase):
    
    def __init__(self, args, data_dim, data_spacing, perf_pattern, in_channels, device):
        super(PIANOinD, self).__init__(args, data_dim, perf_pattern, in_channels, device)
        self.BC = self.args.BC
        self.partials = AdvDiffPartial(data_spacing, device) 
        self.V_time_varying = args.V_time
        if self.V_time_varying:
            self.flowV_PDE = FlowVPartial(args, data_spacing, device)
        self.it = 0 # Only for initialization #
        self.stochastic = args.stochastic 
        
    def forward(self, t, batch_C):
        '''
        t: (batch_size,)
        batch_C: (batch_size, slc, row, col)
        return (dC/dt) := \\div(D * \\nabla C) - V * \\nabla C
        '''
        batch_size  = batch_C.size(0) 
        batch_C = self.set_BC(batch_C) 

        if 'diff' not in self.perf_pattern: # NOTE: NOT up-to-date
            V, delta_V, Sigma = self.get_Vt(self.it)
            if self.args.predict_deviation:
                V = V + delta_V
            if self.args.predict_value_mask:
                value_mask = self.get_value_mask()
                V = V * value_mask
            Vlst = self.get_Vlst(V)
            out = self.partials.Grad_Vs[self.V_type](batch_C, Vlst)
            if self.stochastic:  
                out = out + Sigma * math.sqrt(self.args.dt) * torch.randn_like(batch_C).to(batch_C)
        elif 'adv' not in self.perf_pattern: # NOTE: NOT up-to-date
            D, delta_D, Sigma = self.get_D()
            if self.args.predict_deviation:
                D = D + delta_D
            if self.args.predict_value_mask:
                value_mask = self.get_value_mask()
                D = D * value_mask
            Dlst = self.get_Dlst(D)
            out = self.partials.Grad_Ds[self.D_type](batch_C, Dlst)
            if self.stochastic:  
                out = out + Sigma * math.sqrt(self.args.dt) * torch.randn_like(batch_C).to(batch_C)
        else:
            if self.args.joint_predict: # NOTE: archived, not work well
                V, D, delta_V, delta_D, Sigma = self.get_VDt(self.it) 
            else:
                V, delta_V, Sigma = self.get_Vt(self.it)
                D, delta_D, _ = self.get_D()
            if self.args.predict_deviation:
                V = V + delta_V
                D = D + delta_D
            if self.args.predict_value_mask:
                if self.args.separate_DV_value_mask:
                    D_value_mask, V_value_mask = self.get_value_mask()
                    D, V = D * D_value_mask, V * V_value_mask
                else:
                    value_mask = self.get_value_mask()
                    D, V = D * value_mask, V * value_mask
            Vlst, Dlst = self.get_VDlst(V, D)

            if self.stochastic: 
                out_V = self.partials.Grad_Vs[self.V_type](batch_C, Vlst) 
                out_D = self.partials.Grad_Ds[self.D_type](batch_C, Dlst) 
                out = out_D + out_V + Sigma[:, 0] * math.sqrt(self.args.dt) * torch.randn_like(batch_C).to(batch_C) # Sigma: (n_batch, n_channe1, s, r, c) --> (n_batch, s, r, c) 
            else:
                out_V = self.partials.Grad_Vs[self.V_type](batch_C, Vlst)  
                out_D = self.partials.Grad_Ds[self.D_type](batch_C, Dlst)
                out = out_V + out_D 

        return out



#############################################################################################################



class PIANO_Skeleton(nn.Module):
    '''
    Plain advection-diffusion PDE solver for pre-set V_lst and D_lst (1D, 2D, 3D) for forward movie simulation
    '''
    def __init__(self, args, data_spacing, perf_pattern, D_type, V_type, device):
        super(PIANO_Skeleton, self).__init__()
        self.args = args
        self.BC = args.BC
        self.dimension = len(data_spacing)
        self.perf_pattern = perf_pattern
        self.partials = AdvDiffPartial(data_spacing, device)
        self.V_time_varying = args.V_time 
        if self.V_time_varying:
            self.flowV_PDE = FlowVPartial(args, data_spacing, device)
        self.D_type, self.V_type = D_type, V_type
        self.stochastic = args.stochastic 
        self.Vlst, self.Dlst = {}, {} # Only for initialization #
        self.Sigma, self.Sigma_V, self.Sigma_D = 0., 0., 0. # Only for initialization # 
        if self.dimension == 1:
            self.neumann_BC = torch.nn.ReplicationPad1d(1)
        elif self.dimension == 2:
            self.neumann_BC = torch.nn.ReplicationPad2d(1)
        elif self.dimension == 3:
            self.neumann_BC = torch.nn.ReplicationPad3d(1)
        else:
            raise ValueError('Unsupported dimension: %d' % self.dimension)
                 
    @property
    def set_BC(self):
    # NOTE For bondary condition of mass concentration #
        '''X: (n_batch, spatial_shape)'''
        if self.BC == 'neumann' or self.BC == 'cauchy':
            if self.dimension == 1:
                return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        elif self.BC == 'dirichlet_neumann' or self.BC == 'source_neumann':
            ctrl_wdth = self.args.dirichlet_width if self.BC == 'dirichlet_neumann' else self.args.source_width
            if self.dimension == 1:
                self.dirichlet_BC = torch.nn.ReplicationPad1d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                self.dirichlet_BC = torch.nn.ReplicationPad2d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                self.dirichlet_BC = torch.nn.ReplicationPad3d(ctrl_wdth)
                return lambda X: self.neumann_dirichlet_BCBC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        else:
            return lambda X: X 
        
    def forward(self, t, batch_C):
        '''
        t: (batch_size,)
        batch_C: (batch_size, (slc,) row, col)
        ''' 
        batch_size = batch_C.size(0)
        batch_C = self.set_BC(batch_C)
        if 'diff' not in self.perf_pattern:
            out = self.partials.Grad_Vs[self.V_type](batch_C, self.Vlst) 
            if self.stochastic:  
                out = out + self.Sigma * math.sqrt(self.args.dt) * torch.randn_like(batch_C).to(batch_C)
        elif 'adv' not in self.perf_pattern:
            out = self.partials.Grad_Ds[self.D_type](batch_C, self.Dlst)
            if self.stochastic:  
                out = out + self.Sigma * math.sqrt(self.args.dt) * torch.randn_like(batch_C).to(batch_C)
        else:
            if self.stochastic:  
                out_D = self.partials.Grad_Ds[self.D_type](batch_C, self.Dlst)
                out_V = self.partials.Grad_Vs[self.V_type](batch_C, self.Vlst) 
                out = out_D + out_V + self.Sigma * math.sqrt(self.args.dt) * torch.randn_like(batch_C).to(batch_C) 
            else:
                out_V = self.partials.Grad_Vs[self.V_type](batch_C, self.Vlst)  
                out_D = self.partials.Grad_Ds[self.D_type](batch_C, self.Dlst)
                out = out_V + out_D
        return out

    ##################################
    
    def get_V_series(self, t_tensor): 
        """
        Call once for generating V_series. 
        Then use the V_series to call self.forward() with one-time interval
        """
        # assign actual Vlst and Dlst to PIANO_FlowV
        self.flowV_PDE.update_DVlst(self.Dlst, self.Vlst) # (n_batch=1, r, c)
        
        self.flowV_PDE.axis = "x"
        self.Vx_series = odeint(self.flowV_PDE, self.Vlst['Vx'], t_tensor, method = 'dopri5', options = self.args)[:-1, 0] # (nT, n_batch=1, r, c) -> (nT, r, c)
        self.flowV_PDE.axis = "y"
        self.Vy_series = odeint(self.flowV_PDE, self.Vlst['Vy'], t_tensor, method = 'dopri5', options = self.args)[:-1, 0] # (nT, n_batch=1, r, c) -> (nT, r, c)
        if self.dimension == 3:
            self.flowV_PDE.axis = "z" 
            self.Vz_series = odeint(self.flowV_PDE, self.Vlst['Vz'], t_tensor, method = 'dopri5', options = self.args)[:-1, 0] # (nT, n_batch=1, r, c) -> (nT, r, c)
        
        if self.dimension == 2:
            return self.Vx_series, self.Vy_series # (nT-1, r, c)
        elif self.dimension == 3:
            return self.Vx_series, self.Vy_series, self.Vz_series # (nT-1, r, c)
        else:
            raise NotImplementedError('Dimension not supported!')
    
    ##################################


###############################################################################
############################  V as Time-Dependent  ############################
###############################################################################

class PIANO_EnergyV(nn.Module):
    def __init__(self, args, t_all, data_dim, data_spacing, device, D_param_lst = None, V_param_lst = None):
        super(PIANO_EnergyV, self).__init__() 
        '''
        Start from D_param_lst, V_param_lst when not None
        '''
        self.args = args 
        self.device = device
        self.BC = self.args.BC
        self.data_dim = data_dim 
        self.dimension = len(data_dim)
        self.data_spacing = data_spacing 
        self.V_time_varying = args.V_time
        self.D_type, self.V_type = args.PD_D_type, args.PD_V_type # D:type: scalar / constant; V_type: vector
        self.partials = AdvDiffPartial(data_spacing, device) 
        self.D_param_lst, self.V_param_lst = D_param_lst, V_param_lst
        if self.dimension == 1:
            self.neumann_BC = torch.nn.ReplicationPad1d(1)
        elif self.dimension == 2:
            self.neumann_BC = torch.nn.ReplicationPad2d(1)
        elif self.dimension == 3:
            self.neumann_BC = torch.nn.ReplicationPad3d(1)
        else:
            raise ValueError('Unsupported dimension: %d' % self.dimension) 

        self.nT = len(t_all)
        self.P_series, self.it = None, 0 # Only for initialization, TBD during training #
        self.register_learning_params()

    @property
    def set_BC(self):
    # NOTE For bondary condition of mass concentration #
        '''X: (n_batch, spatial_shape)'''
        if self.BC == 'neumann' or self.BC == 'cauchy':
            if self.dimension == 1:
                return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        elif self.BC == 'dirichlet_neumann' or self.BC == 'source_neumann':
            ctrl_wdth = self.args.dirichlet_width if self.BC == 'dirichlet_neumann' else self.args.source_width
            if self.dimension == 1:
                self.dirichlet_BC = torch.nn.ReplicationPad1d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                self.dirichlet_BC = torch.nn.ReplicationPad2d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                self.dirichlet_BC = torch.nn.ReplicationPad3d(ctrl_wdth)
                return lambda X: self.neumann_dirichlet_BCBC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        else:
            return lambda X: X    

    def forward(self, t, batch_C):
        '''
        batch_C: (batch_size, ((slc,) row,) col)
        return (dC/dt) := \\div(D * \\nabla C) - V * \\nabla C
        '''
        batch_C = self.set_BC(batch_C)
        out_D = self.partials.Grad_Ds[self.D_type](batch_C, self.get_Dlst())
        out_V = self.partials.Grad_Vs['vector'](batch_C, self.get_Vlst(self.it))
        dC = out_D + out_V
        return dC
 
    def register_learning_params(self): 
        if self.args.fix_D: # TODO: under test #
            self.D = torch.ones(self.data_dim).to(self.device) * 0.01
        else:
            if self.D_param_lst:
                print('Register with input D')
                self.register_parameter('D', nn.Parameter(self.D_param_lst['D'])) 
                #print(self.D.mean().item())
            else: 
                print('Register with new D')
                if self.D_type == 'constant':
                    self.register_parameter('D', nn.Parameter(torch.tensor(0., dtype = torch.float, device = self.device, requires_grad = True)))
                elif self.D_type == 'scalar':
                    self.register_parameter('D', nn.Parameter((torch.randn(self.data_dim, dtype = torch.float, device = self.device, requires_grad = True) + 1) * 1e-4))

        if self.V_param_lst:
            print('Register with input Vlst') 
            if self.V_type == 'vector':
                self.register_parameter('Vx', nn.Parameter(self.V_param_lst['Vx'])) 
                self.register_parameter('Vy', nn.Parameter(self.V_param_lst['Vy'])) 
                if self.dimension == 3:
                    self.register_parameter('Vz', nn.Parameter(self.V_param_lst['Vz']))  
            elif self.V_type == 'vector_curl_free':
                self.register_parameter('P', nn.Parameter(self.V_param_lst['P']))
        else:
            print('Register with new Vlst') 
            if self.V_type == 'vector': 
                self.register_parameter('Vx', nn.Parameter((torch.randn(tuple([self.nT-1]) + tuple(self.data_dim), dtype = torch.float, device = self.device, requires_grad = True)) * 1e-4))  
                self.register_parameter('Vy', nn.Parameter((torch.randn(tuple([self.nT-1]) + tuple(self.data_dim), dtype = torch.float, device = self.device, requires_grad = True)) * 1e-4))
                if self.dimension == 3:
                    self.register_parameter('Vz', nn.Parameter((torch.randn(tuple([self.nT-1]) + tuple(self.data_dim), dtype = torch.float, device = self.device, requires_grad = True)) * 1e-4))
            elif self.V_type == 'vector_curl_free':
                self.register_parameter('P', nn.Parameter((torch.randn(tuple([self.nT-1]) + tuple(self.data_dim), dtype = torch.float, device = self.device, requires_grad = True)) * 1e-4)) # V := \nabla P; P: (nT, (s,), r, c)
     
    def get_Dlst(self): 
        if self.D_type == 'constant':
            return {'D': torch.clamp(self.D, min = 0.)}
        elif self.D_type == 'scalar':
            return {'D': torch.clamp(self.D, min = 0.).unsqueeze(0)}

    def get_Vlst(self, it):
        '''
        self.P: (nT, (s,) r, c)
        REQUIRED: self.i_t
        ''' 
        if self.V_type == 'vector':
            Vlst = {'Vx': self.Vx[it].unsqueeze(0), 'Vy': self.Vy[it].unsqueeze(0)} # (n_batch=1, nT-1, r, c)
            if self.dimension == 3:
                Vlst.update({'Vz': self.Vz[it].unsqueeze(0)})
        elif self.V_type == 'vector_curl_free': 
            V = gradient_c(self.P[it].unsqueeze(0), batched = True, delta_lst = self.data_spacing) 
            Vx, Vy = V[..., 0], V[..., 1]
            #print(Vx.mean().item(), Vy.mean().item())
            if self.dimension == 2:
                V = torch.stack([Vx, Vy]) # (2, n_batch=1, r, c)
                Vlst = {'Vx': Vx, 'Vy': Vy} 
            elif self.dimension == 3:
                Vz = V[..., 2]
                V = torch.stack([Vx, Vy, Vz]) # (3, n_batch=1, s, r, c)
                Vlst = {'Vx': Vx, 'Vy': Vy, 'Vz': Vz}
            else:
                print('Dimension == %d not supported!' % self.dimension) 
        else:
            raise NotImplementedError('PD_V_type not supported:', self.V_type)
        return Vlst

    def get_V_series(self, origin = None, spacing = None, direction = None, save_fld = None): # For saving # 
        '''
        P: (nT, (s,) r, c)
        '''
        if self.V_type == 'vector':
            Vx_series, Vy_series = self.Vx, self.Vy
            if self.dimension == 3:
                Vz_series = self.Vz
        elif self.V_type == 'vector_curl_free':
            Vx_series, Vy_series, Vz_series = torch.zeros_like(self.P), torch.zeros_like(self.P), torch.zeros_like(self.P) # (nT, r, c)
            V = gradient_f(self.P, batched = True, delta_lst = self.data_spacing) # nT as batch_channel
            for it in range(self.P.size(0)): 
                Vx_series[it] = V[..., 0] # (r, c)
                Vy_series[it] = V[..., 1]
                if self.dimension == 3:
                    Vz_series[it] = V[..., 2]
        else:
            raise NotImplementedError('PD_V_type not supported:', self.V_type)

        if save_fld is not None:
            nda2img(Vx_series.detach().cpu().numpy(), origin = origin, spacing = spacing, direction = direction, save_path = os.path.join(save_fld, 'Vx.nii'))
            nda2img(Vy_series.detach().cpu().numpy(), origin = origin, spacing = spacing, direction = direction, save_path = os.path.join(save_fld, 'Vy.nii'))
            if self.dimension == 3:
                nda2img(Vz_series.detach().cpu().numpy(), save_path = os.path.join(save_fld, 'Vz.nii'))
                
        if self.dimension == 2:
            return {'Vx': Vx_series, 'Vy': Vy_series}
        elif self.dimension == 3:
            return {'Vx': Vx_series, 'Vy': Vy_series, 'Vz': Vz_series} # (nT, r, c)
        else:
            raise ValueError('Unsupported dimension: %d' % self.dimension) 
 
 
###############################################################################
    
    
class PIANO_FlowV(nn.Module):
    def __init__(self, args, data_dim, data_spacing, perf_pattern, device, mask = None, D_param_lst = None, V_param_lst = None):
        super(PIANO_FlowV, self).__init__() 
        '''
        Supports V as either time-constant or time-varying
        Start from D_param_lst, V_param_lst when not None
        '''
        self.args = args
        self.device = device
        self.BC = self.args.BC
        self.data_dim = data_dim
        self.vessel_mask = 1. # NOTE To assign during learning #
        self.dimension = len(data_dim)
        self.data_spacing = data_spacing
        self.perf_pattern = perf_pattern
        self.D_type, self.V_type = args.PD_D_type, args.PD_V_type
        self.mask = mask if mask is not None else Variable(torch.ones(self.data_dim, dtype = torch.float, device = self.device).to(device))
        self.V_time_varying = args.V_time 
        if self.V_time_varying:
            self.flowV_PDE = FlowVPartial(args, data_spacing, device)
        self.partials = AdvDiffPartial(data_spacing, device) 
        self.D_param_lst, self.V_param_lst = D_param_lst, V_param_lst
        if self.dimension == 1:
            self.neumann_BC = torch.nn.ReplicationPad1d(1)
        elif self.dimension == 2:
            self.neumann_BC = torch.nn.ReplicationPad2d(1)
        elif self.dimension == 3:
            self.neumann_BC = torch.nn.ReplicationPad3d(1)
        else:
            raise ValueError('Unsupported dimension: %d' % self.dimension) 

        self.relu = nn.ReLU()
        self.it = 0 # Initialization, if V is time-varying, self.it needs to be assgined during learning
        self.register_learning_params()
        
    @property
    def set_BC(self):
    # NOTE For bondary condition of mass concentration #
        '''X: (n_batch, spatial_shape)'''
        if self.BC == 'neumann' or self.BC == 'cauchy':
            if self.dimension == 1:
                return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        elif self.BC == 'dirichlet_neumann' or self.BC == 'source_neumann':
            ctrl_wdth = self.args.dirichlet_width if self.BC == 'dirichlet_neumann' else self.args.source_width
            if self.dimension == 1:
                self.dirichlet_BC = torch.nn.ReplicationPad1d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                self.dirichlet_BC = torch.nn.ReplicationPad2d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                self.dirichlet_BC = torch.nn.ReplicationPad3d(ctrl_wdth)
                return lambda X: self.neumann_dirichlet_BCBC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        else:
            return lambda X: X    

    def forward(self, t, batch_C):
        '''
        batch_C: (batch_size, ((slc,) row,) col)
        return (dC/dt) := \\div(D * \\nabla C) - V * \\nabla C
        '''
        batch_C = self.set_BC(batch_C)
        if 'diff' not in self.perf_pattern:
            dC = self.partials.Grad_Vs[self.V_type](batch_C, self.get_Vlst(self.get_Vt(self.it)))
        elif 'adv' not in self.perf_pattern:
            dC = self.partials.Grad_Ds[self.D_type](batch_C, self.get_Dlst(self.get_D()))
        else:
            out_D = self.partials.Grad_Ds[self.D_type](batch_C, self.get_Dlst(self.get_D()))
            out_V = self.partials.Grad_Vs[self.V_type](batch_C, self.get_Vlst(self.get_Vt(self.it)))
            dC = out_D + out_V
        return dC

    def register(self, name_lst): 
        for name in name_lst: # all initialized ~ randn([-1, 1])
            #self.register_parameter(name, nn.Parameter((torch.randn(self.data_dim, dtype = torch.float, device = self.device, requires_grad = True) - 0.5) * 2 * self.mask))
            self.register_parameter(name, nn.Parameter((torch.zeros(self.data_dim, dtype = torch.float, device = self.device, requires_grad = True)) * self.mask))
    
    def register_learning_params(self):
        if self.args.fix_D: 
            self.D = torch.ones(self.data_dim).to(self.device) * 0.01
        elif self.D_param_lst:
            self.register_D(self.D_param_lst)
        elif self.D_type == 'constant':
            self.register_parameter('D', nn.Parameter(torch.tensor(0., dtype = torch.float, device = self.device, requires_grad = True)))
        else:  
            self.register(self.D_names)
        
        if self.V_param_lst:
            self.register_V(self.V_param_lst)
        elif self.V_type == 'constant':
            self.register_parameter('V', nn.Parameter(torch.tensor(0., dtype = torch.float, device = self.device, requires_grad = True)))
        else:
            self.register(self.V_names)

    def register_D(self, D_param_lst):
        if self.D_type == 'constant' or self.D_type == 'scalar':
            self.register_parameter('D', nn.Parameter(D_param_lst['D']))
        elif 'spectral' in self.D_type:
            L = D_param_lst['L'] # (shape, 2/3)
            S = D_param_lst['S'] # (shape, (1)/3)
            self.register_parameter('L1', nn.Parameter(L[..., 0]))
            self.register_parameter('L2', nn.Parameter(L[..., 1]))
            if self.dimension == 2:
                self.register_parameter('S', nn.Parameter(S))
            elif self.dimension == 3:
                self.register_parameter('L3', nn.Parameter(L[..., 2]))
                self.register_parameter('S1', nn.Parameter(S[..., 0]))
                self.register_parameter('S2', nn.Parameter(S[..., 1]))
                self.register_parameter('S3', nn.Parameter(S[..., 2]))
        else:
            D = D_param_lst['D'] # (4/9, shape)
            if self.dimension == 2:
                self.register_parameter('Dxx', nn.Parameter(D[..., 0]))
                self.register_parameter('Dyy', nn.Parameter(D[..., 3]))
                if 'full' in self.D_type:
                    self.register_parameter('Dxy', nn.Parameter(D[..., 1]))
            elif self.dimension == 3:
                self.register_parameter('Dxx', nn.Parameter(D[..., 0]))
                self.register_parameter('Dyy', nn.Parameter(D[..., 4]))
                self.register_parameter('Dzz', nn.Parameter(D[..., 8]))
                if 'full' in self.D_type:
                    self.register_parameter('Dxy', nn.Parameter(D[..., 1]))
                    self.register_parameter('Dxz', nn.Parameter(D[..., 2]))
                    self.register_parameter('Dyz', nn.Parameter(D[..., 5]))
            else:
                raise NotImplementedError

    def register_V(self, V_param_lst):
        if self.V_type == 'constant' or self.V_type == 'scalar':
            self.register_parameter('V', nn.Parameter(V_param_lst['V']))
        elif 'div_free' in self.V_type:
            Phi = V_param_lst['Phi'] # (shape, 1/3)
            if 'HHD' in self.V_type: # (shape)
                self.register_parameter('H', nn.Parameter(V_param_lst['H']))
            if self.dimension == 2:
                self.register_parameter('Phi', nn.Parameter(Phi))
            elif self.dimension == 3:
                self.register_parameter('Phi_a', nn.Parameter(Phi[..., 0]))
                self.register_parameter('Phi_b', nn.Parameter(Phi[..., 1]))
                if 'stream' in self.V_type:
                    self.register_parameter('Phi_c', nn.Parameter(Phi[..., 2]))
        else:
            self.register_parameter('Vx', nn.Parameter(V_param_lst['Vx']))
            self.register_parameter('Vy', nn.Parameter(V_param_lst['Vy']))
            if self.dimension == 3:
                self.register_parameter('Vz', nn.Parameter(V_param_lst['Vz']))

    @property
    def D_names(self):
        if 'diff' not in self.perf_pattern:
            pass
        elif self.D_type == 'constant' or self.D_type == 'scalar':
            Ds = {'D'}
            return Ds 
        else: # 'constant', 'scalar', 'diag', 'full_spectral' 'full_cholesky', 'full_dual', 'full_spectral', 'full_semi_spectral'
            if 'spectral' in self.D_type:
                if self.dimension == 2:
                    Ds = {'L1', 'L2', 'S'}
                elif self.dimension == 3:
                    Ds = {'L1', 'L2', 'L3', 'S1', 'S2', 'S3'}
                else:
                    raise ValueError('Wrong data dimension')
            else:
                Ds = {'Dxx', 'Dyy'}
                if self.dimension == 3:
                    Ds = {'Dxx', 'Dyy', 'Dzz'}
                if 'full' in self.D_type:
                    Ds.update({'Dxy'})
                    if self.dimension == 3:
                        Ds.update({'Dxz'})
                        Ds.update({'Dyz'})
            return Ds

    @property
    def V_names(self): # 'constant', 'vector', 'vector_div_free_clebsch', 'vector_div_free_stream', 'vector_HHD'
        if 'adv' not in self.perf_pattern:
            pass
        elif self.V_type == 'constant' or self.V_type == 'scalar':
            Vs = {'V'}
        elif 'div_free' in self.V_type:
            if self.dimension == 2:
                if 'HHD' in self.V_type:
                    Vs = {'Phi', 'H'}
                else:
                    Vs = {'Phi'}
            elif self.dimension == 3:
                if 'clebsch' in self.V_type:
                    Vs = {'Phi_a', 'Phi_b'}
                elif 'stream' in self.V_type:
                    Vs = {'Phi_a', 'Phi_b', 'Phi_c'}
                elif 'HHD' in self.V_type:
                    Vs = {'Phi_a', 'Phi_b', 'Phi_c', 'H'}
        else:
            Vs = {'Vx', 'Vy'}
            if self.dimension == 3:
                Vs.update({'Vz'})
        return Vs
    
    def get_Vlst(self, V): # V: (dim, n_batch, shape) # Add batch dim for integration # (dim, n_batch, (s,) r, c)
        if self.V_type == 'constant' or self.V_type == 'scalar':
            return {'V': V[None]}
        elif self.dimension == 2:
            return {'Vx': V[None, 0], 'Vy': V[None, 1]} # (n_batch=1, r, c)
        elif self.dimension == 3:
            return {'Vx': V[None, 0], 'Vy': V[None, 1], 'Vz': V[None, 2]} # (n_batch, s, r, c)
        else:
            raise NotImplementedError('Not supported dimension !')

    def get_Dlst(self, D): # D (1/3/6, n_batch, shape) # Add batch dim for integration #
        if self.D_type == 'constant' or self.D_type == 'scalar':
            return {'D': D[None]} # (n_batch=1, (s,) r, c)
        elif 'diag' in self.D_type:
            if self.dimension == 2:
                return {'Dxx': D[None, 0], 'Dyy': D[None, 1]}
            elif self.dimension == 3:
                return {'Dxx': D[None, 0], 'Dyy': D[None, 1], 'Dzz': D[None, 2]}
            else:
                raise NotImplementedError('Not supported dimension !')
        elif 'full' in self.D_type:
            if self.dimension == 2:
                return {'Dxx': D[None, 0], 'Dxy': D[None, 1], 'Dyy': D[None, 2]}
            elif self.dimension == 3:
                return {'Dxx': D[None, 0], 'Dxy': D[None, 1], 'Dxz': D[None, 2], 'Dyy': D[None, 3], 'Dyz': D[None, 4], 'Dzz': D[None, 5]}
            else:
                raise NotImplementedError('Not supported dimension !')

    @property
    def get_D_func(self):
        return {
            'constant': self.get_constantD,
            'scalar': self.get_scalarD,
            'diag': self.get_diagD,
            'full': self.get_fullD,
            'full_spectral': self.get_spectralD,
            'full_cholesky': self.get_choleskyD,
        }
    
    @property
    def get_V_func(self):
        return {
            #'constant': self.get_constantV,
            #'scalar': self.get_scalarV,
            'vector': self.get_vectorV,
            #'vector_div_free_clebsch': self.get_divfree_clebschV,
            'vector_div_free_stream': self.get_divfree_streamV,
            #'vector_div_free_stream_gauge': self.get_divfree_streamV, 
            #'vector_HHD': self.get_HHDV,
        }

    ###########################################
    
    def get_D(self):
        return self.get_D_func[self.D_type]()

    def get_V(self):
        return self.get_V_func[self.V_type]()

    def get_Vt(self, it = 0):
        if it > 0 and self.V_time_varying: # FlowV version: V as time-varying field
            if self.dimension == 2:
                curr_Vx, curr_Vy = self.Vx_series[it], self.Vy_series[it] # (r, c)
                return torch.stack([curr_Vx, curr_Vy], dim = 0) # (2, r, c)
            elif self.dimension == 3:
                curr_Vx, curr_Vy, curr_Vz = self.Vx_series[it], self.Vy_series[it], self.Vz_series[it] # (r, c)
                return torch.stack([curr_Vx, curr_Vy, curr_Vz], dim = 0) # (3, s, r, c)
            else:
                raise NotImplementedError('Dimension not supported!') 
        else: # PIANO version: V as time-independent field
            return self.get_V()
        
    ###########################################

    def get_constantD(self):
        return torch.ones(self.data_dim, dtype = torch.float, device = self.device) * torch.clamp(self.D, min = 0.) * self.mask #** 2
        
    def get_scalarD(self):
        return torch.clamp(self.D, min = 0.) * self.mask
    
    def get_diagD(self):
        if self.dimension == 1:
            raise NotImplementedError('diag_D is not supported for 1D version of diffusivity')
        elif self.dimension == 2:
            return torch.stack([self.Dxx, self.Dyy], dim = 0) * self.mask[None] # (2, r, c)
        elif self.dimension == 3:
            return torch.stack([self.Dxx, self.Dyy, self.Dzz], dim = 0) * self.mask[None] # (3, s, r, c)
        
    def get_fullD(self): 
        if self.dimension == 1:
            raise NotImplementedError('full_D is not supported for 1D version of diffusivity')
        elif self.dimension == 2:
            return torch.stack([self.Dxx, self.Dxy, self.Dyy], dim = 0) * self.mask[None] # (3, r, c)
        elif self.dimension == 3:
            return torch.stack([self.Dxx, self.Dxy, self.Dxz, self.Dyy, self.Dyz, self.Dzz], dim = 0) * self.mask[None] # (6, s, r, c)

    def get_spectralD(self):
        '''Spectral decomposition'''
        if self.dimension == 2:
            U = cayley_map(self.S1) # (r, c)
            raw_L = torch.clamp(torch.stack([self.L1, self.L2], dim =  0), min = 0.) # (2, r, c)
            L1, L2 =  raw_L[0] + raw_L[1],  raw_L[0] # NOTE: L in descending order #
            return construct_spectralD_2D(U, torch.stack([L1, L2, L3], dim = 0), batched = False) * self.mask[None]  # (3, r, c)
        elif self.dimension == 3:
            U = cayley_map(torch.stack([self.S1, self.S2, self.S3], dim =  0)) # (3, s, r, c)
            raw_L = torch.clamp(torch.stack([self.L1, self.L2, self.L3], dim =  0), min = 0.) # (3, s, r, c)
            L1, L2, L3 = raw_L[0] + raw_L[1] + raw_L[2], raw_L[0] + raw_L[1], raw_L[0] # NOTE: L in descending order #
            return construct_spectralD_3D(U, torch.stack([L1, L2, L3], dim = 0), batched = False) * self.mask[None]  # (6, s, r, c)
        else:
            raise NotImplementedError

    def get_L(self):
        if self.dimension == 2:
            raw_L = torch.clamp(torch.stack([self.L1, self.L2], dim =  0), min = 0.) # (2, r, c)
            L1, L2 = raw_L[:, 0] + raw_L[:, 1],  raw_L[:, 0] # NOTE: L in descending order #
            return torch.stack([L1, L2], dim = 0)
        elif self.dimension == 3:
            U = cayley_map(torch.stack([self.S1, self.S2, self.S3], dim =  0)) # (3, s, r, c)
            raw_L = torch.clamp(torch.stack([self.L1, self.L2, self.L3], dim =  0), min = 0.) # (3, s, r, c)
            L1, L2, L3 = raw_L[:, 0] + raw_L[:, 1] + raw_L[:, 2], raw_L[:, 0] + raw_L[:, 1], raw_L[:, 0] # NOTE: L in descending order #
            return torch.stack([L1, L2, L3], dim = 0)
        else:
            raise NotImplementedError

    def get_U(self):
        if self.dimension == 2:
            U = cayley_map(self.S1) # (r, c, 2, 2)
            return flatten_U_2D(U, batched = False) * self.mask[None] # (4, r, c)
        elif self.dimension == 3:
            U = cayley_map(torch.stack([self.S1, self.S2, self.S3], dim =  0)) # (s, r, c, 3, 3)
            return flatten_U_3D(U, batched = False) * self.mask[None] # (9, s, r, c)
        else:
            raise NotImplementedError

    def get_S(self):
        if self.dimension == 2:
            return self.S1 * self.mask  # (r, c)
        elif self.dimension == 3:
            return torch.stack([self.S1, self.S2, self.S3], dim =  0) * self.mask[None] # (3, s, r, c)
        else:
            raise NotImplementedError

    def get_choleskyD(self):
        '''Cholesky decomposition: ensuring PSD -- D = LL^T, L is a lower-triangular matrix, with non-negative diagonal entries'''
        if self.dimension == 1:
            raise NotImplementedError('full_cholesky is not supported for 1D version of diffusivity')
        elif self.dimension == 2:
            Dxx = self.Dxx ** 2
            Dxy = self.Dxx * self.Dxy
            Dyy = self.Dxy ** 2 + self.Dyy ** 2
            return torch.stack([self.Dxx, self.Dxy, self.Dyy], dim = 0) * self.mask[None] # (3, r, c)
        elif self.dimension == 3:
            Dxx = self.Dxx ** 2
            Dxy = self.Dxx * self.Dxy
            Dxz = self.Dxx * self.Dxz
            Dyy = self.Dxy ** 2 + self.Dyy ** 2
            Dyz = self.Dxy * self.Dxz + self.Dyy * self.Dyz
            Dzz = self.Dxz ** 2 + self.Dyz ** 2 + self.Dzz ** 2
            return torch.stack([self.Dxx, self.Dxy, self.Dxz, self.Dyy, self.Dyz, self.Dzz], dim = 0) * self.mask[None] # (6, s, r, c)
        else:
            raise NotImplementedError
    
    ##################################
    
    def get_V_series(self, t_tensor = None): # Call once upon V(t=0) being updated #
        
        V0_lst = self.get_Vlst(self.get_V()) # (n_batch=1, r, c)
        Dlst = self.get_Dlst(self.get_D()) # (n_batch=1, r, c) 
        
        if self.V_time_varying:

            self.flowV_PDE.update_DVlst(Dlst, V0_lst)
            
            self.flowV_PDE.axis = "x"
            self.Vx_series = odeint(self.flowV_PDE, V0_lst['Vx'], t_tensor, method = 'dopri5', options = self.args)[:-1, 0] # (nT, n_batch=1, r, c) -> (nT-1, r, c)
            self.flowV_PDE.axis = "y"
            self.Vy_series = odeint(self.flowV_PDE, V0_lst['Vy'], t_tensor, method = 'dopri5', options = self.args)[:-1, 0] # (nT, n_batch=1, r, c) -> (nT-1, r, c)
            if self.dimension == 3:
                self.flowV_PDE.axis = "z" 
                self.Vz_series = odeint(self.flowV_PDE, V0_lst['Vz'], t_tensor, method = 'dopri5', options = self.args)[:-1, 0] # (nT, n_batch=1, r, c) -> (nT-1, r, c)
            
            if self.dimension == 2:
                return self.Vx_series, self.Vy_series # (nT, r, c)
            elif self.dimension == 3:
                return self.Vx_series, self.Vy_series, self.Vz_series # (nT, r, c)
            else:
                raise NotImplementedError('Dimension not supported!')

        else:
            if self.dimension == 2:
                return V0_lst["Vx"], V0_lst["Vy"] # (1, r, c)
            elif self.dimension == 3:
                return V0_lst["Vx"], V0_lst["Vy"], V0_lst["Vz"] # (1, r, c)
            else:
                raise NotImplementedError('Dimension not supported!')
            
    
    ##################################
    
    def get_vectorV(self):
        if self.dimension == 2:
            return torch.stack([self.Vx, self.Vy], dim = 0) * self.mask[None] * self.vessel_mask # (2, r, c)
        elif self.dimension == 3:
            return torch.stack([self.Vx, self.Vy, self.Vz], dim = 0) * self.mask[None] * self.vessel_mask # (3, s, r, c)
        else:
            raise NotImplementedError('Dimension not supported!')

    def get_divfree_clebschV(self):
        '''
        Ref: Representation of divergence-free vector fields (https://www.jstor.org/stable/43638978?seq=1)
        '''
        if self.dimension == 1:
            raise NotImplementedError('vector is not supported for 1D version of velocity')
        elif self.dimension == 2:
            Vx, Vy = clebsch_2D(self.Phi, batched = False, delta_lst = self.data_spacing)
            return torch.stack([Vx, Vy], dim = 0) * self.mask[None] * self.vessel_mask
        elif self.dimension == 3:
            Vx, Vy, Vz = clebsch_3D(self.Phi_a, self.Phi_b, batched = False, delta_lst = self.data_spacing)
            return torch.stack([Vx, Vy, Vz], dim = 0) * self.mask[None] * self.vessel_mask
        else:
            raise NotImplementedError

    def get_Phi(self):
        if self.dimension == 2:
            return self.Phi * self.mask
        elif self.dimension == 3:
            if 'clebsch' in self.V_type:
                return torch.stack([self.Phi_a, self.Phi_b], dim = 0) * self.mask[None]
            else:
                return torch.stack([self.Phi_a, self.Phi_b, self.Phi_c], dim = 0) * self.mask[None]
        else:
            raise NotImplementedError

    def get_H(self):
        return self.H * self.mask

    def get_divfree_streamV(self):
        if self.dimension == 2:
            Vx, Vy = stream_2D(self.Phi, batched = False, delta_lst = self.data_spacing)
            return torch.stack([Vx, Vy], dim = 0) * self.mask[None]
        elif self.dimension == 3:
            Vx, Vy, Vz = stream_3D(self.Phi_a, self.Phi_b, self.Phi_c, batched = False, delta_lst = self.data_spacing)
            return torch.stack([Vx, Vy, Vz], dim = 0) * self.mask[None]
        else:
            raise NotImplementedError

    def get_HHDV(self):
        if self.dimension == 3:
            Vx, Vy, Vz = HHD_3D(self.Phi_a, self.Phi_b, self.Phi_c, self.H, batched = False, delta_lst = self.data_spacing)
            return torch.stack([Vx, Vy, Vz], dim = 0) * self.mask[None]
        else:
            raise NotImplementedError



###############################################################################
#########################  Partial Utilities  #################################
###############################################################################



class FlowVPartial(nn.Module):
    '''
    PDE: # TODO: so far ONLY support D as SCALAR, and V as VECTOR
    \delta V / \delta t + 1/2 \nabla || V ||^2 = - \nabla (D V) == - D \laplacian V = - D \div (\nabla V) == - D \nabla \cdot (\nabla V)
    (V is constructed to be curl-free := \nabla p)  ==> 
    \delta p / \delta t + 1/2 || \nabla p ||^2 = - D \laplacian p == - D \div (\nabla P) == - D \nabla \cdot (\nabla p)
    
    DOc: https://www.overleaf.com/read/ddsqbfbpnkzd
    '''
    def __init__(self, args, data_spacing, device):
        super(FlowVPartial, self).__init__()
        self.dimension = len(data_spacing)
        assert self.dimension == 2 or self.dimension == 3
        self.BC = args.BC
        self.device = device
        self.data_spacing = data_spacing
        try:
            self.D_type, self.V_type = args.PD_D_type, args.PD_V_type
        except:
            self.D_type, self.V_type = args.GT_D_type, args.GT_V_type # for data simulation code
        
        self.Dlst = None # Only for initialization, to be updated during training # 
        self.Vlst = None # Only for initialization, to be updated during training # 
        self.axis = None # Only for initialization, to be specified during training # 
        self.norm_V, self.div_DV = None, None

        if self.dimension == 1:
            self.neumann_BC = torch.nn.ReplicationPad1d(1)
        elif self.dimension == 2:
            self.neumann_BC = torch.nn.ReplicationPad2d(1)
        elif self.dimension == 3:
            self.neumann_BC = torch.nn.ReplicationPad3d(1)
        else:
            raise NotImplementedError('Dimension == %d not supported!' % self.dimension)
    
    def update_DVlst(self, Dlst, Vlst):
        self.Dlst = Dlst
        self.Vlst = Vlst
        
        # update intermediate values
        if self.D_type == "constant":
            D = self.Dlst["D"]
            if self.dimension == 2:
                self.div_DV = self.dXf(D * self.Vlst["Vx"]) + self.dYf(D * self.Vlst['Vy'])
            elif self.dimension == 3:
                self.div_DV = self.dXf(D * self.Vlst["Vx"]) + self.dYf(D * self.Vlst['Vy']) + self.dZf(self.Vlst["Vz"])
            #pass
        elif self.D_type == "scalar":
            D = self.Dlst["D"]
            if self.dimension == 2:
                self.div_DV = self.dXf(D * self.Vlst["Vx"]) + self.dYf(D * self.Vlst['Vy'])
            elif self.dimension == 3:
                self.div_DV = self.dXf(D * self.Vlst["Vx"]) + self.dYf(D * self.Vlst['Vy']) + self.dZf(self.Vlst["Vz"])
        else:
            raise NotImplementedError('D_type as %d not currentlyu supported!' % self.D_type)
        
        if self.dimension == 2:
            self.norm_V = self.Vlst["Vx"] ** 2 + self.Vlst["Vy"] ** 2
        elif self.dimension == 3:
            self.norm_V = self.Vlst["Vx"] ** 2 + self.Vlst["Vy"] ** 2 + self.Vlst["Vz"] ** 2
        else:
            raise NotImplementedError('Dimension == %d not supported!' % self.dimension)

    def get_F(self):
        '''
        F: an intermediate value := (\partial [V_axis] / \partial [axis])  -- (n_batch, slc, row, col)
        '''
        if self.axis == "x":
            return 0.5 * self.dXc(self.norm_V)
        elif self.axis == "y":
            return 0.5 * self.dYc(self.norm_V)
        elif self.axis == "z":
            return 0.5 * self.dZc(self.norm_V)
        else:
            raise NotImplementedError('Dimension == %d not supported!' % self.dimension)

    def get_G(self):
        '''
        G: an intermediate value := \laplacian  (D V)
        ''' 
        if self.D_type == "constant" or self.D_type == "scalar":
            return self.get_G_scalarD()
        else:
            raise NotImplementedError("Not supported PD_D_type!")
    
    # NOTE: Archived #
    '''def get_G_constantD(self, V_axis): # Assume V as curl-free (Archived)
        D = self.Dlst["D"]
        if self.axis == "x":
            return D * self.ddXc(V_axis)
        elif self.axis == "y":
            return D * self.ddYc(V_axis)
        elif self.axis == "z":
            return D * self.ddZc(V_axis)
        else:
            raise NotImplementedError('Dimension == %d not supported!' % self.dimension)'''
    
    def get_G_scalarD(self):
        if self.axis == "x":
            return self.dXb(self.div_DV)
        elif self.axis == "y":
            return self.dYb(self.div_DV)
        elif self.axis == "z":
            return self.dZb(self.div_DV)
        else:
            raise NotImplementedError('Dimension == %d not supported!' % self.dimension)

    @property
    def set_BC(self):
    # NOTE For boundary condition of mass concentration #
        '''X: (n_batch, spatial_shape)'''
        if self.BC == 'neumann' or self.BC == 'cauchy':
            if self.dimension == 1:
                return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        elif self.BC == 'dirichlet_neumann' or self.BC == 'source_neumann':
            ctrl_wdth = self.args.dirichlet_width if self.BC == 'dirichlet_neumann' else self.args.source_width
            if self.dimension == 1:
                self.dirichlet_BC = torch.nn.ReplicationPad1d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                self.dirichlet_BC = torch.nn.ReplicationPad2d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                self.dirichlet_BC = torch.nn.ReplicationPad3d(ctrl_wdth)
                return lambda X: self.neumann_dirichlet_BCBC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        else:
            return lambda X: X 
        
    def forward(self, t, batch_V_axis):
        '''
        t: (batch_size,)
        batch_V_axis: (n_batch, slc, row, col)
        ''' 
        #batch_size = batch_P.size(0)
        #batch_V_axis = self.set_BC(batch_V_axis)
        F = self.get_F() # Get intermediate value F #
        G = self.get_G() # Get intermediate value G #
        return - F - G
        
    ################# Utilities #################
    def dXb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 0] 
    def dXc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 0] 
    def dXf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 0] 
    def dYb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 1] 
    def dYc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 1] 
    def dYf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 1] 
    def dZb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 2] 
    def dZc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 2] 
    def dZf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 2] 
    def ddXc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 0], 
        batched = True, delta_lst = self.data_spacing)[..., 0]
    def ddYc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 1], 
        batched = True, delta_lst = self.data_spacing)[..., 1]
    def ddZc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 2], 
        batched = True, delta_lst = self.data_spacing)[..., 2]


# NOTE: archived #
class FlowPPartial(nn.Module):
    '''
    PDE: * Currently only support: D as scalar fields *
    * V as curl-free fields constructed via \nabla p (where p is a scalar field) *
    \delta V / \delta t + 1/2 \nabla || V ||^2 = - \nabla (D V) == - D \laplacian V = - D \div (\nabla V) == - D \nabla \cdot (\nabla V)
    (V is constructed to be curl-free := \nabla p)  ==> 
    \delta p / \delta t + 1/2 || \nabla p ||^2 = - D \laplacian p == - D \div (\nabla P) == - D \nabla \cdot (\nabla p)
    '''
    def __init__(self, args, data_spacing, device):
        super(FlowPPartial, self).__init__()
        self.dimension = len(data_spacing)
        assert self.dimension == 2 or self.dimension == 3
        self.BC = args.BC
        self.device = device
        self.data_spacing = data_spacing
        self.D = None # Only for initialization, to be updated during training # 

        if self.dimension == 1:
            self.neumann_BC = torch.nn.ReplicationPad1d(1)
        elif self.dimension == 2:
            self.neumann_BC = torch.nn.ReplicationPad2d(1)
        elif self.dimension == 3:
            self.neumann_BC = torch.nn.ReplicationPad3d(1)
        else:
            raise ValueError('Unsupported dimension: %d' % self.dimension)
    
    def get_Vlst(self, P, save_path = None):
        '''
        V := \nabla P
        P: (n_batch, (s,) r, c)
        '''
        V = gradient_c(P, batched = True, delta_lst = self.data_spacing) 
        Vx, Vy = V[..., 0], V[..., 1]
        if self.dimension == 2:
            V = torch.stack([Vx, Vy]) # (2, n_batch=1, r, c)
            Vlst = {'Vx': Vx, 'Vy': Vy}
        elif self.dimension == 3:
            Vz = V[..., 2]
            V = torch.stack([Vx, Vy, Vz]) # (3, n_batch=1, s, r, c)
            Vlst = {'Vx': Vx, 'Vy': Vy, 'Vz': Vz}
        else:
            print('Dimension == %d not supported!' % self.dimension) 
        if save_path is not None: 
            V_nda = V[:, 0]
            if self.dimension == 2: 
                V_nda = V_nda.cpu().numpy() # (2, r, c)
                nda2img(V_nda, save_path = save_path)
            elif self.dimension == 3:
                V_nda = V_nda.permute(1, 2, 3, 0).cpu().numpy() # (s, r, c, 3)
                nda2img(V_nda, save_path = save_path)  
        return Vlst

    def get_F(self, P):
        '''
        F: an intermediate value := (\partial P / \partial x)^2 + (\partial P / \partial y)^2 + (\partial P / \partial z)^2 -- (n_batch, slc, row, col)
        '''
        F_1 = self.dXc(P) ** 2
        F_2 = self.dYc(P) ** 2
        if self.dimension == 2:
            F = F_1 + F_2
        elif self.dimension == 3:
            F_3 = self.dZc(P) ** 2
            F = F_1 + F_2 + F_3
        else:
            raise ValueError('Dimension == %d not supported!' % self.dimension)
        return F

    def get_G(self, P):
        '''
        G: an intermediate value := \laplacian P == \div (\nabla P)
        ''' 
        P_xx = self.ddXc(P)
        P_yy = self.ddYc(P)
        if self.dimension == 2:
            laplacian_P = P_xx + P_yy
        elif self.dimension == 3:
            P_zz = self.ddZc(P)
            laplacian_P = P_xx + P_yy + P_zz
        else:
            raise ValueError('Dimension == %d not supported!' % self.dimension)
        return laplacian_P

    @property
    def set_BC(self):
    # NOTE For boundary condition of mass concentration #
        '''X: (n_batch, spatial_shape)'''
        if self.BC == 'neumann' or self.BC == 'cauchy':
            if self.dimension == 1:
                return lambda X: self.neumann_BC(X[:, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                return lambda X: self.neumann_BC(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        elif self.BC == 'dirichlet_neumann' or self.BC == 'source_neumann':
            ctrl_wdth = self.args.dirichlet_width if self.BC == 'dirichlet_neumann' else self.args.source_width
            if self.dimension == 1:
                self.dirichlet_BC = torch.nn.ReplicationPad1d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 2:
                self.dirichlet_BC = torch.nn.ReplicationPad2d(ctrl_wdth)
                return lambda X: self.dirichlet_BC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            elif self.dimension == 3:
                self.dirichlet_BC = torch.nn.ReplicationPad3d(ctrl_wdth)
                return lambda X: self.neumann_dirichlet_BCBC(X[:, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth, ctrl_wdth : -ctrl_wdth].unsqueeze(dim=1))[:,0]
            else:
                raise NotImplementedError('Unsupported B.C.!')
        else:
            return lambda X: X 
        
    def forward(self, t, batch_P):
        '''
        t: (batch_size,)
        batch_P: potential for curl-free V := \nabla P -- (n_batch, slc, row, col)
        ''' 
        #batch_size = batch_P.size(0)
        batch_P = self.set_BC(batch_P)
        F = self.get_F(batch_P) # Get intermediate value F #
        G = self.get_G(batch_P) # Get intermediate value G #
        return - 0.5 * F - self.D * G
        
    ################# Utilities #################
    def dXc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 0] 
    def dYc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 1] 
    def dZc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 2] 
    def ddXc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 0], 
        batched = True, delta_lst = self.data_spacing)[..., 0]
    def ddYc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 1], 
        batched = True, delta_lst = self.data_spacing)[..., 1]
    def ddZc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 2], 
        batched = True, delta_lst = self.data_spacing)[..., 2]



class AdvDiffPartial(nn.Module):
    def __init__(self, data_spacing, device):
        super(AdvDiffPartial, self).__init__()
        self.dimension = len(data_spacing)  # (slc, row, col)
        self.device = device
        self.data_spacing = data_spacing

    @property
    def Grad_Ds(self):
        return {
            'constant': self.Grad_constantD,
            'scalar': self.Grad_scalarD,
            'diag': self.Grad_diagD,
            'full': self.Grad_fullD,
            'full_dual': self.Grad_fullD,
            'full_spectral':self.Grad_fullD,
            'full_cholesky': self.Grad_fullD,
            'full_symmetric': self.Grad_fullD
        }
    @property
    def Grad_Vs(self):
        return {
            'constant': self.Grad_constantV,
            'scalar': self.Grad_scalarV,
            'vector': self.Grad_vectorV, # For general V w/o div-free TODO self.Grad_vectorV
            'vector_div_free': self.Grad_div_free_vectorV,
            'vector_div_free_clebsch': self.Grad_div_free_vectorV,
            'vector_div_free_stream': self.Grad_div_free_vectorV,
            'vector_div_free_stream_gauge': self.Grad_div_free_vectorV, 
        }

    def Grad_constantD(self, C, Dlst):
        if self.dimension == 1:
            return Dlst['D'] * (self.ddXc(C))
        elif self.dimension == 2:
            return Dlst['D'] * (self.ddXc(C) + self.ddYc(C))
        elif self.dimension == 3:
            return Dlst['D'] * (self.ddXc(C) + self.ddYc(C) + self.ddZc(C))

    def Grad_constant_tensorD(self, C, Dlst):
        if self.dimension == 1:
            raise NotImplementedError
        elif self.dimension == 2:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                Dlst['Dxy'] * self.dXb(dC_f[..., 1]) + Dlst['Dxy'] * self.dYb(dC_f[..., 0]) +\
                        Dlst['Dyy'] * self.dYb(dC_f[..., 1])  
        elif self.dimension == 3:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return Dlst['Dxx'] * self.dXb(dC_f[..., 0]) + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) + Dlst['Dzz'] * self.dZb(dC_f[..., 2]) + \
                            Dlst['Dxy'] * (self.dXb(dC_f[..., 1]) + self.dYb(dC_f[..., 0])) + \
                                Dlst['Dyz'] * (self.dYb(dC_f[..., 2]) + self.dZb(dC_f[..., 1])) + \
                                    Dlst['Dxz'] * (self.dZb(dC_f[..., 0]) + self.dXb(dC_f[..., 2]))
        
    def Grad_scalarD(self, C, Dlst): # batch_C: (batch_size, (slc), row, col)
        # Expanded version: \nabla (D \nabla C) => \nabla D \cdot \nabla C (part (a)) + D \Delta C (part (b)) # 
        # NOTE: Work better than Central Differences !!! # 
        # Nested Forward-Backward Difference Scheme in part (b)#
        if self.dimension == 1:
            dC = gradient_c(C, batched = True, delta_lst = self.data_spacing)
            return gradient_c(Dlst['D'], batched = True, delta_lst = self.data_spacing) * dC + \
                Dlst['D'] * gradient_c(dC, batched = True, delta_lst = self.data_spacing)
        else: # (dimension = 2 or 3)
            dC_c = gradient_c(C, batched = True, delta_lst = self.data_spacing)
            dC_f = gradient_f(C, batched = True, delta_lst = self.data_spacing)
            dD_c = gradient_c(Dlst['D'], batched = True, delta_lst = self.data_spacing)
            out = (dD_c * dC_c).sum(-1)
            for dim in range(dC_f.size(-1)):
                out += Dlst['D'] * gradient_b(dC_f[..., dim], batched = True, delta_lst = self.data_spacing)[..., dim]
            return out

    def Grad_diagD(self, C, Dlst):
        # Expanded version #
        if self.dimension == 1:
            raise NotImplementedError('diag_D is not supported for 1D version of diffusivity')
        elif self.dimension == 2:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return self.dXc(Dlst['Dxx']) * dC_c[..., 0] + Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                self.dYc(Dlst['Dyy']) * dC_c[..., 1] + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) 
        elif self.dimension == 3:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return self.dXc(Dlst['Dxx']) * dC_c[..., 0] + Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                self.dYc(Dlst['Dyy']) * dC_c[..., 1] + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) +\
                    self.dZc(Dlst['Dzz']) * dC_c[..., 2] + Dlst['Dzz'] * self.dZb(dC_f[..., 2])

    def Grad_fullD(self, C, Dlst):
        # Expanded version #
        '''https://github.com/uncbiag/PIANOinD/blob/master/Doc/PIANOinD.pdf'''
        if self.dimension == 1:
            raise NotImplementedError('full_D is not supported for 1D version of diffusivity')
        elif self.dimension == 2:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return self.dXc(Dlst['Dxx']) * dC_c[..., 0] + Dlst['Dxx'] * self.dXb(dC_f[..., 0]) +\
                self.dXc(Dlst['Dxy']) * dC_c[..., 1] + Dlst['Dxy'] * self.dXb(dC_f[..., 1]) +\
                    self.dYc(Dlst['Dxy']) * dC_c[..., 0] + Dlst['Dxy'] * self.dYb(dC_f[..., 0]) +\
                        self.dYc(Dlst['Dyy']) * dC_c[..., 1] + Dlst['Dyy'] * self.dYb(dC_f[..., 1])  
        elif self.dimension == 3:
            dC_c = self.dc(C)
            dC_f = self.df(C)
            return (self.dXc(Dlst['Dxx']) + self.dYc(Dlst['Dxy']) + self.dZc(Dlst['Dxz'])) * dC_c[..., 0] + \
                (self.dXc(Dlst['Dxy']) + self.dYc(Dlst['Dyy']) + self.dZc(Dlst['Dyz'])) * dC_c[..., 1] + \
                    (self.dXc(Dlst['Dxz']) + self.dYc(Dlst['Dyz']) + self.dZc(Dlst['Dzz'])) * dC_c[..., 2] + \
                        Dlst['Dxx'] * self.dXb(dC_f[..., 0]) + Dlst['Dyy'] * self.dYb(dC_f[..., 1]) + Dlst['Dzz'] * self.dZb(dC_f[..., 2]) + \
                            Dlst['Dxy'] * (self.dXb(dC_f[..., 1]) + self.dYb(dC_f[..., 0])) + \
                                Dlst['Dyz'] * (self.dYb(dC_f[..., 2]) + self.dZb(dC_f[..., 1])) + \
                                    Dlst['Dxz'] * (self.dZb(dC_f[..., 0]) + self.dXb(dC_f[..., 2]))

    def Grad_constantV(self, C, Vlst):
        if len(Vlst['V'].size()) == 1:
            if self.dimension == 1:
                return - Vlst['V'] * self.dXb(C) if Vlst['V'] > 0 else - Vlst['V'] * self.dXf(C)
            elif self.dimension == 2:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C)) if Vlst['V'] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C))
            elif self.dimension == 3:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C) + self.dZb(C)) if Vlst['V'] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C) + self.dZf(C))
        else:
            if self.dimension == 1:
                return - Vlst['V'] * self.dXb(C) if Vlst['V'][0, 0] > 0 else - Vlst['V'] * self.dXf(C)
            elif self.dimension == 2:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C)) if Vlst['V'][0, 0, 0] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C))
            elif self.dimension == 3:
                return - Vlst['V'] * (self.dXb(C) + self.dYb(C) + self.dZb(C)) if Vlst['V'][0, 0, 0, 0] > 0 else - Vlst['V'] * (self.dXf(C) + self.dYf(C) + self.dZf(C))
    
    def Grad_constant_vectorV(self, C, Vlst):
        if self.dimension == 1:
            raise NotImplementedError
        elif self.dimension == 2:
            out_x = - Vlst['Vx'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vx'][0, 0, 0] > 0 else - Vlst['Vx'] * (self.dXf(C) + self.dYf(C))
            out_y = - Vlst['Vy'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vy'][0, 0, 0] > 0 else - Vlst['Vy'] * (self.dXf(C) + self.dYf(C))
            return out_x + out_y
        elif self.dimension == 3:
            out_x = - Vlst['Vx'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vx'][0, 0, 0] > 0 else - Vlst['Vx'] * (self.dXf(C) + self.dYf(C))
            out_y = - Vlst['Vy'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vy'][0, 0, 0] > 0 else - Vlst['Vy'] * (self.dXf(C) + self.dYf(C))
            out_z = - Vlst['Vz'] * (self.dXb(C) + self.dYb(C)) if Vlst['Vz'][0, 0, 0] > 0 else - Vlst['Vz'] * (self.dXf(C) + self.dYf(C))
            return out_x + out_y + out_z
    
    def Grad_SimscalarV(self, C, Vlst):
        V = Vlst['V']
        Upwind_C = Upwind(C, self.data_spacing)
        if self.dimension == 1:
            C_x = Upwind_C.dX(V)
            return - V * C_x
        if self.dimension == 2:
            C_x, C_y = Upwind_C.dX(V), Upwind_C.dY(V)
            return - V * (C_x + C_y)
        if self.dimension == 3:
            C_x, C_y, C_z = Upwind_C.dX(V), Upwind_C.dY(V), Upwind_C.dZ(V)
            return - V * (C_x + C_y + C_z)

    def Grad_scalarV(self, C, Vlst):
        V = Vlst['V']
        Upwind_C = Upwind(C, self.data_spacing)
        dV = gradient_c(V, batched = True, delta_lst = self.data_spacing)
        if self.dimension == 1:
            C_x = Upwind_C.dX(V)
            return - V * C_x - C * dV
        elif self.dimension == 2:
            C_x, C_y = Upwind_C.dX(V), Upwind_C.dY(V)
            return - V * (C_x + C_y) - C * dV.sum(-1)
        elif self.dimension == 3:
            C_x, C_y, C_z = Upwind_C.dX(V), Upwind_C.dY(V), Upwind_C.dZ(V)
            return - V * (C_x + C_y + C_z) - C * dV.sum(-1)

    def Grad_div_free_vectorV(self, C, Vlst):
        ''' For divergence-free-by-definition velocity'''
        if self.dimension == 1:
            raise NotImplementedError('clebschVector is not supported for 1D version of velocity')
        Upwind_C = Upwind(C, self.data_spacing) 
        C_x, C_y = Upwind_C.dX(Vlst['Vx']), Upwind_C.dY(Vlst['Vy'])
        if self.dimension == 2:
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y)
        elif self.dimension == 3:
            C_z = Upwind_C.dZ(Vlst['Vz'])
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y + Vlst['Vz'] * C_z)
            
    def Grad_vectorV(self, C, Vlst):
        ''' For general velocity'''
        if self.dimension == 1:
            raise NotImplementedError('vector is not supported for 1D version of velocity')
        Upwind_C = Upwind(C, self.data_spacing) 
        C_x, C_y = Upwind_C.dX(Vlst['Vx']), Upwind_C.dY(Vlst['Vy'])
        Vx_x = self.dXc(Vlst['Vx'])
        Vy_y = self.dYc(Vlst['Vy']) 
        if self.dimension == 2:
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y) - C * (Vx_x + Vy_y)
        if self.dimension == 3:
            C_z = Upwind_C.dZ(Vlst['Vz'])
            Vz_z = self.dZc(Vlst['Vz'])
            return - (Vlst['Vx'] * C_x + Vlst['Vy'] * C_y + Vlst['Vz'] * C_z) - C * (Vx_x + Vy_y + Vz_z)
    
    ################# Utilities #################
    def db(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)
    def df(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)
    def dc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)
    def dXb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 0]
    def dXf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 0]
    def dXc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 0]
    def dYb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 1]
    def dYf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 1]
    def dYc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 1]
    def dZb(self, X):
        return gradient_b(X, batched = True, delta_lst = self.data_spacing)[..., 2]
    def dZf(self, X):
        return gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 2]
    def dZc(self, X):
        return gradient_c(X, batched = True, delta_lst = self.data_spacing)[..., 2]
    def ddXc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 0], 
        batched = True, delta_lst = self.data_spacing)[..., 0]
    def ddYc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 1], 
        batched = True, delta_lst = self.data_spacing)[..., 1]
    def ddZc(self, X):
        return gradient_b(gradient_f(X, batched = True, delta_lst = self.data_spacing)[..., 2], 
        batched = True, delta_lst = self.data_spacing)[..., 2]



