import os
import abc
import torch
import torch.nn as nn

#######################################################
# Utils ###############################################
#######################################################

from learning.ODE.FD import FD_torch
from learning.Modules.unet import UNet3D
from learning.Modules.resnet import generate_model
from learning.utils import SetBC, gradient_c, Upwind, nda_save_img



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

def divfree_V(Va, Vb, device):
    dVa = gradient_c(Va, device = device)
    dVb = gradient_c(Vb, device = device)
    Va_x, Va_y, Va_z = dVa[..., 0], dVa[..., 1], dVa[..., 2]
    Vb_x, Vb_y, Vb_z = dVb[..., 0], dVb[..., 1], dVb[..., 2]
    Vx = Va_y * Vb_z - Va_z * Vb_y
    Vy = Va_z * Vb_x - Va_x * Vb_z
    Vz = Va_x * Vb_y - Va_y * Vb_x
    return Vx, Vy, Vz


def HHD_V(Phi_a, Phi_b, Phi_c, H, device):
	'''
	input: (batch,s, r, c)
	'''
	dDa = gradient_c(Phi_a, device)
	dDb = gradient_c(Phi_b, device)
	dDc = gradient_c(Phi_c, device) 
	dH  = gradient_c(H, device)
	Va_x, Va_y, Va_z = dDa[..., 0], dDa[..., 1], dDa[..., 2]
	Vb_x, Vb_y, Vb_z = dDb[..., 0], dDb[..., 1], dDb[..., 2]
	Vc_x, Vc_y, Vc_z = dDc[..., 0], dDc[..., 1], dDc[..., 2]
	Vx = Vc_y - Vb_z + dH[..., 0]
	Vy = Va_z - Vc_x + dH[..., 1]
	Vz = Vb_x - Va_y + dH[..., 2]
	return Vx, Vy, Vz


def stream_V(Phi_a, Phi_b, Phi_c, device):
	'''
	input: (batch,s, r, c)
	'''
	dDa = gradient_c(Phi_a, device)
	dDb = gradient_c(Phi_b, device)
	dDc = gradient_c(Phi_c, device) 
	Va_x, Va_y, Va_z = dDa[..., 0], dDa[..., 1], dDa[..., 2]
	Vb_x, Vb_y, Vb_z = dDb[..., 0], dDb[..., 1], dDb[..., 2]
	Vc_x, Vc_y, Vc_z = dDc[..., 0], dDc[..., 1], dDc[..., 2]
	Vx = Vc_y - Vb_z
	Vy = Va_z - Vc_x
	Vz = Vb_x - Va_y
	return Vx, Vy, Vz


class AdvDiffPDEs(nn.Module):
	'''
	Apply to 3D cases
	D, Vx, Vy, Vz: scalar field in 3D
	'''
	__metaclass__ = abc.ABCMeta
	def __init__(self, args, nT, data_dim, data_spacing, device, mask = None, contour = None):
		super(AdvDiffPDEs, self).__init__()
		self.args = args
		self.perf_pattern = args.perf_pattern
		self.data_dim = data_dim  # (slc, row, col)
		self.data_spacing = data_spacing
		self.BC = self.args.BC
		self.contour = contour
		self.device = device
		self.FDSolver = FD_torch(self.data_spacing, self.device) 

	@abc.abstractmethod
	def run(self, batch_C, param_lst, mask):
		raise NotImplementedError

	def Grad_scalarD(self, C, D):
		return self.FDSolver.dXb(D * self.FDSolver.dXf(C)) + self.FDSolver.dYb(D * self.FDSolver.dYf(C)) + self.FDSolver.dZb(D * self.FDSolver.dZf(C))
	def Grad_SimvectorV(self, C, Vx, Vy, Vz):
		Upwind_C = Upwind(self.FDSolver, C, self.device)
		C_x, C_y, C_z = Upwind_C.dX(Vx), Upwind_C.dY(Vy), Upwind_C.dZ(Vz)
		return - (Vx * C_x + Vy * C_y + Vz * C_z)
	def Grad_vectorV(self, C, Vx, Vy, Vz):
		Upwind_C = Upwind(self.FDSolver, C, self.device)
		C_x, C_y, C_z = Upwind_C.dX(Vx), Upwind_C.dY(Vy), Upwind_C.dZ(Vz)
		Vx_x = self.expand(gradient_c(Vx[0], self.device)[..., 0], self.batch_size)
		Vy_y = self.expand(gradient_c(Vy[0], self.device)[..., 1], self.batch_size)
		Vz_z = self.expand(gradient_c(Vz[0], self.device)[..., 2], self.batch_size)
		return - (Vx * C_x + Vy * C_y + Vz * C_z) - C * (Vx_x + Vy_y + Vz_z)
	def expand(self, X):
		return (X).expand(self.args.batch_size, -1, -1, -1)


class AdvPDE(AdvDiffPDEs):
	def run(self, batch_C, param_lst, mask):
		return SetBC(self.Grad_SimvectorV(batch_C, param_lst['Vx'], param_lst['Vy'], param_lst['Vz']), self.BC, self.contour) * mask

class DiffPDE(AdvDiffPDEs):
	def run(self, batch_C, param_lst, mask):
		return SetBC(self.Grad_scalarD(batch_C, param_lst['D']), self.BC, self.contour) * mask

class AdvDiffPDE(AdvDiffPDEs):
	def run(self, batch_C, param_lst, mask):
		return SetBC(self.Grad_scalarD(batch_C, param_lst['D']) + \
			self.Grad_SimvectorV(batch_C, param_lst['Vx'], param_lst['Vy'], param_lst['Vz']), self.BC, self.contour) * mask


class D_Net(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(D_Net, self).__init__()
        self.features = UNet3D(in_channels, out_channels = 1)
        self.device = device
        
    def get_params(self, perf_patch):
        n_batch = perf_patch.size(0)
        params = self.features(perf_patch)
        return {
            'D': params[:, 0],
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        D = params['D'][0].detach()
        if self.device is not 'cpu':
            D = D.cpu()
        nda_save_img(D.numpy(), save_path = os.path.join(save_dir, 'D.nii'))
        return D, V
        
    def run(self, perf_patch):
        return {
            'D': self.get_params(perf_patch)['D'],
        }

class V_Net_demo(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(V_Net_demo, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels = in_channels * 2, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 2, out_channels = in_channels * 4, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 4, out_channels = in_channels * 8, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 8, out_channels = 3, kernel_size = global_kz, stride = 1, padding = global_padding),
        )
        #self.decoder = nn.Sequential(
        #    nn.Conv3d(in_channels, out_channels = in_channels * 2, kernel_size = global_kz, stride = 1, padding = global_padding),
        #    nn.ReLU(inplace=True),
        #)
        self.device = device
        
    def get_params(self, x):
        x = self.encoder(x)
        #x = self.decoder(x)
        return {
            'Vx': x[:, 0],
            'Vy': x[:, 1],
            'Vz': x[:, 2],
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            V = V.cpu()
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        return {
            'Vx': params['Vx'],
            'Vy': params['Vy'],
            'Vz': params['Vz'],
        }

class AutoEncoder_V(nn.Module):
    def __init__(self, args, in_channels, device):
        super(AutoEncoder_V, self).__init__()
        self.in_channels = in_channels
        self.device = device

    @property
    @abc.abstractmethod
    def encoder(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def decoder(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def final_layer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self, x):
        x = self.encoder(x) # (batch, 3, s, r, c)
        x = self.decoder(x) # (batch, 3, s, r, c) 
        x = self.final_layer(x) # (batch, 3, s, r, c)

        Va, Vb = x[..., 0], x[...,  1]
        dVa = gradient_c(Va, device = self.device)
        dVb = gradient_c(Vb, device = self.device)
        Va_x, Va_y, Va_z = dVa[..., 0], dVa[..., 1], dVa[..., 2]
        Vb_x, Vb_y, Vb_z = dVb[..., 0], dVb[..., 1], dVb[..., 2]
        Vx = Va_y * Vb_z - Va_z * Vb_y
        Vy = Va_z * Vb_x - Va_x * Vb_z
        Vz = Va_x * Vb_y - Va_y * Vb_x
        return {
            'Vx': Vx,
            'Vy': Vy,
            'Vz': Vz,
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            V = V.cpu()
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        return {
            'Vx': params['Vx'],
            'Vy': params['Vy'],
            'Vz': params['Vz'],
        }





class V_Net_sim(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(V_Net_sim, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels = in_channels * 2, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 2, out_channels = in_channels * 4, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 4, out_channels = in_channels * 8, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 8, out_channels = in_channels * 16, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 16, out_channels = in_channels * 8, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 8, out_channels = in_channels * 4, kernel_size = global_kz, stride = 1, padding = global_padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels * 4, out_channels = in_channels * 2, kernel_size = global_kz, stride = 1, padding = global_padding),
            )
        '''
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 2, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 8, in_channels * 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 16, in_channels * 32),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 32, in_channels * 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 16, in_channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 8, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels * 2),
        )
        '''
        self.activation = nn.Linear(in_channels * 2, 2)
        self.device = device
        
    def get_params(self, x):
        x = self.encoder(x) # (batch, 3, s, r, c)
        #x = self.decoder(x.permute(0, 2, 3, 4, 1)) # (batch, 3, s, r, c) -> (batch, s, r, c, 3)
        x = self.activation(x.permute(0, 2, 3, 4, 1))
        #x = self.activation(x)
        Va, Vb = x[..., 0], x[...,  1]
        dVa = gradient_c(Va, device = self.device)
        dVb = gradient_c(Vb, device = self.device)
        Va_x, Va_y, Va_z = dVa[..., 0], dVa[..., 1], dVa[..., 2]
        Vb_x, Vb_y, Vb_z = dVb[..., 0], dVb[..., 1], dVb[..., 2]
        Vx = Va_y * Vb_z - Va_z * Vb_y
        Vy = Va_z * Vb_x - Va_x * Vb_z
        Vz = Va_x * Vb_y - Va_y * Vb_x
        return {
            'Vx': Vx,
            'Vy': Vy,
            'Vz': Vz,
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            V = V.cpu()
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        return {
            'Vx': params['Vx'],
            'Vy': params['Vy'],
            'Vz': params['Vz'],
        }

class V_Net_linear(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(V_Net_linear, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels = in_channels * 2, kernel_size = global_kz, stride = 1, padding = global_padding),
            )
        self.decoder = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 4),
            nn.Linear(in_channels * 4, in_channels * 8),
            nn.Linear(in_channels * 8, 3),
        )
        self.device = device
        
    def get_params(self, x):
        x = self.encoder(x) # (batch, 3, s, r, c)
        x = self.decoder(x.permute(0, 2, 3, 4, 1)) # (batch, 3, s, r, c) -> (batch, s, r, c, 3)

        return {
            'Vx': x[..., 0],
            'Vy': x[..., 1],
            'Vz': x[..., 2],
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            V = V.cpu()
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        return {
            'Vx': params['Vx'],
            'Vy': params['Vy'],
            'Vz': params['Vz'],
        }


class V_Net_Stream(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(V_Net_Stream, self).__init__()
        #self.features = UNet3D(in_channels, out_channels = 3)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels = in_channels * 2, kernel_size = global_kz, stride = 1, padding = global_padding),
            )
        self.decoder = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 4),
            nn.Linear(in_channels * 4, in_channels * 8),
            nn.Linear(in_channels * 8, 3),
        )
        self.device = device
        
    def get_params(self, perf_patch):
        n_batch = perf_patch.size(0)
        #params = self.features(perf_patch)
        params = self.encoder(perf_patch)
        params = self.encoder(params)
        Vx, Vy, Vz = stream_V(params[:, 0], params[:, 1], params[:, 2], self.device)
        return {
            'Vx': Vx,
            'Vy': Vy,
            'Vz': Vz,
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            V = V.cpu()
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        return {
            'Vx': params['Vx'],
            'Vy': params['Vy'],
            'Vz': params['Vz'],
        }




class V_Net_divfree(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(V_Net_divfree, self).__init__()
        #self.features = UNet3D(in_channels, out_channels = 2)
        self.features = generate_model(10, no_max_pool = True, n_input_channels = in_channels, n_classes = 2)
        self.device = device
        
    def get_params(self, perf_patch):
        x = self.features(perf_patch) # (batch, 3, s, r, c) -> (batch, s, r, c, 3)
        Va, Vb = x[..., 0], x[..., 1]
        dVa = gradient_c(Va, device = self.device)
        dVb = gradient_c(Vb, device = self.device)
        Va_x, Va_y, Va_z = dVa[..., 0], dVa[..., 1], dVa[..., 2]
        Vb_x, Vb_y, Vb_z = dVb[..., 0], dVb[..., 1], dVb[..., 2]
        Vx = Va_y * Vb_z - Va_z * Vb_y
        Vy = Va_z * Vb_x - Va_x * Vb_z
        Vz = Va_x * Vb_y - Va_y * Vb_x
        return {
            'Vx': Vx,
            'Vy': Vy,
            'Vz': Vz,
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            V = V.cpu()
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        return {
            'Vx': params['Vx'],
            'Vy': params['Vy'],
            'Vz': params['Vz'],
        }

class V_Net_HHD(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(V_Net_HHD, self).__init__()
        self.features = UNet3D(in_channels, out_channels = 4)
        self.device = device
        
    def get_params(self, perf_patch):
        n_batch = perf_patch.size(0)
        params = self.features(perf_patch)
        return {
            'Phi_a': params[:, 0],
            'Phi_b': params[:, 1],
            'Phi_c': params[:, 2],
            'H': params[:, 3],
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            V = V.cpu()
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        Vx, Vy, Vz = HHD_V(params['Phi_a'], params['Phi_b'], params['Phi_c'], params['H'], self.device)
        return {
            'Vx': Vx,
            'Vy': Vy,
            'Vz': Vz,
        }


class DV_Net(nn.Module):

    def __init__(self, args, in_channels, device,):
        super(DV_Net, self).__init__()
        self.features = UNet3D(in_channels, out_channels = 5)
        self.device = device
        
    def get_params(self, perf_patch):
        n_batch = perf_patch.size(0)
        params = self.features(perf_patch)
        return {
            'D': params[:, 0],
            'Phi_a': params[:, 1],
            'Phi_b': params[:, 2],
            'Phi_c': params[:, 3],
            'H': params[:, 4],
        }
        
    def save_DV(self, perf_patch, save_dir):
        params = self.run(perf_patch)
        D = params['D'][0].detach()
        Vx, Vy, Vz = params['Vx'][0].detach(), params['Vy'][0].detach(), params['Vz'][0].detach()
        V = torch.stack([Vx, Vy, Vz]).permute(1, 2, 3, 0)
        if self.device is not 'cpu':
            D, V = D.cpu(), V.cpu()
        nda_save_img(D.numpy(), save_path = os.path.join(save_dir, 'D.nii'))
        nda_save_img(V.numpy(), save_path = os.path.join(save_dir, 'V.nii'))
        return D, V
        
    def run(self, perf_patch):
        params = self.get_params(perf_patch)
        Vx, Vy, Vz = HHD_V(params['Phi_a'], params['Phi_b'], params['Phi_c'], params['H'], self.device)
        return {
            'D': params['D'],
            'Vx': Vx,
            'Vy': Vy,
            'Vz': Vz,
        }

class ConservLoss(nn.Module):
    '''
    Advection-Diffusion PDE Conservation Loss: 
    ConsLoss = dC/dt - (\\div(D * \\nabla C) - V * \\nabla C)
            \\approx (C ^ {k+1} - C ^ k) - 0.5 * {\
            (\\div(D * \\nabla C ^ {k}) - V * \\nabla C ^ {k}) + \
            (\\div(D * \\nabla C ^ {k + 1}) - V * \\nabla C ^ {k + 1})}
    '''          
    def __init__(self, args, nT, data_dim, data_spacing, device, mask = None, contour = None):
        super(ConservLoss, self).__init__()
        if 'adv' not in args.perf_pattern:
            self.PDEFunc = DiffPDE(args, nT, data_dim, data_spacing, device, mask = None, contour = contour)
        elif 'diff' not in args.perf_pattern:
            self.PDEFunc = AdvPDE(args, nT, data_dim, data_spacing, device, mask = None, contour = contour)
        else:
            self.PDEFunc = AdvDiffPDE(args, nT, data_dim, data_spacing, device, mask = None, contour = contour)
        self.args = args 
        self.device = device


    def get_dC(self, batch_Ct, param_lst, mask):
        batch_C0, batch_C1 = batch_Ct[..., 0], batch_Ct[..., 1]
        dC0, dC1 = self.PDEFunc.run(batch_C0, param_lst, mask), self.PDEFunc.run(batch_C1, param_lst, mask)
        return 0.5 * (dC0 + dC1) * self.args.dt # (batch, s, r, c)

    def predict(self, batch_Ct, param_lst, mask):
        dC = self.get_dC(batch_Ct, param_lst, mask)
        return batch_Ct[..., 0] + dC

    def forward(self, batch_Ct, param_lst, mask):
        '''
        batch_Ct: (batch, s, r, c, 2)
        '''
        batch_C0, batch_C1 = batch_Ct[..., 0], batch_Ct[..., 1]
        dC = self.get_dC(batch_Ct, param_lst, mask)
        return (abs((batch_C1 - batch_C0) - dC) ** 2).mean()

