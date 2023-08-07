import os, sys, argparse, gc, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import SimpleITK as sitk
#from shutil import copyfile, rmtree


from Datasets.demo_IXI import get_movie
from Preprocess.IXI.itk_utils import anisotropic_smoothing
from utils import make_dir, nda2img, divergence3D, divergence3D_numpy, gradient_c, gradient_f, stream_3D, img2nda


#%% Basic settings
parser = argparse.ArgumentParser('Divergence-free Velocity Fields Generation')
parser.add_argument('--velocity_magnitude', type = float, default = 1.)

# For vessel probability #
parser.add_argument('--n_iter', type = int, default = 10000)
parser.add_argument('--save_freq', type = int, default = 500)
parser.add_argument('--gradient_weight', type = float, default = 2)
#parser.add_argument('--div_free_weight', type = float, default = 1e+5)
parser.add_argument('--lr', type = float, default = 5e-3)

args_fit = parser.parse_args()  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DivFreeFitter(nn.Module):

    def __init__(self, save_fld, mask, data_dim, data_spacing, device):
        super(DivFreeFitter, self).__init__()
        self.mask = mask # torch tensor on device #
        self.device = device
        self.save_fld = save_fld
        self.data_dim = data_dim
        self.data_spacing = data_spacing
        self.set_neumann =  torch.nn.ReplicationPad3d(1)

        self.register_parameter('phi_a', nn.Parameter((torch.randn(self.data_dim, device = self.device, requires_grad = True) - 0.5) * 2 * self.mask))
        self.register_parameter('phi_b', nn.Parameter((torch.randn(self.data_dim, device = self.device, requires_grad = True) - 0.5) * 2 * self.mask))
        self.register_parameter('phi_c', nn.Parameter((torch.randn(self.data_dim, device = self.device, requires_grad = True) - 0.5) * 2 * self.mask))

    def get_V(self):
        # NOTE: NO MASK SHOULD BE ADDED ON PHI !!! <-- Cause boundary sharpening issues when computing gradient with masked Phi
        vx, vy, vz = stream_3D(self.phi_a, self.phi_b, self.phi_c, batched = False, delta_lst = self.data_spacing)
        v = torch.stack([vx, vy, vz], dim = -1) * self.mask[..., None]
        v = self.neumann_bc(v, isVector = True)
        return v

    def neumann_bc(self, x, isVector = True):
        # x: (shape, (vector_dim)) #
        if isVector:
            x = x.permute(3, 0, 1, 2).unsqueeze(dim = 0) # (batch = 1, vector_dim, s, r, c)
            return (self.set_neumann(x[:, :, 1:-1, 1:-1, 1:-1])[0]).permute(1, 2, 3, 0)
        else:
            x = x.unsqueeze(0).unsqueeze(0) # (batch = 1, vector_dim = 1, s, r, c)
            return self.set_neumann(x[:, :, 1:-1, 1:-1, 1:-1])[0, 0]
    
    def save(self, save_fld, origin, spacing, direction):
        # NOTE: NO MASK SHOULD BE ADDED ON PHI when saving !!! <-- Boundary sharpening issues when re-computing gradient with masked Phi
        phi = (torch.stack([self.phi_a, self.phi_b, self.phi_c], dim = -1)).detach()
        v = (self.get_V()).detach()

        if self.device is not 'cpu':
            v, phi = v.cpu(), phi.cpu()
        v, phi = v.numpy(), phi.numpy()
        nda2img(phi, origin, spacing, direction, os.path.join(save_fld, 'Phi.mha'))

        nda2img(v, origin, spacing, direction, os.path.join(save_fld, 'V.mha'))
        nda2img(abs(v), origin, spacing, direction, os.path.join(save_fld, 'Abs_V.mha'))
        nda2img(np.linalg.norm(v, axis = -1), origin, spacing, direction, os.path.join(save_fld, 'Norm_V.mha'))
        return os.path.join(save_fld, 'V.mha')

class AbsLoss(nn.Module):
    def __init__(self, mask, weight):
        super(AbsLoss, self).__init__()
        self.mask = mask
        self.weight = weight
    
    def forward(self, v, v_fit):
        abs_loss = (abs(v) - abs(v_fit)) ** 2 * self.mask[..., None]
        return  abs_loss.mean() * self.weight

class DivFreeLoss(nn.Module):
    def __init__(self, mask, weight, data_spacing):
        super(DivFreeLoss, self).__init__()
        self.mask = mask
        self.weight = weight
        self.data_spacing = data_spacing
    
    def forward(self, v):
        divergence = divergence3D(v, vector_dim = -1, batched = False, data_spacing = self.data_spacing)
        return  (divergence ** 2 * self.mask).mean() * self.weight

class GradientLoss(nn.Module):
    def __init__(self, mask, weight, data_spacing):
        super(GradientLoss, self).__init__()
        self.mask = mask[..., None] if mask is not None else 1.
        self.weight = weight
        self.data_spacing = data_spacing
    
    def forward(self, v):
        '''grad_loss = (gradient_f(v[..., 0], batched = False, delta_lst = self.data_spacing) ** 2 + \
                    gradient_f(v[..., 1], batched = False, delta_lst = self.data_spacing) ** 2 +\
                    gradient_f(v[..., 2], batched = False, delta_lst = self.data_spacing) ** 2) # (s, r, c, 3)'''
        grad_loss = (gradient_c(v[..., 0], batched = False, delta_lst = self.data_spacing) ** 2 + \
                    gradient_c(v[..., 1], batched = False, delta_lst = self.data_spacing) ** 2 +\
                    gradient_c(v[..., 2], batched = False, delta_lst = self.data_spacing) ** 2) # (s, r, c, 3)
        return  (grad_loss * self.mask).mean() * self.weight
        


################################################################
########################  Fit one case  ########################
################################################################


def get_bin_vessel_smoothed(vessel_path):
    smoothed_vessel_path = anisotropic_smoothing(vessel_path, n_iter = 1, diffusion_time = 0.00005, \
        anisotropic_lambda = 4, enhancement_type = 3, noise_scale = 0.002, feature_scale = 1, exponent = 1)

    vessel_smoothed, origin, spacing, direction = img2nda(smoothed_vessel_path)
    mask_smoothed = np.zeros(vessel_smoothed.shape)
    mask_smoothed[vessel_smoothed != 0 ] = 1.
    nda2img(mask_smoothed, origin, spacing, direction, save_path = os.path.join(os.path.dirname(vessel_path), 'VesselMask_smoothed.mha'))
    os.remove(smoothed_vessel_path)
    return os.path.join(os.path.dirname(vessel_path), 'VesselMask_smoothed.mha')


def fit(v_path, device, n_iter, lr = 1e-3, save_freq = 500, save_fld = None, mask_path = None):

    save_fld = save_fld if save_fld else make_dir(os.path.join(os.path.dirname(v_path), 'DivFree'))
    def get_mask(nda):
        mask_nda = np.zeros(nda.shape)
        mask_nda[nda != 0] = 1.
        return mask_nda

    v_img = sitk.ReadImage(v_path)
    origin, spacing, direction = v_img.GetOrigin(), v_img.GetSpacing(), v_img.GetDirection()
    data_dim = [v_img.GetSize()[2-i] for i in range(3)]
    data_spacing = [spacing[2-i] for i in range(3)]
    
    v_nda = sitk.GetArrayFromImage(v_img)

    v = torch.from_numpy(v_nda).float().to(device)
    if mask_path:
        print('Mask path from:', mask_path)
        mask = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(mask_path))).float().to(device)
    else:
        mask = torch.from_numpy(get_mask(np.max(abs(v_nda), axis = -1))).float().to(device)

    v_loss_crit = nn.MSELoss() # TODO MSELoss smoother than L1Loss #
    v_loss_crit.to(device)
    
    fitter = DivFreeFitter(save_fld, mask, data_dim, data_spacing, device)
    fitter.to(device)

    mask = None
    gl_crit = GradientLoss(mask, args_fit.gradient_weight, data_spacing)
    gl_crit.to(device)

    optimizer = optim.Adam(fitter.parameters(), lr = lr)

    for i_iter in range(n_iter):
        optimizer.zero_grad()
        v_fit = fitter.get_V()

        v_loss = v_loss_crit(v, v_fit)
        dv_loss = gl_crit(v_fit)
        loss = v_loss + dv_loss
        loss.backward()
        optimizer.step()
        #print('Iter {:04d} |  V {:.6f}'.format(i_iter, v_loss.item()))
        #print('     {:04d} | dV {:.6f}'.format(i_iter, dv_loss.item()))

        # For checking #
        #if (i_iter + 1) % save_freq == 0:
        #    iter_fld = make_dir(os.path.join(save_fld, '%d' % (i_iter + 1)))
        #    fitter.save(iter_fld, origin, spacing, direction)#'''
    fit_v_path = fitter.save(save_fld, origin, spacing, direction)
    return fit_v_path


if __name__ == '__main__':
    
    main_fld = '/media/peirong/PR/IXI'
    processed_fld = '/media/peirong/PR5/IXI_Processed'
    
    names_file = open(os.path.join(main_fld, 'IDs.txt'), 'r')
    case_names = names_file.readlines()
    names_file.close()
    #case_names = ['IXI002-Guys-0828'] # TODO
    #case_names = case_names[:40]

    for i in range(1, len(case_names)):
        case_name = case_names[i].split('\n')[0]
        print('\nStart processing case NO.%d (of %d): %s' % (i+1, len(case_names), case_name))
        case_fld = os.path.join(processed_fld, case_name)

        # For binary V fitting: using binary of smoothed vessel mask #
        vessel_path = os.path.join(processed_fld, case_name, 'Vessel.mha')
        bin_vessel_smoothed_path = get_bin_vessel_smoothed(vessel_path)

        v_path = os.path.join(processed_fld, case_name, 'AdvectionMaps/V.mha')
        #if os.path.isdir(os.path.join(os.path.dirname(v_path), 'DivFree')):
        #    shutil.rmtree(os.path.join(os.path.dirname(v_path), 'DivFree'))
        save_fld = make_dir(os.path.join(os.path.dirname(v_path), 'DivFree'))

        # Fit div-free V #
        fit(v_path, device, args_fit.n_iter, args_fit.lr, args_fit.save_freq, save_fld, mask_path = bin_vessel_smoothed_path)

        # Get advection-diffusino movie #
        get_movie(case_fld)

    gc.collect()



# %%
