import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import abc
import torch
import numpy as np
import SimpleITK as sitk
from shutil import copyfile
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ODE.odeint import odeint
from Preprocess.contour import get_contour
from Preprocess.prepro_utils import cropping
from utils import make_dir, save_sitk, nda2img
from Postprocess.PTIMeasures import PTI, WritePTIImage
from DemoOptions.U0 import tophat_corner, tophat_center, tophat_top, gaussian
U0s = {
    'tophat_corner': tophat_corner, 
    'tophat_center': tophat_center, 
    'tophat_top': tophat_top, 
    'gaussian': gaussian
    }
from DemoOptions.PerfFlag import adv_diff, mix, adv_only, diff_only, mix, circle_adv_only, rect_adv_only
PerfFlags = {
    'adv_diff': adv_diff, 
    'adv_only': adv_only, 
    'diff_only': diff_only, 
    'mix_adv_diff': mix,
    'circle_adv_only': circle_adv_only,
    'rect_adv_only': rect_adv_only
    }   
from DemoOptions.DV import constant, gaussian, stroke
DVs = {
    'constant': constant,
    'gaussian': gaussian,
    'stroke': stroke
} 


def ReadPerfusion(CTC_path, mask_path, contour_path, arrival, peak, end, args, device):
    perf = sitk.ReadImage(CTC_path)
    origin = perf.GetOrigin()
    spacing = perf.GetSpacing()
    direction = perf.GetDirection()
    perf = sitk.GetArrayFromImage(perf) # (s, r, c, t)
    if end == -1:
        perf = perf[..., peak : ]
    else:
        perf = perf[..., peak : end]
    data_dim = perf.shape[0], perf.shape[1], perf.shape[2] 
    T = np.arange(perf.shape[-1]) * args.dt # (nT, )
    #opt_T = np.arange(perf.shape[-1] * int(args.dt / args.opt_dt)) * args.opt_dt
    T = torch.tensor(T, dtype = torch.float, device = device)
    #opt_T = torch.tensor(opt_T, dtype = torch.float, device = device)
    U0 = torch.tensor(perf[..., 0], dtype = torch.float, device = device)
    U0 = U0.expand(1, -1, -1, -1) # (1, s, r, c)
    U  = torch.tensor(perf, dtype = torch.float, device = device)
    U  = U.expand(1, -1, -1, -1, -1) # (1, s, r, c, t)
    if not os.path.isfile(mask_path):
        mask = np.ones(perf[..., 0].shape)
        non_brain = np.where(abs(perf[..., 0]) < 0.01) # TODO
        mask[non_brain] = 0
        for s in range(len(mask)):
            mask[s] = ndimage.binary_fill_holes(mask[s])
        mask_img = sitk.GetImageFromArray(mask, isVector = False)
        mask_img.SetOrigin(origin)
        mask_img.SetSpacing(spacing)
        mask_img.SetDirection(direction)
        sitk.WriteImage(mask_img, mask_path)
    else:
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    if contour_path is not None:
        if os.path.isfile(contour_path):
            contour = sitk.GetArrayFromImage(sitk.ReadImage(contour_path))
        else:
            contour = get_contour(mask_path)
            contour_img = sitk.GetImageFromArray(contour, isVector = False)
            contour_img.SetOrigin(origin)
            contour_img.SetSpacing(spacing)
            contour_img.SetDirection(direction)
            sitk.WriteImage(contour_img, contour_path)
        contour = torch.tensor(contour, dtype = torch.float, device = device)
    else:
        contour = None
    mask = torch.tensor(mask, dtype = torch.float, device = device)
    data_spacing = torch.tensor([spacing[2], spacing[1], spacing[0]], dtype = torch.float, device = device)
    return U0, T, U, mask, contour, data_dim, data_spacing, origin, spacing, direction, np.mean(perf)


class OneDGenerator(object):
    def __init__(self, args, device = 'cpu', fld = None):
        self.args = args
        self.dim = args.dim
        self.data_dim, self.data_spacing = [args.nx], [args.dx]
        self.t0, self.x0 = args.t0, args.x0
        self.t = self.t0 + np.arange(args.nT) * args.dt
        self.x = self.x0 + np.arange(args.nx) * args.dx
        self.init_value, self.U0_pattern = args.init_value, args.GT_U0_pattern.split('_')[0]
        self.velocity, self.V_pattern = args.velocity, args.GT_V_pattern
        self.diffusivity, self.D_pattern = args.diffusivity, args.GT_D_pattern
        self.device = device
        self.fld = fld
        self.D_path = os.path.join(make_dir(os.path.join(fld, '0')), 'D (U0: %s).npy' % args.GT_U0_pattern) 
        self.V_path = os.path.join(make_dir(os.path.join(fld, '0')), 'V (U0: %s).npy' % args.GT_U0_pattern) 
        
    @property
    def T(self):
        return torch.tensor(self.t, dtype = torch.float, device = self.device)
        
    @property
    def X(self):
        return torch.tensor(self.x, dtype = torch.float, device = self.device)
    
    @property
    def D(self):
        D = self.Patterns[self.D_pattern](self.X, self.velocity)
        return D
    
    @property
    def V(self):
        V = self.Patterns[self.V_pattern](self.X, self.velocity)
        print(V.mean().item())
        return V

    @property
    def U0(self):
        U0 = self.Patterns[self.U0_pattern](self.X, self.init_value)
        return U0.expand(1, -1)

    @property
    def Patterns(self):
        return {
            'constant': self.constant,
            'tophat': self.tophat,
            'gaussian': self.gaussian
        } 

    def tensor_save_nda(self, var, file_name):
        if self.device is not 'cpu':
            var = var.cpu()
        var.numpy()
        np.save(file_name, var)
        return var

    def constant(self, grid, value):
        return torch.ones(len(grid), dtype = torch.float, device = self.device) * value
        
    def gaussian(self, grid, value):
        return torch.exp(torch.tensor(- ((grid - 3) / 1) ** 2, dtype = torch.float, device = self.device)) * value

    def tophat(self, grid, value):
        type = self.args.GT_U0_pattern[-1] # L (left), R (right), C (center)
        out = torch.zeros(len(grid), dtype = torch.float, device = self.device)
        if type is 'L':
            out[int(len(self.X) / 8) : int(len(self.X) * 3 / 8)] = value
        elif type is 'R':
            out[int(len(self.X) * 5 / 8) : int(len(self.X) * 7 / 8)] = value
        elif type is 'C':
            out[int(len(self.X) * 3 / 8) : int(len(self.X) * 5 / 8)] = value 
        else:
            raise ValueError('Failed to recognize pattern type, should be L/R/C.')
        return out

    def get_SemiLag_U(self, RecordSum = False):
        if 'diff' in self.args.perf_pattern:
            self.tensor_save_nda(self.D, self.D_path)
        if 'adv' in self.args.perf_pattern:
            self.tensor_save_nda(self.V, self.V_path)
        func = SemiLagrangian1D(self.args, self.X, self.data_dim, V_ResumePath = self.V_path, device = self.device, for_GT = True)
        with torch.no_grad():
            U = func(self.U0, interp_method = 'Linear')
        U = U.permute(1, 2, 0) # (n_time, n_batch, nX) -> (n_batch, nX, n_time)
        if self.args.BC is 'Dirichlet':
            U[:, 0, :] = self.U0[:, 0]
            U[:, -1, :] = self.U0[:, -1]
        fig = plt.figure()
        ims = []
        if self.device is not 'cpu':
            U_lst = (U.permute(0, 2, 1)).cpu() # (1, nX, time) -> (1, time, nX)
        U_lst = U_lst[0].numpy()
        for it in range(U_lst.shape[0]):
            if (it + 1) % self.args.record_freq == 0:
                im = plt.plot(self.x, U_lst[it], 'g')
                ims.append(im)
        ani = animation.ArtistAnimation(fig, ims, interval = 200, blit = True, repeat_delay = 2000)
        ani.save(os.path.join(self.fld, '0/U (U0: %s).gif' % self.args.GT_U0_pattern), writer = 'imagemagick')
        if RecordSum:
            Avg = U.mean()
            for it in range(U.size(-1)):
                temp = U[0, :, it]
                print('Avg. value at time {:03d}: {:.6f}'.format(it, torch.mean(temp).item()))
            print('Domain-averaged value:  {:.6f}'.format(Avg.item()))
        return U # (n_batch, slc, row, col, n_time)

    def get_U(self, Func, RecordSum = False):
        if 'diff' in self.args.perf_pattern:
            self.tensor_save_nda(self.D, self.D_path)
        if 'adv' in self.args.perf_pattern:
            self.tensor_save_nda(self.V, self.V_path)
        PDEFunc = Func(self.args, self.D_path, self.V_path, self.data_dim, self.data_spacing, self.device, for_GT = True)
        with torch.no_grad():
            U = odeint(None, PDEFunc, self.U0, self.T, method = self.args.integ_method)
        U = U.permute(1, 2, 0) # (n_time, n_batch, nX) -> (n_batch, nX, n_time)
        if self.args.BC is 'Dirichlet':
            U[:, 0, :] = self.U0[:, 0]
            U[:, -1, :] = self.U0[:, -1]
        opt_U = []
        opt_nT = int(self.args.dt * self.args.nT / self.args.opt_dt)
        scale = int(self.args.opt_dt / self.args.dt)
        for it in range(opt_nT):
            opt_U.append(U[..., it * scale])
        opt_U.append(U[..., -1])
        opt_U = (torch.stack(opt_U)).permute(1, 2, 0) # (time, n_batch, nX) -> (n_batch, nX, n_time)
        if RecordSum:
            Avg = opt_U.mean()
            for it in range(opt_U.size(-1)):
                temp = opt_U[0, :, it]
                print('Avg. value at time {:03d}: {:.6f}'.format(it, torch.mean(temp).item()))
            print('Domain-averaged value:  {:.6f}'.format(Avg.item()))

        fig = plt.figure()
        ims = []
        if self.device is not 'cpu':
            U_lst = (U.permute(0, 2, 1)).cpu() # (1, nX, time) -> (1, time, nX)
        U_lst = U_lst[0].numpy()
        for it in range(U_lst.shape[0]):
            if (it + 1) % self.args.record_freq == 0:
                im = plt.plot(self.x, U_lst[it], 'g')
                ims.append(im)
        ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 200)
        ani.save(os.path.join(self.fld, '0/U (U0: %s).gif' % self.args.GT_U0_pattern), writer = 'imagemagick')
        return U, opt_U # (n_batch, nX, n_time)



class PIGenerator(object):
    '''
    For 3D Demo
    '''
    def __init__(self, args, D_path, V_path, U0_path, Mask_path = None, Contour_path = None, device = 'cpu', save_fld = None):
        self.args = args
        self.device = device
        self.Mask_path = Mask_path
        self.Contour_path = Contour_path
        self.perf_pattern = args.GT_perf_pattern
        self.U0_path, self.U0_magnitude = U0_path, args.U0_magnitude
        self.D_path, self.D_type, self.D_magnitude = D_path, args.GT_D_type, args.diffusion_magnitude
        self.V_path, self.V_type, self.V_magnitude = V_path, args.GT_V_type, args.velocity_magnitude

        self.main_fld = save_fld
        self.save_fld = self.get_save_fld()

        self.brain_mask_nda, [self.x0, self.y0, self.z0], [self.x1, self.y1, self.z1], [self.origin, self.spacing, self.direction] = self.get_brain_mask()
        self.brain_mask = torch.from_numpy(self.brain_mask_nda).float().to(self.device)
        self.data_spacing = [self.spacing[2-i] for i in range(3)] 
        self.register_U0()


    def get_save_fld(self):
        if self.perf_pattern is 'diff_only':
            self.setting_info = 'D-%s' % self.D_type
        elif self.perf_pattern is 'adv_only':
            self.setting_info = 'V-%s' % self.V_type
        else:
            self.setting_info = 'D-%s_V-%s' % (self.D_type, self.V_type)
        #save_fld = make_dir(os.path.join(self.main_fld, 'Movies', self.perf_pattern, self.setting_info)) if self.main_fld \
        #    else make_dir(os.path.join(os.path.dirname(self.D_path), 'Movies', self.perf_pattern, self.setting_info))
        save_fld = make_dir(os.path.join(self.main_fld, 'Movies')) if self.main_fld \
            else make_dir(os.path.join(os.path.dirname(self.D_path), 'Movies'))
        return save_fld

    def register_U0(self):
        # NOTE: For Numpy / Torch (reverse order to ITK image)
        U0_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.U0_path))
        # Re-aranging to [-1, 1] * args.velocity_magnitude
        U0_nda = U0_nda * self.U0_magnitude / np.max(abs(U0_nda))
        U0 = (torch.from_numpy(U0_nda).float()).to(self.device)[self.x0 : self.x1, self.y0 : self.y1, self.z0 : self.z1] * self.brain_mask
        self.U0 = U0.expand(1, -1, -1, -1) # (batch = 1, s, r, c) for PIANO input
        self.save(self.U0[0], 'U0')

    @property
    def T(self):
        t = self.args.t0 + np.arange(self.args.nT) * self.args.dt
        return torch.tensor(t, dtype = torch.float, device = self.device)

    def get_brain_mask(self):
        if not self.Mask_path:
            U0_img = sitk.ReadImage(self.U0_path)
            spacing, direction = U0_img.GetSpacing(), U0_img.GetDirection()
            U0_nda = sitk.GetArrayFromImage(U0_img)
            mask = np.zeros(U0_nda.shape) # (n_batch, ...) - > (...)
            brain = np.where(U0_nda != 0.)
            mask[brain] = 1
            # TODO: depends on task
            for s in range(len(mask)): 
                mask[s] = ndimage.binary_fill_holes(mask[s])
            '''for r in range(len(mask[0])):
                mask[:, r] = ndimage.binary_fill_holes(mask[:, r])
            for c in range(len(mask[0, 0])):
                mask[:, :, c] = ndimage.binary_fill_holes(mask[:, :, c])'''
            mask_img = sitk.GetImageFromArray(mask, isVector = False)
            cropped_mask_img, [x0, y0, z0], [x1, y1, z1], origin = cropping(mask_img, tol = 0.)
            cropped_mask_img.SetOrigin(origin)
            cropped_mask_img.SetSpacing(spacing)
            cropped_mask_img.SetDirection(direction)

            sitk.WriteImage(cropped_mask_img, os.path.join(self.save_fld, 'BrainMask.nii'))
            mask = sitk.GetArrayFromImage(cropped_mask_img)
        else:
            copyfile(self.Mask_path, os.path.join(self.save_fld, 'BrainMask.nii'))
            mask_img = sitk.ReadImage(self.Mask_path)
            oriign, spacing, direction = U0_img.GetOrigin(), U0_img.GetSpacing(), U0_img.GetDirection()
            self.brain_mask_nda = sitk.GetArrayFromImage(mask_img)
            x0, y0, z0 = 0, 0, 0
            x1, y1, z1 = self.brain_mask_nda.shape[0], self.brain_mask_nda.shape[1], self.brain_mask_nda.shape[2]
        return mask, [x0, y0, z0], [x1, y1, z1], [origin, spacing, direction]
    
    @property
    def contour(self):
        if self.Contour_path is not None and os.path.isfile(self.Contour_path):
            return (torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(self.Contour_path))).float()).to(self.device)
        else:
            return None

    @property
    def Dlst(self):
        if 'diff' in self.args.GT_perf_pattern:
            assert os.path.isfile(self.D_path)
            D_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.D_path))
            # Re-aranging to [-1, 1] * args.velocity_magnitude
            D_nda = D_nda * self.D_magnitude / np.max(abs(D_nda))
            # (batch = 1, s, r, c, (9))
            D = torch.from_numpy(D_nda).unsqueeze(0).float().to(self.device)[:, self.x0 : self.x1, self.y0 : self.y1, self.z0 : self.z1] * self.brain_mask[None, :, :, :, None] 
            D_path = self.save(D[0], 'D')
            if 'scalar' in self.D_type:
                return {'D': D}
            elif 'full' in self.D_type:
                # Compute PTI measures #
                PTIWriter = WritePTIImage(self.save_fld, self.origin, self.spacing, self.direction, self.device, to_smooth = False)
                PTISolver = PTI(os.path.join(self.save_fld, 'ScalarMaps'), self.Mask_path, 'diff_only', D_path = D_path, D_type = self.D_type, device = self.device, EigenRecompute = True)
                PTIWriter.save(PTISolver.Trace(), 'Trace.nii') 
                PTIWriter.save(PTISolver.eva, 'L.nii')
                PTIWriter.save(PTISolver.U(), 'U.nii')
                PTIWriter.save(PTISolver.FA(), 'FA.nii')
                PTIWriter.save(PTISolver.D_Color_Direction(), 'D_Color_Direction.nii')
                return {'Dxx': D[..., 0], 'Dxy': D[..., 1], 'Dxz': D[..., 2], 'Dyy': D[..., 4], 'Dyz': D[..., 5], 'Dzz': D[..., 8]}
            else:
                raise Exception('Unsupported input D_type')
        else:
            return None

    @property
    def Vlst(self):
        if 'adv' in self.args.GT_perf_pattern:
            assert os.path.isfile(self.V_path)
            V_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.V_path))
            # Re-aranging to [-1, 1] * args.velocity_magnitude
            V_nda = V_nda * self.V_magnitude / np.max(abs(V_nda))
            # (batch = 1, s, r, c, 3)
            V = torch.from_numpy(V_nda).unsqueeze(0).float().to(self.device)[:, self.x0 : self.x1, self.y0 : self.y1, self.z0 : self.z1] * self.brain_mask[None, :, :, :, None] 
            self.save(V[0], 'V')
            self.save(abs(V[0]), 'Abs_V')
            if 'vector' in self.V_type:
                # Compute L2-norm of V #
                nda2img(np.linalg.norm(V_nda, axis = -1), self.origin, self.spacing, self.direction, \
                    save_path = os.path.join(make_dir(os.path.join(self.save_fld, 'ScalarMaps')), 'Norm_V.nii'))
                return {'Vx': V[..., 0], 'Vy': V[..., 1], 'Vz': V[..., 2]}
            else:
                raise Exception('Unsupported input V_type')
        else:
            return None

    def get_U(self, Func):
        PDEFunc = Func(self.args, self.data_spacing, self.perf_pattern, self.D_type, self.V_type, self.device)
        PDEFunc.Dlst, PDEFunc.Vlst = self.Dlst, self.Vlst

        # Integrate
        with torch.no_grad():
            U = odeint(PDEFunc, self.U0, self.T, method = self.args.integ_method, options = self.args)
        U = (U.permute(1, 2, 3, 4, 0))[0][..., ::self.args.t_save_freq] * self.brain_mask[..., None] # (n_time, n_batch, slc, row, col) -> (slc, row, col, n_time)
        for it in range(U.size(-1)):
                print('Avg. value at time {:03d}: {:.6f}'.format(it, U[..., it].mean().item()))
        print('Domain-averaged value:  {:.6f}'.format(U.mean().item()))
        self.save(U, 'AdvDiff') 
        self.get_temporal(U, prefix = 'AdvDiff')

        ##################################################################################
        ##### Add Adv_only and Diff_only simulation for GT_perf_pattern as Adv_Diff ######
        ##################################################################################

        if self.args.GT_perf_pattern is 'adv_diff':
            # Pure advection version #
            print('Generate Advection-seperated version')
            AdvFunc = Func(self.args, self.data_spacing, 'adv_only', self.args.GT_D_type, self.args.GT_V_type, self.device)
            AdvFunc.Vlst = self.Vlst
            with torch.no_grad():
                U_adv = odeint(AdvFunc, self.U0, self.T, method = self.args.integ_method, options = self.args)
                #U_adv = odeint(None, AdvFunc, self.U0, self.T, method = self.args.integ_method)
            U_adv = (U_adv.permute(1, 2, 3, 4, 0))[0][..., self.args.t_save_freq] * self.brain_mask[..., None] # (n_time, n_batch, slc, row, col) -> (slc, row, col, n_time)
            for it in range(U_adv.size(-1)):
                print('Avg. value at time {:03d}: {:.6f}'.format(it, U_adv[..., it].mean().item()))
            print('Domain-averaged value:  {:.6f}'.format(U_adv.mean().item()))
            
            self.save(U_adv, 'Adv') 
            self.get_temporal(U_adv, prefix = 'Adv')
            # Pure diffusion version #
            print('Generate Diffusion-seperated version')
            DiffFunc = Func(self.args, self.data_spacing, 'diff_only', self.args.GT_D_type, self.args.GT_V_type, self.device)
            DiffFunc.Dlst = self.Dlst
            with torch.no_grad():
                U_diff = odeint(DiffFunc, self.U0, self.T, method = self.args.integ_method, options = self.args)
                #U_diff = odeint(None, DiffFunc, self.U0, self.T, method = self.args.integ_method)
            U_diff = (U_diff.permute(1, 2, 3, 4, 0))[0][..., self.args.t_save_freq] * self.brain_mask[..., None] # (n_time, n_batch, slc, row, col) -> (slc, row, col, n_time)
            for it in range(U_diff.size(-1)):
                print('Avg. value at time {:03d}: {:.6f}'.format(it, U_diff[..., it].mean().item()))
            print('Domain-averaged value:  {:.6f}'.format(U_diff.mean().item()))
            self.save(U_diff, 'Diff') 
            self.get_temporal(U_diff, prefix = 'Diff')
            
        return U # (n_batch, slc, row, col, n_time)
    
    def save(self, tensor, name):
        s_name = os.path.join(self.save_fld, '%s.nii') % (name)
        print('Save ground truth data in %s' % s_name)
        save_sitk(tensor, s_name, self.origin, self.spacing, self.direction) 
        return os.path.join(self.save_fld, '%s.nii') % (name)
    
    def get_temporal(self, U, prefix = 'T'):
        # U: (s, r, c, t)
        save_fld = make_dir(os.path.join(self.save_fld, 'TimeMachines'))
        axial_temporal    = U.permute(3, 1, 2, 0) # slice
        coronal_temporal  = U.permute(3, 0, 2, 1) # row
        sagittal_temporal = torch.flip(U.permute(3, 0, 1, 2), dims = [2]) # col # TODO Reverse direction for visualization end
        save_sitk(axial_temporal, os.path.join(save_fld, '%s-Axial.nii' % prefix), self.origin, self.spacing, self.direction) 
        save_sitk(coronal_temporal, os.path.join(save_fld, '%s-Coronal.nii' % prefix), self.origin, self.spacing, self.direction) 
        save_sitk(sagittal_temporal, os.path.join(save_fld, '%s-Sagittal.nii' % prefix), self.origin, self.spacing, self.direction)
        return







class PTIGenerator(object):
    def __init__(self, args, D_path, V_path, U0_path, Mask_path, Contour_path, device = 'cpu', save_fld = None):
        self.args = args
        self.dim = args.dim
        self.Mask_path = Mask_path
        self.Contour_path = Contour_path
        self.perf_pattern = args.GT_perf_pattern
        self.U0_path, self.U0_magnitude = U0_path, args.U0_magnitude
        self.D_path, self.D_type, self.D_magnitude = D_path, args.GT_D_type, args.diffusion_magnitude
        self.V_path, self.V_type, self.V_magnitude = V_path, args.GT_V_type, args.velocity_magnitude
        self.t = self.args.t0 + np.arange(self.args.nT) * self.args.dt
        self.device = device
        if save_fld is None:
            save_fld = make_dir(os.path.dirname(self.D_path), 'PWI')
        self.save_fld = save_fld
        self.register_info()

    def register_info(self):
        U0 = sitk.ReadImage(self.U0_path)
        self.origin = U0.GetOrigin()
        self.spacing = U0.GetSpacing()
        self.direction = U0.GetDirection()
        self.data_dim = [U0.GetSize()[2-i] for i in range(3)]
        self.data_spacing = [U0.GetSpacing()[2-i] for i in range(3)]

    @property
    def brain_mask(self):
        if not self.Mask_path:
            mask = np.zeros(self.raw_U0.shape) # (n_batch, ...) - > (...)
            brain = np.where(self.raw_U0 != 0.)
            mask[brain] = 1
            for s in range(len(mask)):
                mask[s] = ndimage.binary_fill_holes(mask[s])
            mask_img = sitk.GetImageFromArray(mask, isVector = False)
            mask_img.SetOrigin(self.origin)
            mask_img.SetSpacing(self.spacing)
            mask_img.SetDirection(self.direction)
            sitk.WriteImage(mask_img, os.path.join(self.save_fld, 'BrainMask.nii'))
            mask = torch.from_numpy(mask).float()
        else:
            mask = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(self.Mask_path))).float()
        return mask.to(self.device)
    
    @property
    def U0(self):
        U0 = sitk.GetArrayFromImage(sitk.ReadImage(self.U0_path)) 
        # Re-aranging to [-1, 1] * args.velocity_magnitude
        U0 = U0 * self.U0_magnitude / np.max(abs(U0))
        U0 = (torch.from_numpy(U0).float()).to(self.device)
        return (U0 * self.brain_mask.expand(1, -1, -1, -1)).to(self.device)

    @property
    def Vlst(self):
        if 'adv' in self.perf_pattern:
            if self.V_path is not None:
                V = sitk.GetArrayFromImage(sitk.ReadImage(self.V_path)) 
                # Re-aranging to [-1, 1] * args.velocity_magnitude
                V = V * self.V_magnitude / np.max(abs(V))
            else:
                raise NotImplementedError('Necessary input "V_path" not found.')
            V = (torch.from_numpy(V).float()).to(self.device)
            return {'Vx': V[..., 0].expand(1, -1, -1, -1), 'Vy': V[..., 1].expand(1, -1, -1, -1), \
            'Vz': V[..., 2].expand(1, -1, -1, -1)}
        else:
            return None
    
    @property
    def Dlst(self):
        if 'diff' in self.perf_pattern:
            if self.D_path is not None:
                D = sitk.GetArrayFromImage(sitk.ReadImage(self.D_path))
                # Re-aranging to [-1, 1] * args.velocity_magnitude
                D = D * self.V_magnitude / np.max(abs(D))
            else:
                raise NotImplementedError('Necessary input "D_path" not found.')

        return {'Dxx': self.D[..., 0, 0].expand(1, -1, -1, -1), 'Dxy': self.D[..., 1, 0].expand(1, -1, -1, -1), \
            'Dxz': self.D[..., 2, 0].expand(1, -1, -1, -1), 'Dyy': self.D[..., 1, 1].expand(1, -1, -1, -1), \
                'Dyz': self.D[..., 2, 1].expand(1, -1, -1, -1), 'Dzz': self.D[..., 2, 2].expand(1, -1, -1, -1)}

    @property
    def T(self):
        return torch.tensor(self.t, dtype = torch.float, device = self.device)

    @property
    def raw_U0(self):
        if self.U0_path is not None:
            raw_U0 = sitk.GetArrayFromImage(sitk.ReadImage(self.U0_path))
        else:
            raw_U0 = np.ones(self.D_nda.shape[:-1])
            print('start', int(self.D_nda.shape[0] * 0.25))
            print('end', int(self.D_nda.shape[0] * 0.75))
            raw_U0[int(self.D_nda.shape[0] * 0.25) : int(self.D_nda.shape[0] * 0.75), \
            int(self.D_nda.shape[1] * 0.25) : int(self.D_nda.shape[1] * 0.75), \
            int(self.D_nda.shape[2] * 0.25) : int(self.D_nda.shape[2] * 0.75)] = 2.
        return raw_U0 * self.args.init_value

    @property
    def contour(self):
        if self.Contour_path is not None:
            contour = sitk.GetArrayFromImage(sitk.ReadImage(self.Contour_path))
            contour = (torch.from_numpy(contour).float()).to(self.device)
        else:
            contour = None
        return contour

    def get_U(self, Func, RecordSum = False):
        PDEFunc = Func(self.args, self.D_path, None, self.data_dim, self.data_spacing, self.device, self.brain_mask, self.contour, for_GT = True)
        with torch.no_grad():
            U = odeint(None, PDEFunc, self.U0, self.T, method = self.args.integ_method)
        U = U.permute(1, 2, 3, 4, 0) # (n_time, n_batch, slc, row, col) -> (n_batch, slc, row, col, n_time)
        
        self.save(U[0], 'AxialPerf (%s)' % self.args.GT_D_type) # (t, slc, row, col) -> (slc, row, col, t)
        self.save(U[0], 'TemporalPerf (%s)' % self.args.GT_D_type) # (t, slc, row, col) -> (t, row, col, slc)
        #self.save(U[0].permute(3, 2, 1, 0), 'AxialPerf') # (t, col, row, slc) -> (slc, row, col, t)
        #self.save(U[0], 'TemporalPerf') # (t, col, row, slc) -> (t, row, col, slc)
        self.save(self.U0[0], 'U0 (%s)' % self.args.GT_D_type, isVector = False)
        if RecordSum:
            Avg = U.mean()
            for it in range(U.size(-1)):
                temp = U[0, :, :, :, it]
                print('Avg. value at time {:03d}: {:.6f}'.format(it, torch.mean(temp).item()))
            print('Domain-averaged value:  {:.6f}'.format(Avg.item()))
        return U # (n_batch, slc, row, col, n_time)
    
    def save(self, tensor, name, isVector = False):
        s_name = os.path.join(self.save_fld, '%s.nii') % (name)
        #print('Save ground truth data in %s' % s_fld)
        save_sitk(tensor, s_name, self.origin, self.spacing, self.direction) 
        return


class DataGenerator(object):
    '''
    For 2D Demo
    '''
    def __init__(self, args, device = 'cpu', isSave = True, save_fld = None):
        '''
        U0_pattern: \{ 'tophat', 'gaussian' \}
        perf_pattern: spatial pattern of perfusion process, chosen from \{ 'adv', 'diff', 'adv_diff', 'mix' \} \
            with assignment \{ 'adv_diff' - 0, 'diff' - 1, 'adv' - -1 \}
        D_type: \{ 'constant', 'scalar', 'diag', 'full' \}
        D_pattern: List consists of choices in \{ 'constant', 'gaussian' \} (Note: must be ["constant"] when D_type is "constant")
        diff: List of initial values for diffusicity at each direction: diff = [diff] or [diff_xx, diff_yy] or [diff_xx, diff_yy, diff_xy]
        V_type: \{ 'constant', 'scalar', 'vector' \}
        V_pattern:  List consists of choices in \{ 'constant',  'gaussian' \} (Note: must be ["constant"] when V_type is "constant")
        vel: List of initial values for velocity at each direction: vel = [vel] or [vel_x, vel_y] 
        integ_method: \{ 'dopri5', 'rk4','euler' \}
        BC (Boundary Condition): \{ None, 'Neumann' \}
        '''
        self.args = args
        self.t0, self.x0, self.y0 = args.t0, args.init_loc[0], args.init_loc[1]
        self.data_dim, self.data_spacing = args.GT_data_dim, args.data_spacing
        self.nX, self.nY = args.GT_data_dim[0], args.GT_data_dim[1]
        self.dx, self.dy = args.data_spacing[0], args.data_spacing[1]
        self.nT, self.dt = args.nT, args.dt
        self.t = self.t0 + np.arange(args.nT) * args.dt 
        self.X = self.x0 + np.arange(self.nX) * self.dx
        self.Y = self.y0 + np.arange(self.nY) * self.dy
        self.perf_pattern = args.GT_perf_pattern
        self.D_type, self.D_pattern, self.diff =  args.GT_D_type, args.D_pattern, args.diffusivity
        self.V_type, self.V_pattern, self.vel =  args.GT_V_type, args.V_pattern, args.velocity
        self.device = device
        self.isSave = isSave
        self.save_fld = save_fld

    @property
    def origin(self):
        return (self.x0, self.y0, self.t0)

    @property
    def spacing(self):
        return (self.dx, self.dy, self.dt)
    
    @property 
    def Perf_Info(self):
        if self.D_type is 'constant' or self.D_type is 'scalar':
            D_info = 'D: %s' % (self.D_pattern['D'])
        elif self.D_type is 'diag':
            D_info = 'Dxx: %s, Dyy: %s' %(self.D_pattern['Dxx'], self.D_pattern['Dyy'])
        elif 'full' in self.D_type:
            D_info = 'Dxx: %s, Dyy: %s, Dxy: %s' % (self.D_pattern['Dxx'], self.D_pattern['Dyy'], self.D_pattern['Dxy'])

        if self.V_type is 'constant' or self.V_type is 'scalar':
            V_info = 'V: %s' % (self.V_pattern['V'])
        elif 'vector' in self.V_type:
            V_info = 'Vx: %s, Vy: %s' % (self.V_pattern['Vx'], self.V_pattern['Vy'])

        if 'diff_only' in self.perf_pattern:
            Perf_Info = '%s' % D_info
        elif 'adv_only' in self.perf_pattern:
            Perf_Info = '%s' % V_info
        else:
            Perf_Info = '%s; %s' % (D_info, V_info)
        return Perf_Info
    
    @property
    def T(self):
        return torch.tensor(self.t, dtype = torch.float, device = self.device)

    @property
    def U0(self):
        U0 = U0s[self.args.U0_pattern]([self.X, self.Y], self.args.init_value, self.device)
        return U0.expand(1, -1, -1) # (n_batch == 1, nX, nY)
    
    @property
    def PerfFlag(self): # has sign
        isPerf = PerfFlags[self.perf_pattern](self.data_dim, self.D_type, self.V_type, self.device)
        if self.isSave:
        	for key in isPerf:
        		self.save(isPerf[key], key)
        return isPerf

    def get_D(self, D_name, D_value, D_pattern):
        D = DVs[D_pattern]([self.X, self.Y], D_value, self.device) * self.PerfFlag['is%s' % D_name]
        if self.isSave:
            self.save(D, '%s' % D_name)
        return D
        #return D.expand(1, -1, -1)
        
    def get_V(self, V_name, V_value, V_pattern):
        V = DVs[V_pattern]([self.X, self.Y], V_value, self.device) * self.PerfFlag['is%s' % V_name]
        if self.isSave:
            self.save(V, '%s' % V_name)
        return V
        #return V.expand(1, -1, -1)

    @property
    def Dlst(self):
        D_lst = []
        if 'adv_only' in self.perf_pattern:
            pass
        elif self.D_type is 'constant' or self.D_type is 'scalar':
            D_lst = {'D': self.get_D('D', self.diff['D'], self.D_pattern['D'])}
            return D_lst
        elif self.D_type is 'diag':
            D_lst = {'Dxx': self.get_D('Dxx', self.diff['Dxx'], self.D_pattern['Dxx']),\
                'Dyy': self.get_D('Dyy', self.diff['Dyy'], self.D_pattern['Dyy'])}
            return D_lst
        elif 'full' in self.D_type:
            D_lst = {'Dxx': self.get_D('Dxx', self.diff['Dxx'], self.D_pattern['Dxx']), \
                'Dyy': self.get_D('Dyy', self.diff['Dyy'], self.D_pattern['Dyy']), \
                    'Dxy': self.get_D('Dxy', self.diff['Dxy'], self.D_pattern['Dxy'])}
        return D_lst
    
    @property
    def Vlst(self):
        V_lst = []
        if 'diff_only' in self.perf_pattern:
            pass
        elif self.V_type is 'constant' or self.V_type is 'scalar':
            V_lst = {'V': self.get_V('V', self.vel['V'], self.V_pattern['V'])}
            return V_lst
        elif 'vector' in self.V_type:
            V_lst = {'Vx': self.get_V('Vx', self.vel['Vx'], self.V_pattern['Vx']), \
                'Vy': self.get_V('Vy', self.vel['Vy'], self.V_pattern['Vy'])}
        return V_lst
    
    def get_U(self, PDEFunc, RecordSum = False, ToNormalize = True):
        PDEFunc = PDEFunc(self.args, self.data_dim, self.data_spacing, self.perf_pattern, self.Dlst, self.Vlst, \
            self.device, mask = None)
        with torch.no_grad():
            U = odeint(PDEFunc, self.U0, self.T, method = self.args.integ_method, options = self.args)
        U = U.permute(1, 0, 2, 3) # (n_time, n_batch, nx, ny) -> (n_batch, n_time, nx, ny)
        if ToNormalize:
            U = U / max(list(abs(U).reshape(-1)))
        if self.isSave:
            self.save(U[0], 'Perf')
        if RecordSum:
            Avg = U.mean()
            for it in range(U.size(1)):
                temp = U[0, it]
                print('Total value at time %d: %.6f' %(it, torch.sum(temp).item()))
            print('Domain-averaged value:   %.6f' % Avg.item())
        return U
    
    def save(self, tensor, name):
        s_name = os.path.join(self.save_fld, '%s.nii') % (name)
        #print('Save ground truth data in %s' % s_fld)
        save_sitk(tensor, s_name, self.origin, self.spacing) # (n_batch, ...) -> (...)
        return


##############################################################################################################################


if __name__ == '__main__':
    # Model settings
    integ_method = 'dopri5'
    BC = None
    gpu = 0
    device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
    isSave = True

    # Data settings
    t0 = 0.0
    init_loc = [-3., -3.]
    data_dim = [151, 151]
    data_spacing = [0.04, 0.04]
    nT = 50
    dt = 0.045
    U0_pattern = 'gaussian'
    init_value = 1.
    perf_pattern = 'adv_diff'
    D_type = 'full_cholesky'
    D_pattern = {'D': 'constant', 'Dxx': 'constant', 'Dyy': 'constant', 'Dxy': 'constant'}
    diffusivity = {'D': 0.4, 'Dxx': 0.3, 'Dyy': 0.3, 'Dxy': 0.3}
    V_type = 'vector'
    V_pattern= {'V': 'constant', 'Vx': 'constant', 'Vy': 'constant'}
    velocity = {'V': 0.8, 'Vx': 0.6, 'Vy': 0.6}

    Generator = DataGenerator(t0, init_loc, data_dim, data_spacing, nT, dt, \
            U0_pattern, init_value, perf_pattern, D_type, D_pattern, diffusivity, \
                V_type, V_pattern, velocity, integ_method, BC, device, isSave = isSave)