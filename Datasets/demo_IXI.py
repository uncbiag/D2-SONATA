import os, sys, argparse, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import numpy as np
import SimpleITK as sitk
from shutil import copyfile, rmtree
import scipy.ndimage as ndimage

from ODE.odeint import odeint
from Preprocess.prepro_utils import cropping
from Preprocess.print_info import write_info
from Learning.Modules.AdvDiffPDE import PIANO_Skeleton
from Postprocess.PTIMeasures import PTI, WritePTIImage
from utils import make_dir, save_sitk, nda2img, stream_3D\

from DemoOptions.ValueMask import gaussian_resize
ValueMasks = {
    'gaussian': gaussian_resize
}

#device = 'cpu'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda:3')


#%% Basic settings
parser = argparse.ArgumentParser('3D demo prep')
parser.add_argument('--case', type = str, default = 'PI_demo')
parser.add_argument('--adjoint', type = bool, default = False)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--is_resume', type=bool, default = False)
parser.add_argument('--integ_method', type = str, default = 'dopri5', choices=['dopri5', 'adams', 'rk4', 'euler'])

# Set as False for samples generation #
parser.add_argument('--stochastic', type=bool, default = False)
parser.add_argument('--separate_stochastic', type = bool, default = False, help = 'Model different stochastic Sigma for D and V')

# For ground truth types
parser.add_argument('--GT_perf_pattern', type = str, default = 'adv_diff', choices = ['mix', 'circle_adv_only', 'adv_diff', 'adv_only', 'diff_only'])
parser.add_argument('--save_only_movie', type = bool, default = False, help = 'Save diff_only and adv_only movies')

parser.add_argument('--t0', type = float, default = 0.0) # the beginning time point
parser.add_argument('--dt', type = float, default = 0.01) # Real dt between scanned images # 0.01

# For random value mask adding -- lesion version #
parser.add_argument('--mask_pattern', type = str, default = 'gaussian', choices = ['gaussian'])

parser.add_argument('--lesion_mask', type = bool, default = True, help = 'determines whether adding random-sized value mask for (stochastic) uncertainty supervision')
    
parser.add_argument('--separate_DV_mask', type = bool, default = False, help = 'D and V share the same mask or not') 
parser.add_argument('--separate_DV_mask_intensity', type = bool, default = True, help = 'D and V share the same lesion segmentation mask but different anomaly intensities') # TODO: under testing # 

parser.add_argument('--save_as_addition', type = bool, default = True, help = 'Whether save V as V = \bar{V} + \delta_V')

# NOTE: total time pointsc := collocation_len * n_collocations
parser.add_argument('--sub_collocation_nt', type = int, default = 5, help = 'time points between two sub_collocation points (ode steps)') # 10
parser.add_argument('--collocation_len', type = int, default = 2, help = '# of sub_collocations between two collocation points collocation_nt / sub_collocation_nt') 
parser.add_argument('--n_collocations', type = int, default = 20, help = 'save on only collocation points')

parser.add_argument('--U0_type', type = str, default = 'Conc', choices = {'MR', 'Conc'}) # DSC: need convert to concentration (Conc) image
parser.add_argument('--U0_magnitude', type = float, default = 1., help = 'max value for U0') # main range ~ 1/4 of max

parser.add_argument('--GT_D_type', type = str, default = 'full', choices = ['scalar', 'full'])
#parser.add_argument('--diffusivity', type = dict, default = {'D': 0.1, 'Dxx': 0.2, 'Dxy': 0.1, 'Dyy': 0.06})
# small diffusion #
#parser.add_argument('--diffusion_magnitude', type = float, default = 0.5, help = 'max value for abs(D)') # 0.5, main range ~ 1/4 of max
# large diffusion #
parser.add_argument('--diffusion_magnitude', type = float, default = 2., help = 'max value for abs(D)')

parser.add_argument('--use_Phi4V', type = bool, default = False) # NOTE: not work well
parser.add_argument('--Phi_magnitude', type = float, default = 300., help = 'max value for abs(Phi): curl(Phi) == V') # V_magnitude ~ Phi_magnitude / 20
parser.add_argument('--GT_V_type', type = str, default = 'vector_div_free', choices = ['vector', 'vector_div_free'])
# original #
#parser.add_argument('--velocity_magnitude', type = float, default = 10., help = 'max value for abs(V)') # 5. main range ~ 1/4 of max
# new try #
parser.add_argument('--velocity_magnitude', type = float, default = 100., help = 'max value for abs(V)') # 5. main range ~ 1/4 of max


parser.add_argument('--time_stepsize', type = float, default = 1e-2, help = 'Time step for integration, \
    if is not None, the assigned value should be able to divide args.time_spacing')
parser.add_argument('--BC', type = str, default = 'neumann', choices = ['None', 'neumann'])

args_ixi = parser.parse_args()  


class PIANOGenerator(object):
    '''
    For 3D Demo
    '''
    def __init__(self, args, save_fld, U0_path, D_path, V_path = None, Phi_path = None, VesselMask_path = None,\
        Mask_path = None, Path4Mask = None, Contour_path = None, device = 'cpu'):
        self.args = args
        self.device = device
        self.Mask_path = Mask_path
        self.Path4Mask = Path4Mask
        self.Contour_path = Contour_path
        self.VesselMask_path = VesselMask_path
        self.perf_pattern = args.GT_perf_pattern
        self.U0_path, self.U0_magnitude = U0_path, args.U0_magnitude
        self.Phi_path, self.Phi_magnitude = Phi_path, args.Phi_magnitude
        self.D_path, self.D_type, self.D_magnitude = D_path, args.GT_D_type, args.diffusion_magnitude
        self.V_path, self.V_type, self.V_magnitude = V_path, args.GT_V_type, args.velocity_magnitude
        self.t0, self.dt, self.n_collocations, self.collocation_len, self.sub_collocation_nt = \
            args.t0, args.dt, args.n_collocations, args.collocation_len, args.sub_collocation_nt

        self.main_fld = save_fld
        self.save_fld, self.scalar_save_fld = self.get_save_fld()  

        self.brain_mask_nda, [self.x0, self.y0, self.z0], [self.x1, self.y1, self.z1], [self.origin, self.spacing, self.direction] = self.get_brain_mask()
        self.brain_mask = torch.from_numpy(self.brain_mask_nda).float().to(self.device)
        self.data_spacing = [self.spacing[2-i] for i in range(3)] 
        self.register_U0(args.U0_type)
        self.set_neumann = torch.nn.ReplicationPad3d(1)

        self.configure_value_mask()

    def div_free_v_cond(self, V):
        # V: (3, s, r, c) 
        return self.set_neumann(V[:, 1:-1, 1:-1, 1:-1].unsqueeze(0))[0] # (3, s, r, c)
    
    def get_save_fld(self):
        if self.perf_pattern == 'diff_only':
            self.movie_name = 'Diff'
            self.setting_info = 'D-%s' % self.D_type
        elif self.perf_pattern == 'adv_only':
            self.movie_name = 'Adv'
            self.setting_info = 'V-%s' % self.V_type
        else:
            self.movie_name = 'AdvDiff'
            self.setting_info = 'D-%s_V-%s' % (self.D_type, self.V_type) 
        save_fld = self.main_fld if self.main_fld \
            else make_dir(os.path.join(os.path.dirname(self.D_path), 'Movies'))
        return save_fld, make_dir(os.path.join(save_fld, 'ScalarMaps'))

    def register_U0(self, img_type = 'MR'):
        # NOTE: For Numpy / Torch (reverse order to ITK image)
        U0_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.U0_path))
        if 'MR' in img_type:
            U0_nda = self.signal2ctc(U0_nda)
        # Re-aranging to [-1, 1] * args.velocity_magnitude
        U0_nda = U0_nda * self.U0_magnitude / np.max(abs(U0_nda))
        U0 = (torch.from_numpy(U0_nda).float()).to(self.device)[self.x0 : self.x1, self.y0 : self.y1, self.z0 : self.z1] * self.brain_mask
        self.U0 = U0.expand(1, -1, -1, -1) # (batch = 1, s, r, c) for PIANO input
        self.save(self.U0[0], 'U0')

    def signal2ctc(self, signal_nda, s0 = 1., k = 1., TE = 0.025):
        #print('   Convert signal image to concentration image')
        ctc_nda = - k/TE * np.log(signal_nda / s0 + 1e-14)
        return ctc_nda

    @property
    def T_subcollocation(self):
        '''
        T between two collocation points
        '''
        t = self.t0 + np.arange(self.sub_collocation_nt) * self.dt
        return torch.tensor(t, dtype = torch.float, device = self.device)

    def get_brain_mask(self):
        if not self.Mask_path:
            if not self.Path4Mask:
                img4msk = sitk.ReadImage(self.U0_path) # If image for masking not provided, use U0 image for masking instead #
            else:
                img4msk = sitk.ReadImage(self.Path4Mask)
            spacing, direction = img4msk.GetSpacing(), img4msk.GetDirection()
            nda4mask = sitk.GetArrayFromImage(img4msk)
            mask = np.zeros(nda4mask.shape) # (n_batch, ...) - > (...)
            brain = np.where(nda4mask != 0.)
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

            sitk.WriteImage(cropped_mask_img, os.path.join(self.save_fld, 'BrainMask.mha'))
            mask = sitk.GetArrayFromImage(cropped_mask_img)
        else:
            copyfile(self.Mask_path, os.path.join(self.save_fld, 'BrainMask.mha'))
            mask_img = sitk.ReadImage(self.Mask_path)
            mask = sitk.GetArrayFromImage(mask_img)
            origin, spacing, direction = mask_img.GetOrigin(), mask_img.GetSpacing(), mask_img.GetDirection()
            x0, y0, z0 = 0, 0, 0
            x1, y1, z1 = mask.shape[0], mask.shape[1], mask.shape[2]
        return mask, [x0, y0, z0], [x1, y1, z1], [origin, spacing, direction]
    
    @property
    def contour(self):
        if self.Contour_path is not None and os.path.isfile(self.Contour_path):
            return (torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(self.Contour_path))).float()).to(self.device) 
        else:
            return None

    def configure_value_mask(self):
        if self.args.lesion_mask: 
            if not self.args.separate_DV_mask and not self.args.separate_DV_mask_intensity:
                strength = 1. - np.random.random_sample() * 0.6 # determine the lesion mask center value
                ValueMask, _, _ = ValueMasks[self.args.mask_pattern]([self.x1, self.y1, self.z1], ctr_lst = None, r_lst = None, strength = strength, device = self.device) # tensor: (s, r, c) 
                self.save(ValueMask, 'ValueMask')
                lesion_seg_nda = (ValueMask < 0.8).float()
                self.save(lesion_seg_nda, 'LesionSeg')

                self.ValueMask = ValueMask[None, :, :, :, None].to(self.device) # (1, s, r, c, 1) 
            else:
                strength = 1. - np.random.random_sample() * 0.6 # determine the lesion mask center value
                ValueMask_V, center, radius= ValueMasks[self.args.mask_pattern]([self.x1, self.y1, self.z1], ctr_lst = None, r_lst = None, strength = strength, device = self.device) # tensor: (s, r, c) 
                self.save(ValueMask_V, 'ValueMask_V')
                lesion_seg_v_nda = (ValueMask_V < 0.8).float()
                self.save(lesion_seg_v_nda, 'LesionSeg_V') 

                self.ValueMask_V = ValueMask_V[None, :, :, :, None].to(self.device) # (1, s, r, c, 1)
                
                if self.args.separate_DV_mask_intensity:
                    strength = 1. - np.random.random_sample() * 0.6 # re-new anomaly strength for D #
                    ValueMask_D, _, _ = ValueMasks[self.args.mask_pattern]([self.x1, self.y1, self.z1], ctr_lst = center, r_lst = radius, strength = strength, device = self.device) # tensor: (s, r, c) 
                else:
                    ValueMask_D, _, _ = ValueMasks[self.args.mask_pattern]([self.x1, self.y1, self.z1], ctr_lst = None, r_lst = None, strength = strength, device = self.device) # tensor: (s, r, c) 
                
                self.save(ValueMask_D, 'ValueMask_D')
                lesion_seg_d_nda = (ValueMask_D < 0.8).float()
                self.save(lesion_seg_d_nda, 'LesionSeg_D') 

                self.ValueMask_D = ValueMask_D[None, :, :, :, None].to(self.device) # (1, s, r, c, 1)
        else:
            self.ValueMask = 1.
        return
    
    @property
    def Dlst(self):
        if 'diff' in self.args.GT_perf_pattern:
            assert os.path.isfile(self.D_path)
            orig_D_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.D_path))
            # Re-aranging to [-1, 1] * args.velocity_magnitude
            orig_D_nda = orig_D_nda * self.D_magnitude / np.max(abs(orig_D_nda))
            # (batch = 1, s, r, c, (9))
            orig_D = torch.from_numpy(orig_D_nda).unsqueeze(0).float().to(self.device)[:, self.x0 : self.x1, self.y0 : self.y1, self.z0 : self.z1] * self.brain_mask[None, :, :, :, None]  
            if self.args.lesion_mask:
                ValueMask_D = self.ValueMask if (not self.args.separate_DV_mask and not self.args.separate_DV_mask_intensity) else self.ValueMask_D 
                
                D = orig_D * ValueMask_D
                delta_D = D - orig_D

                D_path = self.save(D[0], 'D') 
                orig_D_path = self.save(orig_D[0], 'orig_D')
                delta_D_path = self.save(delta_D[0], 'delta_D')
            else:
                D = orig_D
                orig_D_path = self.save(D[0], 'D')
                   
            if 'scalar' in self.D_type:
                return {'D': orig_D}
            elif 'full' in self.D_type:
                # Compute PTI measures #
                PTIWriter = WritePTIImage(self.save_fld, self.origin, self.spacing, self.direction, self.device, to_smooth = False)
                PTISolver = PTI(self.scalar_save_fld, self.Mask_path, 'diff_only', D_path = orig_D_path, D_type = self.D_type, device = self.device, EigenRecompute = True)
                PTIWriter.save(PTISolver.FA(), 'FA.mha', mask_nda = self.brain_mask_nda)
                PTIWriter.save(PTISolver.Trace(), 'Trace.mha', mask_nda = self.brain_mask_nda) 
                PTIWriter.save(PTISolver.eva, 'L.mha', mask_nda = self.brain_mask_nda[..., None])
                PTIWriter.save(PTISolver.U(), 'U.mha', mask_nda = self.brain_mask_nda[..., None])
                PTIWriter.save(PTISolver.D_Color_Direction(), 'D_Color_Direction.mha', mask_nda = self.brain_mask_nda[..., None])

                Dlst = {'Dxx': D[..., 0], 'Dxy': D[..., 1], 'Dxz': D[..., 2], 'Dyy': D[..., 4], 'Dyz': D[..., 5], 'Dzz': D[..., 8]}
                if self.args.lesion_mask: # lesion_mask: (1, s, r, c, 1)
                    ValueMask_D = ValueMask_D.cpu()
                    PTIWriter.save(PTISolver.Trace(), 'orig_Trace.mha', mask_nda = self.brain_mask_nda)
                    PTIWriter.save(PTISolver.Trace() * ValueMask_D[0, :, :, :, 0], 'Trace.mha', mask_nda = self.brain_mask_nda)
                    PTIWriter.save(PTISolver.Trace() * (1. - ValueMask_D[0, :, :, :, 0]), 'delta_Trace.mha', mask_nda = self.brain_mask_nda)
                    PTIWriter.save(PTISolver.eva, 'orig_L.mha', mask_nda = self.brain_mask_nda[..., None])
                    PTIWriter.save(PTISolver.eva * ValueMask_D[0], 'L.mha', mask_nda = self.brain_mask_nda[..., None])
                    PTIWriter.save( - PTISolver.eva * (1. - ValueMask_D[0]), 'delta_L.mha', mask_nda = self.brain_mask_nda[..., None]) # NOTE: eva >= 0, while delta_L >= 0 
                    Dlst.update({'orig_Dxx': orig_D[..., 0], 'orig_Dxy': orig_D[..., 1], 'orig_Dxz': orig_D[..., 2], 'orig_Dyy': orig_D[..., 4], 'orig_Dyz': orig_D[..., 5], 'orig_Dzz': orig_D[..., 8], \
                        'delta_Dxx': delta_D[..., 0], 'delta_Dxy': delta_D[..., 1], 'delta_Dxz': delta_D[..., 2], 'delta_Dyy': delta_D[..., 4], 'delta_Dyz': delta_D[..., 5], 'delta_Dzz': delta_D[..., 8]})
                return Dlst
            else:
                raise Exception('Unsupported input D_type')
        else:
            return None

    @property
    def Vlst(self):
        if 'adv' in self.args.GT_perf_pattern:
            if self.args.use_Phi4V: # TODO: not up-to-date: not working
                assert os.path.isfile(self.Phi_path) and os.path.isfile(self.VesselMask_path)
                copyfile(self.VesselMask_path, os.path.join(self.save_fld, 'Vessel_smoothed.mha'))
                VesselMask = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(self.VesselMask_path))).float().to(self.device) # (s, r, c)

                Phi_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.Phi_path)) # (s, r, c, 3)
                # Re-aranging to [-1, 1] * args.Phi_magnitude
                Phi = torch.from_numpy(Phi_nda * self.Phi_magnitude / np.max(abs(Phi_nda))).float().to(self.device)
                self.save(Phi, 'ScalarMaps/Phi')

                Vx, Vy, Vz = stream_3D(Phi[..., 0], Phi[..., 1], Phi[..., 2], batched = False, delta_lst = self.data_spacing)
                V = torch.stack([Vx, Vy, Vz], dim = 0) * VesselMask[None] # (3, s, r, c)
                V = self.div_free_v_cond(V).permute(1, 2, 3, 0) # (s, r, c, 3)
            else:
                assert os.path.isfile(self.V_path)
                orig_V_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.V_path))
                # Re-aranging to [-1, 1] * args.velocity_magnitude
                orig_V_nda = orig_V_nda * self.V_magnitude / np.max(abs(orig_V_nda))
                # (batch = 1, s, r, c, 3)
                orig_V = torch.from_numpy(orig_V_nda).float().to(self.device)
            # Match dimension for PDE solver #
            orig_V = orig_V.unsqueeze(0)[:, self.x0 : self.x1, self.y0 : self.y1, self.z0 : self.z1] * self.brain_mask[None, :, :, :, None] 
            if self.args.lesion_mask:
                ValueMask_V = self.ValueMask if (not self.args.separate_DV_mask and not self.args.separate_DV_mask_intensity) else self.ValueMask_V

                V = orig_V * ValueMask_V
                delta_V = V - orig_V

                V_path = self.save(V[0], 'V') 
                orig_V_path = self.save(orig_V[0], 'orig_V')
                delta_V_path = self.save(delta_V[0], 'delta_V')

                self.save(abs(V[0]), 'ScalarMaps/Abs_V')
                self.save(abs(orig_V[0]), 'ScalarMaps/orig_Abs_V')
                self.save(abs(delta_V[0]), 'ScalarMaps/delta_Abs_V')

                self.save(torch.norm(V[0], dim = -1), 'ScalarMaps/Norm_V')
                self.save(torch.norm(orig_V[0], dim = -1), 'ScalarMaps/orig_Norm_V')
                self.save(torch.norm(delta_V[0], dim = -1), 'ScalarMaps/delta_Norm_V')
            else:
                V = orig_V
                orig_V_path = self.save(V[0], 'V')
                self.save(abs(V[0]), 'ScalarMaps/Abs_V')
                self.save(torch.norm(V[0], dim = -1), 'ScalarMaps/Norm_V')

            return {'Vx': V[..., 0], 'Vy': V[..., 1], 'Vz': V[..., 2]} 
        else:
            return None
    
    def integrate(self, PDEFunc):
        U = torch.stack([self.U0] * self.n_collocations, dim = 1) # (n_batch = 1, n_collocations, s, r, c)
        # Integrate by collocations #
        with torch.no_grad():
            for i_coll in range(1, self.n_collocations):
                U[:, i_coll] = U[:, i_coll - 1] * self.brain_mask[None] # (n_batch = 1, s, r, c)
                for i_sub_coll in range(self.collocation_len): # (nt = -1, batch = 1, slc, row, col)
                    U[:, i_coll] = odeint(PDEFunc, U[:, i_coll], self.T_subcollocation, method = self.args.integ_method, options = self.args)[-1]
                #print('   - Avg. value at time {:03d}: {:.6f}'.format(i_coll, U[:, i_coll].mean().item()))
        U = (U[0] * self.brain_mask[None]).permute(1, 2, 3, 0) # (n_collocations, slc, row, col) -> (slc, row, col, n_collocations)
        #print('   - Domain-averaged value:  {:.6f}'.format(U.mean().item()))
        return U # (s, r, c, nT = args.n_collocations)


    def get_U(self, Func):
        PDEFunc = Func(self.args, self.data_spacing, self.perf_pattern, self.D_type, self.V_type, self.device)
        PDEFunc.Dlst, PDEFunc.Vlst = self.Dlst, self.Vlst

        U = self.integrate(PDEFunc)
        U_path = self.save(U, self.movie_name) 
        write_info(U_path, 'Info.txt', show_info = False) # not showing concentration information (BAT, TTP, etc.)
        self.get_temporal(U, prefix = self.movie_name)

        ###### Save adv_only and diff_only simulation if GT_perf_pattern == adv_diff ######

        if self.perf_pattern == 'adv_diff' and self.args.save_only_movie:

            print('   Generate Advection-seperated version')
            PDEFunc.perf_pattern = 'adv_only'
            U = self.integrate(PDEFunc)
            self.save(U, 'Adv') 
            self.get_temporal(U, prefix = 'Adv')

            print('   Generate Diffusion-seperated version')
            PDEFunc.perf_pattern = 'diff_only'
            U = self.integrate(PDEFunc)
            self.save(U, 'Diff') 
            self.get_temporal(U, prefix = 'Diff')
        return 
    
    def save(self, tensor, name):
        s_name = os.path.join(self.save_fld, '%s.mha') % (name)
        #print('   Save ground truth data in %s' % s_name)
        save_sitk(tensor, s_name, self.origin, self.spacing, self.direction) 
        return os.path.join(self.save_fld, '%s.mha') % (name)
    
    def get_temporal(self, U, prefix = 'T'):
        # U: (s, r, c, t)
        save_fld = make_dir(os.path.join(self.save_fld, 'TimeMachines'))
        axial_temporal    = U.permute(3, 1, 2, 0) # slice
        coronal_temporal  = U.permute(3, 0, 2, 1) # row
        sagittal_temporal = torch.flip(U.permute(3, 0, 1, 2), dims = [2]) # col # TODO Reverse direction for visualization end
        save_sitk(axial_temporal, os.path.join(save_fld, '%s-Axial.mha' % prefix), self.origin, self.spacing, self.direction) 
        save_sitk(coronal_temporal, os.path.join(save_fld, '%s-Coronal.mha' % prefix), self.origin, self.spacing, self.direction) 
        save_sitk(sagittal_temporal, os.path.join(save_fld, '%s-Sagittal.mha' % prefix), self.origin, self.spacing, self.direction)
        return


def get_movie(case_fld):

    #print('Start generating movie')

    D_path = os.path.join(case_fld, 'DTI.mha')
    V_path = None
    Phi_path = os.path.join(case_fld, 'AdvectionMaps/DivFree/Phi.mha') # Phi for computing divergence-free V # 
    BrainMask_path = os.path.join(case_fld, 'BrainMask.mha')
    U0_path = os.path.join(case_fld, 'MRA.mha') # NOTE MRA as U0: directly treat as concentration map without signal converting #

    #if not os.path.isfile(D_path) or not os.path.isfile(V_path) or not os.path.isfile(U0_path) or not os.path.isfile(BrainMask_path):
    #    print('Check file existence in case NO.%d (of %d): %s\n' % (i + 1, len(case_names), case_name))
    #    continue
    #else:
    save_fld = make_dir(os.path.join(case_fld, 'Movies(DivFree)'))

    # Copy GT maps in Movie folder for future use #
    VesselMask_path = os.path.join(case_fld, 'VesselMask_smoothed.mha') # NOTE Binary map of smoothed vessel for fitting div_free V #
    Phi_path = os.path.join(case_fld, 'AdvectionMaps/DivFree/Phi.mha') # Phi for computing divergence-free V # 

    #save_fld = make_dir(os.path.join(case_fld, 'Movies'))
    Generator = PIANOGenerator(args_ixi, save_fld, U0_path, D_path, V_path = V_path, Phi_path = Phi_path, \
        VesselMask_path = VesselMask_path, Mask_path = BrainMask_path, device = device)
    Generator.get_U(PIANO_Skeleton)

    return



def main():

    on_server = True
    if on_server:
        processed_fld = '/playpen-raid2/peirong/Data/IXI_Processed'
        names_file = open(os.path.join(processed_fld, 'IDs.txt'), 'r')
    else:
        processed_fld = '/media/peirong/PR5/IXI_Processed'
        names_file = open(os.path.join(processed_fld, 'IDs.txt'), 'r')
    all_names = names_file.readlines()
    names_file.close()
    print('Number of all cases:', len(all_names))

    start = 0
    #case_names = all_names[start : 130] 
    
    #case_names = all_names[start:]  
    #case_names = all_names[:20] 
    #case_names = all_names[20:40]  
    #case_names = all_names[40:60]  
    #case_names = all_names[60:80] 
    #case_names = all_names[80:100]  

    #case_names = all_names[80:87] + all_names[120:131]
    #case_names = all_names[87:100]  
    
    #case_names = all_names[100:120]  
    case_names = all_names[120:131]  
    
    #case_names = ['IXI280-HH-1860']

    print('Number of cases to process:', len(case_names))


    if args_ixi.lesion_mask:
        if args_ixi.separate_DV_mask:
            prefix = 'Movies-SepShpLesion'
        elif args_ixi.separate_DV_mask_intensity:
            prefix = 'Movies-SepLesion'
        else:
            prefix = 'Movies-Lesion'
    else:
        prefix = 'Movies' 


    for i in range(len(case_names)):
        case_name = case_names[i].split('\n')[0]
        case_fld = os.path.join(processed_fld, case_name)

        if not os.path.isfile(os.path.join(processed_fld, case_name, 'DTI.mha')):
            print('\n---Skip non-existed case NO.%d (of %d): %s\n' % (i + 1, len(case_names), case_name))
        #elif os.path.isdir(os.path.join(processed_fld, case_name, prefix)): # TODO
        #    print('Skip processed case NO.%d (of %d): %s\n' % (i + 1, len(case_names), case_name))
        else:
            try:
                D_path = os.path.join(case_fld, 'DTI.mha')
                V_path = os.path.join(case_fld, 'AdvectionMaps/V_smoothed.mha')
                Phi_path = os.path.join(case_fld, 'AdvectionMaps/DivFree/Phi.mha') # Phi for computing divergence-free V # 
                BrainMask_path = os.path.join(case_fld, 'BrainMask.mha')
                VesselMask_path = os.path.join(case_fld, 'VesselMask_smoothed.mha') # NOTE Binary map of smoothed vessel for fitting div_free V #
                U0_path = os.path.join(case_fld, 'MRA.mha') # NOTE MRA as U0: directly treat as concentration map without signal converting #

                if args_ixi.use_Phi4V:
                    save_fld = make_dir(os.path.join(case_fld, prefix+'(DivFree)'))
                else:
                    if os.path.isdir(os.path.join(case_fld, prefix)):
                        rmtree(os.path.join(case_fld, prefix))
                    save_fld = make_dir(os.path.join(case_fld, prefix))

                print('Start processing case NO.%d (of %d): %s' % (i + 1, len(case_names), case_name))
                print('     Save folder:', save_fld)
                
                Generator = PIANOGenerator(args_ixi, save_fld, U0_path, D_path, V_path = V_path, Phi_path = Phi_path, \
                    VesselMask_path = VesselMask_path, Mask_path = BrainMask_path, device = device) 
                Generator.get_U(PIANO_Skeleton)

                # Archieved: for small diffusion maps #
                '''mag_lst = [1.] 
                #mag_lst = [0.2, 0.4, 0.6, 0.8, 1.] # TODO
                for magnitude in mag_lst:
                    print('Current scale: %.1f' % magnitude)
                    temp_save_fld = make_dir(os.path.join(save_fld, 'Mag(%.1f)' % magnitude))
                    Generator = PIANOGenerator(args_ixi, temp_save_fld, U0_path, D_path, V_path = V_path, Phi_path = Phi_path, \
                        VesselMask_path = VesselMask_path, Mask_path = BrainMask_path, device = device) 
                    Generator.get_U(PIANO_Skeleton)'''
                
                #if os.path.isdir(os.path.join(case_fld, 'Movies')):
                #    shutil.rmtree(os.path.join(case_fld, 'Movies'))

                '''elif not os.path.isdir(os.path.join(case_fld, 'Movies')):
                    print('Start processing case NO.%d (of %d): %s\n' % (i+1, len(case_names), case_name))
                    save_fld = make_dir(os.path.join(case_fld, 'Movies'))
                    #save_fld = make_dir(os.path.join(case_fld, 'Movies (div_free)'))
                    get_movie(args_ixi, save_fld, D_path, V_path, U0_path, BrainMask_path, Path4Mask, device)
                else:
                    print('Skip finished case NO.%d (of %d): %s\n' % (i + 1, len(case_names), case_name))'''
                
            except:
                print('\n---Failed case: %s\n' % case_name)


##############################################################################################################################
if __name__ == '__main__':

    main()
 