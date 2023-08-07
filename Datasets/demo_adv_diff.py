import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import abc
import torch
import numpy as np
#import SimpleITK as sitk

from ODE.odeint import odeint
from Learning.Modules.AdvDiffPDE import PIANO_Skeleton
from utils import make_dir, save_sitk, cayley_map, construct_spectralD_2D

from DemoOptions.U0 import *
from DemoOptions.DV import *
from DemoOptions.PerfFlag import *

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
from DemoOptions.ValueMask import gaussian
ValueMasks = {
    'gaussian': gaussian
}


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


#%% Basic settings
parser = argparse.ArgumentParser('2D demo: anisotropic flying points')
parser.add_argument('--case', type = str, default = 'demo_2D')
parser.add_argument('--adjoint', type = bool, default = False)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--is_resume', type=bool, default = False)
parser.add_argument('--integ_method', type = str, default = 'dopri5', choices=['dopri5', 'adams', 'rk4', 'euler'])

# Set as False for samples generation #
parser.add_argument('--stochastic', type=bool, default = False)
parser.add_argument('--separate_stochastic', type = bool, default = False, help = 'Model different stochastic Sigma for D and V')

# For random value mask adding #
parser.add_argument('--mask_pattern', type = str, default = 'gaussian', choices = ['gaussian'])

parser.add_argument('--lesion_mask', type = bool, default = True, help = 'determines whether adding random-sized value mask for (stochastic) uncertainty supervision')
parser.add_argument('--mask_portion', type = float, default = 1., help = '\in [0, 1]: percentage of mask-added samples if lesion_mask == True') 


## TODO: under testing
parser.add_argument('--separate_DV_mask_intensity', type = bool, default = True, help = 'D and V share the same lesion mask, but with different value mask intensity')
parser.add_argument('--test_separate_DV_mask_intensity', type = bool, default = True, help = 'D and V share the same lesion mask, but with different value mask intensity')
parser.add_argument('--V_time', type = bool, default = False)



parser.add_argument('--test_lesion_mask', type = bool, default = True, help = 'determines whether adding random-sized value mask during testing sample generation')
parser.add_argument('--test_mask_portion', type = float, default = 1., help = '\in [0, 1]: [For testing samples only] percentage of mask-added samples') 

# NOTE: archived: not work well #
parser.add_argument('--separate_DV_mask', type = bool, default = False, help = 'D and V share the same mask or not')
parser.add_argument('--test_separate_DV_mask', type = bool, default = False, help = '[For testing samples only] D and V share the same mask or not')
parser.add_argument('--save_as_addition', type = bool, default = False, help = 'Whether save V as V = \bar{V} + \delta_V') 


# For ground truth types

init_loc = [-10, -10] # [-20, -20] 
GT_perf_pattern = 'adv_diff' # 'mix', 'circle_adv_only', 'adv_diff', 'adv_only', 'diff_only' 

parser.add_argument('--init_loc', type = list, default = init_loc)
parser.add_argument('--GT_perf_pattern', type = str, default = GT_perf_pattern, choices = ['mix', 'circle_adv_only', 'adv_diff', 'adv_only', 'diff_only'])

parser.add_argument('--t0', type = float, default = 0.0) # the beginning time point
parser.add_argument('--dt', type = float, default = 0.01) # Real dt between scanned images # NOTE default = 0.01
# NOTE: total time pointsc := collocation_len * n_collocations
parser.add_argument('--sub_collocation_nt', type = int, default = 2, help = 'time points between two sub_collocation points (ode steps)') # 10
parser.add_argument('--collocation_len', type = int, default = 1, help = '# of sub_collocations between two collocation points collocation_nt / sub_collocation_nt') 
parser.add_argument('--n_collocations', type = int, default = 40, help = 'save on only collocation points') # NOTE default = 40 (for samples: 200)

parser.add_argument('--GT_data_dim', type = list, default = [64, 64]) # (151, 151), (128, 128)
parser.add_argument('--data_spacing', type = list, default = [0.5, 0.5])  # (dx, dy): (0.1, 0.1)

parser.add_argument('--U0_pattern', type = str, default = 'gaussian', choices = ['tophat_corner', 'tophat_top', 'tophat_center', 'gaussian'])
parser.add_argument('--init_value', type = float, default = 1., help = 'Initial max value for tophat-type U0') 

parser.add_argument('--GT_D_type', type = str, default = 'full_spectral', choices = ['constant', 'scalar', 'diag', 'full_cholesky', 'full_soectral'])
parser.add_argument('--D_pattern', type = dict, default = {'D': 'constant', 'Dxx': 'constant', 'Dyy': 'constant', 'Dxy': 'constant'}, choices = ['constant', 'gaussian'])
# NOTE: Dxx: stretch across rows; Dyy: stretch across columns #
#parser.add_argument('--D_magnitude', type = dict, default = {'D': 0.1, 'Dxx': 0.1, 'Dxy': 0.02, 'Dyy': 0.02})
parser.add_argument('--D_magnitude', type = dict, default = {'D': 0.1, 'Dxx': 1, 'Dxy': 0.1, 'Dyy': 0.1})

parser.add_argument('--GT_L_type', type = str, default = 'constant_diag', choices = ['constant', 'constant_diag', 'scalar', 'diag'])
parser.add_argument('--L_pattern', type = dict, default = {'L': 'constant', 'L1': 'constant', 'L2': 'constant'}, choices = ['constant', 'gaussian']) 
#parser.add_argument('--L_magnitude', type = dict, default = {'L': 0.1, 'L1': 2, 'L2': 0.1}) # L1 >= L2
parser.add_argument('--L_magnitude', type = dict, default = {'L': 0.1, 'L1': 1., 'L2': 1.2}) # L1 >= L2 # NOTE: temp

parser.add_argument('--GT_S_type', type = str, default = 'constant', choices = ['constant', 'scalar'])
parser.add_argument('--S_pattern', type = dict, default = {'S': 'constant'}, choices = ['constant', 'gaussian']) 
# NOTE: S > 0: clock-wise rotation of principal axes; S < 0: anti-clock-wise rotation of principal axes #
#parser.add_argument('--S_magnitude', type = dict, default = {'S': -0.5}, help = "S should be chosen within (-1, 1)")
parser.add_argument('--S_magnitude', type = dict, default = {'S': 0.8}, help = "S should be chosen within (-1, 1)") # TODO: temp

parser.add_argument('--GT_V_type', type = str, default = 'vector', choices = ['constant', 'scalar_solenoidal', 'scalar', 'vector_solenoidal', 'vector', 'div_free'])
parser.add_argument('--V_pattern', type = dict, default = {'V': 'constant', 'Vx': 'constant', 'Vy': 'constant'}, choices = ['constant', 'gaussian'])
parser.add_argument('--V_magnitude', type = float, default = {'V': 1, 'Vx': 5, 'Vy': 15}) # 1.25


#parser.add_argument('--time_stepsize', type = float, default = 1e-2, help = 'Time step for integration, \
#    if is not None, the assigned value should be able to divide args.time_spacing')
parser.add_argument('--BC', type = str, default = 'neumann', choices = [None, 'neumann'])

args_demo = parser.parse_args()  


class AniPointsGenerator(object):
    '''
    For 2D Demo
    '''
    def __init__(self, args, device = 'cpu', save_fld = None, to_save = True, for_test = False):
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
        self.device = device 
        self.to_save = to_save
        self.save_fld = save_fld 
        self.t0, self.x0, self.y0 = args.t0, args.init_loc[0], args.init_loc[1]
        self.data_dim, self.data_spacing = args.GT_data_dim, args.data_spacing
        self.nX, self.nY = args.GT_data_dim[0], args.GT_data_dim[1]
        self.dx, self.dy = args.data_spacing[0], args.data_spacing[1]
        self.t0, self.dt, self.n_collocations, self.collocation_len, self.sub_collocation_nt = \
            args.t0, args.dt, args.n_collocations, args.collocation_len, args.sub_collocation_nt
        self.X = self.x0 + np.arange(self.nX) * self.dx
        self.Y = self.y0 + np.arange(self.nY) * self.dy
        self.perf_pattern = args.GT_perf_pattern
        self.V_type, self.V_pattern, self.V_magnitude = args.GT_V_type, args.V_pattern, args.V_magnitude 
        self.D_type, self.D_pattern, self.D_magnitude = args.GT_D_type, args.D_pattern, args.D_magnitude
        self.S_type, self.S_pattern, self.S_magnitude = args.GT_S_type, args.S_pattern, args.S_magnitude
        self.L_type, self.L_pattern, self.L_magnitude = args.GT_L_type, args.L_pattern, args.L_magnitude

        self.configure_value_mask(for_test)

    @property
    def origin(self):
        return (self.x0, self.y0, self.t0)

    @property
    def spacing(self):
        return (self.dx, self.dy, self.dt)
    
    @property 
    def Perf_Info(self):
        if self.D_type == 'constant' or self.D_type == 'scalar':
            D_info = 'D: %s' % (self.D_pattern['D'])
        elif self.D_type == 'diag':
            D_info = 'Dxx: %s, Dyy: %s' %(self.D_pattern['Dxx'], self.D_pattern['Dyy'])
        elif 'full' in self.D_type:
            D_info = 'Dxx: %s, Dyy: %s, Dxy: %s' % (self.D_pattern['Dxx'], self.D_pattern['Dyy'], self.D_pattern['Dxy'])

        if self.V_type == 'constant' or self.V_type == 'scalar':
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
    def T_subcollocation(self):
        '''
        T between two collocation points
        '''
        t = self.t0 + np.arange(self.sub_collocation_nt) * self.dt
        return torch.tensor(t, dtype = torch.float, device = self.device)

    @property
    def U0(self):
        U0 = U0s[self.args.U0_pattern]([self.X, self.Y], self.args.init_value, self.device)
        return U0.expand(1, -1, -1) # (n_batch == 1, nX, nY)
    
    '''
    * Archived *
    @property
    def PerfFlag(self): # has sign
        isPerf = PerfFlags[self.perf_pattern](self.data_dim, self.D_type, self.V_type, self.device) 
        for key in isPerf:
            self.save(isPerf[key], key)
        return isPerf''' 

    def get_scalar(self, name, value, pattern):
        x = DVs[pattern]([self.X, self.Y], value, self.device) 
        if self.to_save:
            self.save(x, '%s' % name)
        return x 

    @property
    def Llst(self):
        Llst = {'L1': self.get_scalar('L1', self.L_magnitude['L1'], self.L_pattern['L1']), 'L2': self.get_scalar('L2', self.L_magnitude['L2'], self.L_pattern['L2'])}
        # Update L according to the value mask
        if self.args.save_as_addition:
            Llst.update({'orig_L1': Llst['L1'], 'orig_L2': Llst['L2']})
            Llst.update({'delta_L1': - Llst['L1'] * (1 - self.ValueMask_D), 'delta_L2': - Llst['L2'] * (1 - self.ValueMask_D)}) # NOTE: eva >= 0, while delta_L >= 0 
        Llst['L1'], Llst['L2'] = Llst['L1'] * self.ValueMask_D, Llst['L2'] * self.ValueMask_D
        if self.to_save:
            for key in Llst.keys():
                self.save(Llst[key], key)
        if self.D_type == 'full_spectral':
            return Llst
        else:
            raise NotImplementedError

    @property
    def Slst(self):
        if self.D_type == 'full_spectral':
            return {'S': self.get_scalar('S', self.S_magnitude['S'], self.S_pattern['S'])} 
        else:
            raise NotImplementedError

    @property
    def U(self):
        if self.D_type == 'full_spectral': 
            return cayley_map(self.Slst['S']) # (r, c, 2, 2) 
        else:
            raise NotImplementedError

    @property
    def Ulst(self): 
        return {'Uxx': self.U[..., 0, 0],'Uxy': self.U[..., 0, 1], 'Uyx': self.U[..., 1, 0], 'Uyy': self.U[..., 1, 1]}
    
    
    def get_random_mask(self, ctr_lst = None, r_lst = None):
        if ctr_lst is None:
            mask_x0 = self.x0 + (np.random.random_sample() * 0.5 + 0.2) * self.nX * self.dx  # NOTE: random.random_sample() \in [0., 1.]
            mask_y0 = self.y0 + (np.random.random_sample() * 0.5 + 0.2) * self.nY * self.dy

            mask_rx = (np.random.random_sample() * 0.3 + 0.1) * self.nX * self.dx
            mask_ry = (np.random.random_sample() * 0.3 + 0.1) * self.nY * self.dy
        else: # assign fixed mask centers and radius
            mask_x0, mask_y0 = ctr_lst
            mask_rx, mask_ry = r_lst 
        if np.random.random_sample() <= 0.5: # strength > 0: anomaly < 1; strenth < 0: anomaly > 1; strength = 0: normal.
            strength = np.random.random_sample() #1. - np.random.random_sample() * 0.6 # determine the lesion mask center value 
        else:
            strength = - np.random.random_sample() * 0.

        #print(' Applied value mask: center = [%.1f, %.1f], radius = [%.1f, %.1f]' % (mask_x0, mask_y0, mask_rx, mask_ry)) # For checking
        
        return ValueMasks[self.args.mask_pattern]([self.X, self.Y], [mask_x0, mask_y0], [mask_rx, mask_ry], strength, self.device), [mask_x0, mask_y0], [mask_rx, mask_ry]

    def configure_value_mask(self, for_test = False):
        
        def get_mask(ctr_lst = None, r_lst = None):
            ValueMask = torch.ones(self.data_dim, dtype = torch.float, device = self.device)
            if for_test and self.args.test_lesion_mask:
                if np.random.random_sample() <= self.args.test_mask_portion:
                    ValueMask, ctr_lst, r_lst = self.get_random_mask(ctr_lst = ctr_lst, r_lst = r_lst)
            elif self.args.lesion_mask:
                if np.random.random_sample() <= self.args.mask_portion:
                    ValueMask, ctr_lst, r_lst = self.get_random_mask(ctr_lst = ctr_lst, r_lst = r_lst)
           # ValueMask = torch.where(ValueMask < 0.8, ValueMask, torch.ones_like(ValueMask)) # large than 0.8 => normal
            UncertaintyMask = abs(1. - ValueMask)
            return UncertaintyMask, ValueMask, ctr_lst, r_lst

        UncertaintyMask, ValueMask, mask_ctr_lst, mask_r_lst = get_mask(ctr_lst = None, r_lst = None)
        self.UncertaintyMask_D, self.ValueMask_D = UncertaintyMask, ValueMask
        if (self.args.separate_DV_mask and not for_test) or (self.args.test_separate_DV_mask and for_test):
            self.UncertaintyMask_V, self.ValueMask_V, _, _ = get_mask(ctr_lst = None, r_lst = None)
        else:
            if (self.args.separate_DV_mask_intensity and not for_test) or (self.args.test_separate_DV_mask_intensity and for_test):
                #print('assign new value mask intensity')
                self.UncertaintyMask_V, self.ValueMask_V, _, _ = get_mask(ctr_lst = mask_ctr_lst, r_lst = mask_r_lst)
            else:
                self.UncertaintyMask_V, self.ValueMask_V = UncertaintyMask, ValueMask 
        self.LesionSegMask_D = (self.ValueMask_D < 0.8).float() + (self.ValueMask_D > 1.2).float() # large than 0.8 and smaller than 1.2 => normal
        self.LesionSegMask_V = (self.ValueMask_V < 0.8).float() + (self.ValueMask_V > 1.2).float() # large than 0.8 and smaller than 1.2 => normal
        
        # TODO: TESTING: set the shared uncertainty mask as the larger ones between UncertaintyMasks of D and V 
        if (self.UncertaintyMask_D - self.UncertaintyMask_V).mean() < 0.: 
            self.UncertaintyMask = self.UncertaintyMask_V
        else:
            self.UncertaintyMask = self.UncertaintyMask_D
        
        if self.to_save:
            self.save(self.ValueMask_D, 'ValueMask_D')
            self.save(self.ValueMask_V, 'ValueMask_V')
            self.save(self.LesionSegMask_D, 'LesionSegMask_D')
            self.save(self.LesionSegMask_V, 'LesionSegMask_V')
            self.save(self.UncertaintyMask, 'Uncertainty')
        return
    
    @property
    def Dlst(self):
        Dlst = {}
        if 'adv_only' in self.perf_pattern:
            return
        if self.D_type == 'constant' or self.D_type == 'scalar':
            D = self.get_scalar('D', self.D_magnitude['D'], self.D_pattern['D'])
            if self.args.save_as_addition:
                delta_D = D * self.ValueMask_D - D
                Dlst.update({'orig_D': D.unsqueeze(0), 'delta_D': delta_D.unsqueeze(0)})
            Dlst.update({'D': (D * self.ValueMask_D).unsqueeze(0)}) 
        elif self.D_type == 'diag':
            Dxx = self.get_scalar('Dxx', self.D_magnitude['Dxx'], self.D_pattern['Dxx'])
            Dyy = self.get_scalar('Dyy', self.D_magnitude['Dyy'], self.D_pattern['Dyy']) 
            if self.args.save_as_addition:
                delta_Dxx = Dxx * self.ValueMask_D - Dxx
                delta_Dyy = Dyy * self.ValueMask_D - Dyy
                Dlst.update({'orig_Dxx': Dxx.unsqueeze(0), 'orig_Dyy': Dyy.unsqueeze(0), 'delta_Dxx': delta_Dxx.unsqueeze(0), 'delta_Dyy': delta_Dyy.unsqueeze(0)})
            Dlst.update({'Dxx': (Dxx * self.ValueMask_D).unsqueeze(0), 'Dyy': (Dyy * self.ValueMask_D).unsqueeze(0)}) 
        elif 'spectral' in self.D_type:
            U = self.U
            if self.to_save:
                self.save(U.view(U.size(0), U.size(1), 4).permute(2, 0, 1), 'U')
            if self.args.save_as_addition:
                D = construct_spectralD_2D(U, torch.stack([self.Llst['orig_L1'], self.Llst['orig_L2']], dim = 0), batched = False) # (3, r, c): Dxx, Dxy, Dyy
            else:
                D = construct_spectralD_2D(U, torch.stack([self.Llst['L1'], self.Llst['L2']], dim = 0), batched = False) # (3, r, c): Dxx, Dxy, Dyy
            if self.args.save_as_addition:
                delta_D = D * self.ValueMask_D - D
                if self.to_save:
                    self.save(D, 'orig_D')
                    self.save(delta_D, 'delta_D')
                Dlst.update({'orig_Dxx': D[0].unsqueeze(0), 'orig_Dxy': D[1].unsqueeze(0), 'orig_Dyy': D[2].unsqueeze(0), \
                    'delta_Dxx': delta_D[0].unsqueeze(0), 'delta_Dxy': delta_D[1].unsqueeze(0), 'delta_Dyy': delta_D[2].unsqueeze(0)})
            
            D = D * self.ValueMask_D
            if self.to_save:
                self.save(D, 'D')
            Dlst.update({'Dxx': D[0].unsqueeze(0), 'Dxy': D[1].unsqueeze(0), 'Dyy': D[2].unsqueeze(0)})
        else:
            raise ValueError('Not supported D type:', self.D_type)
            '''Dlst = {'Dxx': self.get_scalar('Dxx', self.D_magnitude['Dxx'], self.D_pattern['Dxx']) * self.ValueMask_D, \
                'Dyy': self.get_scalar('Dyy', self.D_magnitude['Dyy'], self.D_pattern['Dyy']) * self.ValueMask_D, \
                    'Dxy': self.get_scalar('Dxy', self.D_magnitude['Dxy'], self.D_pattern['Dxy']) * self.ValueMask_D}
            return Dlst'''
        return Dlst
    
    @property
    def Vlst(self):
        Vlst = {}
        if 'diff_only' in self.perf_pattern:
            pass
        elif self.V_type == 'constant' or self.V_type == 'scalar':
            V = self.get_scalar('V', self.V_magnitude['V'], self.V_pattern['V'])
            if self.args.save_as_addition:
                delta_V = V * self.ValueMask_V - V
                Vlst.update({'orig_V': V, 'delta_V': delta_V})
            Vlst.update({'V': (V * self.ValueMask_V).unsqueeze(0)}) # (n_batch=1, r, c)
        elif 'vector' in self.V_type:
            Vx = self.get_scalar('Vx', self.V_magnitude['Vx'], self.V_pattern['Vx']) * self.ValueMask_V
            Vy = self.get_scalar('Vy', self.V_magnitude['Vy'], self.V_pattern['Vy']) * self.ValueMask_V
            if self.args.save_as_addition:
                delta_Vx = Vx * self.ValueMask_V - Vx
                delta_Vy = Vy * self.ValueMask_V - Vy
                if self.to_save:
                    self.save(torch.stack([Vx, Vy], dim = 0), 'orig_V')
                    self.save(torch.stack([delta_Vx, delta_Vy], dim = 0), 'delta_V')
                Vlst.update({'orig_Vx': Vx.unsqueeze(0), 'orig_Vy': Vy.unsqueeze(0), 'delta_Vx': delta_Vx.unsqueeze(0), 'delta_Vy': delta_Vy.unsqueeze(0)}) 
            Vx = Vx * self.ValueMask_V
            Vy = Vy * self.ValueMask_V
            if self.to_save:
                self.save(torch.stack([Vx, Vy], dim = 0), 'V') 
            Vlst.update({'Vx': Vx.unsqueeze(0), 'Vy': Vy.unsqueeze(0)})
        return Vlst 
    
    def integrate(self, PDEFunc, to_print = True):
        U = torch.stack([self.U0] * self.n_collocations, dim = 1) # (n_batch = 1, n_collocations, r, c)
        # Integrate by collocations #
        with torch.no_grad():
            for i_coll in range(1, self.n_collocations):
                U[:, i_coll] = U[:, i_coll - 1]  # (n_batch = 1, r, c)
                for i_sub_coll in range(self.collocation_len): # (nt = -1, batch = 1, slc, row, col)
                    U[:, i_coll] = odeint(PDEFunc, U[:, i_coll], self.T_subcollocation, method = self.args.integ_method, options = self.args)[-1]
                if to_print:
                    print('   - Avg. value at time {:03d}: {:.6f}'.format(i_coll, U[:, i_coll].mean().item()))
        U = U[0] # (nT, r, c)
        if to_print:
            print('   - Domain-averaged value:  {:.6f}'.format(U.mean().item()))
        return U # (nT, r, c)

    def get_movie(self, PDEFunc, ToNormalize = True, to_print = True):
        PDEFunc = PDEFunc(self.args, self.data_spacing, self.perf_pattern, self.D_type, self.V_type, self.device)
        PDEFunc.Dlst, PDEFunc.Vlst = self.Dlst, self.Vlst

        movie = self.integrate(PDEFunc, to_print = to_print)
        
        if ToNormalize:
            movie = movie / max(abs(movie[0]))
        if self.to_save:
            self.save(movie, 'Perf')
        return movie # (nT, r, c)
    
    def save(self, tensor, name, isVector = None):
        s_name = os.path.join(self.save_fld, '%s.nii') % (name)
        #print('Save ground truth data in %s' % s_fld)
        save_sitk(tensor, s_name, self.origin, self.spacing, isVector = isVector) # (n_batch, ...) -> (...)
        return
 

def randn_generate(save_fld = None, to_save = False, to_print = True, for_test = False, device = 'cpu'):   
    args_demo.GT_perf_pattern = 'adv_diff'
    args_demo.n_collocations = 20
    args_demo.init_loc = [np.random.uniform(-20., -10, 1).astype(float)[0], np.random.uniform(-20., -10, 1).astype(float)[0]]
    args_demo.S_magnitude['S'] = np.random.uniform(-1., 1., 1).astype(float)[0]
    L1 = np.random.uniform(0., 2, 1).astype(float)[0]
    #args_demo.L_magnitude = {'L1': L1, 'L2': np.random.uniform(0., L1, 1).astype(float)[0]}
    #args_demo.V_magnitude = {'Vx': np.random.uniform(-15, 15, 1).astype(float)[0], 'Vy': np.random.uniform(-15, 15, 1).astype(float)[0]}
    Generator = AniPointsGenerator(args_demo, device, save_fld = save_fld, to_save = to_save, for_test = for_test)
    movie = Generator.get_movie(PIANO_Skeleton, ToNormalize = False, to_print = to_print)
    param_lst = {'Dlst': Generator.Dlst, 'Llst': Generator.Llst, 'Ulst': Generator.Ulst, 'Slst': Generator.Slst, 'Vlst': Generator.Vlst, \
        'Sigma_V': Generator.UncertaintyMask_V, 'Sigma_D': Generator.UncertaintyMask_D, 'Sigma': Generator.UncertaintyMask, 'Seg_V': Generator.LesionSegMask_V, 'Seg_D': Generator.LesionSegMask_D, \
            'ValueMask_D': Generator.ValueMask_D, 'ValueMask_V': Generator.ValueMask_V}
    return movie, param_lst, Generator.origin, Generator.spacing # (nT, r, c)


def fix_generate(args, save_fld, to_save = True, start_i = 0, for_test = False):
    S_magnitudes = [-1]
    #S_magnitudes = [-1, -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.] 
    for i in range(len(S_magnitudes)):
        temp_fld = make_dir(os.path.join(save_fld, str(i + start_i)))
        #for dim in range(len(args.init_loc)):
        #    args.init_loc[dim] += 0.08
        '''args.init_loc[0] = args.init_loc[0] + 0.005 * i
        args.init_loc[1] = args.init_loc[1] + 0.005 * i

        args.velocity['Vx'] = args.velocity['Vx'] - 0.02 * i
        args.velocity['Vy'] = args.velocity['Vx'] - 0.02 * i
        if args.GT_perf_pattern == 'full_spectral':
            args.L_magnitude['L1'] = args.L_magnitude['L1'] + 0.005 * i
            args.L_magnitude['L2'] = args.L_magnitude['L2'] + 0.005 * i
            args.S_magnitude['S']  = args.S_magnitude['S'] + 0.005 * i
        else:
            args.D_magnitude['Dxx'] = args.D_magnitude['Dxx'] + 0.005 * i
            #args.D_magnitude['Dxy'] = args.D_magnitude['Dxy'] + 0.005 * i
            args.D_magnitude['Dyy'] = args.D_magnitude['Dxx'] - 0.005 * i'''
        args.S_magnitude['S'] = S_magnitudes[i]
        Generator = AniPointsGenerator(args, device, save_fld = temp_fld, to_save = to_save, for_test = for_test)
        GT_u = Generator.get_movie(PIANO_Skeleton, ToNormalize = False)    



##############################################################################################################################


if __name__ == '__main__':

    on_server = True
    if on_server:
        main_dir = '/playpen-raid2/peirong/Data/2d_Demo'
    else: 
        main_dir = '/home/peirong/raid2/Data/2d_Demo' 
    #save_dir = os.path.join(main_dir, 'adv_diff_LU')
    save_dir = os.path.join(main_dir, args_demo.GT_perf_pattern)

    n_samples = 5
    for i in range(n_samples):
        fix_generate(args_demo, save_dir, start_i = i, for_test = False)

    #for i in range(50):
    #    randn_generate(make_dir(os.path.join(save_dir, str(i))), to_save = True)

