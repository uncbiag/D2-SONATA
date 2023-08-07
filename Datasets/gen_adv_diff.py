import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import abc
import torch
import numpy as np
import SimpleITK as sitk

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


#%% Basic settings
parser = argparse.ArgumentParser('2D demo: anisotropic flying points')
parser.add_argument('--case', type = str, default = 'demo_2D')
parser.add_argument('--adjoint', type = bool, default = False)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--is_resume', type=bool, default = False)
parser.add_argument('--integ_method', type = str, default = 'dopri5', choices=['dopri5', 'adams', 'rk4', 'euler'])

# For ground truth types

init_loc = [-10, -10] # [-20, -20] 
GT_perf_pattern = 'adv_only' # 'mix', 'circle_adv_only', 'adv_diff', 'adv_only', 'diff_only' 

parser.add_argument('--init_loc', type = list, default = init_loc)
parser.add_argument('--GT_perf_pattern', type = str, default = GT_perf_pattern, choices = ['mix', 'circle_adv_only', 'adv_diff', 'adv_only', 'diff_only'])

parser.add_argument('--t0', type = float, default = 0.0) # the beginning time point
parser.add_argument('--dt', type = float, default = 0.02) # Real dt between scanned images
# NOTE: total time pointsc := collocation_len * n_collocations
parser.add_argument('--sub_collocation_nt', type = int, default = 2, help = 'time points between two sub_collocation points (ode steps)') # 10
parser.add_argument('--collocation_len', type = int, default = 1, help = '# of sub_collocations between two collocation points collocation_nt / sub_collocation_nt') 
parser.add_argument('--n_collocations', type = int, default = 40, help = 'save on only collocation points')

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
    def __init__(self, args, device = 'cpu', save_fld = None, to_save = True):
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
    
    @property
    def PerfFlag(self): # has sign
        isPerf = PerfFlags[self.perf_pattern](self.data_dim, self.D_type, self.V_type, self.device) 
        for key in isPerf:
            self.save(isPerf[key], key)
        return isPerf

    def get_scalar(self, name, value, pattern):
        x = DVs[pattern]([self.X, self.Y], value, self.device) 
        if self.to_save:
            self.save(x, '%s' % name)
        return x  

    @property
    def U(self):
        if self.D_type == 'full_spectral': 
            return cayley_map(self.Slst['S']) # (r, c, 2, 2) 
        else:
            raise NotImplementedError

    @property
    def Ulst(self): 
        return {'Uxx': self.U[..., 0, 0],'Uxy': self.U[..., 0, 1], 'Uyx': self.U[..., 1, 0], 'Uyy': self.U[..., 1, 1]}

    @property
    def Dlst(self):
        D_lst = []
        if 'adv_only' in self.perf_pattern:
            pass
        elif self.D_type == 'constant' or self.D_type == 'scalar':
            return {'D': self.get_scalar('D', self.D_magnitude['D'], self.D_pattern['D'])} 
        elif self.D_type == 'diag':
            return {'Dxx': self.get_scalar('Dxx', self.D_magnitude['Dxx'], self.D_pattern['Dxx']),\
                'Dyy': self.get_scalar('Dyy', self.D_magnitude['Dyy'], self.D_pattern['Dyy'])} 
        elif 'spectral' in self.D_type:
            U = self.U
            if self.to_save:
                self.save(U.view(U.size(0), U.size(1), 4).permute(2, 0, 1), 'U')
            D = construct_spectralD_2D(U, torch.stack([self.Llst['L1'], self.Llst['L2']], dim = 0), batched = False) # (3, r, c): Dxx, Dxy, Dyy
            if self.to_save:
                self.save(D, 'D')
            return {'Dxx': D[0], 'Dxy': D[1], 'Dyy': D[2]}
        else:
            D_lst = {'Dxx': self.get_scalar('Dxx', self.D_magnitude['Dxx'], self.D_pattern['Dxx']) * self.PerfFlag['isDxx'], \
                'Dyy': self.get_scalar('Dyy', self.D_magnitude['Dyy'], self.D_pattern['Dyy']) * self.PerfFlag['isDyy'], \
                    'Dxy': self.get_scalar('Dxy', self.D_magnitude['Dxy'], self.D_pattern['Dxy']) * self.PerfFlag['isDxy']}
            return D_lst
    
    @property
    def Vlst(self):
        V_lst = []
        if 'diff_only' in self.perf_pattern:
            pass
        else:
            Vx_path, Vy_path = os.path.join(self.save_fld, 'Vx.nii'), os.path.join(self.save_fld, 'Vy.nii')
            Vx_img, Vy_img = sitk.ReadImage(Vx_path), sitk.ReadImage(Vy_path)

            return {'Vx': sitk.GetArrayFromImage(Vx_img), \
                'Vy': sitk.GetArrayFromImage(Vy_img)} 
    
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
            movie = movie / max(list(abs(U).reshape(-1)))
        if self.to_save:
            self.save(movie, 'Perf')
        return movie # (nT, r, c)
    
    def save(self, tensor, name, isVector = None):
        s_name = os.path.join(self.save_fld, '%s.nii') % (name)
        #print('Save ground truth data in %s' % s_fld)
        save_sitk(tensor, s_name, self.origin, self.spacing, isVector = isVector) # (n_batch, ...) -> (...)
        return


def generate(args, save_fld, to_save = True):
    args_demo.GT_perf_pattern = 'adv_only'
    args_demo.V_magnitude = {'Vx': 10, 'Vy': 10}
    Generator = AniPointsGenerator(args, device, save_fld = save_fld, to_save = to_save)
    GT_u = Generator.get_movie(PIANO_Skeleton, ToNormalize = False)
    
        



##############################################################################################################################
if __name__ == '__main__':

    on_server = False
    if on_server:
        main_dir = '/playpen-raid1/peirong/Data/2d_Demo'
    else: 
        main_dir = make_dir('/home/peirong/Downloads/flow2') 
    #save_dir = os.path.join(main_dir, 'adv_diff_LU')
    save_dir = make_dir(os.path.join(main_dir, args_demo.GT_perf_pattern))

    generate(args_demo, save_dir, to_save = True)

    #for i in range(50):
    #    randn_generate(make_dir(os.path.join(save_dir, str(i))), to_save = True)
