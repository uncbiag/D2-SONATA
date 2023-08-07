import os, sys, argparse 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import torch
from torch.autograd import Variable

from utils import *
from ODE.adjoint import odeint_adjoint as odeint
from Learning.Modules.AdvDiffPDE import FlowVPartial, PIANO_Skeleton, PIANO_FlowV


main_fld = '/playpen-raid2/peirong/Data/Allen/2d-sim'
#V_fld = make_dir(os.path.join(main_fld, 'V'))


n_batch = 1

nT = 30
dt = 0.05 # 0.01, 0.05

x0, y0 = [-6., -4.] # TODO
data_dim = [64, 64] #, 64]
data_spacing = [0.2, 0.2] #, 1.]

##########################################

#%% Basic settings
parser = argparse.ArgumentParser('2D FlowV - Simulation')
parser.add_argument('--V_time', type = bool, default = True)
parser.add_argument('--adjoint', type = bool, default = True) 
parser.add_argument('--stochastic', type = bool, default = False)
parser.add_argument('--dt', type = float, default = dt, help = 'time interval unit') # 0.01 , 0.02
parser.add_argument('--BC', type = str, default = None, \
    choices = ['None', 'neumann', 'dirichlet', 'cauchy', 'source', 'source_neumann', 'dirichlet_neumann'])
parser.add_argument('--perf_pattern', type = str, default = 'adv_diff', choices = ['adv_diff', 'adv_only', 'diff_only'])
parser.add_argument('--PD_D_type', type = str, default = 'scalar', \
    choices = ['constant', 'scalar', 'diag', 'full_spectral' 'full_cholesky', 'full_dual', 'full_spectral', 'full_semi_spectral'])
# TODO: note: in 2D, vector_div_free_clebsch == vector_div_free_stream
parser.add_argument('--PD_V_type', type = str, default = 'vector', \
    choices = ['constant', 'vector', 'vector_div_free_clebsch', 'vector_div_free_stream', 'vector_div_free_stream_gauge', 'vector_HHD'])
parser.add_argument('--C_magnitude', type = float, default = 10)
parser.add_argument('--V_magnitude', type = float, default = 1.)
parser.add_argument('--D_magnitude', type = float, default = 0.01)
parser.add_argument('--gpu', type = str, required = True, help = 'Select which gpu to use')
args = parser.parse_args()


device = torch.device('cuda:%s' % str(args.gpu))


def gaussian(data_grid, value, device):
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        out = torch.exp(torch.tensor([[- X[i] ** 2 - Y[j] ** 2 for j in range(len(Y))] for i in range(len(X))], \
            dtype = torch.float, device = device)) * value
    else:
        raise NotImplementedError
    return out

##########################################

FlowV_PDE = FlowVPartial(args, data_spacing, device) 
FlowV_PDE.to(device)

AdvDiff_PDE = PIANO_Skeleton(args, data_spacing, args.perf_pattern, args.PD_D_type, args.PD_V_type, device) 
AdvDiff_PDE.to(device)

##########################################

t_all = torch.from_numpy(np.arange(nT) * dt).float().to(device) # (nT)
X = x0 + np.arange(data_dim[0]) * data_spacing[0]
Y = y0 + np.arange(data_dim[1]) * data_spacing[1]
init_C = Variable(gaussian([X, Y], value = 1., device = device)).unsqueeze(0) * args.C_magnitude # (n_batch = 1, r, c)

#D = Variable((torch.randn(n_batch, data_dim[0], data_dim[1]) + 1.) * 0.5).float().to(device) * args.D_magnitude # D >= 0
D = Variable((torch.ones(data_dim[0], data_dim[1]) + 1.) * 0.5).float().to(device) * args.D_magnitude # D >= 0 
FlowV_PDE.D = D

#init_P = Variable(torch.ones(n_batch, data_dim[0], data_dim[1])).float().to(device) * args.V_magnitude
#init_P = Variable(gaussian([X, Y], value = 1., device = device)).unsqueeze(0) * args.V_magnitude # (n_batch = 1, r, c)
init_Px = Variable(torch.stack([torch.arange(0, data_dim[1])] * data_dim[0], dim = 0)).permute(1, 0).float().to(device) * args.V_magnitude # Make Vx != 0
init_Py = Variable(torch.stack([torch.arange(0, data_dim[0])] * data_dim[1], dim = 0)).float().to(device) * args.V_magnitude # Make Vy != 0 

init_P = init_Px
init_P = init_Py
init_P = (init_Px + init_Py) * 0.5

init_Vlst = FlowV_PDE.get_Vlst(init_P.unsqueeze(0))

#####################################
## C(t) with time-INDEPENDENT V, D ##
#####################################
AdvDiff_PDE.Dlst = {'D': D.unsqueeze(0)}
AdvDiff_PDE.Vlst =  init_Vlst
C_orig_series = odeint(AdvDiff_PDE, init_C, t_all, method = 'dopri5', options = args) 

#####################################
##  C(t) with time-DEPENDENT V, D  ##
#####################################
Flow_PDE = PIANO_FlowV(args, t_all, data_dim, data_spacing, device, D_param_lst = {'D': D}, V_param_lst = {'P0': init_P})
Flow_PDE.to(device)

P_series = Flow_PDE.get_P_series(t_all) # get P at all time points
C_series = torch.stack([init_C] * nT, dim = 0) # (nT, n_batch, r, c)
for nt in range(1, nT):
    Flow_PDE.it = nt-1 # update current time point # 
    C_series[nt] = odeint(Flow_PDE, C_series[nt-1], t_all[:2], method = 'dopri5', options = args)[-1]
    #print(C_test_series[nt].mean().item())  

#####################################
############# Save all ##############
#####################################
D_nda = D
nda2img(D_nda.cpu().numpy(), save_path = os.path.join(main_fld, 'D.nii'))

P0_nda = init_P
nda2img(Flow_PDE.P0.detach().cpu().numpy(), save_path = os.path.join(main_fld, 'P0.nii'))

Flow_PDE.get_V_series(P_series, save_fld = main_fld)

P_nda = P_series[:, 0].detach().cpu().numpy() # (nT, n_batch=1, r, c) -> (nT, r, c)
nda2img(P_nda, save_path = os.path.join(main_fld, 'P.nii'))
nda2img(P_nda[0], save_path = os.path.join(main_fld, 'P0.nii'))


C_nda = C_series[:, 0].detach().cpu().numpy() # (nT, n_batch=1, r, c) -> (nT, r, c)
nda2img(C_nda, save_path = os.path.join(main_fld, 'C.nii'))

C_orig_nda = C_orig_series[:, 0].cpu().numpy() # (nT, n_batch=1, r, c) -> (nT, r, c)
nda2img(C_orig_nda, save_path = os.path.join(main_fld, 'C_orig.nii'))

C_diff = abs(C_orig_series - C_series).mean().item()
print('Abs. Difference:', C_diff)

