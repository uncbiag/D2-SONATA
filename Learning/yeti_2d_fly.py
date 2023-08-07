import os, sys, datetime, gc, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
import numpy as np
#import SimpleITK as sitk

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import RcParams
plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
matplotlib.colors.ColorConverter.colors['carolinablue'] = '#7BAFD4'
matplotlib.colors.ColorConverter.colors['darkblue'] = '#00275D'
matplotlib.colors.ColorConverter.colors['lightgrey'] = '#E1E1E1'
matplotlib.colors.ColorConverter.colors['mygreen']= '#DCEDC8'
matplotlib.colors.ColorConverter.colors['myyellow']= '#FFE082'
matplotlib.colors.ColorConverter.colors['myorange']= '#FFAB91'
matplotlib.colors.ColorConverter.colors['myblue']= '#BBDEFB'

import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from utils import *
from Losses.losses import *
#from Learning.Modules.unet2d import *

from Datasets.demo_adv_diff import randn_generate, args_demo # NOTE: use config from args_demo for training samples generation
from Learning.Modules.AdvDiffPDE import PIANOinD, PIANO_Skeleton
from Datasets.dataset_movie import extract_BC_2D, extract_dBC_2D

#device = 'cpu'
#device = torch.device('cuda:0')
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)

'''
Core Options:
patch_data_dim : predict patch data dim
max_down_scales: maximum dowmsampling scales for coarse prediction;
niters_step1   : num_iters for coarser-only training phase;
'''

patch_data_spacing = args_demo.data_spacing # NOTE GT Spacng [0.07, 0.07] #
print('data_spacing: %s' % patch_data_spacing)
#patch_data_spacing = [0.5, 0.5] # NOTE PD Spacng #

sub_collocation_nt = 2
collocation_len = 5
dt = args_demo.dt / collocation_len
print('dt: %s' % dt)


args_demo.mask_portion = 0.6 # 0.5
args_demo.lesion_mask = False # training: sample generation
args_demo.separate_DV_mask_intensity = True # TODO: under testing

args_demo.test_mask_portion = 1.
args_demo.test_lesion_mask = True # testing: using samples with lesion mask
args_demo.test_separate_DV_mask_intensity = True # TODO: under testing


#####################
# NOTE: not working #
args_demo.separate_DV_mask = False # training: generate separate anomaly masks for D and V 
args_demo.test_separate_DV_mask = False # testing: using samples with lesion masks separate for D and DV
#####################


patch_data_dim = [64, 64]
input_time_frame = 10

loss_time_frame = 10
increase_loss_time_frame = False
increase_loss_time_frame_freq = 100
max_loss_time_frame = 10 # NOTE max_loss_time_frame<= input_time_frame #


# Testing settings #
n_test_sample = 10 # 10

lr = 5e-4
test_freq = 500 # 500, 1000
save_test_perf_after_itr = 0

max_dowm_scales = 0
niters_step1 = 100000000

is_resume = True
resume_root_fld = '/playpen-raid2/peirong/Results/2D_Results/adv_diff/same_LV/WR-[N-sepL]' 
ResumeModelPath = os.path.join(resume_root_fld, 'latest_checkpoint.pth')
ResumePaths = {'root': resume_root_fld, 'model': ResumeModelPath}

#%% Basic settings
parser = argparse.ArgumentParser('2D YETI on-the-fly')
parser.add_argument('--gpu', type = str, required = True) ## NOTE: gpu selection ## 
parser.add_argument('--for_test', type = bool, default = False) 
parser.add_argument('--is_resume', type = bool, default = is_resume)
parser.add_argument('--adjoint', type = bool, default = True)
parser.add_argument('--perf_loss', type = str, default = 'abs', choices = {'dyn', 'abs', 'both'})
parser.add_argument('--model_type', type = str, default = 'unet', choices = {'unet', 'vae'})

# Model main configurations #

#####################
## NOTE Under Test ##
#####################
parser.add_argument('--V_time', type = bool, default = False)
#####################

parser.add_argument('--predict_value_mask', type = bool, default = False, help = 'If True, predict V(or D) as V := \bar{V} * value_mask') # Default: True
parser.add_argument('--separate_DV_value_mask', type = bool, default = True, help = 'Whether separately predict anomaly value mask for D and V') # Default: True
parser.add_argument('--actual_physics_loss', type = bool, default = True, help = 'If True, compute loss for base_D * value_mask') # Default: True

parser.add_argument('--stochastic', type = bool, default = False, help = 'Model as normal PDE or stochastic-PDE') # Default: True
parser.add_argument('--stochastic_separate_net', type = bool, default = True, help = 'Predict the stochastic term in a separate net than ') # Default: True


# NOTE: Archieved: not working #
parser.add_argument('--joint_predict', type = bool, default = False, help = 'Whether use joint decoder for V and D')  
parser.add_argument('--value_mask_separate_net', type = bool, default = False, help = 'If True, predict delta_V and delta_D in an independent network')  # NOTE not work well: vm -> 1
parser.add_argument('--vm_sde_net', type = bool, default = False, help = 'sde and value_mask predicted in one separate network') # NOTE: not work well
parser.add_argument('--predict_deviation', type = bool, default = False, help = 'If True, predict V(or D) as V := \bar{V} + \delta{V}')    
parser.add_argument('--deviation_separate_net', type = bool, default = False, help = 'If True, predict delta_V and delta_D in an independent network')     
parser.add_argument('--deviation_extra_weight', type = float, default = 0., help = 'Extra weights for physics deviation supervision (if predict_deviation)') 
parser.add_argument('--predict_segment', type = bool, default = False, help = 'Whether add a separate segmentation network')  
parser.add_argument('--segment_condition_on_physics', type = bool, default = False, help = 'Whether multiply predicted SegMask with delta_out')  
parser.add_argument('--segment_net_type', type = str, default = 'conc', choices = {'conc', 'dev'}, help = 'input choices: conc time-series or physics deviation (must w/ predict_deviation)')  

# TODO: SDE configs
parser.add_argument('--VM_weight', type = float, default = 1., help = 'useful if predict_value_mask == True') 
parser.add_argument('--seg_weight', type = float, default = 1., help = 'useful if predict_segment == True')
parser.add_argument('--SDE_loss', type = bool, default = True)
parser.add_argument('--SDE_weight', type = float, default = 1)
parser.add_argument('--compute_uncertainty', type = int, default = 0) # n_samples (0: not computing uncertainty when testing)

#%% Learning model settings
parser.add_argument('--integ_method', type = str, default = 'dopri5', choices=['dopri5', 'adams', 'rk4', 'euler'])
parser.add_argument('--initial_method', type = str, default = 'cayley', choices = ['henaff', 'cayley'])   

parser.add_argument('--perf_loss_4train', type = bool, default = False)
parser.add_argument('--gradient_loss', type = bool, default = False) # NOTE: not work well for anisotropic diffusion #
parser.add_argument('--gl_weight', type = float, default = 0.0001) # 500 # 1
# Choosing sgl_weight => SG loss ~ Perf loss #
# SGL 0.5 # NOTE:  SGL 0.1 only ==> ill-posed !!! #
# SGL 0.1 + AL 1E-7: NOTE: SGL no need to be too large: otherwise perf_loss hard to reduce #
parser.add_argument('--sgl_weight', type =  float, default = 1.) # NOTE: 0.1 not work! 1 for scalar D, (5 for full_choeslky), 0.5 for full_spectral
parser.add_argument('--spatial_gradient_loss', type = bool, default = True)   


parser.add_argument('--GT_V', type = bool, default = True, help = 'Add loss on V for supervised learning') 
parser.add_argument('--GT_V_weight', type = float, default = 1)  
parser.add_argument('--GT_D', type = bool, default = True, help = 'Add loss on D for supervised learning')
parser.add_argument('--GT_D_weight', type = float, default = 1) #200
parser.add_argument('--GT_LU', type = bool, default = False, help = 'Add loss on L & U for supervised learning') # For spectral 
parser.add_argument('--GT_L_weight', type = float, default = 1) #300
parser.add_argument('--GT_U_weight', type = float, default = 1) #100

parser.add_argument('--GT_Phi', type = bool, default = False, help = 'Add loss on Phi for supervised learning') 
parser.add_argument('--GT_Phi_weight', type = float, default = 1)  


# For ground truth types
parser.add_argument('--data_dim', type = list, default = patch_data_dim) # (64, 64): 62 + 2 (Neumann B.C. padding) = 64
parser.add_argument('--data_spacing', type = list, default = patch_data_spacing) # (dx, dy), (0.04, 0.04), (0.1, 0.1)
parser.add_argument('--max_down_scales', type = int, default = max_dowm_scales) # max down_sampling scales (>= 1)
# NOTE: (1) 2 * boundary_crop_training + stride_testing <= stride_testing; (2) 2 * boundary_crop_training <= data_dim
parser.add_argument('--stride_testing', type = list, default = [8, 8]) # [16, 16]. for Large(0.6). [8, 8] spatial stride for generating testing dataset
#parser.add_argument('--boundary_crop_training', type = list, default = [16, 16]) # [10, 10] for Large(0.6). stride_testing >= 2 * boundary_crop + stride_testing
# NOTE: boundary_crop_training >= 1 for BC == neumann 
# NOTE: boundary_crop_training >= 1 for BC == cauchy
# NOTE: boundary_crop_training >= dirichlet_width for BC == dirichlet
# NOTE: boundary_crop_testing >= boundary_crop_training
parser.add_argument('--boundary_crop_training', type = list, default = [0, 0]) # [4, 4]. stride_testing >= 2 * boundary_crop + stride_testing
parser.add_argument('--boundary_crop_testing', type = list, default = [0, 0]) # [4, 4]. stride_testing >= 2 * boundary_crop + stride_testing
parser.add_argument('--dirichlet_width', type = int, default = 1) # boundaries viewed as source function
parser.add_argument('--source_width', type = int, default = 1) # 1. boundaries viewed as source function

parser.add_argument('--perf_pattern', type = str, default = 'adv_diff', choices = ['adv_diff', 'adv_only', 'diff_only'])
parser.add_argument('--PD_D_type', type = str, default = 'full_spectral', choices = ['constant', 'constant_spectral', 'scalar', 'diag', 'full_cholesky', 'full_symmetric', 'full_spectral'])
# TODO: note: in 2D, vector_div_free_clebsch == vector_div_free_stream 
parser.add_argument('--PD_V_type', type = str, default = 'vector_div_free_stream', choices = ['constant', 'constant_vector', 'vector', 'vector_div_free_clebsch', 'vector_div_free_stream', 'vector_HHD'])
parser.add_argument('--D_magnitude', type = float, default = 1.)
parser.add_argument('--V_magnitude', type = float, default = 1.)
parser.add_argument('--t0', type = float, default = 0.0) # the beginning time point
# NOTE: fake one, used for ensuring numerical stability.
# NOTE: Remember to scale to real-scale afterwards
parser.add_argument('--dt', type = float, default = dt) # NOTE: 0.01 / 0.02 is the BEST 
#parser.add_argument('--time_stepsize', type = float, default = 0.01, help = 'Time step for integration, \
#    if is not None, the assigned value should be able to divide args.time_spacing')

# TODO: B.C. must be 'neumann' for patch-based strategy
# NOTE: 'dirichlet_neumann' == 'cauchy' when dirichlet_width == 1 #
parser.add_argument('--BC', type = str, default = 'cauchy', \
    choices = ['neumann', 'dirichlet', 'cauchy', 'source', 'source_neumann', 'dirichlet_neumann'])
#%% Training and testing settings
parser.add_argument('--n_filter', type = int, default = 64)  
parser.add_argument('--latent_variable_size', type = int, default = 128) # 5000 # For VAE #
parser.add_argument('--sub_collocation_nt', type = int, default = sub_collocation_nt)
parser.add_argument('--collocation_len', type = int, default = collocation_len)

parser.add_argument('--input_time_frame', type = int, default = input_time_frame) # input time_frame as GT_nT for V, D prediction # NOTE: set to super large if want input the entire availabvle time frame
parser.add_argument('--loss_time_frame', type = int, default = loss_time_frame) # 2 
parser.add_argument('--increase_loss_time_frame', type = bool, default = increase_loss_time_frame) #2
parser.add_argument('--increase_loss_time_frame_freq', type = int, default = increase_loss_time_frame_freq) # 25 (scalar D), 50 (full D). TODO: what is the best freq ? 50 for adv_only (20: too fast !!!)
parser.add_argument('--max_loss_time_frame', type = int, default = max_loss_time_frame) # 8, 15

parser.add_argument('--n_test_sample', type = int, default = n_test_sample, help = '# of test samples to generate')
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--opt_type', type = str, default = 'Adam')
parser.add_argument('--lr', type = float, default = lr, help = 'Model learning rate')
parser.add_argument('--lr_weight_decay', type = float, default = 0.) 
parser.add_argument('--max_num_lr_reduce', type = int, default = 2)
parser.add_argument('--lr_reduce_rate', type = int, default = 0.8)
parser.add_argument('--lr_reduce_criterion', type = float, default = 0.2) 
parser.add_argument('--niters_total', type = int, default = 100000000)
parser.add_argument('--niters_adv_only', type = int, default = 0)
parser.add_argument('--niters_step1', type = int, default = niters_step1)
parser.add_argument('--print_freq', type = int, default = 1) # 5, 10, 25
parser.add_argument('--test_freq', type = int, default = test_freq) # 50, 100, 500  
parser.add_argument('--use_stop_crit', type = bool, default = False) 
parser.add_argument('--stop_crit', type = float, default = 0.001) # Loss reduce rate for one iteration
parser.add_argument('--max_stop_count', type = int, default = 1, help  = 'Stop when stop_count >= max_stop_count') 

args = parser.parse_args()  

args_demo.save_as_addition = args.predict_deviation or args.predict_value_mask

def get_movie(input_time_frame, loss_time_frame, loss4perf = False, save_fld = None, to_save = False, for_test = False, device = 'cpu'):
    full_movie, param_lst, origin, spacing = randn_generate(save_fld, to_save, to_print = False, for_test = for_test) # (nT, r, c)
    input_start_t = np.random.randint(0, full_movie.shape[0] - input_time_frame + 1)
    movie4input = full_movie[input_start_t : input_start_t + input_time_frame]
    if for_test:
        return movie4input, full_movie, param_lst, origin, spacing
    else:
        if loss4perf:
            loss_start_t = np.random.randint(0, movie4input.shape[0] - loss_time_frame + 1)
            movie4loss = movie4input[loss_start_t : loss_start_t + loss_time_frame]
            return movie4input, movie4loss, param_lst
        return movie4input, param_lst

def get_max_len(list_of_list):
    # each list: (n_test, n_sample)
    max_len = 0
    len_list = []
    for lst in list_of_list:
        len_list.append(len(lst))
        max_len = max(max_len, len(lst))
    
    for i in range(len(list_of_list)):
        if len(list_of_list[i]) < max_len:
            list_of_list[i] = list_of_list[i] + [list_of_list[i][-1]] * (max_len - len(list_of_list[i])) 
    print(len(list_of_list))
    return list_of_list


def main(args, SaveFld, ResumePaths = None):

    device = torch.device('cuda:%s' % str(args.gpu))

    ###########################  Saving Path  ###########################

    print(args)
    if args.adjoint:
        from ODE.adjoint import odeint_adjoint as odeint
        print('Using adjoint method...')
    else:
        print('Not using adjoint method...')
        from ODE.odeint import odeint

    model_info = '' 

    if args.stochastic and args.predict_value_mask and args.vm_sde_net:
        model_info += '_[sep-SDE-VM'
        if args.actual_physics_loss:
            model_info += '+'
        model_info += ']'
    else:
        if args.stochastic:
            sde_prefix = 'SDE' if not args.stochastic_separate_net else 'sep-SDE'
            if args.SDE_loss:
                model_info += '_%s(%d)' % (sde_prefix, args.SDE_weight)
            else:
                model_info += '_%%' % (sde_prefix)    

        if args.predict_value_mask:
            vm_prefix = 'sep-' if args.value_mask_separate_net else '' # archived: not working #
            model_info = model_info + '_[%sDV-VM' % vm_prefix if args.separate_DV_value_mask else model_info + '_[%sVM' % vm_prefix # TODO: under testing # 
            if args.actual_physics_loss:
                model_info += '+'
            model_info += ']'

    ##############################
    #### Archived Not Working ####
    if args.predict_deviation:
        model_info = model_info + '_[Dev' if not args.deviation_separate_net else model_info + '_[sep-Dev' 
        if args.deviation_extra_weight > 0.:
            model_info += '_%d' % int(args.deviation_extra_weight)
        model_info += ']'

    if args.predict_segment:
        model_info = model_info + '_[free-' if not args.segment_condition_on_physics else model_info + '_[cond-'
        if args.segment_net_type == 'conc':
            model_info += 'ConcSeg]'  
        elif args.segment_net_type == 'dev':
            assert args.predict_deviation
            model_info += 'DevSeg]'
    ##############################
    ##############################

    if args.perf_loss_4train:
        model_info += '_[Perf]'  

    if args_demo.lesion_mask:
        data_prefix = 'sepL' if args_demo.separate_DV_mask_intensity else 'L'
    else:
        data_prefix = 'N'
    if args_demo.test_lesion_mask:
        data_prefix = data_prefix + '-sepL' if args_demo.test_separate_DV_mask_intensity else data_prefix + '-L' 
    else:
        data_prefix += '-N'
    
    model_info = 'WR-[%s]%s' % (data_prefix, model_info)

    n_test_iter = 0

    if args.is_resume:
        print('Resume model_info: ', os.path.basename(ResumePaths['root']))
        print('Current model_info:', model_info)
        print('Resume training from %s' % ResumePaths['model'])
        checkpoint = torch.load(ResumePaths['model'], map_location = device)
        #model_info = 'Res(%s)-%s' % (checkpoint['epoch'], model_info)

        # Re-store test error list
        test_loss_perf_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_C.npy')))
        test_loss_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_D.npy'))) 
        test_loss_L_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_L.npy'))) 
        test_loss_U_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_U.npy'))) 
        test_loss_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_V.npy'))) 
    
        list_of_list = [test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst]
        list_of_list = get_max_len(list_of_list)

        if args.predict_segment:
            test_loss_Seg_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_Seg.npy'))) 
            list_of_list += [test_loss_Seg_lst]

            list_of_list = get_max_len(list_of_list)
            
            list_of_list = get_max_len(list)

        if args.stochastic:
            test_loss_Sigma_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_Sigma.npy'))) 
            list_of_list += [test_loss_Sigma_lst]

            if not args.predict_value_mask:
                list_of_list = get_max_len(list_of_list)
                [test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst, test_loss_Sigma_lst] = list_of_list 
 
        if args.predict_value_mask: 
            test_loss_VM_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_VM_D.npy')))
            test_loss_VM_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'Err_VM_V.npy')))
            list_of_list += [test_loss_VM_D_lst, test_loss_VM_V_lst]
            
            list_of_list = get_max_len(list_of_list)

            if args.stochastic:
                [test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst, test_loss_Sigma_lst, test_loss_VM_D_lst, test_loss_VM_V_lst] = list_of_list
            else:
                [test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst, test_loss_VM_D_lst, test_loss_VM_V_lst] = list_of_list

        epoch_len = len(list_of_list[0])
        for lst in list_of_list:
            assert len(lst) == epoch_len

        n_test_iter = len(test_loss_perf_lst) 
        print(len(test_loss_perf_lst))


    if args.is_resume or args.for_test: 
        assert ResumePaths is not None
        main_fld = ResumePaths['root'] if not args.for_test else make_dir(os.path.join(ResumePaths['root'], '0-Test')) 
    else: 
        main_fld = make_dir(os.path.join(SaveFld, args.perf_pattern, 'same_LV/%s' % (model_info)))
    
    print('Main folder: %s' % main_fld)
    print('PID - %s' % os.getpid())
    
    
    ###########################  V, D Prediction  ###########################

    in_channels = args.input_time_frame    
    print('Input channels:', str(in_channels))
    loss_time_frame = min(args.input_time_frame, args.loss_time_frame)  
    max_loss_time_frame = min(in_channels, args.max_loss_time_frame) 
    print('max_loss_time_frame: %s' % str(max_loss_time_frame))
    sub_collocation_t = torch.from_numpy(np.arange(args.sub_collocation_nt) * args.dt).float().to(device) 

    PIANO = PIANOinD(args, args.data_dim, args.data_spacing, args.perf_pattern, in_channels, device)
    PIANO.to(device)
    optimizer_PIANO = optim.Adam(PIANO.parameters(), lr = args.lr, weight_decay = args.lr_weight_decay) #, betas = [0.9, 0.999], weight_decay = 0.01

    if args.is_resume or args.for_test:
        PIANO.load_state_dict(checkpoint['model_state_dict'])
        #optimizer_PIANO.load_state_dict(checkpoint['optimizer_state_dict']) # NOTE: use new-defined optimizer #
        optimizer_PIANO.load_state_dict(checkpoint['optimizer_state_dict']) 
        PIANO.train()
        if args.for_test:
            epoch = 0
            args.n_epochs_total = 1
            main_fld = make_dir(os.path.join(main_fld, str(checkpoint['epoch']))) 
        else:
            epoch = checkpoint['epoch'] 
    else:
        epoch = 0

    test_PDE = PIANO_Skeleton(args, args.data_spacing, args.perf_pattern, args.PD_D_type, args.PD_V_type, device) 
    test_PDE.to(device)

    ###########################  Setting Info  ###########################

    setting_info = 'in%s-out%s_bz%s' % (in_channels, loss_time_frame, args.batch_size)
    if args.joint_predict:
        setting_info = 'Joint_%s' % setting_info 
    if args.gradient_loss:
        setting_info = '%s_GL%s' % (setting_info, args.gl_weight) 
    if args.spatial_gradient_loss and args.perf_loss_4train:
        setting_info = '%s_SG%s' % (setting_info, args.sgl_weight)
    setting_info = 'Crop(%s)_%s' % (args.boundary_crop_training[0], setting_info)
    if 'dirichlet' in args.BC:
        setting_info = 'Dri(%s)_%s' % (args.dirichlet_width, setting_info)
    elif 'source' in args.BC:
        setting_info = 'Src(%s)_%s' % (args.source_width, setting_info)
    
    file_path = os.path.join(main_fld, 'info (%s)' % setting_info)
    file = open(file_path, 'a+')

    print('Number of parameters to optimize - %d' % (len(list(PIANO.parameters()))))
    file.write('\nNumber of parameters to optimize - %d' % (len(list(PIANO.parameters()))))

    print('Starting epoch - %d' % epoch)
    file.write('\nStarting epoch - %d' % epoch)

    ###########################  Losses and Functions  ###########################

    loss_param_criterion = nn.L1Loss()
    #loss_param_criterion = nn.MSELoss() # NOTE cannot learn the anisotropy well #
    loss_param_criterion.to(device)    
    loss_perf_criterion = nn.MSELoss()
    loss_perf_criterion.to(device)   
    if args.stochastic:
        #loss_SDE_criterion = nn.MSELoss() # NOTE: TBD
        loss_SDE_criterion = nn.L1Loss() # NOTE: TBD
        loss_SDE_criterion.to(device)
        loss_SDE_test_criterion = nn.MSELoss() # NOTE
        loss_SDE_test_criterion.to(device)
    if args.model_type == 'vae':
        if args.perf_loss == 'dyn':
            def perf_loss_function(pd, gt, mu_lst, logvar_lst):
                MSE = loss_perf_criterion(pd[:, 1:] - pd[:, :-1], gt[:, 1:] - gt[:, :-1])
                # https://arxiv.org/abs/1312.6114 (Appendix B)
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD = 0.
                for i in range(len(mu_lst)):
                    KLD_element = mu_lst[i].pow(2).add_(logvar_lst[i].exp()).mul_(-1).add_(1).add_(logvar_lst[i])
                    KLD += torch.sum(KLD_element).mul_(-0.5)
                return MSE + KLD 
        elif args.perf_loss == 'abs':
            def perf_loss_function(pd, gt, mu_lst, logvar_lst):
                MSE = loss_perf_criterion(pd, gt)
                # https://arxiv.org/abs/1312.6114 (Appendix B)
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD = 0.
                for i in range(len(mu_lst)):
                    KLD_element = mu_lst[i].pow(2).add_(logvar_lst[i].exp()).mul_(-1).add_(1).add_(logvar_lst[i])
                    KLD += torch.sum(KLD_element).mul_(-0.5)
                return MSE + KLD
        else:
            def perf_loss_function(pd, gt, mu_lst, logvar_lst):
                MSE = loss_perf_criterion(pd, gt) + loss_perf_criterion(pd[:, 1:] - pd[:, :-1], gt[:, 1:] - gt[:, :-1])
                # https://arxiv.org/abs/1312.6114 (Appendix B)
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD = 0.
                for i in range(len(mu_lst)):
                    KLD_element = mu_lst[i].pow(2).add_(logvar_lst[i].exp()).mul_(-1).add_(1).add_(logvar_lst[i])
                    KLD += torch.sum(KLD_element).mul_(-0.5)
                return MSE + KLD
    elif args.model_type == 'unet':
        if args.perf_loss == 'dyn':
            def perf_loss_function(pd, gt, mu_lst, logvar_lst):
                MSE = loss_perf_criterion(pd[:, 1:] - pd[:, :-1], gt[:, 1:] - gt[:, :-1])
                return MSE
        elif args.perf_loss == 'abs':
            def perf_loss_function(pd, gt, mu_lst, logvar_lst):
                MSE = loss_perf_criterion(pd, gt)
                return MSE
        else:
            def perf_loss_function(pd, gt, mu_lst, logvar_lst):
                MSE = loss_perf_criterion(pd, gt) + loss_perf_criterion(pd[:, 1:] - pd[:, :-1], gt[:, 1:] - gt[:, :-1])
                return MSE
    
    if args.gradient_loss:
        loss_grad_criterion = GradientLoss(args.gl_weight).to(device)
        if 'scalar' in args.PD_D_type:
            def grad_loss_function(V, D, batched = True, perf_pattern = 'adv_diff'): # V: (batch, channel, r, c); D: (batch, r, c)
                if 'diff' in args.perf_pattern:
                    return loss_grad_criterion(list(V.permute(1, 0, 2, 3)) + [D], batched) # -> list: channels: (batch, r, c)
                else:
                    return loss_grad_criterion(list(V.permute(1, 0, 2, 3)), batched) # -> list: channels: (batch, r, c)
        else:
            def grad_loss_function(V, D, batched = False, perf_pattern = 'adv_diff'): # (batch, channel, r, c)
                if 'diff' in args.perf_pattern:
                    return loss_grad_criterion(list(torch.cat([V, D], dim = 1).permute(1, 0, 2, 3)), batched)
                else:
                    return loss_grad_criterion(list(V.permute(1, 0, 2, 3)), batched)

    if args.spatial_gradient_loss:
        loss_spatial_grad_criterion = SpatialGradientLoss(args.sgl_weight)
        loss_spatial_grad_criterion.to(device) 

    if args.predict_segment:
        loss_segment_criterion = nn.BCELoss()
        loss_segment_criterion.to(device)
     
    def zero_boundary(X): # X: (batch, channel, r, c)
        X[:, : args.boundary_crop_training[0], : args.boundary_crop_training[1]] = 0
        X[:, : args.boundary_crop_training[0], - args.boundary_crop_training[1] : ] = 0
        X[:, - args.boundary_crop_training[0] :, : args.boundary_crop_training[1]] = 0
        X[:, - args.boundary_crop_training[0] :, - args.boundary_crop_training[1] : ] = 0
        return X
    def patch_cropping(X): # X: (batch, channel, r, c)
        return X[:, :, args.boundary_crop_training[0] : X.size(2) - args.boundary_crop_training[0], args.boundary_crop_training[1] : X.size(3) - args.boundary_crop_training[1]]
    def get_Vlst(V):
        Vlst = {'V': V} if 'scalar' in args.PD_V_type else {'Vx': V[:, 0], 'Vy': V[:, 1]}
        return Vlst
    def get_Dlst(D):
        Dlst = {'D': D} if 'scalar' in args.PD_D_type else {'Dxx': D[:, 0], 'Dxy': D[:, 1], 'Dyy': D[:, 2]}
        return Dlst
    
    def plot_loss(test_loss_lst, current_test_loss, label):
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        current_test_loss[-2] = np.percentile(current_test_loss[:-2], 25)
        current_test_loss[-1] = np.percentile(current_test_loss[:-2], 75)
        test_loss_lst.append(current_test_loss) # (n_iter, n_test_sample + 2)
        test_loss_nda = np.array(test_loss_lst)
        np.save(os.path.join(main_fld, '%s.npy' % label), np.array(test_loss_nda))

        t = list(np.arange(test_loss_nda.shape[0]))  
        ax.plot(t, np.mean(test_loss_nda[:, :-2], axis = -1), 'r--', label = label)
        x_label = 'Iter (X%d)' % args.test_freq
        ax.set_xlabel(x_label) 
        ax.set_ylabel('Loss')
        #ax.set_yscale('log')
        #ax.legend()
        fig_name = '%s(%s)' % (label, model_info)
        ax.title.set_text(fig_name) 
        plt.savefig(os.path.join(main_fld, '%s.png' % fig_name))
        plt.close(fig)
        return test_loss_lst

    ########################### Record settings ###########################

    file.write('\n\nPID - %s' % os.getpid())  
    file.write('\nDevice - %s' % device) 
    file.write('\nMain folder - %s' % main_fld)
    file.write('\nSetting info - %s' % setting_info)
    file.write('\nMethod for integration - %s' % args.integ_method)
    file.write('\nStride of testing - %s' % str(args.stride_testing))
    file.write('\nCrop size - %s' % str(args.boundary_crop_training))
    print('PID - %s' % os.getpid())
    print('Device - %s' % device) 
    print('Main folder - %s' % main_fld)
    print('Setting info - %s' % setting_info)
    print('Method for integration - %s' % args.integ_method)
    print('Stride of testing - %s' % str(args.stride_testing))
    print('Crop size - %s' % str(args.boundary_crop_training))

    #log_dir = os.path.join(make_dir(os.path.join(main_fld, 'TensorBoard')), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #writer = SummaryWriter(log_dir) 
    #print('TensorBoard open command: tensorboard --logdir="%s"' % log_dir)

    ###########################  Training  ###########################

    if not args.is_resume:
        test_loss_D_lst  = []
        test_loss_L_lst  = []
        test_loss_U_lst  = [] 
        test_loss_S_lst  = []
        test_loss_V_lst  = []
        test_loss_dV_lst = []
        test_loss_Seg_lst    = []
        test_loss_perf_lst   = []
        test_loss_Sigma_lst  = [] 
        #test_loss_Sigma_D_lst  = []  
        #test_loss_Sigma_V_lst  = []    
        if args.predict_value_mask: 
            test_loss_VM_D_lst = [] 
            test_loss_VM_V_lst = [] 

    stop_count = 0
    Vlst, Dlst = [0.], [0.]
    mu_lst, logvar_lst = [], []

    lr_to_change = True
    num_lr_reduce = 0
    lr_reduce_criterion = args.lr_reduce_criterion 

    end = time.time()
    file.close()
    
    while epoch < args.niters_total: 
        epoch += 1
        file = open(file_path, 'a')

        optimizer_PIANO.zero_grad()
        
        if args.perf_loss_4train:
            u_4input, u_4loss, param_lst = get_movie(in_channels, loss_time_frame, loss4perf = True, save_fld = None, to_save = False, for_test = False, device = device) # (nT, r, c) 
            u_4input, u_4loss = Variable(u_4input[None].float().to(device), requires_grad = True), Variable(u_4loss[None].float().to(device), requires_grad = True) # (n_batch = 1, nT, r, c) 
        else:
            u_4input, param_lst = get_movie(in_channels, loss_time_frame, loss4perf = False, save_fld = None, to_save = False, for_test = False, device = device) # (nT, r, c) 
            u_4input = Variable(u_4input[None].float().to(device), requires_grad = True) # (n_batch = 1, nT, r, c) 
        
        PIANO.perf_pattern = args.perf_pattern
        PIANO.input_features = u_4input

        # Sigma ~ (n_batch = 1, 1, r, c)  
        base_V, base_D, delta_V, delta_D, Sigma = PIANO.get_VD() # n_batch = 1
        if args.predict_deviation: # archived: not working #
            V, D = base_V + delta_V, base_D + delta_D # (n_batch, 3, r, c)
        else:
            V, D = base_V, base_D

        ## For Losses ##

        if args.predict_value_mask:  
            
            if args.separate_DV_value_mask:
                GT_VM_D, GT_VM_V = param_lst['ValueMask_D'].float().to(device), param_lst['ValueMask_V'].float().to(device) # (r, c)
                D_value_mask, V_value_mask = PIANO.get_value_mask() # (n_batch, 1, r, c)
                
                #loss_VM = (loss_param_criterion(D_value_mask[0, 0], GT_VM_D) + loss_param_criterion(V_value_mask[0, 0], GT_VM_V)) * args.VM_weight / 2.
                loss_VM_D = loss_param_criterion(D_value_mask[0, 0], GT_VM_D) * args.VM_weight
                loss_VM_V = loss_param_criterion(V_value_mask[0, 0], GT_VM_V) * args.VM_weight
                loss_VM_D.backward(retain_graph = True)
                loss_VM_V.backward(retain_graph = True)
                epoch_loss_VM_D = loss_VM_D.item()
                epoch_loss_VM_V = loss_VM_V.item()
            else:
                GT_VM_D, GT_VM_V = param_lst['ValueMask_D'].float().to(device), param_lst['ValueMask_V'].float().to(device) # (r, c) 
                #GT_VM = param_lst['ValueMask_D'].float().to(device) # (r, c) # NOTE: In this case: ValueMask_D == ValueMask_V # 
                value_mask = PIANO.get_value_mask() # (n_batch, 1, r, c)
                D_value_mask, V_value_mask = value_mask, value_mask
                loss_VM = (loss_param_criterion(value_mask[0, 0], GT_VM_D) + loss_param_criterion(value_mask[0, 0], GT_VM_V)) * args.VM_weight   

                loss_VM.backward(retain_graph = True)
                epoch_loss_VM_D = loss_VM.item()
                epoch_loss_VM_V = loss_VM.item()
            
        if args.GT_D:
            if args.predict_deviation: # archived: not working #
                assert not args.predict_value_mask
                GT_orig_Dxx, GT_orig_Dxy, GT_orig_Dyy, GT_delta_Dxx, GT_delta_Dxy, GT_delta_Dyy = param_lst['Dlst']['orig_Dxx'].float().to(device), \
                    param_lst['Dlst']['orig_Dxy'].float().to(device), param_lst['Dlst']['orig_Dyy'].float().to(device), \
                        param_lst['Dlst']['delta_Dxx'].float().to(device), param_lst['Dlst']['delta_Dxy'].float().to(device), param_lst['Dlst']['delta_Dyy'].float().to(device) # (r, c)
                loss_D = (loss_param_criterion(base_D[0, 0], GT_orig_Dxx) + loss_param_criterion(base_D[0, 1], GT_orig_Dxy) * 2 + loss_param_criterion(base_D[0, 2], GT_orig_Dyy)) * args.GT_D_weight + \
                    (loss_param_criterion(delta_D[0, 0], GT_delta_Dxx) + loss_param_criterion(delta_D[0, 1], GT_delta_Dxy) * 2 + loss_param_criterion(delta_D[0, 2], GT_delta_Dyy)) * (args.GT_D_weight + args.deviation_extra_weight)
            
            elif args.predict_value_mask:
                GT_orig_Dxx, GT_orig_Dxy, GT_orig_Dyy = param_lst['Dlst']['orig_Dxx'].float().to(device), \
                    param_lst['Dlst']['orig_Dxy'].float().to(device), param_lst['Dlst']['orig_Dyy'].float().to(device) # (r, c)
                loss_D = (loss_param_criterion(base_D[0, 0], GT_orig_Dxx) + loss_param_criterion(base_D[0, 1], GT_orig_Dxy) * 2 + loss_param_criterion(base_D[0, 2], GT_orig_Dyy)) * args.GT_D_weight
                if args.actual_physics_loss:
                    GT_Dxx, GT_Dxy, GT_Dyy = param_lst['Dlst']['Dxx'].float().to(device), param_lst['Dlst']['Dxy'].float().to(device), param_lst['Dlst']['Dyy'].float().to(device) # (r, c)
                    loss_D = loss_D + (loss_param_criterion(base_D[0, 0] * D_value_mask[0, 0], GT_Dxx) + loss_param_criterion(base_D[0, 1] * D_value_mask[0, 0], GT_Dxy) * 2 + \
                        loss_param_criterion(base_D[0, 2] * D_value_mask[0, 0], GT_Dyy)) * args.GT_D_weight

            else:
                GT_Dxx, GT_Dxy, GT_Dyy = param_lst['Dlst']['Dxx'].float().to(device), param_lst['Dlst']['Dxy'].float().to(device), param_lst['Dlst']['Dyy'].float().to(device) # (r, c)
                loss_D = (loss_param_criterion(D[0, 0], GT_Dxx) + loss_param_criterion(D[0, 1], GT_Dxy) * 2 + loss_param_criterion(D[0, 2], GT_Dyy)) * args.GT_D_weight
                #loss_D = loss_param_criterion(D[0], torch.stack([GT_Dxx, GT_Dxy, GT_Dyy], dim = 0)) * args.GT_D_weight
                
            loss_D.backward(retain_graph = True)
            epoch_loss_D = loss_D.item()
            
        if args.GT_LU:
            base_L, delta_L = PIANO.get_L() # (batch = 1, 2, r, c)
            U = PIANO.get_U() # (batch = 1, 4, r, c)

            if args.predict_deviation:
                L = base_L + delta_L
                GT_orig_L1, GT_orig_L2, GT_delta_L1, GT_delta_L2 = param_lst['Llst']['orig_L1'].float().to(device),\
                     param_lst['Llst']['orig_L2'].float().to(device), param_lst['Llst']['delta_L1'].float().to(device), param_lst['Llst']['delta_L2'].float().to(device)
                loss_L = (loss_param_criterion(base_L[0, 0], GT_orig_L1) + loss_param_criterion(base_L[0, 1], GT_orig_L2)) * args.GT_L_weight + \
                    (loss_param_criterion(delta_L[0, 0], GT_delta_L1) + loss_param_criterion(delta_L[0, 1], GT_delta_L2)) * (args.GT_L_weight + args.deviation_extra_weight)
            elif args.predict_value_mask: 
                GT_orig_L1, GT_orig_L2 = param_lst['Llst']['orig_L1'].float().to(device), param_lst['Llst']['orig_L2'].float().to(device) 
                loss_L = (loss_param_criterion(base_L[0, 0], GT_orig_L1) + loss_param_criterion(base_L[0, 1], GT_orig_L2)) * args.GT_L_weight 
                if args.actual_physics_loss:
                    GT_L1, GT_L2 = param_lst['Llst']['L1'].float().to(device), param_lst['Llst']['L2'].float().to(device) # (r, c) 
                    loss_L = loss_L + (loss_param_criterion(base_L[0, 0] * D_value_mask[0, 0], GT_L1) + loss_param_criterion(base_L[0, 1] * D_value_mask[0, 0], GT_L2)) * args.GT_L_weight 
            else:
                L = base_L
                GT_L1, GT_L2 = param_lst['Llst']['L1'].float().to(device), param_lst['Llst']['L2'].float().to(device) # (r, c) 
                loss_L = (loss_param_criterion(L[0, 0], GT_L1) + loss_param_criterion(L[0, 1], GT_L2)) * args.GT_L_weight
                #loss_L = loss_param_criterion(L[0], torch.stack([GT_L1, GT_L2], dim = 0)) * args.GT_L_weight 

            GT_Uxx, GT_Uxy, GT_Uyx, GT_Uyy = param_lst['Ulst']['Uxx'].float().to(device), param_lst['Ulst']['Uxy'].float().to(device), \
                param_lst['Ulst']['Uyx'].float().to(device), param_lst['Ulst']['Uyy'].float().to(device) 
            loss_U = (loss_param_criterion(U[0, 0], GT_Uxx) + loss_param_criterion(U[0, 1], GT_Uxy) + \
                loss_param_criterion(U[0, 2], GT_Uyx) + loss_param_criterion(U[0, 3], GT_Uyy)) * args.GT_U_weight
            #loss_U = loss_param_criterion(U[0], torch.stack([GT_Uxx, GT_Uxy, GT_Uyx, GT_Uyy], dim = 0)) * args.GT_U_weight

            loss_L.backward(retain_graph = True)
            loss_U.backward(retain_graph = True)
            epoch_loss_L = loss_L.item()
            epoch_loss_U = loss_U.item()

        if args.GT_V:
            if args.predict_deviation:
                GT_orig_Vx, GT_orig_Vy, GT_delta_Vx, GT_delta_Vy = param_lst['Vlst']['orig_Vx'].float().to(device), param_lst['Vlst']['orig_Vy'].float().to(device), \
                    param_lst['Vlst']['delta_Vx'].float().to(device), param_lst['Vlst']['delta_Vy'].float().to(device) # (r, c) 
                loss_V = (loss_param_criterion(base_V[0, 0], GT_orig_Vx) + loss_param_criterion(base_V[0, 1], GT_orig_Vy)) * args.GT_V_weight + \
                    (loss_param_criterion(delta_V[0, 0], GT_delta_Vx) + loss_param_criterion(delta_V[0, 1], GT_delta_Vy)) * (args.GT_V_weight + args.deviation_extra_weight)
            elif args.predict_value_mask:
                GT_orig_Vx, GT_orig_Vy = param_lst['Vlst']['orig_Vx'].float().to(device), param_lst['Vlst']['orig_Vy'].float().to(device) 
                loss_V = (loss_param_criterion(base_V[0, 0], GT_orig_Vx) + loss_param_criterion(base_V[0, 1], GT_orig_Vy)) * args.GT_V_weight
                if args.actual_physics_loss:
                    GT_Vx, GT_Vy = param_lst['Vlst']['Vx'].float().to(device), param_lst['Vlst']['Vy'].float().to(device) # (r, c) 
                    loss_V = loss_V + (loss_param_criterion(base_V[0, 0] * V_value_mask[0, 0], GT_Vx) + loss_param_criterion(base_V[0, 1] * V_value_mask[0, 0], GT_Vy)) * args.GT_V_weight
            else:
                GT_Vx, GT_Vy = param_lst['Vlst']['Vx'].float().to(device), param_lst['Vlst']['Vy'].float().to(device) # (r, c) 
                loss_V = (loss_param_criterion(V[0, 0], GT_Vx) + loss_param_criterion(V[0, 1], GT_Vy)) * args.GT_V_weight
                #loss_V = loss_param_criterion(V[0], torch.stack([GT_Vx, GT_Vy], dim = 0)) * args.GT_V_weight 

            loss_V.backward(retain_graph = True)
            epoch_loss_V = loss_V.item()

        if args.gradient_loss: 
            loss_grad = grad_loss_function(V, D, batched = True, perf_pattern = args.perf_pattern)
            loss_grad.backward(retain_graph = True)
            epoch_loss_grad = loss_grad.item() 

        if args.stochastic:
            # TODO: TESTING shared uncertainty #
            GT_Sigma = param_lst['Sigma'].float().to(device)
            loss_SDE = loss_SDE_criterion(GT_Sigma, Sigma[0, 0]) * args.SDE_weight 
            epoch_loss_SDE = loss_SDE.item()  
            '''GT_Sigma_V = param_lst['Sigma_V'].float().to(device)
            GT_Sigma_D = param_lst['Sigma_D'].float().to(device)
            if args.separate_DV_value_mask:
                loss_SDE_D = loss_SDE_criterion(GT_Sigma_D, Sigma[0, 0]) * args.SDE_weight 
                loss_SDE_V = loss_SDE_criterion(GT_Sigma_V, Sigma[0, 1]) * args.SDE_weight 
                loss_SDE = loss_SDE_D + loss_SDE_V
                epoch_loss_SDE_D = loss_SDE_D.item()  
                epoch_loss_SDE_V = loss_SDE_V.item()  
            else:
                loss_SDE = loss_SDE_criterion(GT_Sigma_V, Sigma[0, 0]) * args.SDE_weight 
                epoch_loss_SDE = loss_SDE.item()  '''

            if args.SDE_loss:
                loss_SDE.backward(retain_graph = True)
        
        if args.predict_segment: 
            if args.segment_net_type == 'conc':
                seg_mask = PIANO.get_segment(threshold = None) # (n_batch, 1, s, r, c)
            elif args.segment_net_type == 'dev':
                if not args.GT_LU:
                    base_L, delta_L = PIANO.get_L() # (batch = 1, 2, r, c)
                seg_mask = PIANO.get_segment(threshold = None, physics_deviation = torch.cat([delta_V, delta_L], dim = 1))
            GT_Seg  = param_lst['Seg_V'].float().to(device) 
            loss_seg = loss_segment_criterion(seg_mask[0, 0], GT_Seg) * args.seg_weight 
            loss_seg.backward(retain_graph = True)
            epoch_loss_seg = loss_seg.item()  

        if args.perf_loss_4train: 

            if 'dirichlet' in args.BC or 'cauchy' in args.BC:
                contours = extract_BC_2D(u_4loss[0], 1) 
            elif 'source' in args.BC:
                dcontours = extract_dBC_2D(u_4loss[0], 1)

            pred_u = torch.stack([u_4loss[:, 0]] * u_4loss.size(1), dim = 1) # (n_batch, loss_n_collocations, r, c)
            for i_coll in range(1, u_4loss.size(1)):  # Loop for n_collocations #
                #print(i_coll+1, '/', movie4loss.size(1).item())
                pred_u[:, i_coll] = pred_u[:, i_coll - 1]
                # (collocation_nt, n_batch, r, c) -> (collocation_it = -1, n_batch, r, c)
                for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (n_batch, r, c)
                    #print(i_sub_coll+1, '/', sub_collocation_t.size().item())
                    pred_u[:, i_coll] = odeint(PIANO, pred_u[:, i_coll], sub_collocation_t, method = args.integ_method, options = args)[-1] 
                if 'dirichlet' in args.BC or 'cauchy' in args.BC: # BC list: [[BC0_0, BC0, 1], [BC1_0, BC1_1]], [BC2_0, BC2_1]]: each: ((n_batch), nT, BC_size, rest_dim_remain)
                    pred_u[:, i_coll] = apply_BC_2D(pred_u[0, i_coll], contours, i_coll, args.BC, batched = False)[None] # (n_batch = 1, r, c)

            if args.spatial_gradient_loss:
                loss_spatial_grad = loss_spatial_grad_criterion(pred_u, u_4loss, batched = True)
                if args.perf_loss_4train:
                    loss_spatial_grad.backward(retain_graph = True)
                else:
                    loss_spatial_grad.backward()
                epoch_loss_sgl = loss_spatial_grad.item()

            if args.model_type == 'vae':
                mu_lst, logvar_lst = PIANO.get_vars() 
            
            loss_perf = perf_loss_function(pred_u, u_4loss, mu_lst, logvar_lst) # (batch, batch_nT, r, c)
            loss_perf.backward()
            epoch_loss_perf = loss_perf.item()

        optimizer_PIANO.step()
 
        
        if epoch % args.print_freq == 0 or epoch == 1:
            print('\nEpoch #{:d}'.format(epoch))
            if args.perf_loss_4train:
                file.write('\n      | Perf {:.9f}'.format(epoch_loss_perf))
                print('      | Perf {:.9f}'.format(epoch_loss_perf))
                if args.spatial_gradient_loss:
                    file.write('\n      | SpGd {:.9f}'.format(epoch_loss_sgl))
                    print('      | SpGd {:.9f}'.format(epoch_loss_sgl))
            if args.GT_V:
                file.write('\n      |    V {:.9f}'.format(epoch_loss_V))
                print('      |    V {:.9f}'.format(epoch_loss_V))
            if args.GT_D:
                file.write('\n      |    D {:.9f}'.format(epoch_loss_D))
                print('      |    D {:.9f}'.format(epoch_loss_D))
            if args.GT_LU:
                file.write('\n      |    L {:.9f}'.format(epoch_loss_L))
                print('      |    L {:.9f}'.format(epoch_loss_L)) 
                file.write('\n      |    U {:.9f}'.format(epoch_loss_U))
                print('      |    U {:.9f}'.format(epoch_loss_U)) 
            if args.stochastic: 
                '''if args.separate_DV_value_mask:
                    file.write('\n      | StoD {:.9f}'.format(epoch_loss_SDE_D))
                    file.write('\n      | StoV {:.9f}'.format(epoch_loss_SDE_V))
                    print('      | StoD {:.9f}'.format(epoch_loss_SDE_D)) 
                    print('      | StoV {:.9f}'.format(epoch_loss_SDE_V)) 
                else:'''
                file.write('\n      |  Sto {:.9f}'.format(epoch_loss_SDE))
                print('      |  Sto {:.9f}'.format(epoch_loss_SDE)) 
            if args.predict_value_mask:  
                file.write('\n      | VM_D {:.9f}'.format(epoch_loss_VM_D))
                file.write('\n      | VM_V {:.9f}'.format(epoch_loss_VM_V))
                print('      | VM_D {:.9f}'.format(epoch_loss_VM_D)) 
                print('      | VM_V {:.9f}'.format(epoch_loss_VM_V))  
            if args.predict_segment:
                file.write('\n      |  Seg {:.9f}'.format(epoch_loss_seg))
                print('      |  Seg {:.9f}'.format(epoch_loss_seg)) 
            if args.gradient_loss:
                file.write('\n      | Grad {:.9f}'.format(epoch_loss_grad))
                print('      | Grad {:.9f}'.format(epoch_loss_grad)) 
            grad_PIANO = avg_grad(PIANO.parameters())
            file.write('\n      |  AvG {:.6f}'.format(grad_PIANO)) 
            print('      |  AvG {:.6f}'.format(grad_PIANO))


        if epoch % args.test_freq == 0:
            n_test_iter += 1

            print('Exp. main folder: %s' % main_fld)
            
            main_test_fld = make_dir(os.path.join(main_fld, '%d' % epoch))

            # Save Model #
            torch.save({
                'model_state_dict': PIANO.state_dict(),
                'optimizer_state_dict': optimizer_PIANO.state_dict(),
                'in_channels': in_channels,
                'epoch': epoch,
                }, os.path.join(main_test_fld, 'checkpoint (%s).pth' % setting_info))

            torch.save({
                'model_state_dict': PIANO.state_dict(),
                'optimizer_state_dict': optimizer_PIANO.state_dict(),
                'in_channels': in_channels,
                'epoch': epoch,
                }, os.path.join(main_fld, 'latest_checkpoint.pth'))

            with torch.no_grad(): 

                test_loss_D = np.zeros(n_test_sample + 2) # Add 2 stat (0.25, 0.75 percentile) at the end #
                test_loss_L = np.zeros(n_test_sample + 2)  
                test_loss_S = np.zeros(n_test_sample + 2)  
                test_loss_U = np.zeros(n_test_sample + 2)   
                test_loss_V = np.zeros(n_test_sample + 2) 
                test_loss_Seg = np.zeros(n_test_sample + 2) 
                test_loss_perf = np.zeros(n_test_sample + 2) 
                if args.stochastic:
                    test_loss_Sigma = np.zeros(n_test_sample + 2)
                    #test_loss_Sigma_D = np.zeros(n_test_sample + 2)  
                    #test_loss_Sigma_V = np.zeros(n_test_sample + 2)
                if args.predict_value_mask:
                    test_loss_VM_D = np.zeros(n_test_sample + 2)  
                    test_loss_VM_V = np.zeros(n_test_sample + 2)  

                for i_test in range(args.n_test_sample):
                    print('{:05d} | Test-{:d}'.format(epoch, i_test))
                    file.write('\n{:05d} | Test-{:d}'.format(epoch, i_test)) 

                    save_test_fld = make_dir(os.path.join(main_test_fld, 'test_%d' % i_test))
                    save_movie_fld = make_dir(os.path.join(save_test_fld, 'Movies'))
                      
                    test_u_4input, full_movie, param_lst, origin, spacing = get_movie(in_channels, loss_time_frame, \
                        loss4perf = False, save_fld = make_dir(os.path.join(save_test_fld, 'GT')), to_save = True, for_test = True, device = device) # (nT, r, c) 
                    test_u_4input, full_movie = Variable(test_u_4input[None].float().to(device), requires_grad = True), Variable(full_movie.float().to(device), requires_grad = True) # (n_batch = 1, nT, r, c) 
         
                    if 'dirichlet' in args.BC or 'cauchy' in args.BC:
                        contours = extract_BC_2D(full_movie, 1) # BC list
                    elif 'source' in args.BC:
                        contours = extract_dBC_2D(full_movie, 1)
                    
                    PIANO.input_features = test_u_4input                
                    base_V, base_D, delta_V, delta_D, Sigma = PIANO.get_VD() # n_batch = 1
                    
                    if args.predict_deviation:
                        V, D = base_V + delta_V, base_D + delta_D
                    elif args.predict_value_mask:
                        if args.separate_DV_value_mask:
                            D_value_mask, V_value_mask = PIANO.get_value_mask() # (n_batch, 1, r, c)
                            V, D = base_V * V_value_mask, base_D * D_value_mask
                            save_sitk(D_value_mask[0, 0], os.path.join(save_test_fld, 'ValueMask_D (%s).mha' % setting_info), origin, spacing, toCut = False)
                            save_sitk(V_value_mask[0, 0], os.path.join(save_test_fld, 'ValueMask_V (%s).mha' % setting_info), origin, spacing, toCut = False)

                            GT_VM_D, GT_VM_V = param_lst['ValueMask_D'].float().to(device), param_lst['ValueMask_V'].float().to(device) # (r, c) 
                            full_VM_D_loss = loss_param_criterion(GT_VM_D, D_value_mask[0, 0]).item()
                            full_VM_V_loss = loss_param_criterion(GT_VM_V, V_value_mask[0, 0]).item()

                            print('      |  VM_D {:.9f}'.format(full_VM_D_loss)) 
                            print('      |  VM_V {:.9f}'.format(full_VM_V_loss)) 
                            file.write('\n      |  VM_D {:.9f}'.format(full_VM_D_loss))
                            file.write('\n      |  VM_V {:.9f}'.format(full_VM_V_loss))
                            test_loss_VM_D[i_test] = full_VM_D_loss 
                            test_loss_VM_V[i_test] = full_VM_V_loss 
                        else:
                            value_mask = PIANO.get_value_mask() # (n_batch, 1, r, c)
                            #D_value_mask, V_value_mask = value_mask, value_mask
                            V, D = base_V * value_mask, base_D * value_mask
                            save_sitk(value_mask[0, 0], os.path.join(save_test_fld, "ValueMask (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (row, column)

                            GT_VM_D, GT_VM_V = param_lst['ValueMask_D'].float().to(device), param_lst['ValueMask_V'].float().to(device) # (r, c) 
                            #GT_VM = param_lst['ValueMask_D'].float().to(device) # (r, c) # NOTE: In this case: ValueMask_D == ValueMask_V #
                            full_VM_D_loss = loss_param_criterion(GT_VM_D, D_value_mask[0, 0]).item()
                            full_VM_V_loss = loss_param_criterion(GT_VM_V, V_value_mask[0, 0]).item()

                            print('      |  VM_D {:.9f}'.format(full_VM_D_loss)) 
                            print('      |  VM_V {:.9f}'.format(full_VM_V_loss)) 
                            file.write('\n      |  VM_D {:.9f}'.format(full_VM_D_loss))
                            file.write('\n      |  VM_V {:.9f}'.format(full_VM_V_loss)) 
                            test_loss_VM_D[i_test] = full_VM_D_loss
                            test_loss_VM_V[i_test] = full_VM_V_loss
                    else:
                        V, D = base_V, base_D

                    test_PDE.Vlst = get_Vlst(V) # (n_batch = 1, 2, r, c) 
                    test_PDE.Dlst = get_Dlst(D)
                    V, D = V[0] , D[0] # (n_channel, r, c)  

                    if args.stochastic:
                        test_PDE.Sigma = Sigma[:, 0]
                        
                    # Save all testing results # 

                    if args.stochastic:
                        Sigma = Sigma[0, 0] # (1, r, c) 
                        save_sitk(Sigma, os.path.join(save_test_fld, "Sigma (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (row, column)
                        
                        GT_Sigma = param_lst['Sigma'].float().to(device) # (r, c) 
                        full_Sigma_loss = loss_SDE_test_criterion(GT_Sigma, Sigma).item()
                        #GT_Sigma_V = param_lst['Sigma_V'].float().to(device) # (r, c) 
                        #full_Sigma_loss = loss_SDE_test_criterion(GT_Sigma_V, Sigma).item()
                        print('      |   SDE {:.9f}'.format(full_Sigma_loss)) 
                        file.write('\n      |   SDE {:.9f}'.format(full_Sigma_loss))
                        test_loss_Sigma[i_test] = full_Sigma_loss 
                             
                    if 'adv' in args.perf_pattern:  

                        if args.predict_deviation:
                            save_sitk(base_V[0], os.path.join(save_test_fld, "orig_V (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                            save_sitk(delta_V[0], os.path.join(save_test_fld, "delta_V (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column) 
                        elif args.predict_value_mask: 
                            save_sitk(base_V[0], os.path.join(save_test_fld, "orig_V (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                        save_sitk(V, os.path.join(save_test_fld, "V (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)

                        GT_Vx, GT_Vy = param_lst['Vlst']['Vx'].float().to(device), param_lst['Vlst']['Vy'].float().to(device) # (r, c)  
                        full_V_loss = ((loss_param_criterion(V[0], GT_Vx) + loss_param_criterion(V[1], GT_Vy))).item()  
                        
                        print('      |     V {:.9f}'.format(full_V_loss)) 
                        file.write('\n      |     V {:.9f}'.format(full_V_loss))
                        test_loss_V[i_test] = full_V_loss 

                        if 'stream' in args.PD_V_type:
                            Phi = PIANO.get_Phi()[0] # (r, c) 
                            save_sitk(Phi, os.path.join(save_test_fld, "Phi (%s).mha" % (setting_info)), origin, spacing, toCut = False) 

                    if 'diff' in args.perf_pattern: # Dxx, Dxy, Dyy    

                        if args.predict_deviation:
                            save_sitk(base_D[0], os.path.join(save_test_fld, "orig_D (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                            save_sitk(delta_D[0], os.path.join(save_test_fld, "delta_D (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                        elif args.predict_value_mask:
                            save_sitk(base_D[0], os.path.join(save_test_fld, "orig_D (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)

                        save_sitk(D, os.path.join(save_test_fld, "D (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (3, row, column)
                        GT_Dxx, GT_Dxy, GT_Dyy = param_lst['Dlst']['Dxx'].float().to(device), param_lst['Dlst']['Dxy'].float().to(device), param_lst['Dlst']['Dyy'].float().to(device) # (r, c)
                        full_D_loss = ((loss_param_criterion(D[0], GT_Dxx) + loss_param_criterion(D[1], GT_Dxy) * 2 + loss_param_criterion(D[2], GT_Dyy))).item()

                        print('      |     D {:.9f}'.format(full_D_loss)) 
                        file.write('\n      |     D {:.9f}'.format(full_D_loss))
                        test_loss_D[i_test] = full_D_loss

                        if 'cholesky' in args.PD_D_type: # TODO
                            L = PIANO.get_L()[0] 
                            save_sitk(L, os.path.join(save_test_fld, "L (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (3, row, column)

                        elif 'full_spectral' in args.PD_D_type:
                            base_L, delta_L = PIANO.get_L() # (batch = 1, 2, r, c)
                            U = PIANO.get_U()[0] # (batch = 1, 4, r, c) 

                            if args.predict_deviation:
                                L = base_L + delta_L
                                save_sitk(base_L[0], os.path.join(save_test_fld, "orig_L (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                                save_sitk(delta_L[0], os.path.join(save_test_fld, "delta_L (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                            elif args.predict_value_mask:
                                L = base_L * D_value_mask[0]
                                save_sitk(base_L[0], os.path.join(save_test_fld, "orig_L (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                            else:
                                L = base_L
                            
                            GT_L1, GT_L2 = param_lst['Llst']['L1'].float().to(device), param_lst['Llst']['L2'].float().to(device)
                            full_L_loss = (loss_param_criterion(L[0, 0], GT_L1) + loss_param_criterion(L[0, 1], GT_L2)).item()
                            
                            print('      |     L {:.9f}'.format(full_L_loss)) 
                            file.write('\n      |     L {:.9f}'.format(full_L_loss))
                            test_loss_L[i_test] = full_L_loss

                            GT_Uxx, GT_Uxy, GT_Uyx, GT_Uyy = param_lst['Ulst']['Uxx'].float().to(device), param_lst['Ulst']['Uxy'].float().to(device), \
                                param_lst['Ulst']['Uyx'].float().to(device), param_lst['Ulst']['Uyy'].float().to(device) 
                            full_U_loss = (loss_param_criterion(U[0], GT_Uxx) + loss_param_criterion(U[1], GT_Uxy) + \
                                loss_param_criterion(U[2], GT_Uyx) + loss_param_criterion(U[3], GT_Uyy)).item() 

                            print('      |     U {:.9f}'.format(full_U_loss)) 
                            file.write('\n      |     U {:.9f}'.format(full_U_loss))
                            test_loss_U[i_test] = full_U_loss
 
                            save_sitk(L, os.path.join(save_test_fld, "L (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (2, row, column)
                            save_sitk(U, os.path.join(save_test_fld, "U (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (3, row, column)
                            #S = PIANO.get_S()[0] 
                            #save_sitk(S, os.path.join(save_test_fld, "S (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (3, row, column)

                    if args.predict_segment: 
                        if args.segment_net_type == 'conc':
                            seg_mask = PIANO.get_segment(threshold = None)[0, 0] # (n_batch, 1, r, c)
                        elif args.segment_net_type == 'dev':
                            seg_mask = PIANO.get_segment(threshold = None, physics_deviation = torch.cat([delta_V, delta_L], dim = 1))[0, 0]
                            
                        GT_Seg  = param_lst['Seg_V'].float().to(device) 
                        full_loss_seg = (loss_segment_criterion(GT_Seg, seg_mask) * args.seg_weight).item()
                        print('      |   Seg {:.9f}'.format(full_loss_seg)) 
                        file.write('\n      |   Seg {:.9f}'.format(full_loss_seg))
                        test_loss_Seg[i_test] = full_loss_seg
                        save_sitk(seg_mask, os.path.join(save_test_fld, "Seg (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (row, column) 
                    

                    if epoch >= save_test_perf_after_itr:
                        
                        ############################################
                        ###### Integrated concentration looses #####
                        ############################################
                            
                        test_PDE.perf_pattern = args.perf_pattern
                        test_PDE.BC = 'neumann' # TODO # 
                        test_PDE.stochastic = False # NOTE: disable stochastic term(s) when computing concentration loss #

                        pred_full_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # full_movie: (nT, r, c)
                        for i_coll in range(1, full_movie.size(0)):
                            pred_full_movie[i_coll] = pred_full_movie[i_coll - 1]
                            #print(i_coll) # NOTE: (nT, n_batch = 1, r, c) -> (nT = -1, r, c)
                            for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (r, c)
                                pred_full_movie[i_coll] = odeint(test_PDE, pred_full_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                            if 'dirichlet' in  args.BC or 'cauchy' in  args.BC or 'source' in args.BC:
                                pred_full_movie[i_coll] = apply_BC_2D(pred_full_movie[i_coll], contours, i_coll, args.BC, batched = False) # (r, c)

                        save_sitk(pred_full_movie, os.path.join(save_movie_fld, "Full (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (time, row, column)
                        
                        full_perf_loss = loss_param_criterion(full_movie, pred_full_movie).item()
                        print('      |     C {:.9f}'.format(full_perf_loss)) 
                        file.write('\n      |     C {:.9f}'.format(full_perf_loss))
                        test_loss_perf[i_test] = full_perf_loss

                        ############################################
                        ########### Computing Uncertainty ##########
                        ############################################

                        if args.compute_uncertainty > 0 and args.stochastic:
                            test_PDE.stochastic = True # NOTE: reset stochastic term(s) #
                            
                            pred_movie_samples = torch.zeros(tuple([args.compute_uncertainty]) + full_movie[:-1].size()) # NOTE (n_samples, nT-1, r, c): exclude t=0 #
                            for i_sample in range(args.compute_uncertainty):
                                #print('Uncertainty test %d/%d' % (i_sample + 1, args.compute_uncertainty))
                                pred_full_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, r, c)
                                for i_coll in range(1, full_movie.size(0)):
                                    pred_full_movie[i_coll] = pred_full_movie[i_coll - 1]
                                    #print(i_coll) # NOTE: (nT, n_batch = 1, r, c) -> (nT = -1, r, c)
                                    for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (r, c)
                                        pred_full_movie[i_coll] = odeint(test_PDE, pred_full_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                                    if 'dirichlet' in  args.BC or 'cauchy' in  args.BC or 'source' in  args.BC:
                                        pred_full_movie[i_coll] = apply_BC_2D(pred_full_movie[i_coll], contours, i_coll, args.BC, batched = False) # (r, c)
                                pred_movie_samples[i_sample] = pred_full_movie[:-1]
                            
                            pred_uncertainty = torch.mean(torch.var(pred_movie_samples, dim = 0), dim = 0) # (r, c)
                            save_sitk(pred_uncertainty, os.path.join(save_test_fld, 'Uncertainty (%s).mha' % setting_info), origin, spacing, toCut = False)
                        
                        
                        ## Save adv_only, diff_only part prediction ##
                        args.contours, args.dcontours = None, None # NOTE: should not imposing abs B.C., only apply Neumann
                        if args.perf_pattern == 'adv_diff':
                            test_PDE.stochastic = False # NOTE: disable stochastic term(s) when computing concentration loss #
                            test_PDE.perf_pattern = 'adv_only'
                            # adv movie predict from t0 #
                            pred_adv_movie =torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, r, c)
                            for i_coll in range(1, full_movie.size(0)):
                                pred_adv_movie[i_coll] = pred_adv_movie[i_coll - 1]
                                #print(i_coll) # NOTE: (nT, n_batch = 1, r, c) -> (nT = -1, r, c)
                                for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (r, c)
                                    pred_adv_movie[i_coll] = odeint(test_PDE, pred_adv_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                                # NOTE: Collocation_t for testing is 2 as entire testing image is too large #
                                if 'dirichlet' in  args.BC or 'cauchy' in  args.BC or 'source' in  args.BC:
                                    pred_adv_movie[i_coll] = apply_BC_2D(pred_adv_movie[i_coll], contours, i_coll, args.BC, batched = False) # (r, c)

                        
                            # adv movie predict from each intermediate time point #
                            '''pred_track_adv_nT = [pred_full_movie[0]]
                            for it in range(full_T.size(0) - 1):
                                pred_track_adv_nT.append(odeint(test_PDE, pred_full_movie[it].unsqueeze(0), full_T[:2], method = args.integ_method, options = args)[1, 0]) # (nT = 2, batch = 1, r, c) -> (r, c)
                            pred_track_adv_nT = torch.stack(pred_track_adv_nT, dim = 0)
                            save_sitk(pred_track_adv_nT, os.path.join(save_movie_fld, "AdvTrack (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (time, row, column)
                            '''
                            save_sitk(pred_adv_movie, os.path.join(save_movie_fld, "AdvFull (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (time, row, column)
                        
                            test_PDE.perf_pattern = 'diff_only'
                            pred_diff_movie =torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, r, c)
                            for i_coll in range(1, full_movie.size(0)):
                                pred_diff_movie[i_coll] = pred_diff_movie[i_coll - 1]
                                #print(i_coll) # NOTE: (nT, n_batch = 1, r, c) -> (nT = -1, r, c)
                                for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (r, c)
                                    pred_diff_movie[i_coll] = odeint(test_PDE, pred_diff_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                                # NOTE: Collocation_t for testing is 2 as entire testing image is too large #
                                if 'dirichlet' in  args.BC or 'cauchy' in  args.BC or 'source' in  args.BC:
                                    pred_diff_movie[i_coll] = apply_BC_2D(pred_diff_movie[i_coll], contours, i_coll, args.BC, batched = False) # (r, c)

                            # diff movie predict from each intermediate time point #
                            '''pred_track_diff_nT = [pred_full_movie[0]]
                            for it in range(full_T.size(0) - 1):
                                pred_track_diff_nT.append(odeint(test_PDE, pred_full_movie[it].unsqueeze(0), full_T[:2], method = args.integ_method, options = args)[1, 0]) # (batch, r, c)
                            pred_track_diff_nT = torch.stack(pred_track_diff_nT, dim = 0)
                            save_sitk(pred_track_diff_nT, os.path.join(save_test_fld, "Movies/DiffTrack (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (time, row, column)
                            '''
                            save_sitk(pred_diff_movie, os.path.join(save_test_fld, "Movies/DiffFull (%s).mha" % (setting_info)), origin, spacing, toCut = False) # (time, row, column)
                
                test_PDE.stochastic = args.stochastic # NOTE: reset stochastic term(s) #

                #writer.add_scalar('testing_loss', test_loss_perf, epoch)

                #if epoch >= save_test_perf_after_itr:

                test_loss_perf_lst = plot_loss(test_loss_perf_lst, test_loss_perf, label = 'Err_C')  
                test_loss_D_lst = plot_loss(test_loss_D_lst, test_loss_D, label = 'Err_D')   
                test_loss_V_lst = plot_loss(test_loss_V_lst, test_loss_V, label = 'Err_V')  
                test_loss_L_lst = plot_loss(test_loss_L_lst, test_loss_L, label = 'Err_L')  
                test_loss_U_lst = plot_loss(test_loss_U_lst, test_loss_U, label = 'Err_U')  
                if args.predict_segment:
                    test_loss_Seg_lst = plot_loss(test_loss_U_lst, test_loss_Seg, label = 'Err_Seg') 
                if args.predict_value_mask: 
                    test_loss_VM_D_lst = plot_loss(test_loss_VM_D_lst, test_loss_VM_D, label = 'Err_VM_D')   
                    test_loss_VM_V_lst = plot_loss(test_loss_VM_V_lst, test_loss_VM_V, label = 'Err_VM_V')   
                if args.stochastic: 
                    test_loss_Sigma_lst = plot_loss(test_loss_Sigma_lst, test_loss_Sigma, label = 'Err_Sigma')   
         
                gc.collect()
        
        # Change learning rate when model tends to converge #
        if lr_to_change:
            if len(test_loss_perf_lst) > 10 and np.array(test_loss_perf_lst[-5:]).mean() < lr_reduce_criterion * np.array(test_loss_perf_lst[:2]).mean():
                print('Reduce learning rate')
                for g in optimizer_PIANO.param_groups:
                    g['lr'] = g['lr'] * (1 - args.lr_reduce_rate)
                num_lr_reduce += 1
                lr_reduce_criterion *= 0.5
                if num_lr_reduce >= args.max_num_lr_reduce:
                    lr_to_change = False
                    
        # Extend batch_nT while approaching to GT_V
        if epoch % args.increase_loss_time_frame_freq == 0 and args.increase_loss_time_frame:
            if loss_time_frame < max_loss_time_frame:
                loss_time_frame = loss_time_frame + 1 
                print('Training batch_nT:', str(loss_time_frame))
        file.close()
    
    end = time.time()
    return

##############################################################################################################################

if __name__ == '__main__':

    on_server = True

    if on_server:
        SaveFld = make_dir('/playpen-raid2/peirong/Results/2D_Results')
        #DataFld = '/playpen-raid/peirong/Data/2d_Demo/adv_diff'
    else:
        SaveFld = make_dir('/home/peirong/biag-raid2/Results/2D_Results')
        #DataFld = '/home/peirong/biag-raid/Data/2d_Demo/adv_diff'

    main(args, SaveFld, ResumePaths = ResumePaths) 