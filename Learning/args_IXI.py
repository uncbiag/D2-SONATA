import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import SimpleITK as sitk

from utils import make_dir
from Learning.yeti_3d import main

'''
Core Options:
patch_data_dim : predict patch data dim
max_down_scales: maximum dowmsampling scales for coarse prediction;
niters_step1   : num_iters for coarser-only training phase;

'''

# TODO: Exp. settings


img_type = 'IXI' # NOTE choices = {"ISLES-MRP", "ISLES-CTP", "IXI", "DTISample"}
IXI_perf_pattern = 'diff_only' # NOTE: For IXI synthetic demo training -GT- movie type #
PD_perf_pattern = 'diff_only'
patch_data_dim = [32, 32, 32]

max_dowm_scales = 0
niters_step1 = 1000000

spacing = 2 # mm # For ISLES #
is_joint = False if 'only' in PD_perf_pattern else True

#%% Basic settings
parser = argparse.ArgumentParser('3D YETI')
parser.add_argument('--img_type', type = str, default = img_type, choices = {"ISLES-MRP", "ISLES-CTP", "IXI", "DTISample"})
parser.add_argument('--resolution', type = float, default = spacing, help = 'ground ttruth voxel resolutino level (mm)')
parser.add_argument('--is_resume', type = bool, default = False)
parser.add_argument('--adjoint', type = bool, default = True)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--model_type', type = str, default = 'unet', choices = {'unet', 'vae'})
parser.add_argument('--joint_predict', type = bool, default = is_joint)
parser.add_argument('--add_gan', type = bool, default = False)
parser.add_argument('--add_identifier', type = bool, default = False)
parser.add_argument('--add_perf_flag', type = bool, default = False)
parser.add_argument('--ident_weight', type = float, default = 0.002)
parser.add_argument('--ident_dt', type = int, default = 8)
parser.add_argument('--perf_loss', type = str, default = 'abs', choices = {'dyn', 'abs', 'both'})
parser.add_argument('--perf_loss_func', type = str, default = 'L1', choices = {'L1', 'L2'})
parser.add_argument('--smart_timing', type = bool, default = True)

#%% Add ground truth D for supervised learning during warm-up
parser.add_argument('--no_concentration_loss', type = bool, default = True, help = 'No training loss on concentration')

parser.add_argument('--GT_D', type = bool, default = False, help = 'Add loss on D for supervised learning')
parser.add_argument('--GT_D_weight', type = float, default = 10) #200
parser.add_argument('--GT_D_CO', type = bool, default = False, help = 'Add loss on Color-by-orientation of D for supervision')
parser.add_argument('--GT_D_CO_weight', type = float, default = 10) #200

parser.add_argument('--Plus_L', type = bool, default = False)
parser.add_argument('--GT_LU', type = bool, default = True) # For structure-informed supervision
parser.add_argument('--GT_L_weight', type = float, default = 10) #200
parser.add_argument('--GT_U_weight', type = float, default = 10) #200

parser.add_argument('--GT_V', type = bool, default = True) 
parser.add_argument('--GT_Phi', type = bool, default = False) 
parser.add_argument('--GT_V_weight', type = float, default = 10.)  

#%% Learning model settings
parser.add_argument('--integ_method', type = str, default = 'dopri5', choices=['dopri5', 'adams', 'rk4', 'euler'])
parser.add_argument('--initial_method', type = str, default = 'cayley', choices = ['henaff', 'cayley'])
parser.add_argument('--add_gaussian_kernel', type = bool, default = False)
parser.add_argument('--smooth_param_type', type = str, default = 'diff', choices = ['adv', 'diff', 'adv_diff'])
parser.add_argument('--kernel_size', type = int, default = 3)
parser.add_argument('--sigma', type = float, default = 0.75)
parser.add_argument('--laplacian_loss_weight', type = float, default = 10000)
parser.add_argument('--gradient_loss', type = bool, default = False)
parser.add_argument('--diff_gl_only', type = bool, default = False)
parser.add_argument('--edge_pres', type = bool, default = False) # False !!! otherwise would result in edge-enhancing
parser.add_argument('--gl_weight', type = float, default = 1.) # 500 # 1
# Choosing sgl_weight => SG loss ~ Perf loss #
parser.add_argument('--spatial_gradient_loss', type = bool, default = False)
parser.add_argument('--sgl_weight', type =  float, default = 0.01) # 0.1
parser.add_argument('--fa_loss', type = bool, default = False)  # TODO
parser.add_argument('--fa_weight', type = float, default = 1e-3) # 5e-4: magnitude ~ 0.1 of concentration loss (2: PI demo)
parser.add_argument('--ta_loss', type = bool, default = False)
parser.add_argument('--ta_weight', type = float, default = 0.1) # 1.
parser.add_argument('--color_orientation_loss', type = bool, default = False)  # TODO
parser.add_argument('--co_weight', type = float, default = 1e-2) # 1e-1
parser.add_argument('--frobenius_loss', type = bool, default = False)
parser.add_argument('--VesselAttention', type = bool, default = False) 
parser.add_argument('--va_weight', type = float, default = 1e-2) 
parser.add_argument('--fl_weight', type = float, default = 5e-4)  # 5e-3

# For ground truth types
parser.add_argument('--data_dim', type = list, default = patch_data_dim) # [32, 32, 32]
parser.add_argument('--max_down_scales', type = int, default = max_dowm_scales) # max down_sampling scales (>= 1)
parser.add_argument('--data_spacing', type = list, default = [1., 1., 1.]) # [spacing, spacing, spacing],[1., 1., 1.] # [0.9231, 0.8984, 0.8984] orig: [0.9, 1.8, 1.8], [1.846, 1.797, 1.797]: d_slc, d_row, s_col
#parser.add_argument('--stride_training', type = list, default = [20, 20, 20]) # spatial stride for generating testing dataset
# NOTE: (1) 2 * boundary_crop + stride_testing <= stride_testing; (2) 2 * boundary_crop <= data_dim
#parser.add_argument('--stride_testing', type = list, default = [6, 6, 6]) # [8, 8, 8], spatial stride for generating testing dataset
#parser.add_argument('--stride_testing', type = list, default = [16, 16, 16]) # [8, 8, 8], [16, 16, 16] spatial stride for generating testing dataset
parser.add_argument('--stride_testing', type = list, default = [16, 16, 16]) # [8, 8, 8], [16, 16, 16] spatial stride for generating testing dataset
# NOTE: boundary_crop >= 1 for BC == neumann
parser.add_argument('--boundary_crop_training', type = list, default = [4, 4, 4]) # [8, 8, 8]
parser.add_argument('--boundary_crop_testing', type = list, default = [4, 4, 4]) # [8, 8, 8]
parser.add_argument('--dirichlet_width', type = int, default = 1) # boundaries viewed as source function
parser.add_argument('--source_width', type = int, default = 1) # 1. boundaries viewed as source function

parser.add_argument('--perf_pattern', type = str, default = PD_perf_pattern, choices = ['adv_diff', 'adv_only', 'diff_only'])
parser.add_argument('--IXI_perf_pattern', type = str, default = IXI_perf_pattern, choices = ['adv_diff', 'adv_only', 'diff_only'])
parser.add_argument('--PD_D_type', type = str, default = 'full_spectral', \
    choices = ['constant', 'scalar', 'diag', 'full_spectral' 'full_cholesky', 'full_dual', 'full_spectral', 'full_semi_spectral'])
# TODO: note: in 2D, vector_div_free_clebsch == vector_div_free_stream
parser.add_argument('--PD_V_type', type = str, default = 'vector_div_free_stream', \
    choices = ['constant', 'vector', 'vector_div_free_clebsch', 'vector_div_free_stream', 'vector_div_free_stream_gauge', 'vector_HHD'])
parser.add_argument('--gauge_weight', type = float, default = 10.)

parser.add_argument('--t0', type = float, default = 0.0) # the beginning time point
# NOTE: fake one, used for ensuring numerical stability.
# NOTE: Remember to scale to real-scale afterwards
parser.add_argument('--dt', type = float, default = 0.01) # 0.01 
parser.add_argument('--nT', type = int, default = 1000) # 200: data_dim = 128, 100: data_dim = 64
parser.add_argument('--time_stepsize', type = float, default = 0.01, help = 'Time step for integration, \
    if is not None, the assigned value should be able to divide args.time_spacing')

parser.add_argument('--BC', type = str, default = 'cauchy', \
    choices = ['neumann', 'dirichlet', 'cauchy', 'source', 'source_neumann', 'dirichlet_neumann'])


#%% Training and testing settings
parser.add_argument('--n_filter', type = int, default = 64)  
parser.add_argument('--latent_variable_size', type = int, default = 128) # 5000
# batch_nT <= input_time_frame
parser.add_argument('--batch_nT', type=int, default = 40) # 5, 10, 16 time frame for integration in computing losses
parser.add_argument('--input_time_frame', type=int, default = 40) # 10 # input time_frame as GT_nT for V, D prediction
parser.add_argument('--increase_batch_nT', type=bool, default = False) #2
parser.add_argument('--increase_batch_nT_freq', type=int, default = 20) # 50. NOTE: batch_nT <= input_time_frame during training
parser.add_argument('--max_batch_nT', type=int, default = 25) # 5, 10, 16
parser.add_argument('--batch_size', type=int, default = 1)
parser.add_argument('--opt_type', type = str, default = 'Adam') # Adam does not perform well
parser.add_argument('--lr', type = float, default = 2e-4, help = 'Model learning rate') # 1e-4 for spa == 2 mm; 5e-4 for 1 mm, 1e-3
parser.add_argument('--lr_weight_decay', type = float, default = 0.001)
parser.add_argument('--lr_ident', type = float, default = 5e-4, help = 'Model learning rate')
parser.add_argument('--max_num_lr_reduce', type = int, default = 2)
parser.add_argument('--lr_reduce_rate', type = int, default = 0.8)
parser.add_argument('--lr_reduce_criterion', type = float, default = 0.5)
parser.add_argument('--n_epochs_total', type = int, default = 100000000)
parser.add_argument('--niters_adv_only', type = int, default = 0)
parser.add_argument('--niters_step1', type = int, default = niters_step1)
parser.add_argument('--print_freq', type = int, default = 5) # 10
parser.add_argument('--test_freq', type = int, default = 100) # 20
parser.add_argument('--smooth_when_learn', type = bool, default = False)
parser.add_argument('--use_stop_crit', type = bool, default = False) 
parser.add_argument('--stop_crit', type = float, default = 0.001) # Loss reduce rate for one iteration
parser.add_argument('--max_stop_count', type = int, default = 1, help  = 'Stop when stop_count >= max_stop_count') 

args_ixi = parser.parse_args()  







