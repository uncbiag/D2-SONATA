import os, sys, argparse, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

#import SimpleITK as sitk

from utils import make_dir

import torch

from Learning.yeti_3d import main

'''
source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peirong/anaconda3/lib/
'''


'''
Core Options:
patch_data_dim : predict patch data dim
max_down_scales: maximum dowmsampling scales for coarse prediction;
'''



#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
#print('Device:', device)



on_server = True


#############################
# TODO: Data input settings #

img_type = 'IXI' # NOTE choices = {"MRP", "CTP", "IXI", "DTISample"} 

train_use_normal = True
train_use_joint_lesion = True # joint lesion mask for D and V
train_use_separate_lesion = True # separate lesio masks (same lesion segmentation mask, different anomaly field intensities) for D and V

test_use_normal = True # False if args.for_test  
test_use_joint_lesion = True 
test_use_separate_lesion = True # separate lesion masks (same lesion segmentation mask, different anomaly field intensities) for D and V

#############################
#############################

img2nda_rewrite = False # Set as True when processed images are updated #  
IXI_perf_pattern = 'adv_diff' # NOTE: For IXI synthetic demo training -GT- movie type # choices = ['adv_diff', 'adv_only', 'diff_only']
PD_perf_pattern = 'adv_diff' # choices = ['adv_diff', 'adv_only', 'diff_only']

input_n_collocations = 5
loss_n_collocations  = 5

if img_type == 'IXI':
    dt = 0.01

    batch_size = 10 # TODO 

    sub_collocation_nt = 5 # >= 2
    collocation_len = 2
    V_magnitude = 1.
    D_magnitude = 1.
     
    is_resume = False 
    if is_resume:
        test_freq = 20 # 20
        lr = 1e-3 # TODO
        #lr = 5e-4 # TODO
    else:
        test_freq = 20 # 20
        lr = 1e-3

else:
    is_resume = True 

    lr = 1e-4 # TODO: test on small lr #
    dt = 0.05
    batch_size = 1
    sub_collocation_nt = 5 # >= 2
    collocation_len = 2
 
    V_magnitude = 1
    D_magnitude = 0.05

    test_freq = 1  # 1, 20


if PD_perf_pattern == 'diff_only':
    resume_fld = '/media/peirong/PR5/IXI_Results/diff_only/Old_Processed/[NoConc]_[GT_LU(100, 100)]_[37]-unet_full_spectral - cauchy/4600'
    ResumeModelPath = os.path.join(resume_fld, 'checkpoint (nCol=10).pth')
elif PD_perf_pattern == 'adv_only':
    resume_fld = '/media/peirong/PR5/IXI_Results/adv_only/[NoConc]_Vess_[GT_V(1000]_[37]-unet_vector_div_free_stream - cauchy/4000'
    ResumeModelPath = os.path.join(resume_fld, 'checkpoint (nCol=10).pth')
elif PD_perf_pattern == 'adv_diff':
    resume_fld = '/playpen-raid2/peirong/Results/IXI_Results/adv_diff/[5]_[N_L_sepL--N_L_sepL]_[NoConc]_sep-SDE(10)_[DV-VM+]'   
    ResumeModelPath = os.path.join(resume_fld, 'latest_checkpoint.pth') 
    ResumePaths = {'root': resume_fld, 'model': ResumeModelPath}


patch_data_dim = [32, 32, 32]

#%% Basic settings
parser = argparse.ArgumentParser('3D PIANOinD')
parser.add_argument('--img_type', type = str, default = img_type, choices = {"MRP", "CTP", "IXI", "DTISample"})
parser.add_argument('--model_type', type = str, default = 'unet', choices = {'unet', 'vae'})
parser.add_argument('--is_resume', type = bool, default = is_resume) # TODO
parser.add_argument('--place_holder', type = str, default = False)
parser.add_argument('--adjoint', type = bool, default = True)  
parser.add_argument('--perf_loss', type = str, default = 'abs', choices = {'dyn', 'abs', 'both'})
parser.add_argument('--perf_loss_func', type = str, default = 'L1', choices = {'L1', 'L2'})
parser.add_argument('--smart_timing', type = bool, default = True)

# Model main configurations #  

######################
## TODO: Under Test ##
######################
parser.add_argument('--V_time', type = bool, default = False)
######################

parser.add_argument('--predict_value_mask', type = bool, default = True, help = 'If True, predict V(or D) as V := \bar{V} * value_mask') # default: True #
parser.add_argument('--separate_DV_value_mask', type = bool, default = True, help = 'Whether separately predict anomaly value mask for D and V') # default: True #
parser.add_argument('--actual_physics_loss', type = bool, default = True, help = 'If True, compute loss for base_D * value_mask') # default: True #

parser.add_argument('--stochastic', type = bool, default = True, help = 'Model as normal PDE or stochastic-PDE') # default: True #
parser.add_argument('--stochastic_separate_net', type = bool, default = True, help = 'Model as normal PDE or stochastic-PDE') # default: True #

## NOTE: Archieved: not working ##
parser.add_argument('--joint_predict', type = bool, default = False, help = 'Whether use joint decoder for V and D')  
parser.add_argument('--vm_sde_net', type = bool, default = False, help = 'sde and value_mask predicted in one separate network')
parser.add_argument('--value_mask_separate_net', type = bool, default = False, help = 'If True, predict delta_V and delta_D in an independent network') # NOTE: not work well
parser.add_argument('--predict_segment', type = bool, default = False, help = 'Whether add a separate segmentation network')  
parser.add_argument('--segment_condition_on_physics', type = bool, default = False, help = 'Whether multiply predicted SegMask with delta_out')  
parser.add_argument('--segment_net_type', type = str, default = 'conc', choices = {'conc', 'dev'}, help = 'input choices: conc time-series or physics deviation (must w/ predict_deviation)')  
parser.add_argument('--predict_deviation', type = bool, default = False, help = 'If True, predict V(or D) as V := \bar{V} + \delta{V}')    
parser.add_argument('--deviation_separate_net', type = bool, default = False, help = 'If True, predict delta_V and delta_D in an independent network')     
parser.add_argument('--deviation_extra_weight', type = float, default = 0., help = 'Extra weights for physics deviation supervision (if predict_deviation)')
parser.add_argument('--jbld_loss', type = bool, default = False, help = 'JBLD Distance loss for diffusion tensor')
parser.add_argument('--jbld_loss_only', type = bool, default = False, help = 'Using JBLD_distance as the ONLY loss for tensors') ## NOTE -- easy to explode 

## SDE configs ##
parser.add_argument('--VM_weight', type = float, default = 10., help = 'useful if predict_value_mask == True')
parser.add_argument('--seg_weight', type = float, default = 10., help = 'useful if predict_segment == True')
parser.add_argument('--SDE_loss', type = bool, default = True)
parser.add_argument('--SDE_weight', type = float, default = 10)
parser.add_argument('--compute_uncertainty', type = int, default = 0) # n_samples (0: not computing uncertainty when testing)

## NOTE: only for real data ##
parser.add_argument('--VesselMasking', type = bool, default = False)  # Set to True for real data
parser.add_argument('--DiffusionMasking', type = bool, default = False)

parser.add_argument('--D_magnitude', type = float, default = D_magnitude)
parser.add_argument('--V_magnitude', type = float, default = V_magnitude)

#%% Add ground truth D for supervised learning during warm-up

parser.add_argument('--no_concentration_loss', type = bool, default = True, help = 'No training loss on concentration')
# Choosing sgl_weight => SG loss ~ Perf loss #
parser.add_argument('--spatial_gradient_loss', type = bool, default = True)
parser.add_argument('--sgl_weight', type =  float, default = 1.) # 0.1
parser.add_argument('--gradient_loss', type = bool, default = False)
parser.add_argument('--gl_weight', type = float, default = 1.) # 500 # 1  

parser.add_argument('--GT_V', type = bool, default = True, help = 'Add loss on V for supervised learning') 
parser.add_argument('--GT_V_weight', type = float, default = 10.)  

parser.add_argument('--GT_D', type = bool, default = True, help = 'Add loss on D for supervised learning')
parser.add_argument('--GT_D_weight', type = float, default = 10) #200
parser.add_argument('--GT_LU', type = bool, default = True, help = 'Add loss on L & U for supervised learning') # For spectral 
parser.add_argument('--GT_L_weight', type = float, default = 10) #300
parser.add_argument('--GT_U_weight', type = float, default = 10) #100

# NOTE: Archieved: not working # 
parser.add_argument('--GT_D_CO', type = bool, default = False, help = 'Add loss on Color-by-orientation of D for supervision')
parser.add_argument('--GT_D_CO_weight', type = float, default = 1) #200 
parser.add_argument('--GT_Phi', type = bool, default = False, help = 'Add loss on Phi for supervised learning') 
parser.add_argument('--GT_Phi_weight', type = float, default = 1.)  


#%% Learning model settings
parser.add_argument('--integ_method', type = str, default = 'dopri5', choices=['dopri5', 'adams', 'rk4', 'euler'])
parser.add_argument('--initial_method', type = str, default = 'cayley', choices = ['henaff', 'cayley'])     

# For ground truth types
parser.add_argument('--data_dim', type = list, default = patch_data_dim) # [32, 32, 32]
parser.add_argument('--data_spacing', type = list, default = [1., 1., 1.]) # [spacing, spacing, spacing],[1., 1., 1.] # [0.9231, 0.8984, 0.8984] orig: [0.9, 1.8, 1.8], [1.846, 1.797, 1.797]: d_slc, d_row, s_col
#parser.add_argument('--stride_training', type = list, default = [20, 20, 20]) # spatial stride for generating testing dataset


# NOTE: (1) 2 * boundary_crop + stride_testing <= stride_testing; (2) 2 * boundary_crop <= data_dim 
parser.add_argument('--stride_testing', type = list, default = [16, 16, 16]) # [8, 8, 8], [16, 16, 16] spatial stride for generating testing dataset # NOTE: For testing during training ##
#parser.add_argument('--stride_testing', type = list, default = [8, 8, 8]) # [8, 8, 8], [16, 16, 16] spatial stride for generating testing dataset # NOTE: For final results to look smoother ##
# NOTE: boundary_crop >= 1 for BC == neumann
parser.add_argument('--boundary_crop_training', type = list, default = [4, 4, 4]) # [8, 8, 8]
parser.add_argument('--boundary_crop_testing', type = list, default = [8, 8, 8]) # [8, 8, 8] ## NOTE: For testing during training ##
#parser.add_argument('--boundary_crop_testing', type = list, default = [4, 4, 4]) # [8, 8, 8] ## NOTE: For final results ##

parser.add_argument('--dirichlet_width', type = int, default = 1) # boundaries viewed as source function
parser.add_argument('--source_width', type = int, default = 1) # 1. boundaries viewed as source function

parser.add_argument('--perf_pattern', type = str, default = PD_perf_pattern, choices = ['adv_diff', 'adv_only', 'diff_only'])
parser.add_argument('--IXI_perf_pattern', type = str, default = IXI_perf_pattern, choices = ['adv_diff', 'adv_only', 'diff_only'])

parser.add_argument('--train_use_joint_lesion', type = bool, default = train_use_joint_lesion) 
parser.add_argument('--test_use_joint_lesion', type = bool, default = test_use_joint_lesion) 
parser.add_argument('--train_use_separate_lesion', type = bool, default = train_use_separate_lesion) 
parser.add_argument('--test_use_separate_lesion', type = bool, default = test_use_separate_lesion) 

parser.add_argument('--PD_D_type', type = str, default = 'full_spectral', \
    choices = ['constant', 'scalar', 'diag', 'full_spectral' 'full_cholesky', 'full_dual', 'full_spectral', 'full_semi_spectral'])
# TODO: note: in 2D, vector_div_free_clebsch == vector_div_free_stream
parser.add_argument('--PD_V_type', type = str, default = 'vector_div_free_stream', \
    choices = ['constant', 'vector', 'vector_div_free_clebsch', 'vector_div_free_stream', 'vector_div_free_stream_gauge', 'vector_HHD'])
parser.add_argument('--gauge_weight', type = float, default = 10.)


# NOTE Time points settings #
parser.add_argument('--t0', type = float, default = 0.0) # the beginning time point
# NOTE: fake one, used for ensuring numerical stability, scale to real-scale afterwards #
parser.add_argument('--dt', type = float, default = dt, help = 'time interval unit') # 0.01 , 0.02
# NOTE: time points (\Delta t) between two collocation points = sub_collocation_nt * collocation_len #
parser.add_argument('--sub_collocation_nt', type = int, default = sub_collocation_nt, help = 'time points between two sub_collocation points (ode steps)') # 25
parser.add_argument('--collocation_len', type = int, default = collocation_len, help = '# of sub_collocations between two collocation points collocation_nt / sub_collocation_nt') 
# loss_n_collocations (collocations for computing concentration losses) <= input_n_collocations
parser.add_argument('--input_n_collocations', type = int, default = input_n_collocations, help = 'Number of collocations as network input for V, D prediction') 
parser.add_argument('--loss_n_collocations', type = int, default = loss_n_collocations,  help = 'Initial collocation time points used in training samples')  # 10
parser.add_argument('--increase_input_n_collocations', type = bool, default = False,  help = 'whether increase input collocation points during training') 
parser.add_argument('--max_loss_n_collocations', type = int, default = 25, \
    help = '(if increase_loss_n_collocations_freq) max collocation time points used in training samples') # 5, 10, 16
parser.add_argument('--increase_loss_n_collocations_freq', type = int, default = 20) # 50. NOTE: batch_nT <= input_n_collocations during training

'''parser.add_argument('--time_stepsize', type = float, default = 0.01, help = 'Time step for integration, \
    if is not None, the assigned value should be able to divide args.time_spacing') # Useless for adaptive time integration #'''

parser.add_argument('--BC', type = str, default = 'cauchy', \
    choices = ['None', 'neumann', 'dirichlet', 'cauchy', 'source', 'source_neumann', 'dirichlet_neumann'])


#%% Training and testing settings
parser.add_argument('--n_filter', type = int, default = 64)  
parser.add_argument('--latent_variable_size', type = int, default = 128) # 5000, For VAE
parser.add_argument('--batch_size', type = int, default = batch_size) # 16, 5
parser.add_argument('--opt_type', type = str, default = 'Adam')
parser.add_argument('--lr', type = float, default = lr, help = 'Model learning rate') # 1e-3, 5e-3, 1-2
parser.add_argument('--lr_weight_decay', type = float, default = 0.001) # 0.001
parser.add_argument('--lr_ident', type = float, default = 5e-4, help = 'Model learning rate')
parser.add_argument('--max_num_lr_reduce', type = int, default = 2)
parser.add_argument('--lr_reduce_rate', type = int, default = 0.8)
parser.add_argument('--lr_reduce_criterion', type = float, default = 0.5)
parser.add_argument('--n_epochs_total', type = int, default = 100000000)
parser.add_argument('--niters_adv_only', type = int, default = 0)
parser.add_argument('--print_freq', type = int, default = 1) # 10
parser.add_argument('--test_freq', type = int, default = test_freq) # 100. 5
parser.add_argument('--smooth_when_learn', type = bool, default = False)
parser.add_argument('--use_stop_crit', type = bool, default = False) 
parser.add_argument('--stop_crit', type = float, default = 0.001) # Loss reduce rate for one iteration
parser.add_argument('--max_stop_count', type = int, default = 1, help  = 'Stop when stop_count >= max_stop_count') 


parser.add_argument('--gpu', type = str, required = True, help = 'Select which gpu to use')
parser.add_argument('--for_test', type = bool, default = False, help = 'for testing') 


args_3D = parser.parse_args()  

if 'MRP' in args_3D.img_type or 'CTP' in args_3D.img_type:
    args_3D.GT_V = False 
    args_3D.GT_D = False 
    args_3D.GT_LU = False 
    args_3D.GT_Phi = False 
    args_3D.GT_D_CO = False 
    args_3D.no_concentration_loss = False  
    args_3D.VesselMasking = True


##############################################################################################################################


def img2nda(img_path, is_rewrite = False):
    nda_path = '%s.npy' % img_path[:-4]
    if not os.path.isfile(nda_path) or is_rewrite:
        np.save(nda_path, sitk.GetArrayFromImage(sitk.ReadImage(img_path)))
    return nda_path


def get_MRP_paths(patient_fld):
    '''
    Paths reader for ISLES2017 MRP folder
    '''
    for module in os.listdir(patient_fld):
        MaskPath = img2nda(os.path.join(patient_fld, 'BrainMask.mha'))
        if "MRP" in module and not module.startswith('.'):
            #CTC_path = img2nda(os.path.join(patient_fld, "MRP/CTC.mha"), is_rewrite = img2nda_rewrite)
            #InfoPath = os.path.join(patient_fld, 'MRP/Info.txt')
            CTC_path = img2nda(os.path.join(patient_fld, "MRP/CTC_fromTTP.mha"), is_rewrite = img2nda_rewrite)
            InfoPath = os.path.join(patient_fld, 'MRP/Info_fromTTP.txt')
            #VesselPath =  img2nda(os.path.join(patient_fld, 'Vessel.mha'))
            #VesselPath =  img2nda(os.path.join(patient_fld, 'Vessel_mirrored.mha'))
            #VesselPath =  img2nda(os.path.join(patient_fld, 'Vessel_mirrored_smoothed.mha')) # TODO
            #VesselPath =  img2nda(os.path.join(patient_fld, 'Vessel_mirrored_smoothed_bin.mha')) # TODO
            VesselPath =  img2nda(os.path.join(patient_fld, 'VesselEnhanced_normalized.mha')) # TODO
            VesselMirrorPath =  img2nda(os.path.join(patient_fld, 'VesselEnhanced_normalized_mirrored.mha')) # TODO
            ValueMaskPath =  img2nda(os.path.join(patient_fld, 'ValueMask_BS.mha')) # TODO

    return {'name': os.path.basename(patient_fld), 'movie': CTC_path, 'mask': MaskPath, 'vessel_mask': VesselPath, 'vessel_mirror_mask': VesselMirrorPath, 'info': InfoPath, \
        'value_mask': ValueMaskPath}
 

def get_CTP_paths(patient_fld):
    '''
    Paths reader for ISLES2018 CTP folder
    '''
    for module in os.listdir(patient_fld): # TODO
        if "PWI" in module and not module.startswith('.'):
            CTC_path = img2nda(os.path.join(patient_fld, "%s/CTC_Axial_TTPtoTTD_norm_Res(%s).npy" % (module, spacing)), is_rewrite = img2nda_rewrite)
            InfoPath = os.path.join(patient_fld, module, 'Info_Res(%s).txt' % spacing)
            
            MaskPath = img2nda(os.path.join(patient_fld, module, 'Mask_Res(%s).nii' % spacing), is_rewrite = img2nda_rewrite)
            VesselPath = None
            ContourPath = img2nda(os.path.join(os.path.dirname(MaskPath), 'Contour_Res(%s).nii' % spacing), is_rewrite = img2nda_rewrite)
    return {'name': os.path.basename(patient_fld), 'movie': CTC_path, 'mask': MaskPath, 'vessel_mask': VesselPath, 'info': InfoPath}


def get_IXI_paths(args, case_fld, movie_fld, movie_type = 'adv_diff', PD_perf_pattern = 'adv_diff', lesion_type = None):
    '''
    Paths reader for IXI synthetic movies folder
    '''
    InfoPath = os.path.join(movie_fld, 'Info.txt')
    MaskPath = img2nda(os.path.join(movie_fld, 'BrainMask.mha'), is_rewrite = img2nda_rewrite)
    VesselPath = img2nda(os.path.join(case_fld, 'VesselMask_smoothed.mha'), is_rewrite = img2nda_rewrite)

    if lesion_type is None:
        ValueMask_Path, LesionSeg_Path = None, None 
    elif lesion_type == 'joint':
        ValueMask_Path = img2nda(os.path.join(movie_fld, 'ValueMask.mha'), is_rewrite = img2nda_rewrite)
        LesionSeg_Path = img2nda(os.path.join(movie_fld, 'LesionSeg.mha'), is_rewrite = img2nda_rewrite) 
    elif lesion_type == 'separate':
        ValueMask_D_Path = img2nda(os.path.join(movie_fld, 'ValueMask_D.mha'), is_rewrite = img2nda_rewrite)
        ValueMask_V_Path = img2nda(os.path.join(movie_fld, 'ValueMask_V.mha'), is_rewrite = img2nda_rewrite)
        LesionSeg_D_Path = img2nda(os.path.join(movie_fld, 'LesionSeg_D.mha'), is_rewrite = img2nda_rewrite) 
        LesionSeg_V_Path = img2nda(os.path.join(movie_fld, 'LesionSeg_V.mha'), is_rewrite = img2nda_rewrite) 
        
        ValueMask_Path = {'D': ValueMask_D_Path, 'V': ValueMask_V_Path}
        LesionSeg_Path = {'D': LesionSeg_D_Path, 'V': LesionSeg_V_Path}
    else:
        ValueError('Not supported lesion_type:', lesion_type)

    if 'adv_only' in movie_type:
        MoviePath = img2nda(os.path.join(movie_fld, 'Adv.mha'), is_rewrite = img2nda_rewrite)
    elif 'diff_only' in movie_type:
        MoviePath = img2nda(os.path.join(movie_fld, 'Diff.mha'), is_rewrite = img2nda_rewrite)
    elif 'adv_diff' in movie_type:
        MoviePath = img2nda(os.path.join(movie_fld, 'AdvDiff.mha'), is_rewrite = img2nda_rewrite)
    else:
        raise ValueError('Unsupported IXI GT movie type.')

    D_Path = img2nda(os.path.join(movie_fld, 'D.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    L_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/L.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    U_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/U.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    FA_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/FA.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    Trace_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/Trace.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    D_CO_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/D_Color_Direction.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None

    V_Path = img2nda(os.path.join(movie_fld, 'V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None
    AbsV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/Abs_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None
    NormV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/Norm_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None 
    Phi_Path = None # NOTE: archived, not work well

    path_dict = {'name': os.path.basename(case_fld), 'movie': MoviePath, 'mask': MaskPath, 'vessel_mask': VesselPath, 'info': InfoPath, \
        'D': D_Path, 'L': L_Path, 'U': U_Path, 'D_CO': D_CO_Path, 'FA': FA_Path, 'Trace': Trace_Path, \
        'V': V_Path, 'Abs_V': AbsV_Path, 'Norm_V': NormV_Path, 'Phi': Phi_Path, \
        'value_mask': ValueMask_Path, 'lesion_seg': LesionSeg_Path}  
 
    if args.predict_deviation and 'Lesion' in movie_fld:
        orig_D_Path = img2nda(os.path.join(movie_fld, 'orig_D.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
        delta_D_Path = img2nda(os.path.join(movie_fld, 'delta_D.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None

        orig_L_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/orig_L.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None 
        delta_L_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/delta_L.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None 

        orig_Trace_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/orig_Trace.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None 
        delta_Trace_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/delta_Trace.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None 

        orig_V_Path = img2nda(os.path.join(movie_fld, 'orig_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None
        delta_V_Path = img2nda(os.path.join(movie_fld, 'delta_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None

        orig_AbsV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/orig_Abs_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None
        delta_AbsV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/delta_Abs_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None

        orig_NormV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/orig_Norm_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None
        delta_NormV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/delta_Norm_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None

        path_dict.update({'orig_D': orig_D_Path, 'delta_D': delta_D_Path, 'orig_L': orig_L_Path, 'delta_L': delta_L_Path, 'orig_Trace': orig_Trace_Path, 'delta_Trace': delta_Trace_Path,\
            'orig_V': orig_V_Path, 'delta_V': delta_V_Path, 'orig_Abs_V': orig_AbsV_Path, 'delta_Abs_V': delta_AbsV_Path, 'orig_Norm_V': orig_NormV_Path, 'delta_Norm_V': delta_NormV_Path})
    else:
        path_dict.update({'orig_D': None, 'delta_D': None, 'orig_L': None, 'delta_L': None, 'orig_Trace': None, 'delta_Trace': None,\
            'orig_V': None, 'delta_V': None, 'orig_Abs_V': None, 'delta_Abs_V': None, 'orig_Norm_V': None, 'delta_Norm_V': None})

    return  path_dict



def learn_IXI(args, AllFolder, SaveFolder, TrainCases, TestCases, ResumePaths = None):
    '''
    Trainer for ISLES Datasets (2017 - MRP or 2018 - CTP)
    '''
    print("# of training cases:", len(TrainCases))
    TrainCasePaths = [os.path.join(AllFolder, case_name.split('\n')[0]) for case_name in TrainCases]
    TrainMovies, TrainInfos, TrainMasks, TrainVesselMasks, TrainDs, TrainD_COs, TrainD_Ls, TrainD_Traces, TrainD_Us, TrainVs, TrainAbs_Vs, TrainNorm_Vs, TrainV_Phis, \
        TrainValueMasks, TrainValueMasks_V, TrainValueMasks_D, TrainLesionSegs = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    Trainorig_Ds, Traindelta_Ds, TrainD_orig_Ls, TrainD_delta_Ls, TrainD_orig_Traces, TrainD_delta_Traces, \
        Trainorig_Vs, Traindelta_Vs, Trainorig_Abs_Vs, Traindelta_Abs_Vs, Trainorig_Norm_Vs, Traindelta_Norm_Vs = [], [], [], [], [], [], [], [], [], [], [], [],
    for i_case in range(len(TrainCasePaths)):
        if not os.path.isdir(TrainCasePaths[i_case]):
            print('   Skip non-existed case #%d: %s' % (i_case+1, TrainCases[i_case].split('\n')[0]))
        else:
            print('   #%d: %s' % (i_case+1, TrainCases[i_case].split('\n')[0]))

            ####################################
            if train_use_normal:  
                movie_main_fld = os.path.join(TrainCasePaths[i_case], 'Movies')  
                    
                paths_dict_lst = []
                ####################################
                ########### Normal cases ###########
                #################################### 
                paths_dict = get_IXI_paths(args, TrainCasePaths[i_case], movie_main_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern) 
                paths_dict_lst.append(paths_dict)

            ###########################################
            ### Additionally add joint lesion cases ###
            ###########################################
            if train_use_joint_lesion and os.path.isdir(os.path.join(TrainCasePaths[i_case], 'Movies-Lesion')):  
                movie_main_fld = os.path.join(TrainCasePaths[i_case], 'Movies-Lesion') 
                paths_dict = get_IXI_paths(args, TrainCasePaths[i_case], movie_main_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern, lesion_type = 'joint')
                paths_dict_lst.append(paths_dict)

            #########################################
            ### Additionally add sep-lesion cases ###
            #########################################
            if train_use_separate_lesion and os.path.isdir(os.path.join(TrainCasePaths[i_case], 'Movies-SepLesion')):  
                movie_main_fld = os.path.join(TrainCasePaths[i_case], 'Movies-SepLesion') 
                paths_dict = get_IXI_paths(args, TrainCasePaths[i_case], movie_main_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern, lesion_type = 'separate')
                paths_dict_lst.append(paths_dict)

            # Append all path_dicts from the current case #
            for paths_dict in paths_dict_lst:
                TrainInfos.append(paths_dict['info'])
                TrainMovies.append(paths_dict['movie'])
                TrainMasks.append(paths_dict['mask'])
                TrainVesselMasks.append(paths_dict['vessel_mask'])

                TrainDs.append(paths_dict['D'])
                Trainorig_Ds.append(paths_dict['orig_D'])
                Traindelta_Ds.append(paths_dict['delta_D'])

                TrainD_Ls.append(paths_dict['L'])
                TrainD_orig_Ls.append(paths_dict['orig_L'])
                TrainD_delta_Ls.append(paths_dict['delta_L'])

                TrainD_Traces.append(paths_dict['Trace'])
                TrainD_orig_Traces.append(paths_dict['orig_Trace'])
                TrainD_delta_Traces.append(paths_dict['delta_Trace'])

                TrainD_Us.append(paths_dict['U'])
                TrainD_COs.append(paths_dict['D_CO'])
                
                TrainVs.append(paths_dict['V'])
                Trainorig_Vs.append(paths_dict['orig_V'])
                Traindelta_Vs.append(paths_dict['delta_V']) 

                TrainAbs_Vs.append(paths_dict['Abs_V'])
                Trainorig_Abs_Vs.append(paths_dict['orig_Abs_V'])
                Traindelta_Abs_Vs.append(paths_dict['delta_Abs_V'])

                TrainNorm_Vs.append(paths_dict['Norm_V'])
                Trainorig_Norm_Vs.append(paths_dict['orig_Norm_V'])
                Traindelta_Norm_Vs.append(paths_dict['delta_Norm_V'])

                TrainV_Phis.append(paths_dict['Phi'])

                TrainValueMasks.append(paths_dict['value_mask'])
                TrainLesionSegs.append(paths_dict['lesion_seg'])
            
            #####################################
            TrainPaths = {'movies': TrainMovies, 'masks': TrainMasks, 'vessel_masks': TrainVesselMasks, 'infos': TrainInfos, \
                'D': {'D': TrainDs, 'D_CO':TrainD_COs, 'L': TrainD_Ls, 'U': TrainD_Us, 'Trace': TrainD_Traces, \
                    'orig_D': Trainorig_Ds, 'delta_D': Traindelta_Ds, 'orig_L': TrainD_orig_Ls, 'delta_L': TrainD_delta_Ls, 'orig_Trace': TrainD_orig_Traces, 'delta_Trace': TrainD_delta_Traces}, \
                'V': {'V': TrainVs, 'Abs_V': TrainAbs_Vs, 'Norm_V': TrainNorm_Vs, 'Phi': TrainV_Phis, \
                    'orig_V': Trainorig_Vs, 'delta_V': Traindelta_Vs, 'orig_Abs_V': Trainorig_Abs_Vs, 'delta_Abs_V': Traindelta_Abs_Vs, 'orig_Norm_V': Trainorig_Norm_Vs, 'delta_Norm_V': Traindelta_Norm_Vs},
                'value_masks': TrainValueMasks, 'lesion_seg': TrainLesionSegs}


    print("# of testing cases:", len(TestCases))
    TestCasePaths = [os.path.join(AllFolder, case_name.split('\n')[0]) for case_name in TestCases]
    TestPaths = []
    for i_case in range(len(TestCasePaths)):
        print('   #%d: %s' % (i_case+1, TestCases[i_case].split('\n')[0]))
        if test_use_normal: 
            movie_fld = os.path.join(TestCasePaths[i_case], 'Movies') 
            paths_dict = get_IXI_paths(args, TestCasePaths[i_case], movie_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern)
            TestPaths.append(paths_dict)

        if test_use_joint_lesion and os.path.isdir(os.path.join(TestCasePaths[i_case], 'Movies-Lesion')):
            movie_fld = os.path.join(TestCasePaths[i_case], 'Movies-Lesion')
            paths_dict = get_IXI_paths(args, TestCasePaths[i_case], movie_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern, lesion_type = 'joint')
            TestPaths.append(paths_dict) 

        if test_use_separate_lesion and os.path.isdir(os.path.join(TestCasePaths[i_case], 'Movies-SepLesion')):
            movie_fld = os.path.join(TestCasePaths[i_case], 'Movies-SepLesion')
            paths_dict = get_IXI_paths(args, TestCasePaths[i_case], movie_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern, lesion_type = 'separate')
            TestPaths.append(paths_dict) 

    main(args, SaveFolder, TrainPaths, TestPaths, ResumePaths, have_GT = True) 



def learn_ISLES(args, AllFolder, SaveFolder, TrainPatients, TestPatients, ResumePaths = None, img_type = 'MRP'):
    '''
    Trainer for ISLES Datasets (2017 - MRP or 2018 - CTP)
    '''
    print("# of training cases:", len(TrainPatients))
    print("# of testing cases:", len(TestPatients))

    TrainPatientPaths = [os.path.join(AllFolder, patient) for patient in TrainPatients]
    TrainMovies = []
    TrainInfos = []
    TrainMasks = []
    TrainVesselMasks, TrainVesselMirrorMasks = [], []
    TrainValueMasks = []
    for i_patient in range(len(TrainPatientPaths)):
        print('   #%d: %s' % (i_patient+1, TrainPatients[i_patient].split('\n')[0]))
        if not TrainPatientPaths[i_patient].startswith('.') and 'training' in TrainPatientPaths[i_patient] or 'case' in TrainPatientPaths[i_patient]:
            if 'MRP' in img_type:
                paths_dict = get_MRP_paths(TrainPatientPaths[i_patient])
            elif 'CTP' in img_type:
                paths_dict = get_CTP_paths(TrainPatientPaths[i_patient])
            TrainInfos.append(paths_dict['info'])
            TrainMasks.append(paths_dict['mask'])
            TrainMovies.append(paths_dict['movie'])
            TrainVesselMasks.append(paths_dict['vessel_mask'])
            TrainVesselMirrorMasks.append(paths_dict['vessel_mirror_mask'])
            TrainValueMasks.append(paths_dict['value_mask'])
    TrainPaths = {'movies': TrainMovies, 'masks': TrainMasks, 'vessel_masks': TrainVesselMasks, 'value_masks': TrainValueMasks, 'vessel_mirror_masks': TrainVesselMirrorMasks, 'infos': TrainInfos}

    TestPatientPaths = [os.path.join(AllFolder, patient) for patient in TestPatients]
    TestPaths = []
    for i_patient in range(len(TestPatientPaths)):
        print('   #%d: %s' % (i_patient + 1, TestPatients[i_patient].split('\n')[0]))
        if not TestPatientPaths[i_patient].startswith('.') and 'training' in TestPatientPaths[i_patient] or 'case' in TestPatientPaths[i_patient]:
            if 'MRP' in img_type:
                paths_dict = get_MRP_paths(TestPatientPaths[i_patient])
            elif 'CTP' in img_type:
                paths_dict = get_CTP_paths(TestPatientPaths[i_patient])
            TestPaths.append(paths_dict)

    main(args, SaveFolder, TrainPaths, TestPaths, ResumePaths, have_GT = False)

    '''print('Now analyzing results for', Patients[i_patient]) # TODO: move this to main function
    param_analyze_Linear(PatientPaths[i_patient], result_fld, interp_method,
    to_smooth = True, V_sigma = 4, D_sigma = 0.5, use_BS = True)'''



##############################################################################################################################


if __name__ == '__main__':
    
    if 'CTP' in args_3D.img_type:

        print('######################### CTP #########################')

        ISLES2018 = '/media/peirong/PR5/ISLES2018/ISLES2018_Training/TRAINING'
        ISLES2018_SaveFld = make_dir('/media/peirong/PR5/ISLES2018_Results_DL')
        Patients = os.listdir(ISLES2018)
        Patients.sort()

        TestName = 'case_80'
        TestPatients = [TestName]

        TrainPatients = os.listdir(ISLES2018) # Analyze all cases
        TrainPatients.sort()
        TrainPatients = ['case_80']

        learn_ISLES(args_3D, ISLES2018, ISLES2018_SaveFld, TrainPatients, TestPatients, ResumePaths = ResumePaths, img_type = img_type)

    if 'MRP' in args_3D.img_type:

        print('######################### MRP #########################')

        if on_server:
            ISLES2017 = '/playpen1/peirong/Data/ISLES2017_Processed'
            #ISLES2017 = '/playpen-raid1/peirong/Data/ISLES2017_Processed'
            #ISLES2017 = '/playpen1/peirong/Data/ISLES2017_Processed_rotated' # rotated 
            #ISLES2017 = '/playpen-raid2/peirong/Data/ISLES2017_Processed_rotated' # rotated

            #ISLES2017_SaveFld = make_dir('/playpen-raid1/peirong/Results/ISLES2017_rotated_Results') # rotated
            ISLES2017_SaveFld = make_dir('/playpen-raid2/peirong/Results/ISLES2017_Results') 
        else:
            ISLES2017 = '/media/peirong/PR5/ISLES2017_Processed'
            ISLES2017_SaveFld = make_dir('/media/peirong/PR5/ISLES2017_Results_DL')

        TrainPatients = os.listdir(ISLES2017) # Analyze all cases
        TrainPatients = []
        for i_name in os.listdir(ISLES2017):
            if 'training_' in i_name: 
                TrainPatients.append(i_name)
        #TrainPatients.remove('IDs.txt')
        TrainPatients.remove('training_42') # Bad CTC trend # 
        #TrainPatients.remove('training_14') # For Testing # 
        TrainPatients.sort()

        TrainPatients = ['training_14']
        '''TrainPatients = ['training_1', 'training_2', 'training_4', 'training_5', \
            'training_7', 'training_8', 'training_9', 
            'training_10', 'training_11', 'training_12', 'training_15', 'training_18', 
            'training_20', 'training_21', 'training_27', 'training_28', 'training_30', 
            'training_31', 'training_32', 'training_33', 'training_36', 'training_37', 
            'training_38', 'training_39', 'training_40', 'training_41', 'training_42', 
            'training_43', 'training_44', 'training_45', 'training_47', 'training_48']'''  #  >32 cases
        '''TrainPatients = ['training_4', 'training_5',  'training_7',  'training_8', 
            'training_13', 'training_14', 'training_16', 'training_18', 
            'training_22', 'training_24', 'training_27', 
            'training_32', 'training_36', 'training_38', 'training_38', 'training_39', 
            'training_40', 'training_41', 'training_42', 'training_47', 'training_48']'''  # Good cases

        TestPatients = []
        for i_name in os.listdir(ISLES2017):
            if 'training_' in i_name: 
                TestPatients.append(i_name)

        TestPatients = ['training_14'] 
        TestPatients = ['training_13', 'training_22', 'training_32', 'training_38'] # Showcases for PAMI #
        '''TestPatients = ['training_4', 'training_5',  'training_7',  'training_8', 
            'training_13', 'training_14', 'training_16', 'training_18', 
            'training_22', 'training_24', 'training_27', 
            'training_32', 'training_36', 'training_38', 'training_38', 'training_39', 
            'training_40', 'training_41', 'training_42', 'training_47', 'training_48'] #'''  # Good cases

        learn_ISLES(args_3D, ISLES2017, ISLES2017_SaveFld, TrainPatients, TestPatients, ResumePaths)
        
    if 'IXI' in args_3D.img_type:
        
        # Synthesized IXI Demo # 
        if on_server:
            IXI = '/playpen1/peirong/Data/IXI_Processed' 
            #IXI = '/playpen-raid1/peirong/Data/IXI_Processed'
            #IXI_SaveFld = make_dir('/playpen-raid1/peirong/Results/IXI_Results')
            IXI_SaveFld = make_dir('/playpen-raid2/peirong/Results/IXI_Results-TEST')
        else:
            IXI = '/media/peirong/PR5/IXI_Processed'
            IXI_SaveFld = make_dir('/media/peirong/PR5/IXI_Results')

        print('######################### IXI #########################')

        names_file = open(os.path.join(IXI, 'IDs.txt'), 'r')
        case_names = names_file.readlines()
        names_file.close()
        #case_names = ['IXI002-Guys-0828'] # TODO
 
        # TODO: Set training and testing cases #
        TrainCases = case_names 
        #TrainCases.sort()# 
        TrainCases = case_names[:131]
        #TrainCases = ['IXI170-Guys-0843']

        TestCases = case_names 
        TestCases.sort()

        TestCases = case_names[:1]   
        TestCases = ['IXI033-HH-1259'] # Showcase for CVPR22 #
        TestCases = ['IXI052-HH-1343'] # Showcase for PAMI22 #
        TestCases = ['IXI033-HH-1259', 'IXI034-HH-1260', 'IXI052-HH-1343', 'IXI095-HH-1390', 'IXI148-HH-1453']

        learn_IXI(args_3D, IXI, IXI_SaveFld, TrainCases, TestCases, ResumePaths)

    '''if 'DTISample' in args_3D.img_type:
        
        # Synthesized Demo #
        
        # TODO: Set Paths #
        temp_fld = '/media/peirong/PR5/DTISample'
        movie_fld = os.path.join(temp_fld, 'Movies/%s' % PD_perf_pattern)
        Demo_SaveFld = make_dir(os.path.join(movie_fld, 'Results'))

        train_info_path = os.path.join(movie_fld, 'Info_Full.txt') # TODO
        #train_movie_path = os.path.join(movie_fld, 'GT_AxialPerf_D-full_cholesky_V-vector.nii') # TODO
        train_movie_path = img2nda(os.path.join(movie_fld, 'GT_AxialPerf_D-full_cholesky.nii'), is_rewrite = img2nda_rewrite) # TODO
        train_mask_path = img2nda(os.path.join(temp_fld, 'Mask_cropped.nii'), is_rewrite = img2nda_rewrite) # TODO

        GT_D_path = img2nda(os.path.join(movie_fld, 'D.nii'), is_rewrite = img2nda_rewrite)
        GT_V_path = img2nda(os.path.join(movie_fld, 'V.nii'), is_rewrite = img2nda_rewrite)
        GT_L_path = img2nda(os.path.join(movie_fld, 'PTIMeasures/L (GT).nii'), is_rewrite = img2nda_rewrite)
        GT_U_path = img2nda(os.path.join(movie_fld, 'PTIMeasures/U (GT).nii'), is_rewrite = img2nda_rewrite)
        L_base_path = img2nda(os.path.join(movie_fld, 'PTIMeasures/L_MD (GT).nii'), is_rewrite = img2nda_rewrite)
        D_base_path = img2nda(os.path.join(movie_fld, 'PTIMeasures/FullD_MD (GT).nii'), is_rewrite = img2nda_rewrite)
        GT_Dco_path = img2nda(os.path.join(movie_fld, 'PTIMeasures/D_Color_Direction (GT).nii'), is_rewrite = img2nda_rewrite)

        TrainPaths = {'movies': [train_movie_path], 'masks': [train_mask_path], 'infos': [train_info_path], \
            'D': {'D':[GT_D_path], 'L_base': [L_base_path], 'D_CO': [GT_Dco_path], 'L': [GT_L_path], 'U': [GT_U_path]},\
            'V': {'V': [GT_V_path]}}
        TestPaths = [{'movie': train_movie_path, 'mask': train_mask_path, 'info': train_info_path, 'L_base': GT_L_path}]

        main(args_3D, Demo_SaveFld, TrainPaths, TestPaths, ResumePaths)'''