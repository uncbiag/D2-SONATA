import os, sys, argparse, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
#import SimpleITK as sitk

from utils import make_dir

import torch

from Learning.piano_3d import main

'''
source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peirong/anaconda3/lib/
'''


'''
Core Options:
patch_data_dim : predict patch data dim
max_down_scales: maximum dowmsampling scales for coarse prediction;
'''


on_server = True

# TODO: Data input settings #

img_type = 'MRP' # NOTE choices = {"MRP", "CTP", "IXI", "DTISample"}

img2nda_rewrite = False # Set as True when processed images are updated # 
IXI_div_free = False
IXI_perf_pattern = 'adv_diff' # NOTE: For IXI synthetic demo training -GT- movie type # choices = ['adv_diff', 'adv_only', 'diff_only']
PD_perf_pattern = 'adv_diff' # choices = ['adv_diff', 'adv_only', 'diff_only']

input_n_collocations = 16
loss_n_collocations  = 16

if img_type == 'IXI':
    lr = 1e-3
    dt = 0.01
    batch_size = 50
    sub_collocation_nt = 5 # >= 2
    collocation_len = 2
else:
    lr = 1e-2
    dt = 0.05
    batch_size = 5
    sub_collocation_nt = 2 # >= 2
    collocation_len = 5 


test_freq = 20 # 100, 20
is_resume = False # TODO #
if PD_perf_pattern == 'diff_only':
    resume_fld = '/media/peirong/PR5/IXI_Results/diff_only/Old_Processed/[NoConc]_[GT_LU(100, 100)]_[37]-unet_full_spectral - cauchy/4600'
    ResumeModelPath = os.path.join(resume_fld, 'checkpoint (nCol=10).pth')
elif PD_perf_pattern == 'adv_only':
    resume_fld = '/media/peirong/PR5/IXI_Results/adv_only/[NoConc]_Vess_[GT_V(1000]_[37]-unet_vector_div_free_stream - cauchy/4000'
    ResumeModelPath = os.path.join(resume_fld, 'checkpoint (nCol=10).pth')
elif PD_perf_pattern == 'adv_diff':
    resume_fld = '/playpen-raid/peirong/Results/IXI_Results/adv_diff/[NoConc]_Vess_[GT_V(100]_[GT_LU(100, 100)]_[98]-vector_div_free_stream_full_spectral - cauchy/3500'
    ResumeModelPath = os.path.join(resume_fld, 'checkpoint (nCol=5).pth')
    #resume_fld = '/media/peirong/PR5/IXI_Results/adv_diff/[NoConc]_Vess_[GT_V(100]_[GT_LU(100, 100)]_[98]-vector_div_free_stream_full_spectral - cauchy/Res(1600)_[NoConc]_Vess_[GT_V(100]_[GT_LU(100, 100)]_[98]-vector_div_free_stream_full_spectral - cauchy/Res(800)_[NoConc]_Vess_[GT_V(100]_[GT_LU(100, 100)]_[98]-vector_div_free_stream_full_spectral - cauchy/1500'
    #ResumeModelPath = os.path.join(resume_fld, 'checkpoint (nCol=10).pth')


patch_data_dim = [32, 32, 32]

#%% Basic settings
parser = argparse.ArgumentParser('3D PIANOinD')
parser.add_argument('--img_type', type = str, default = img_type, choices = {"MRP", "CTP", "IXI", "DTISample"})
parser.add_argument('--is_resume', type = bool, default = is_resume) # TODO
parser.add_argument('--adjoint', type = bool, default = True)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--model_type', type = str, default = 'unet', choices = {'unet', 'vae'})
parser.add_argument('--joint_predict', type = bool, default = False, help = 'Joint decoder or split decoder for V, D prediction')
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
parser.add_argument('--GT_D_weight', type = float, default = 1) #200
parser.add_argument('--GT_D_CO', type = bool, default = False, help = 'Add loss on Color-by-orientation of D for supervision')
parser.add_argument('--GT_D_CO_weight', type = float, default = 1) #200

parser.add_argument('--VesselMasking', type = bool, default = False) 
parser.add_argument('--DiffusionMasking', type = bool, default = False)

parser.add_argument('--Plus_L', type = bool, default = False)
parser.add_argument('--GT_LU', type = bool, default = True, help = 'Add loss on L & U for supervised learning') # For spectral 
parser.add_argument('--GT_L_weight', type = float, default = 1) #300
parser.add_argument('--GT_U_weight', type = float, default = 1) #100

parser.add_argument('--GT_V', type = bool, default = True, help = 'Add loss on V for supervised learning') 
parser.add_argument('--GT_V_weight', type = float, default = 1.)  
parser.add_argument('--GT_Phi', type = bool, default = False, help = 'Add loss on Phi for supervised learning') 
parser.add_argument('--GT_Phi_weight', type = float, default = 1.)  

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
parser.add_argument('--va_weight', type = float, default = 1e-2) 
parser.add_argument('--fl_weight', type = float, default = 5e-4)  # 5e-3

# For ground truth types
parser.add_argument('--data_dim', type = list, default = patch_data_dim) # [32, 32, 32]
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

parser.add_argument('--BC', type = str, default = 'neumann', \
    choices = ['None', 'neumann', 'dirichlet', 'cauchy', 'source', 'source_neumann', 'dirichlet_neumann'])


#%% Training and testing settings
parser.add_argument('--n_filter', type = int, default = 64)  
parser.add_argument('--latent_variable_size', type = int, default = 128) # 5000, For VAE
parser.add_argument('--batch_size', type=int, default = batch_size) # 16, 5
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

args_3D = parser.parse_args()  

if 'MRP' in args_3D.img_type or 'CTP' in args_3D.img_type:
    args_3D.GT_V = False 
    args_3D.GT_D = False 
    args_3D.GT_LU = False 
    args_3D.GT_Phi = False 
    args_3D.GT_D_CO = False 
    args_3D.no_concentration_loss = False 

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

    return {'name': os.path.basename(patient_fld), 'movie': CTC_path, 'mask': MaskPath, 'vessel_mask': VesselPath, 'info': InfoPath}


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


def get_IXI_paths(case_fld, movie_fld, movie_type = 'adv_diff', PD_perf_pattern = 'adv_diff'):
    '''
    Paths reader for IXI synthetic movies folder
    '''
    InfoPath = os.path.join(movie_fld, 'Info.txt')
    MaskPath = img2nda(os.path.join(movie_fld, 'BrainMask.mha'))
    VesselPath = img2nda(os.path.join(case_fld, 'VesselMask_smoothed.mha'))
    if 'adv_only' in movie_type:
        MoviePath = img2nda(os.path.join(movie_fld, 'Adv.mha'), is_rewrite = img2nda_rewrite)
    elif 'diff_only' in movie_type:
        MoviePath = img2nda(os.path.join(movie_fld, 'Diff.mha'), is_rewrite = img2nda_rewrite)
    elif 'adv_diff' in movie_type:
        MoviePath = img2nda(os.path.join(movie_fld, 'AdvDiff.mha'), is_rewrite = img2nda_rewrite)
    else:
        raise ValueError('Unsupported IXI GT movie type.')

    D_Path = img2nda(os.path.join(movie_fld, 'D.mha')) if 'diff' in PD_perf_pattern else None
    L_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/L.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    U_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/U.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    FA_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/FA.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    Trace_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/Trace.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None
    D_CO_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/D_Color_Direction.mha'), is_rewrite = img2nda_rewrite) if 'diff' in PD_perf_pattern else None

    V_Path = img2nda(os.path.join(movie_fld, 'V.mha')) if 'adv' in PD_perf_pattern else None
    
    if IXI_div_free:
        Phi_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/Phi.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None
    else:
        Phi_Path = None # TODO
    AbsV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/Abs_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None
    NormV_Path = img2nda(os.path.join(movie_fld, 'ScalarMaps/Norm_V.mha'), is_rewrite = img2nda_rewrite) if 'adv' in PD_perf_pattern else None

    return {'name': os.path.basename(case_fld), 'movie': MoviePath, 'mask': MaskPath, 'vessel_mask': VesselPath, 'info': InfoPath, \
        'D': D_Path, 'L': L_Path, 'U': U_Path, 'D_CO': D_CO_Path, 'FA': FA_Path, 'Trace': Trace_Path, \
        'V': V_Path, 'Abs_V': AbsV_Path, 'Norm_V': NormV_Path, 'Phi': Phi_Path} # 'Phi'



def learn_IXI(args, AllFolder, SaveFolder, TrainCases, TestCases, ResumeModelPath = None):
    '''
    Trainer for ISLES Datasets (2017 - MRP or 2018 - CTP)
    '''
    print("# of training cases:", len(TrainCases))
    TrainCasePaths = [os.path.join(AllFolder, case_name.split('\n')[0]) for case_name in TrainCases]
    TrainMovies, TrainInfos, TrainMasks, TrainVesselMasks, TrainDs, TrainD_COs, TrainD_Ls, TrainD_Us, TrainVs, TrainAbs_Vs, TrainNorm_Vs, TrainV_Phis = [], [], [], [], [], [], [], [], [], [], [], []
    for i_case in range(len(TrainCasePaths)):
        print('   #%d: %s' % (i_case+1, TrainCases[i_case].split('\n')[0]))

        if IXI_div_free:
            movie_main_fld = os.path.join(TrainCasePaths[i_case], 'Movies(DivFree)')
        else:
            #movie_fld = os.path.join(case_fld, 'Movies')
            movie_main_fld = os.path.join(TrainCasePaths[i_case], 'Movies') # TODO
        for mag_fld in os.listdir(movie_main_fld):
            temp_mpovie_fld = os.path.join(movie_main_fld, mag_fld)
            paths_dict = get_IXI_paths(TrainCasePaths[i_case], temp_mpovie_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern)

            TrainInfos.append(paths_dict['info'])
            TrainMovies.append(paths_dict['movie'])
            TrainMasks.append(paths_dict['mask'])
            TrainVesselMasks.append(paths_dict['vessel_mask'])

            TrainDs.append(paths_dict['D'])
            TrainD_Ls.append(paths_dict['L'])
            TrainD_Us.append(paths_dict['U'])
            TrainD_COs.append(paths_dict['D_CO'])
            
            TrainVs.append(paths_dict['V'])
            TrainAbs_Vs.append(paths_dict['Abs_V'])
            TrainNorm_Vs.append(paths_dict['Norm_V'])
            TrainV_Phis.append(paths_dict['Phi'])
        TrainPaths = {'movies': TrainMovies, 'masks': TrainMasks, 'vessel_masks': TrainVesselMasks, 'infos': TrainInfos, \
            'D': {'D': TrainDs, 'D_CO':TrainD_COs, 'L': TrainD_Ls, 'U': TrainD_Us}, \
            'V': {'V': TrainVs, 'Abs_V': TrainAbs_Vs, 'Norm_V': TrainNorm_Vs, 'Phi': TrainV_Phis}}

    print("# of testing cases:", len(TestCases))
    TestCasePaths = [os.path.join(AllFolder, case_name.split('\n')[0]) for case_name in TestCases]
    TestPaths = []
    for i_case in range(len(TestCasePaths)):
        print('   #%d: %s' % (i_case+1, TestCases[i_case].split('\n')[0]))
        movie_fld = os.path.join(TestCasePaths[i_case], 'Movies/Mag(1.0)') # TODO
        paths_dict = get_IXI_paths(TestCasePaths[i_case], movie_fld, movie_type = args.IXI_perf_pattern, PD_perf_pattern = args.perf_pattern)
        TestPaths.append(paths_dict)

    main(args, SaveFolder, TrainPaths, TestPaths, ResumeModelPath)

    
def learn_ISLES(args, AllFolder, SaveFolder, TrainPatients, TestPatients, ResumeModelPath = None, img_type = 'MRP'):
    '''
    Trainer for ISLES Datasets (2017 - MRP or 2018 - CTP)
    '''
    print("# of training cases:", len(TrainPatients))
    print("# of testing cases:", len(TestPatients))

    TrainPatientPaths = [os.path.join(AllFolder, patient) for patient in TrainPatients]
    TrainMovies = []
    TrainInfos = []
    TrainMasks = []
    TrainVesselMasks = []
    for i_patient in range(len(TrainPatientPaths)):
        if not TrainPatientPaths[i_patient].startswith('.') and 'training' in TrainPatientPaths[i_patient] or 'case' in TrainPatientPaths[i_patient]:
            if 'MRP' in img_type:
                paths_dict = get_MRP_paths(TrainPatientPaths[i_patient])
            elif 'CTP' in img_type:
                paths_dict = get_CTP_paths(TrainPatientPaths[i_patient])
            TrainInfos.append(paths_dict['info'])
            TrainMasks.append(paths_dict['mask'])
            TrainMovies.append(paths_dict['movie'])
            TrainVesselMasks.append(paths_dict['vessel_mask'])
    TrainPaths = {'movies': TrainMovies, 'masks': TrainMasks, 'vessel_masks': TrainVesselMasks, 'infos': TrainInfos}

    TestPatientPaths = [os.path.join(AllFolder, patient) for patient in TestPatients]
    TestPaths = []
    for i_patient in range(len(TestPatientPaths)):
        if not TestPatientPaths[i_patient].startswith('.') and 'training' in TestPatientPaths[i_patient] or 'case' in TestPatientPaths[i_patient]:
            if 'MRP' in img_type:
                paths_dict = get_MRP_paths(TestPatientPaths[i_patient])
            elif 'CTP' in img_type:
                paths_dict = get_CTP_paths(TestPatientPaths[i_patient])
            TestPaths.append(paths_dict)

    main(args, SaveFolder, TrainPaths, TestPaths, ResumeModelPath)

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

        learn_ISLES(args_3D, ISLES2018, ISLES2018_SaveFld, TrainPatients, TestPatients, ResumeModelPath = ResumeModelPath, img_type = img_type)

    if 'MRP' in args_3D.img_type:

        print('######################### MRP #########################')

        if on_server:
            ISLES2017 = '/playpen-raid/peirong/Data/ISLES2017_Processed'
            ISLES2017_SaveFld = make_dir('/playpen-raid/peirong/Results/ISLES2017_Results')
        else:
            ISLES2017 = '/media/peirong/PR5/ISLES2017_Processed'
            ISLES2017_SaveFld = make_dir('/media/peirong/PR5/ISLES2017_Results_DL')

        TrainPatients = os.listdir(ISLES2017) # Analyze all cases
        TrainPatients.remove('IDs.txt')
        TrainPatients.remove('training_42') # Bad CTC trend # 
        TrainPatients.remove('training_14') # For Testing # 
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

        TestPatients = ['training_14']
        #TestPatients = ['training_14', 'training_16']

        learn_ISLES(args_3D, ISLES2017, ISLES2017_SaveFld, TrainPatients, TestPatients, ResumeModelPath = ResumeModelPath, img_type = img_type)

    if 'DTISample' in args_3D.img_type:
        
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

        main(args_3D, Demo_SaveFld, TrainPaths, TestPaths, ResumeModelPath)

    if 'IXI' in args_3D.img_type:
        
        # Synthesized IXI Demo #

        if on_server:
            IXI = '/playpen-raid/peirong/Data/IXI_Processed'
            IXI_SaveFld = make_dir('/playpen-raid/peirong/Results/IXI_Results')
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
        TrainCases.sort()#
        TrainCases = case_names[2:100] # TODO
        #TrainCases = ['IXI170-Guys-0843'] # TODO

        #TrainCases = case_names[3:54] # TODO

        TestCases = case_names 
        TestCases.sort()
        TestCases = case_names[:3] # TODO

        learn_IXI(args_3D, IXI, IXI_SaveFld, TrainCases, TestCases, ResumeModelPath)











