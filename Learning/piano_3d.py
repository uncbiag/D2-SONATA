import os, sys, datetime, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from shutil import rmtree, copyfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import SimpleITK as sitk

from utils import *
from Losses.losses import *
from Postprocess.PTIMeasures import PTI, WritePTIImage
from Learning.Modules.AdvDiffPDE import PIANO_FlowV, PIANO_Skeleton
from Datasets.dataset_movie import MovieDataset_3D_SmartTiming as MovieDataset_3D
 


device = torch.device('cuda:0')
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


PD_perf_pattern = 'adv_diff' # choices = ['adv_diff', 'adv_only', 'diff_only']

test_freq = 500
n_epochs_total = 5000

input_n_collocations = 2 #5
loss_n_collocations  = 2 #5

lr = 1e-3
dt = 0.05 

sub_collocation_nt = 2 # >= 2
collocation_len = 2
 



parser = argparse.ArgumentParser('3D PIANO') 
parser.add_argument('--adjoint', type = bool, default = True) 

parser.add_argument('--VesselMasking', type = bool, default = False) # TODO 
parser.add_argument('--DiffusionMasking', type = bool, default = False)
parser.add_argument('--V_time', type = bool, default = False)
parser.add_argument('--stochastic', type = bool, default = False)
parser.add_argument('--fix_D', type = bool, default = False)
parser.add_argument('--predict_value_mask', type = bool, default = False, help = 'If True, predict V(or D) as V := \bar{V} * value_mask') 
parser.add_argument('--predict_segment', type = bool, default = False) 

#%% Add ground truth D for supervised learning during warm-up

parser.add_argument('--Plus_L', type = bool, default = False)


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


parser.add_argument('--perf_pattern', type = str, default = PD_perf_pattern, choices = ['adv_diff', 'adv_only', 'diff_only']) 
parser.add_argument('--PD_D_type', type = str, default = 'scalar', \
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

parser.add_argument('--BC', type = str, default = 'neumann', choices = ['None', 'neumann', 'dirichlet', 'cauchy', 'source', 'source_neumann', 'dirichlet_neumann'])


#%% Training and testing settings
parser.add_argument('--n_filter', type = int, default = 64)  
parser.add_argument('--latent_variable_size', type = int, default = 128) # 5000, For VAE 
parser.add_argument('--opt_type', type = str, default = 'Adam')
parser.add_argument('--lr', type = float, default = lr, help = 'Model learning rate') # 1e-3, 5e-3, 1-2
parser.add_argument('--lr_weight_decay', type = float, default = 0.001) # 0.001 
parser.add_argument('--max_num_lr_reduce', type = int, default = 2)
parser.add_argument('--lr_reduce_rate', type = int, default = 0.8)
parser.add_argument('--lr_reduce_criterion', type = float, default = 0.5)
parser.add_argument('--n_epochs_total', type = int, default = n_epochs_total)
parser.add_argument('--niters_adv_only', type = int, default = 0)
parser.add_argument('--print_freq', type = int, default = 10) # 10
parser.add_argument('--test_freq', type = int, default = test_freq) # 100. 5
parser.add_argument('--smooth_when_learn', type = bool, default = False)
parser.add_argument('--use_stop_crit', type = bool, default = False) 
parser.add_argument('--stop_crit', type = float, default = 0.001) # Loss reduce rate for one iteration
parser.add_argument('--max_stop_count', type = int, default = 1, help  = 'Stop when stop_count >= max_stop_count') 

args_opt = parser.parse_args()  


def save_temporal(movie, save_fld, origin, spacing, direction, prefix, postfix):
    if not isinstance(movie, (np.ndarray)):
        if movie.requires_grad:
            movie = movie.detach()
        if movie.device != 'cpu': 
            movie = movie.cpu()
        movie = movie.numpy()
    axial_temporal = np.transpose(movie, (3, 1, 2, 0)) # slice
    coronal_temporal = np.transpose(movie, (3, 0, 2, 1)) # row
    sagittal_temporal = np.transpose(movie, (3, 0, 1, 2))[..., ::-1] # col # TODO Reverse direction for visualization end
    save_sitk(axial_temporal, os.path.join(save_fld, "%s-Axial%s" % (prefix, postfix)), origin, spacing, direction) # (time, row, column)
    save_sitk(coronal_temporal, os.path.join(save_fld,  "%s-Coronal%s" % (prefix, postfix)), origin, spacing, direction) # (time, row, column)
    save_sitk(sagittal_temporal, os.path.join(save_fld,  "%s-Sagittal%s" % (prefix, postfix)), origin, spacing, direction) # (time, row, column)
    return


def main(args, case_name, SaveFld, CTC_path, Mask_path, Info_path, Vessel_path = None, ResumePathList = None):
    
    #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')
    
    ###########################  Saving Path  ###########################
    
    print(args)
    if args.adjoint:
        from ODE.adjoint import odeint_adjoint as odeint
        print('Using adjoint method...')
    else:
        print('Not using adjoint method...')
        from ODE.odeint import odeint

    if 'adv_only' in args.perf_pattern:
        model_info = args.PD_V_type
    elif 'diff_only' in args.perf_pattern:
        model_info = args.PD_D_type
    elif 'adv_diff' in args.perf_pattern:
        model_info = '%s_%s' % (args.PD_V_type, args.PD_D_type)
    model_info = '%s - %s' % (model_info, args.BC)
    
    if args.VesselMasking:
        model_info = 'Vess_%s' % model_info
    if args.DiffusionMasking:
        model_info = 'Diff-%s' % model_info
    if ResumePathList is not None:
        print('Resume training')
        model_info = 'Res_%s' % model_info
    model_info = 'nCol[%s]_%s' % (args.loss_n_collocations, model_info)
    main_fld = make_dir(os.path.join(SaveFld, args.perf_pattern, model_info, case_name))
    print('Main folder: %s' % main_fld)


    ###########################  Datasets  ###########################
    
        
    GT_fld = make_dir(os.path.join(main_fld, '0-GT'))

    CTC_nda = np.load(CTC_path)
    data_dim = CTC_nda.shape[:-1]
    dataset = MovieDataset_3D(args, args.loss_n_collocations, data_dim, [1, 1, 1], None, os.path.basename(CTC_path)[:-4], CTC_path, \
            Mask_path, VesselPath = Vessel_path, InfoPath = Info_path, device = device) 
    data_loader = DataLoader(dataset, batch_size = 1, shuffle = False)
        

    origin, spacing, direction, data_spacing = dataset.origin, dataset.spacing, dataset.direction, dataset.data_spacing
    save_sitk(np.load(CTC_path), os.path.join(GT_fld, 'CTC.mha'), origin, spacing, direction)
    save_sitk(np.load(Mask_path), os.path.join(GT_fld, 'Mask.mha'), origin, spacing, direction)
    if args.VesselMasking:
        save_sitk(np.load(Vessel_path), os.path.join(GT_fld, 'Vessel.mha'), origin, spacing, direction)

    ###########################  PIANO SetUp  ###########################

    mask = torch.from_numpy(dataset.get_mask()).float().to(device)
    if ResumePathList:
        if 'diff' in args.perf_pattern:
            if 'spectral' in args.PD_D_type:
                S = torch.from_numpy(np.load(ResumePathList['S'])).float().to(device)
                L = torch.from_numpy(np.load(ResumePathList['L'])).float().to(device)
                D_param_lst = {'S': S, 'L': L}
                save_sitk(S, os.path.join(GT_fld, 'Resume_S.mha'), origin, spacing, direction)
                save_sitk(L, os.path.join(GT_fld, 'Resume_L.mha'), origin, spacing, direction)
            else:
                D = torch.from_numpy(np.load(ResumePathList['D'])).float().to(device)
                D_param_lst = {'D': D}
                save_sitk(D, os.path.join(GT_fld, 'Resume_D.mha'), origin, spacing, direction)
        if 'adv' in args.perf_pattern:
            if 'div_free' in args.PD_V_type:
                Phi = torch.from_numpy(np.load(ResumePathList['Phi'])).float().to(device)
                save_sitk(Phi, os.path.join(GT_fld, 'Resume_Phi.mha'), origin, spacing, direction)
                if 'HHD' in args.PD_V_type:
                    H = torch.from_numpy(np.load(ResumePathList['H'])).float().to(device)
                    V_param_lst = {'Phi': Phi, 'H': H}
                    save_sitk(H, os.path.join(GT_fld, 'Resume_H.mha'), origin, spacing, direction)
                else:
                    V_param_lst = {'Phi': Phi}
            else:
                V = torch.from_numpy(np.load(ResumePathList['V'])).float().to(device)
                V_param_lst = {'V': V}
                save_sitk(V, os.path.join(GT_fld, 'Resume_V.mha'), origin, spacing, direction)
    else:
        D_param_lst, V_param_lst = None, None
    
    piano = PIANO_FlowV(args, data_dim, data_spacing, args.perf_pattern, device, mask = mask, D_param_lst = D_param_lst, V_param_lst = V_param_lst)
    piano.to(device)
    optimizer_piano = optim.Adam(piano.parameters(), lr = args.lr, weight_decay = args.lr_weight_decay) # betas = [0.9, 0.999]
    
    steps = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_piano, steps)

    ##  Setting Info  ##
    setting_info = 'nCol=%s' % (args.loss_n_collocations) 
    if args.gradient_loss:
        setting_info = '%s_GL%s' % (setting_info, args.gl_weight)
    if args.ta_loss and 'diff' in args.perf_pattern:
        setting_info = '%s_TA%s' % (setting_info, args.ta_weight)
    if args.fa_loss and 'diff' in args.perf_pattern:
        setting_info = '%s_FA%s' % (setting_info, args.fa_weight)
    if args.color_orientation_loss and 'diff' in args.perf_pattern:
        setting_info = '%s_CO[max sqr + w. gl]%s' % (setting_info, args.co_weight)
    if args.spatial_gradient_loss:
        setting_info = '%s_SG%s' % (setting_info, args.sgl_weight) 

    file = open(os.path.join(main_fld, 'info (%s)' % setting_info), 'w')
    print('Number of parameters to optimize - %d' % (len(list(piano.parameters()))))
    file.write('\nNumber of parameters to optimize - %d' % (len(list(piano.parameters()))))

    ###########################  Losses and Functions  ###########################
   
    loss_perf_criterion = nn.MSELoss() # Losses for concentration prediction #
    loss_perf_criterion.to(device) 
    
    if args.gradient_loss:
        loss_grad_criterion = GradientLoss(args.gl_weight).to(device)
        if 'scalar' in args.PD_D_type:
            def grad_loss_function(V, D, batched = True, perf_pattern = 'adv_diff'): # V: (batch, channel, s, r, c); D: (batch, s, r, c)
                if 'diff' in args.perf_pattern:
                    if args.diff_gl_only:
                        return loss_grad_criterion([D], batched)
                    else:
                        return loss_grad_criterion(list(V.permute(1, 0, 2, 3, 4)) + [D], batched) # -> list: channels: (batch, s, r, c)
                else:
                    return loss_grad_criterion(list(V.permute(1, 0, 2, 3, 4)), batched) # -> list: channels: (batch, s, r, c)
        else:
            def grad_loss_function(V, D, batched = False, perf_pattern = 'adv_diff'): # (batch, channel, s, r, c)
                if 'diff' in args.perf_pattern:
                    if args.diff_gl_only:
                        return loss_grad_criterion(list(D.permute(1, 0, 2, 3, 4)), batched)
                    else:
                        return loss_grad_criterion(list(torch.cat([V, D], dim = 1).permute(1, 0, 2, 3, 4)), batched)
                else:
                    return loss_grad_criterion(list(V.permute(1, 0, 2, 3, 4)), batched)

    if args.spatial_gradient_loss:
        loss_spatial_grad_criterion = SpatialGradientLoss(args.sgl_weight)
        loss_spatial_grad_criterion.to(device)

    if args.color_orientation_loss:
        loss_color_orient_criterion = SpectralColorOrientLoss3D(args.co_weight)
        loss_color_orient_criterion.to(device)

    if args.ta_loss:
        loss_ta_criterion = TensorAnisotropyLoss3D(args.ta_weight)

    if args.fa_loss:
        if 'dual' in args.PD_D_type:
            loss_fa_criterion = DualAnisotropyLoss3D(args.fa_weight)
        elif 'spectral' in args.PD_D_type:
            loss_fa_criterion = SpectralAnisotropyLoss3D(args.fa_weight)
        loss_fa_criterion.to(device)

    if args.frobenius_loss:
        if 'scalar' in args.PD_D_type:
            loss_frobenius_criterion = ScalarFrobenius(args.fl_weight)
        elif 'full' in args.PD_D_type:
            loss_frobenius_criterion = TensorFrobenius3D(args.fl_weight)
        loss_frobenius_criterion.to(device)
    
    if 'gauge' in args.PD_V_type and 'adv' in args.perf_pattern:
        loss_gauge_criterion = DivergenceLoss3D(args.gauge_weight) 
        loss_gauge_criterion.to(device)

    ########################### Record settings ###########################

    file.write('\n\nPID-%s' % os.getpid())  
    file.write('\nDevice-%s' % device) 
    file.write('\nMain folder-%s' % main_fld)
    file.write('\nSetting info-%s' % setting_info)
    print('PID-%s' % os.getpid())
    print('Device-%s' % device) 
    print('Main folder-%s' % main_fld)
    print('Setting info-%s' % setting_info)


    ###########################  Training  ###########################

    #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')
    #time.sleep(4)

    test_loss_perf_lst = []
    fig = plt.figure()
    stop_count = 0

    lr_to_change = True
    num_lr_reduce = 0
    lr_reduce_criterion = args.lr_reduce_criterion

    end = time.time()

    for epoch in range(1, args.n_epochs_total + 1):

        piano.perf_pattern = args.perf_pattern

        ####################################################################################

        epoch_loss_perf  = 0.
        epoch_loss_sgl   = 0.
        epoch_loss_grad  = 0.
        epoch_loss_co    = 0.
        epoch_loss_fa    = 0.
        epoch_loss_ta    = 0.
        epoch_loss_frob  = 0.
        epoch_loss_gauge = 0.

        for i, batch in enumerate(data_loader):

            optimizer_piano.zero_grad()

            sub_collocation_t, movie4loss = \
                Variable(batch['sub_collocation_t'][0].float().to(device), requires_grad = True), \
                    Variable(batch['movie4input'].float().to(device), requires_grad = True)

            if args.VesselMasking:
                piano.vessel_mask = Variable(batch['vessel_mask'].float().to(device)) #  # (n_batch, s, r, c) 
                if args.DiffusionMasking:
                    piano.diffusion_mask = 1. - piano.vessel_mask

            if 'dirichlet' in args.BC or 'cauchy' in args.BC:
                contours = batch['movie_BC'] 
            elif 'source' in args.BC:
                contours = batch['movie_dBC'] 


            if args.gradient_loss:
                V, D = piano.get_VD()
                loss_grad = grad_loss_function(V, D, batched = True, perf_pattern  = args.perf_pattern)
                loss_grad.backward(retain_graph = True)
                epoch_loss_grad = loss_grad.item()
                print('      | Grad {:.9f}'.format(loss_grad.item()))
            
            if args.ta_loss and 'diff' in args.perf_pattern:
                D = piano.get_D()
                loss_ta = loss_ta_criterion(D, batched = True)
                loss_ta.backward(retain_graph = True)
                epoch_loss_ta = loss_ta.item()
                print('      |   TA {:.9f}'.format(loss_ta.item()))

            if args.fa_loss and 'diff' in args.perf_pattern:
                L = piano.get_L()
                loss_anisotropic = loss_fa_criterion(L, batched = True)
                loss_anisotropic.backward(retain_graph = True)
                epoch_loss_fa = loss_anisotropic.item()
                print('      |   FA {:.9f}'.format(loss_anisotropic.item()))

            if args.color_orientation_loss and 'diff' in args.perf_pattern:
                L = piano.get_L()
                U = piano.get_U()
                loss_color_orient = loss_color_orient_criterion(L, U, batched = True)
                loss_color_orient.backward(retain_graph = True)
                epoch_loss_co = loss_color_orient.item()
                print('      |   CO {:.9f}'.format(loss_color_orient.item()))

            if args.frobenius_loss and 'diff' in args.perf_pattern:
                D = piano.get_D()
                loss_frobenius = loss_frobenius_criterion(D, batched = True)
                loss_frobenius.backward(retain_graph = True)
                epoch_loss_frob = loss_frobenius.item()
                print('      | Frob {:.9f}'.format(loss_frobenius.item()))
            
            if 'gauge' in args.PD_V_type and 'adv' in args.perf_pattern:
                Phi = piano.get_Phi() # (batch, 3, s, r, c)
                loss_gauge = loss_gauge_criterion(Phi, batched = True)
                loss_gauge.backward(retain_graph = True)
                epoch_loss_gauge = loss_gauge.item()
        

            pred_full_movie = torch.stack([movie4loss[:, 0]] * movie4loss.size(1), dim = 1) # (n_batch, loss_n_collocations, s, r, c)
            for i_coll in range(1, movie4loss.size(1)):  # Loop for n_collocations #
                #print(i_coll+1, '/', movie4loss.size(1).item())
                pred_full_movie[:, i_coll] = pred_full_movie[:, i_coll - 1]
                # (collocation_nt, n_batch, s, r, c) -> (collocation_it = -1, n_batch, s, r, c)
                for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (n_batch, s, r, c)
                    #print(i_sub_coll+1, '/', sub_collocation_t.size().item())
                    pred_full_movie[:, i_coll] = odeint(piano, pred_full_movie[:, i_coll], sub_collocation_t, method = args.integ_method, options = args)[-1] 
                if 'dirichlet' in args.BC or 'cauchy' in args.BC: # BC list: [[BC0_0, BC0, 1], [BC1_0, BC1_1]], [BC2_0, BC2_1]]: each: ((n_batch), nT, BC_size, rest_dim_remain)
                    pred_full_movie[:, i_coll] = apply_BC_3D(pred_full_movie[:, i_coll], contours, i_coll, args.BC, batched = True) # (n_batch, s, r, c)
                else:
                    pred_full_movie[:, i_coll] = pred_full_movie[:, i_coll]
                    
            #pred_full_movie = movie4loss.clone()
            #pred_full_movie = odeint(piano, pred_full_movie[:, 0], sub_collocation_t, method = args.integ_method, options = args).permute(1, 0, 2, 3, 4)
            if args.spatial_gradient_loss:
                loss_spatial_grad = loss_spatial_grad_criterion(pred_full_movie, movie4loss, batched = True)
                loss_spatial_grad.backward(retain_graph = True)
                epoch_loss_sgl = loss_spatial_grad.item()
                print('      | SpGd {:.9f}'.format(loss_spatial_grad.item()))

            loss_perf = loss_perf_criterion(pred_full_movie, movie4loss) #* 1e+7 # TODO (batch, collocation_nt, s, r, c)
            
            loss_perf.backward()
            epoch_loss_perf = loss_perf.item()
            print('      | Perf {:.9f}'.format(loss_perf.item()))

            optimizer_piano.step()

        if epoch % args.print_freq == 0:
            print('\nEpoch #{:d}'.format(epoch))
            file.write('      | Perf {:.9f}'.format(epoch_loss_perf))
            print('      | Perf {:.9f}'.format(epoch_loss_perf))
            if args.gradient_loss:
                file.write('\n      | Grad {:.9f}'.format(epoch_loss_grad))
                print('      | Grad {:.9f}'.format(epoch_loss_grad))
            if args.ta_loss and 'diff' in args.perf_pattern:
                file.write('\n      |   TA {:.9f}'.format(epoch_loss_ta))
                print('      |   TA {:.9f}'.format(epoch_loss_ta))
            if args.fa_loss and 'diff' in args.perf_pattern:
                file.write('\n      |   FA {:.9f}'.format(epoch_loss_fa))
                print('      |   FA {:.9f}'.format(epoch_loss_fa))
            if args.color_orientation_loss and 'diff' in args.perf_pattern:
                file.write('\n      |   CO {:.9f}'.format(epoch_loss_co))
                print('      |   CO {:.9f}'.format(epoch_loss_co))
            if args.frobenius_loss and 'diff' in args.perf_pattern:
                file.write('\n      | Frob {:.9f}'.format(epoch_loss_frob))
                print('      | Frob {:.9f}'.format(epoch_loss_frob))
            if args.spatial_gradient_loss:
                file.write('\n      | SpGd {:.9f}'.format(epoch_loss_sgl))
                print('      | SpGd {:.9f}'.format(epoch_loss_sgl))
            if 'gauge' in args.PD_V_type and 'adv' in args.perf_pattern:
                file.write('\n      | Gaug {:.9f}'.format(epoch_loss_gauge))
                print('      | Gaug {:.9f}'.format(epoch_loss_gauge))
            grad_piano = avg_grad(piano.parameters())
            file.write('\n      |  AvG {:.9f}'.format(grad_piano)) 
            print('      |  AvG {:.9f}\n'.format(grad_piano))


        if epoch % args.test_freq == 0:
            
            #for param_group in optimizer_piano.param_groups:
            #    print('Current learning rate: %.5f' % param_group['lr'])

            main_test_fld = make_dir(os.path.join(main_fld, '%d' % epoch))
            scalar_fld = make_dir(os.path.join(main_test_fld, 'ScalarMaps'))
            movie_fld = make_dir(os.path.join(main_test_fld, 'Movies'))
            termporal_movie_fld = make_dir(os.path.join(main_test_fld, 'TimeMachines'))

            V_path = os.path.join(main_test_fld, "V.mha")
            D_path = os.path.join(main_test_fld, "D.mha")
            if 'adv' in args.perf_pattern:
                V = piano.get_V().permute(1, 2, 3, 0) # (s, r, c, 3)
                V_measures(V, save_fld = scalar_fld, origin = origin, spacing = spacing, direction = direction)
                save_sitk(V, V_path, origin, spacing, direction)
                if 'div_free' in args.PD_V_type:
                    Phi = piano.get_Phi().permute(1, 2, 3, 0) # (s, r, c, 3)
                    save_sitk(Phi, os.path.join(main_test_fld, 'Phi.mha'), origin, spacing, direction) # (time, row, column)
                    if 'HHD' in args.PD_V_type:
                        H = piano.get_H().permute(1, 2, 3, 0) # (s, r, c, 3)
                        save_sitk(H, os.path.join(main_test_fld, 'H.mha'), origin, spacing, direction) # (time, row, column)
            if 'diff' in args.perf_pattern:
                if 'full' in args.PD_D_type:
                    D = piano.get_D().permute(1, 2, 3, 0) # (s, r, c, 6): order: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
                    D = torch.stack([D[..., 0], D[..., 1], D[..., 2], \
                                    D[..., 1], D[..., 3], D[..., 4], \
                                    D[..., 2], D[..., 4], D[..., 5]], dim = -1) # (s, r, c, 9): order: [Dxx, Dxy, Dxz, Dxy, Dyy, Dyz, Dxz, Dyz, Dzz]
                else:
                    D = piano.get_D()
                save_sitk(D, D_path, origin, spacing, direction) 
                # Compute PTI measures of D #
                PTIWriter = WritePTIImage(main_test_fld, origin, spacing, direction, device, to_smooth = args.smooth_when_learn)
                PTISolver = PTI(main_test_fld, Mask_path, 'diff_only', D_path = D_path, D_type = args.PD_D_type, \
                    V_path = V_path, V_type = args.PD_V_type, device = device, EigenRecompute = True)
                if 'full' in args.PD_D_type:
                    Trace_path = PTIWriter.save(PTISolver.Trace(), 'Trace (PTI).mha')
                    L_path = PTIWriter.save(PTISolver.eva, 'L (PTI).mha')
                    U_path = PTIWriter.save(PTISolver.U(), 'U (PTI).mha')
                    FA_path = PTIWriter.save(PTISolver.FA(), 'FA (PTI).mha')
                    DColorDirection_path = PTIWriter.save(PTISolver.D_Color_Direction(), 'D_Color_Direction (PTI).mha')
                    
                if 'spectral' in args.PD_D_type:
                    U = piano.get_U().permute(1, 2, 3, 0) # (s, r, c, 9)
                    save_sitk(U, os.path.join(main_test_fld, 'U.mha'), origin, spacing, direction)
                    L = piano.get_L().permute(1, 2, 3, 0) # (s, r, c, 3)
                    save_sitk(L, os.path.join(main_test_fld, 'L.mha'), origin, spacing, direction)
                    S = piano.get_S().permute(1, 2, 3, 0) # (s, r, c, 3)
                    save_sitk(S, os.path.join(main_test_fld, 'S.mha'), origin, spacing, direction)


            ## Save CTC prediction ##
            '''sub_collocation_t, full_movie = dataset.get_full_sample()['sub_collocation_t'].float().to(device), dataset.get_full_sample()['full_movie'].float().to(device) 
            sub_collocation_t, full_movie = Variable(sub_collocation_t).to(device), Variable(full_movie).to(device)

            if 'dirichlet' in args.BC or 'cauchy' in args.BC:
                contours = dataset.get_full_sample()['movie_BC'] 
            elif 'source' in args.BC:
                contours = dataset.get_full_sample()['movie_dBC'] 

            pred_full_movie =torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, s, r, c)
            for i_coll in range(1, full_movie.size(0)):
                pred_full_movie[i_coll] = pred_full_movie[i_coll - 1] * mask
                #print(i_coll) # NOTE: (nT, n_batch = 1, s, r, c) -> (nT = -1, s, r, c)
                for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (s, r, c)
                    pred_full_movie[i_coll] = odeint(piano, pred_full_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                    
            full_perf_loss = loss_perf_criterion(full_movie, pred_full_movie).item()
            print('TEST  | Perf {:.9f}'.format(full_perf_loss)) 
            file.write('\nTEST  | Perf {:.9f}'.format(full_perf_loss))

            pred_full_movie = pred_full_movie.permute(1, 2, 3, 0) # (nT, s, r, c) -> (s, r, c, nT)
            save_sitk(pred_full_movie, os.path.join(movie_fld, "PD.mha"), origin, spacing, direction) # (time, row, column)
            save_temporal(pred_full_movie, termporal_movie_fld, origin, spacing, direction, prefix = 'PD', postfix = '.mha')'''

            ## Save adv_only, diff_only part prediction ##
            '''if args.perf_pattern == 'adv_diff':
                #args.contours, args.dcontours = None, None # NOTE: should not imposing abs B.C., only apply Neumann
                # adv movie predict from t0 #
                piano.perf_pattern = 'adv_only'
                pred_adv_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, s, r, c)
                for i_coll in range(1, full_movie.size(0)):
                    pred_adv_movie[i_coll] = pred_adv_movie[i_coll - 1]
                    #print(i_coll) # NOTE: (nT, n_batch = 1, s, r, c) -> (nT = -1, s, r, c)
                    for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (s, r, c)
                        pred_adv_movie[i_coll] = odeint(piano, pred_adv_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                    
                pred_adv_movie = pred_adv_movie.permute(1, 2, 3, 0) # (nT, s, r, c) -> (s, r, c, nT)
                save_sitk(pred_adv_movie, os.path.join(movie_fld, "PD_Adv.mha"), origin, spacing) # (time, row, column)
                save_temporal(pred_adv_movie, termporal_movie_fld, origin, spacing, direction, prefix = 'PD_Adv', postfix = '.mha')
            
                # diff movie predict from t0 #
                piano.perf_pattern = 'diff_only'
                pred_diff_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, s, r, c)
                for i_coll in range(1, full_movie.size(0)):
                    pred_diff_movie[i_coll] = pred_diff_movie[i_coll - 1]
                    #print(i_coll) # NOTE: (nT, n_batch = 1, s, r, c) -> (nT = -1, s, r, c)
                    for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (s, r, c)
                        pred_diff_movie[i_coll] = odeint(piano, pred_diff_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                    
                pred_diff_movie = pred_diff_movie.permute(1, 2, 3, 0) # (nT, s, r, c) -> (s, r, c, nT)
                save_sitk(pred_diff_movie, os.path.join(movie_fld, "PD_Diff.mha"), origin, spacing) # (time, row, column)
                save_temporal(pred_diff_movie, termporal_movie_fld, origin, spacing, direction, prefix = 'PD_Diff', postfix = '.mha')
                del pred_adv_movie, pred_diff_movie'''


            #writer.add_scalar('testing_loss', test_loss_perf, epoch)
            # Plot updated loss
            #t = [i for i in range(0, int(epoch/args.test_freq) + 1)]
            t = [i for i in range(0, int(epoch/args.test_freq))]
            #test_loss_perf_lst.append(full_perf_loss)
            test_loss_perf_lst.append(epoch_loss_perf)
            plt.plot(t, test_loss_perf_lst, 'r--', label = 'loss_perf')
            x_label = 'Iter (%d)' % args.test_freq
            plt.xlabel(x_label, fontsize = 16) 
            plt.ylabel('Loss', fontsize = 16)
            plt.legend()
            fig_name = '%s' % (setting_info)
            plt.title(fig_name, fontsize = 12)
            plt.savefig(os.path.join(main_fld, '%s.png' % fig_name))
            plt.clf()


            # Change learning rate when model tends to converge #
            if lr_to_change: # TODO: Add param (D, V) loss in testing # 
                if len(test_loss_perf_lst) > 10 and np.array(test_loss_perf_lst[-5:]).mean() < lr_reduce_criterion * np.array(test_loss_perf_lst[:2]).mean():
                    print('Reduce learning rate')
                    for g in optimizer_piano.param_groups:
                        g['lr'] = g['lr'] * (1 - args.lr_reduce_rate)
                    num_lr_reduce += 1
                    lr_reduce_criterion *= 0.5
                    if num_lr_reduce >= args.max_num_lr_reduce:
                        lr_to_change = False
                
            # Extend batch_nT while approaching to GT_V
            #if epoch % args.increase_loss_n_collocations_freq == 0 and args.increase_input_n_collocations:
            #    if DataSet_Training.loss_n_collocations < args.max_loss_n_collocations:
            #        DataSet_Training.loss_n_collocations = DataSet_Training.loss_n_collocations + 1 
            #        print('Training batch_nT:', str(DataSet_Training.loss_n_collocations))


            gc.collect()
    
    end = time.time()
    file.close()
    return

##############################################################################################################################

def img2nda(img_path, is_rewrite = False):
    nda_path = '%s.npy' % img_path[:-4]
    if not os.path.isfile(nda_path) or is_rewrite:
        np.save(nda_path, sitk.GetArrayFromImage(sitk.ReadImage(img_path)))
    return nda_path

if __name__ == '__main__':

    on_server = True

    img_type = 'MRP' # 'MRP' 

    if img_type == 'MRP':
    
        if on_server:
            #AllFld = '/playpen-raid2/peirong/ISLES2017_Processed'  
            AllFld = '/playpen-raid2/peirong/Data/ISLES2017_Processed_rotated'
            SaveFld = make_dir('/playpen-raid2/peirong/Results/ISLES2017_Results') 
            resume_fld = None # '/playpen-raid2/peirong/Results/ISLES2017_Results/adv_diff/[Vess_enhanced_normalied]-[D-0.05,V-1]-nCol[5]_Res(1100)_[Conc]_Vess_[1]-vector_div_free_stream_full_spectral - cauchy/1/' + case_name
        else:
            AllFld = '/media/peirong/PR5/ISLES2017_Processed'
            SaveFld = make_dir('/media/peirong/PR5/ISLES2017_Results_DL')
            resume_fld  = '/home/peirong/biag-raid/Results/ISLES2017_Results/adv_diff/nCol[5]_[Conc]_Vess_[1]-vector_div_free_stream_full_spectral - cauchy/260'

        args_opt.lr = 1e-3
        args_opt.dt = 0.05
        
        case_names = []
        for i_name in os.listdir(AllFld):
            if 'training_' in i_name: 
                case_names.append(i_name) 
        #case_names.remove('training_42') # Bad CTC trend # 
        #case_names.remove('training_14') # For Testing # 
        case_names.sort()
        print('Total cases to process:', len(case_names))
        

        
        for case_name in case_names:
            
            print('----- Now processing', case_name)

            Mask_path = img2nda(os.path.join(AllFld, case_name, 'BrainMask.mha'))
            CTC_path = img2nda(os.path.join(AllFld, case_name, 'MRP/CTC_fromTTP.mha'))
            Info_path = os.path.join(AllFld, case_name, 'MRP/Info_fromTTP.txt')
            
            #Vessel_path = img2nda(os.path.join(AllFld, case_name, 'VesselEnhanced_normalized.mha'))
            Vessel_path = None
            #ResumePathList = {'Phi': img2nda(os.path.join(resume_fld, 'Phi.mha')), \
            #                    'S': img2nda(os.path.join(resume_fld, 'S.mha')), 
            #                    'L': img2nda(os.path.join(resume_fld, 'L.mha'))}
            ResumePathList = None
            
            main(args_opt, case_name, SaveFld, CTC_path, Mask_path, Info_path, Vessel_path = Vessel_path, ResumePathList = ResumePathList) 
    
    else:

        args_opt.lr = 1e-4
        args_opt.dt = 0.01

        case_name = 'IXI002-Guys-0828'

        CaseFld = '/playpen-raid/peirong/Data/IXI_Processed/' + case_name
        SaveFld = make_dir(os.path.join('/playpen-raid1/peirong/Results/IXI_Results/MICCAI/' + case_name))

        Mask_path = img2nda(os.path.join(CaseFld, 'BrainMask.mha'))
        CTC_path = img2nda(os.path.join(CaseFld, 'Movies(test)/AdvDiff.mha'))
        Info_path = os.path.join(CaseFld, 'Movies(test)/Info.txt')
        Vessel_path = img2nda(os.path.join(CaseFld, 'VesselMask_smoothed.mha'))

        resume_fld = '/playpen-raid1/peirong/Results/IXI_Results/adv_diff/LargeD-SplitLoss/Forward-Stream/L1-nCol[5]_[NoConc]_Vess_[GT_V(10]_[GT_LU(10, 10)]_[GT_D(10)]_[98]-vector_div_free_stream_full_spectral - cauchy/1660'
        ResumePathList = {'Phi': img2nda(os.path.join(resume_fld, case_name, 'Phi.mha')), \
                            'S': img2nda(os.path.join(resume_fld, case_name, 'S.mha')), 
                            'L': img2nda(os.path.join(resume_fld, case_name, 'L.mha'))}

        main(args_opt, case_name, SaveFld, CTC_path, Mask_path, Info_path, Vessel_path = Vessel_path, ResumePathList = ResumePathList) 