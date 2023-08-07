import os, sys, datetime, gc, shutil
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
from Learning.Modules.AdvDiffPDE import PIANOinD, PIANO_Skeleton
    

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

 


def main(args, SaveFld, TrainPaths, TestPaths, ResumePaths = None, have_GT = False):
 
    device = torch.device('cuda:%s' % str(args.gpu))
    
    #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')
    
    ###########################  Saving Path  ###########################
    
    print(args)
    if args.adjoint:
        from ODE.adjoint import odeint_adjoint as odeint
        print('Using adjoint method...')
    else:
        print('Not using adjoint method...')
        from ODE.odeint import odeint

    model_info = '%s' % len(TrainPaths['movies'])

    ###########################
    ##### Dataset Settings ####
    
    data_prefix = 'N' # training with normal data
    if args.train_use_joint_lesion:
        data_prefix += '_L'
    if args.train_use_separate_lesion:
        data_prefix += '_sepL' 
    data_prefix += '--N' # testing with normal data
    if args.test_use_joint_lesion:
        data_prefix += '_L' 
    if args.test_use_separate_lesion:
        data_prefix += '_sepL' 
    model_info = '[%s]_[%s]' % (args.input_n_collocations, data_prefix)
    
    if args.no_concentration_loss and 'IXI' in args.img_type:
        model_info += '_[NoConc]'
    elif 'MRP' in args.img_type:
        model_info += '_[Conc]'
    ###########################


    ###########################
    ## archived: not working ##
    if args.jbld_loss:
        model_info = model_info + '_[JBLD-only]' if args.jbld_loss_only else model_info + '_[+JBLD]'

    if args.predict_deviation: # NOTE: archived: not work well
        assert not args.predict_value_mask 
        model_info = model_info + '_[Dev' if not args.deviation_separate_net else model_info + '_[sep-Dev' 
        if args.deviation_extra_weight > 0.:
            model_info += '_%d' % int(args.deviation_extra_weight)
        model_info += ']'

    if args.predict_segment: # NOTE: archived: not work well
        model_info = model_info + '_[free-' if not args.segment_condition_on_physics else model_info + '_[cond-'
        if args.segment_net_type == 'conc':
            model_info += 'ConcSeg]'  
        elif args.segment_net_type == 'dev':
            assert args.predict_deviation
            model_info += 'DevSeg]'
    ###########################
    
    if args.stochastic and args.predict_value_mask and args.vm_sde_net: # archived: not work well
        model_info += '_[sep-SDE-VM'
        if args.actual_physics_loss:
            model_info += '+'
        model_info += ']'
    else:
        if args.stochastic:
            sde_prefix = 'SDE-Sigmoid' if not args.stochastic_separate_net else 'sep-SDE' 
            if args.SDE_loss:
                model_info += '_%s(%d)' % (sde_prefix, args.SDE_weight)
            else:
                model_info += '_%%' % (sde_prefix)  

        if args.predict_value_mask:
            vm_prefix = 'sep-' if args.value_mask_separate_net else ''
            model_info = model_info + '_[%sDV-VM' % vm_prefix if args.separate_DV_value_mask else model_info + '_[%sVM' % vm_prefix
            if args.actual_physics_loss:
                model_info += '+'
            model_info += ']'

    '''if not args.no_concentration_loss:
        if args.GT_V:
            model_info  = 'Phase-II/%s' % model_info
        else:
            model_info = 'No-Phase-I/%s' % model_info '''

    n_test_iter = 0

    if args.is_resume:
        print('Resume model_info: ', os.path.basename(ResumePaths['root']))
        print('Current model_info:', model_info)
        if args.img_type == 'IXI' and not args.for_test:
            assert os.path.basename(ResumePaths['root']) == model_info
        print('Resume training from %s' % ResumePaths['model'])
        checkpoint = torch.load(ResumePaths['model'], map_location = device)

        if args.img_type == 'IXI':
            # Re-store test error list
            test_loss_perf_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_perf.npy'), allow_pickle = True))
            test_loss_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_D.npy'), allow_pickle = True)) 
            test_loss_L_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_L.npy'), allow_pickle = True)) 
            test_loss_U_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_U.npy'), allow_pickle = True)) 
            test_loss_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_V.npy'), allow_pickle = True)) 
            list_of_list = [test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst]

            if args.predict_segment: # NOTE: archived: not work well
                test_loss_Seg_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_Seg.npy'))) 

                list_of_list += [test_loss_Seg_lst]

                list_of_list = get_max_len(list_of_list)

            if args.stochastic:
                # TODO: TESTING shared uncertainty # 
                test_loss_Sigma_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_Sigma.npy'), allow_pickle = True)) 
                #test_loss_Sigma_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_Sigma_D.npy'))) 
                #test_loss_Sigma_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_Sigma_V.npy'))) 
                
                #list_of_list += [test_loss_Sigma_D_lst, test_loss_Sigma_V_lst]
                list_of_list += [test_loss_Sigma_lst]

                list_of_list = get_max_len(list_of_list)
                #[test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst, test_loss_Sigma_lst] = list_of_list 
            
            if args.predict_deviation: # NOTE: archived: not work well
                test_loss_orig_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_orig_V.npy'), allow_pickle = True))
                test_loss_delta_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_delta_V.npy'), allow_pickle = True))

                test_loss_orig_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_orig_D.npy'), allow_pickle = True))
                test_loss_delta_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_delta_D.npy'), allow_pickle = True))

                test_loss_orig_L_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_orig_L.npy'), allow_pickle = True))
                test_loss_delta_L_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_delta_L.npy'), allow_pickle = True))

                list_of_list += [test_loss_orig_V_lst, test_loss_delta_V_lst, test_loss_orig_D_lst, test_loss_delta_D_lst, test_loss_orig_L_lst, test_loss_delta_L_lst]
                list_of_list = get_max_len(list_of_list)

            if args.predict_value_mask: 
                try: 
                    test_loss_VM_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_VM_D.npy'), allow_pickle = True))  
                    test_loss_VM_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_VM_V.npy'), allow_pickle = True))  
                    test_loss_orig_V_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_orig_V.npy'), allow_pickle = True))  
                    test_loss_orig_D_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_orig_D.npy'), allow_pickle = True))  
                    test_loss_orig_L_lst = list(np.load(os.path.join(ResumePaths['root'], 'loss_orig_L.npy'), allow_pickle = True)) 
                    list_of_list += [test_loss_VM_D_lst, test_loss_VM_V_lst, test_loss_orig_V_lst, test_loss_orig_D_lst, test_loss_orig_L_lst]
                    list_of_list = get_max_len(list_of_list)
                except: # Applicable to old version 
                    test_loss_VM_D_lst = []
                    test_loss_VM_V_lst = []
                    test_loss_orig_V_lst = []
                    test_loss_orig_D_lst = []
                    test_loss_orig_L_lst = []


                '''if args.stochastic:
                    [test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst, test_loss_Sigma_lst, test_loss_VM_lst, test_loss_orig_V_lst, test_loss_orig_D_lst, test_loss_orig_L_lst] = list_of_list
                else:
                    [test_loss_perf_lst, test_loss_D_lst, test_loss_L_lst, test_loss_U_lst, test_loss_V_lst, test_loss_VM_lst, test_loss_orig_V_lst, test_loss_orig_D_lst, test_loss_orig_L_lst] = list_of_list'''
                 
            '''epoch_len = len(list_of_list[0])
            for i in range(len(list_of_list)):
                lst = list_of_list[i]
                print(len(lst))
                if len(lst) > epoch_len:
                    list_of_list[i] = list_of_list[i][:epoch_len]
                elif len(lst) < epoch_len:
                    for j in range(i):
                        epoch_len = len(lst)
                        list_of_list[j] = list_of_list[j][:epoch_len]
                else:
                    pass
                assert len(lst) == epoch_len'''
            print(len(test_loss_perf_lst))

            n_test_iter = len(test_loss_perf_lst)  


    if args.place_holder:  
        args.test_freq = 100000000
        args.predict_deviation, args.predict_segment, args.stochastic = False, False, False
        main_fld = make_dir(os.path.join(SaveFld, args.perf_pattern, 'place_holder', str(args.gpu)))
    elif (args.is_resume or args.for_test) and args.img_type == 'IXI': 
        assert ResumePaths is not None
        print('----- Update Model Info -----')
        #model_info = 'Res(%s)_%s' % (checkpoint['epoch'], model_info) # NOTE: in new folder
        #main_fld = make_dir(os.path.join(SaveFld, args.perf_pattern, model_info))
        main_fld = ResumePaths['root'] if not args.for_test else make_dir(os.path.join(ResumePaths['root'], '0-Test')) # NOTE: in orig. folder
    else: 
        main_fld = make_dir(os.path.join(SaveFld, args.perf_pattern, model_info))
    
    print('Main folder: %s' % main_fld)
    print('PID - %s' % os.getpid())
    
    ###########################  Datasets  ###########################
    
    if args.smart_timing:
        from Datasets.dataset_movie import MoviesDataset_3D_SmartTiming as MoviesDataset_3D
        from Datasets.dataset_movie import MovieDataset_3D_SmartTiming as MovieDataset_3D
    else:
        from Datasets.dataset_movie import MoviesDataset_3D
        from Datasets.dataset_movie import MovieDataset_3D

    TrainGT_Ds = TrainPaths['D']['D'] if args.GT_D and 'diff' in args.perf_pattern else None 
    TrainGT_D_COs = TrainPaths['D']['D_CO'] if args.GT_D_CO and 'diff' in args.perf_pattern else None
    TrainGT_Ls = TrainPaths['D']['L'] if args.GT_LU and 'diff' in args.perf_pattern else None
    TrainGT_Us = TrainPaths['D']['U'] if args.GT_LU and 'diff' in args.perf_pattern else None
    TrainGT_Vs = TrainPaths['V']['V'] if args.GT_V and 'adv' in args.perf_pattern else None
    TrainGT_Phis = TrainPaths['V']['Phi'] if args.GT_Phi and 'adv' in args.perf_pattern else None
    VesselMasks = TrainPaths['vessel_masks'] if args.VesselMasking else None
    VesselMirrorMasks = TrainPaths['vessel_mirror_masks'] if args.VesselMasking else None
    ValueMasks = TrainPaths['value_masks'] if args.predict_value_mask else None

    deviation_path_dict = {}
    if (args.predict_deviation or args.predict_value_mask) and args.img_type == 'IXI':
        deviation_path_dict['orig_D'] = TrainPaths['D']['orig_D'] 
        deviation_path_dict['delta_D'] = TrainPaths['D']['delta_D'] 
        deviation_path_dict['orig_L'] = TrainPaths['D']['orig_L'] 
        deviation_path_dict['delta_L'] = TrainPaths['D']['delta_L'] 
        deviation_path_dict['orig_V'] = TrainPaths['V']['orig_V'] 
        deviation_path_dict['delta_V'] = TrainPaths['V']['delta_V'] 
    
    if args.img_type == 'IXI':
        Seg_paths = TrainPaths['lesion_seg']
    else: 
        Seg_paths = None
    

    DataSet_Training = MoviesDataset_3D(args, args.input_n_collocations, args.loss_n_collocations, args.data_dim, TrainPaths['movies'], VesselMasks, VesselMirrorMasks, ValueMasks, Seg_paths, \
        TrainPaths['infos'], TrainGT_Ds, TrainGT_D_COs, TrainGT_Ls, TrainGT_Us, TrainGT_Vs, TrainGT_Phis, deviation_path_dict, device = device) # TODO: add mask in training?
    train_loader = DataLoader(DataSet_Training, batch_size = args.batch_size, shuffle = True)


    DataSets_Testing = []
    test_loaders = [] 
    GT_fld = make_dir(os.path.join(main_fld, '0-GT'))
    in_channels = DataSet_Training.input_n_collocations
    for i_test in range(len(TestPaths)):
        TestPath = TestPaths[i_test]
        temp_dataset = MovieDataset_3D(args, args.input_n_collocations, args.data_dim, args.stride_testing, TestPath, TestPath['name'], TestPath['movie'], \
            TestPath['mask'], ValueMaskPath = TestPath['value_mask'], InfoPath = TestPath['info'], device = device)
        test_loaders.append(DataLoader(temp_dataset, batch_size = 1, shuffle = False))
        temp_dataset.BC = 'neumann' # TODO: neumann for full testing

        GT_i_fld = make_dir(os.path.join(GT_fld, str(temp_dataset.CaseName)))
        DataSets_Testing.append(temp_dataset)
        in_channels = min(temp_dataset.N_collocations, in_channels)

        origin, spacing, direction = temp_dataset.origin, temp_dataset.spacing, temp_dataset.direction
        save_sitk(np.load(TestPath['mask']), os.path.join(GT_i_fld, 'Mask.mha'), origin, spacing, direction)
        if TestPath['value_mask'] is not None:
            if isinstance(TestPath['value_mask'], str): 
                save_sitk(np.load(TestPath['value_mask']), os.path.join(GT_i_fld, 'ValueMask.mha'), origin, spacing, direction) 
            else:
                save_sitk(np.load(TestPath['value_mask']['D']), os.path.join(GT_i_fld, 'ValueMask_D.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['value_mask']['V']), os.path.join(GT_i_fld, 'ValueMask_V.mha'), origin, spacing, direction)
        if have_GT and TestPath['lesion_seg'] is not None:
            if isinstance(TestPath['lesion_seg'], str): 
                save_sitk(np.load(TestPath['lesion_seg']), os.path.join(GT_i_fld, 'LesionSeg.mha'), origin, spacing, direction)
            else:
                save_sitk(np.load(TestPath['lesion_seg']['D']), os.path.join(GT_i_fld, 'LesionSeg_D.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['lesion_seg']['V']), os.path.join(GT_i_fld, 'LesionSeg_V.mha'), origin, spacing, direction)
        if args.VesselMasking:
            save_sitk(np.load(TestPath['vessel_mask']), os.path.join(GT_i_fld, 'Vessel.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['vessel_mirror_mask']), os.path.join(GT_i_fld, 'Vessel_mirrored.mha'), origin, spacing, direction)
        if have_GT and 'diff' in args.perf_pattern: 
            save_sitk(np.load(TestPath['D']), os.path.join(GT_i_fld, 'D.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['L']), os.path.join(GT_i_fld, 'L.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['U']), os.path.join(GT_i_fld, 'U.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['FA']), os.path.join(GT_i_fld, 'FA.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['Trace']), os.path.join(GT_i_fld, 'Trace.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['D_CO']), os.path.join(GT_i_fld, 'D_Color_Direction.mha'), origin, spacing, direction)
            if (args.predict_deviation or args.predict_value_mask) and TestPath['orig_D'] is not None:  
                save_sitk(np.load(TestPath['orig_D']), os.path.join(GT_i_fld, 'orig_D.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['delta_D']), os.path.join(GT_i_fld, 'delta_D.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['orig_L']), os.path.join(GT_i_fld, 'orig_L.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['delta_L']), os.path.join(GT_i_fld, 'delta_L.mha'), origin, spacing, direction) 

        if have_GT and 'adv' in args.perf_pattern: 
            save_sitk(np.load(TestPath['V']), os.path.join(GT_i_fld, 'V.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['Abs_V']), os.path.join(GT_i_fld, 'Abs_V.mha'), origin, spacing, direction)
            save_sitk(np.load(TestPath['Norm_V']), os.path.join(GT_i_fld, 'Norm_V.mha'), origin, spacing, direction)
            if (args.predict_deviation or args.predict_value_mask) and TestPath['orig_V'] is not None:
                save_sitk(np.load(TestPath['orig_V']), os.path.join(GT_i_fld, 'orig_V.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['delta_V']), os.path.join(GT_i_fld, 'delta_V.mha'), origin, spacing, direction) 
                save_sitk(np.load(TestPath['orig_Abs_V']), os.path.join(GT_i_fld, 'orig_Abs_V.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['delta_Abs_V']), os.path.join(GT_i_fld, 'delta_Abs_V.mha'), origin, spacing, direction) 
                save_sitk(np.load(TestPath['orig_Norm_V']), os.path.join(GT_i_fld, 'orig_Norm_V.mha'), origin, spacing, direction)
                save_sitk(np.load(TestPath['delta_Norm_V']), os.path.join(GT_i_fld, 'delta_Norm_V.mha'), origin, spacing, direction)   

        movie_nda = np.transpose(temp_dataset.movie_nda, (1, 2, 3, 0)) # (nT, s, r, c) -> (s, r, c, nT)
        save_sitk(movie_nda, os.path.join(GT_i_fld, 'GT.mha'), origin, spacing, direction)
        save_temporal(movie_nda, GT_i_fld, origin, spacing, direction, prefix = 'Temporal', postfix = '.mha')

        temp_dataset.VesselPath = TestPath['vessel_mask'] if args.VesselMasking else None
        temp_dataset.VesselMirrorPath = TestPath['vessel_mirror_mask'] if args.VesselMasking else None
        if 'diff' in args.perf_pattern and have_GT:
            temp_dataset.DPath = TestPath['D'] 
            temp_dataset.LPath = TestPath['L'] 
            temp_dataset.UPath = TestPath['U'] 
            if args.predict_deviation or args.predict_value_mask:
                temp_dataset.orig_DPath = TestPath['orig_D']
                temp_dataset.delta_DPath = TestPath['delta_D']
                temp_dataset.orig_LPath = TestPath['orig_L']
                temp_dataset.delta_LPath = TestPath['delta_L']
        if 'adv' in args.perf_pattern and have_GT:
            temp_dataset.VPath = TestPath['V'] 
            temp_dataset.PhiPath = TestPath['Phi']  
            if args.predict_deviation or args.predict_value_mask:
                temp_dataset.orig_VPath = TestPath['orig_V']
                temp_dataset.delta_VPath = TestPath['delta_V'] 


    '''if args.is_resume: # TODO
        in_channels = checkpoint['in_channels']'''
    print('# of training cases:', len(DataSet_Training))
    DataSet_Training.input_time_frame = in_channels
    
    args.max_loss_n_collocations = min(in_channels, args.max_loss_n_collocations) # max available n_collocations used for PIANO integration # 
    print('Max collocation time points for training: %s' % str(args.max_loss_n_collocations))
    
    for i_test in range(len(TestPaths)):
        DataSets_Testing[i_test].input_time_frame = in_channels
    print('\nInput channels:', str(in_channels))
    print('# of testing cases:', len(DataSets_Testing))
    

    ###########################  V, D Prediction Nets  ###########################

    ##  Setting Info  ##
    setting_info = 'nCol=%s' % (args.input_n_collocations)
    if args.joint_predict:
        setting_info = 'Joint_%s' % setting_info
    if args.gradient_loss:
        setting_info = '%s_GL%s' % (setting_info, args.gl_weight)   
    if args.spatial_gradient_loss:
        setting_info = '%s_SG%s' % (setting_info, args.sgl_weight) 

    # For V, D prediction #
    PIANO = PIANOinD(args, args.data_dim, args.data_spacing, args.perf_pattern, in_channels, device) # NOTE: initialize net as adv+diff prediction
    PIANO.to(device)
    optimizer_PIANO = optim.Adam(PIANO.parameters(), lr = args.lr, weight_decay = args.lr_weight_decay) # betas = [0.9, 0.999]

    if args.is_resume or args.for_test:
        PIANO.load_state_dict(checkpoint['model_state_dict'])
        #optimizer_PIANO.load_state_dict(checkpoint['optimizer_state_dict']) # NOTE: use new-defined optimizer #
        PIANO.train()
        if args.for_test:
            epoch = 0
            args.n_epochs_total = 1
            main_fld = make_dir(os.path.join(main_fld, str(checkpoint['epoch']))) 
            print('-------------- For Test --------------')
        else:
            epoch = checkpoint['epoch'] 
    else:
        epoch = 0
    
    if args.img_type == 'MRP' or args.img_type == 'CTP':
        new_main_fld = os.path.join(os.path.dirname(main_fld), '[Res-%d]_%s' % (epoch, os.path.basename(main_fld)))
        old_main_fld = main_fld
        if not os.path.isdir(new_main_fld):
            os.rename(main_fld, new_main_fld) 
        if os.path.isdir(old_main_fld):
            shutil.rmtree(old_main_fld) # NOTE: in this way, ISLES cannot be started learning from scratch
        main_fld = new_main_fld
        print('Updated main_fld:', main_fld)
  
    file_path = os.path.join(main_fld, 'info (%s)' % setting_info) 
    file = open(file_path, 'a+')

    steps = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_PIANO, steps)
    print('Number of parameters to optimize - %d' % (len(list(PIANO.parameters()))))
    file.write('\nNumber of parameters to optimize - %d' % (len(list(PIANO.parameters()))))

    print('Starting epoch - %d' % epoch)
    file.write('\nStarting epoch - %d' % epoch)

    # For time integration # 
    test_PDE = PIANO_Skeleton(args, args.data_spacing, args.perf_pattern, args.PD_D_type, args.PD_V_type, device) 
    test_PDE.to(device)

    ###########################  Losses and Functions  ###########################

    loss_param_criterion = nn.L1Loss() # Losses for supervised learning in V, D #
    #loss_param_criterion = nn.MSELoss() # Losses for supervised learning in V, D # NOTE: not work well
    loss_param_criterion.to(device)  
    if args.stochastic:
        loss_SDE_criterion = nn.MSELoss() # NOTE
        #loss_SDE_criterion = nn.L1Loss() # NOTE: archived, not work well
        loss_SDE_criterion.to(device)   
        loss_SDE_test_criterion = nn.MSELoss() # NOTE
        loss_SDE_test_criterion.to(device)   
    loss_perf_criterion = nn.MSELoss() # Losses for concentration prediction #
    loss_perf_criterion.to(device)      
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
    
    if args.predict_segment:
        loss_segment_criterion = nn.BCELoss()
        loss_segment_criterion.to(device)
        
    if args.gradient_loss:
        loss_grad_criterion = GradientLoss(args.gl_weight).to(device)
        if 'scalar' in args.PD_D_type:
            def grad_loss_function(V, D, batched = True, perf_pattern = 'adv_diff'): # V: (batch, channel, s, r, c); D: (batch, s, r, c)
                return loss_grad_criterion(list(V.permute(1, 0, 2, 3, 4)) + [D], batched) # -> list: channels: (batch, s, r, c)
        else:
            def grad_loss_function(V, D, batched = False, perf_pattern = 'adv_diff'): # (batch, channel, s, r, c)
                return loss_grad_criterion(list(torch.cat([V, D], dim = 1).permute(1, 0, 2, 3, 4)), batched)

    if args.spatial_gradient_loss:
        loss_spatial_grad_criterion = SpatialGradientLoss(args.sgl_weight)
        loss_spatial_grad_criterion.to(device) 
    
    if 'gauge' in args.PD_V_type and 'adv' in args.perf_pattern:
        loss_gauge_criterion = DivergenceLoss3D(args.gauge_weight) 
        loss_gauge_criterion.to(device)
    def patch_cropping(X): # X: (batch, channel, s, r, c)
        return X[:, :, args.boundary_crop_training[0] : args.data_dim[0] - args.boundary_crop_training[0], \
            args.boundary_crop_training[1] : args.data_dim[1] - args.boundary_crop_training[1], \
                args.boundary_crop_training[2] : args.data_dim[2] - args.boundary_crop_training[2]]

    '''def get_VDlst(V, D):
        Vlst = {'V': V} if 'scalar' in args.PD_V_type else {'Vx': V[:, 0], 'Vy': V[:, 1], 'Vz': V[:, 2]}
        Dlst = {'D': D} if 'scalar' in args.PD_D_type else {'Dxx': D[:, 0], 'Dxy': D[:, 1], 'Dxz': D[:, 2], \
            'Dyy': D[:, 3], 'Dyz': D[:, 4], 'Dzz': D[:, 5]}
        return Vlst, Dlst'''
    def get_Vlst(V):
        Vlst = {'V': V} if 'scalar' in args.PD_V_type else {'Vx': V[:, 0], 'Vy': V[:, 1], 'Vz': V[:, 2]}
        return Vlst
    def get_Dlst(D):
        Dlst = {'D': D} if 'scalar' in args.PD_D_type else {'Dxx': D[:, 0], 'Dxy': D[:, 1], 'Dxz': D[:, 2], \
            'Dyy': D[:, 3], 'Dyz': D[:, 4], 'Dzz': D[:, 5]}
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
    file.write('\nCase - %s' % args.img_type)
    file.write('\nMain folder - %s' % main_fld)
    file.write('\nSetting info - %s' % setting_info)
    file.write('\nStride of testing - %s' % str(args.stride_testing))
    file.write('\nCrop size-%s' % str(args.boundary_crop_training))
    print('PID - %s' % os.getpid())
    print('Device - %s' % device)
    print('Case - %s' % args.img_type)
    print('Main folder - %s' % main_fld)
    print('Setting info - %s' % setting_info)
    print('Stride of testing - %s' % str(args.stride_testing))
    print('Crop size - %s' % str(args.boundary_crop_training))

    #log_dir = os.path.join(make_dir(os.path.join(main_fld, 'TensorBoard')), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #writer = SummaryWriter(log_dir) 
    #print('TensorBoard open command: tensorboard --logdir="%s"' % log_dir)

    #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')
    #time.sleep(4)

    ###########################  Training  ###########################


    if not args.is_resume or args.img_type != 'IXI':
        test_loss_D_lst  = []
        test_loss_L_lst  = []
        test_loss_U_lst  = [] 
        test_loss_V_lst  = []
        test_loss_Seg_lst   = []
        test_loss_perf_lst  = []
        test_loss_Sigma_lst = []
        #test_loss_Sigma_D_lst = []
        #test_loss_Sigma_V_lst = []
        if args.predict_deviation:
            test_loss_orig_D_lst = []
            test_loss_orig_V_lst = []
            test_loss_orig_L_lst = []
            test_loss_delta_D_lst = []
            test_loss_delta_V_lst = []
            test_loss_delta_L_lst = []
        if args.predict_value_mask:
            test_loss_VM_D_lst = []
            test_loss_VM_V_lst = []
            test_loss_orig_V_lst = []
            test_loss_orig_D_lst = []
            test_loss_orig_L_lst = []

    mu_lst, logvar_lst = [], []

    lr_to_change = True
    num_lr_reduce = 0
    lr_reduce_criterion = args.lr_reduce_criterion

    end = time.time()
    file.close()

    while epoch < args.n_epochs_total:
        epoch += 1 
        
        file = open(file_path, 'a')

        if epoch % args.test_freq == 0 or epoch == 1: 
            n_test_iter += 1

        #if epoch % args.test_freq == 0:
            #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')

            print('Main folder: %s' % main_fld)
            
            main_test_fld = make_dir(os.path.join(main_fld, '%d' % epoch)) if not args.for_test else main_fld
            
            for param_group in optimizer_PIANO.param_groups:
                print('Current learning rate: %.5f' % param_group['lr'])

            # Save Model #
            if not args.for_test:
                torch.save({
                    'model_state_dict': PIANO.state_dict(),
                    'optimizer_state_dict': optimizer_PIANO.state_dict(),
                    'in_channels': in_channels,
                    'model_info': model_info,
                    'epoch': epoch,
                    }, os.path.join(main_test_fld, 'checkpoint (%s).pth' % setting_info))

                torch.save({
                    'model_state_dict': PIANO.state_dict(),
                    'optimizer_state_dict': optimizer_PIANO.state_dict(),
                    'in_channels': in_channels,
                    'model_info': model_info,
                    'epoch': epoch,
                    }, os.path.join(main_fld, 'latest_checkpoint.pth'))

            with torch.no_grad():   
                
                test_loss_D  = np.zeros(len(DataSets_Testing) + 2) # Add 2 stats: 25 & 75 percentiles #
                test_loss_L  = np.zeros(len(DataSets_Testing) + 2)  
                test_loss_U  = np.zeros(len(DataSets_Testing) + 2) 
                test_loss_V  = np.zeros(len(DataSets_Testing) + 2) 
                test_loss_perf  = np.zeros(len(DataSets_Testing) + 2)
                test_loss_Sigma = np.zeros(len(DataSets_Testing) + 2)
                #test_loss_Sigma_D = np.zeros(len(DataSets_Testing) + 2)
                #test_loss_Sigma_V = np.zeros(len(DataSets_Testing) + 2)
                if args.predict_segment:
                    test_loss_Seg = np.zeros(len(DataSets_Testing) + 2) 
                if args.predict_value_mask:
                    test_loss_VM_D = np.zeros(len(DataSets_Testing) + 2) 
                    test_loss_VM_V = np.zeros(len(DataSets_Testing) + 2) 
                if args.predict_deviation:
                    test_loss_orig_D = np.zeros(len(DataSets_Testing) + 2)
                    test_loss_orig_V = np.zeros(len(DataSets_Testing) + 2)
                    test_loss_orig_L = np.zeros(len(DataSets_Testing) + 2) 
                    test_loss_delta_D = np.zeros(len(DataSets_Testing) + 2)
                    test_loss_delta_V = np.zeros(len(DataSets_Testing) + 2)
                    test_loss_delta_L = np.zeros(len(DataSets_Testing) + 2) 
                if args.predict_value_mask:
                    test_loss_orig_D = np.zeros(len(DataSets_Testing) + 2)
                    test_loss_orig_V = np.zeros(len(DataSets_Testing) + 2)
                    test_loss_orig_L = np.zeros(len(DataSets_Testing) + 2)  

                for i_test in range(len(DataSets_Testing)):
                    temp_dataset = DataSets_Testing[i_test]
                    print('Testing %d/%d: %s' % (i_test+1, len(DataSets_Testing), temp_dataset.CaseName))
                    file.write('\nTesting %d/%d: %s' % (i_test+1, len(DataSets_Testing), temp_dataset.CaseName))

                    save_test_fld = make_dir(os.path.join(main_test_fld, '%s' % temp_dataset.CaseName))
                    D_path = os.path.join(save_test_fld, "D.mha")
                    V_path = os.path.join(save_test_fld, "V.mha")

                    origin, spacing, direction, data_dim, mask_nda, mask_path = temp_dataset.origin, temp_dataset.spacing, \
                        temp_dataset.direction, temp_dataset.data_dim, temp_dataset.get_mask(), temp_dataset.MaskPath
                    mask = torch.from_numpy(mask_nda).to(device)
                    def masked(tensor): # nda:  (s, r, c, channel) or (s, r, c); mask: (s, r, c)
                        return tensor * mask if tensor.size() == mask.size() else tensor * mask.unsqueeze(-1)
                    
                    #nT = temp_dataset.N_collocations
                    #cnt = torch.zeros((max(nT, 9), data_dim[0], data_dim[1], data_dim[2]), device = device)
                    cnt = torch.zeros((9, data_dim[0], data_dim[1], data_dim[2]), device = device)
                    if args.predict_segment:
                        merge_Seg = torch.zeros((data_dim[0], data_dim[1],data_dim[2]), device = device)
                    if args.predict_value_mask:
                        if args.separate_DV_value_mask:
                            merge_VM_D = torch.zeros((data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_VM_V = torch.zeros((data_dim[0], data_dim[1], data_dim[2]), device = device)
                        else:
                            merge_VM = torch.zeros((data_dim[0], data_dim[1],data_dim[2]), device = device)
                    if 'adv' in args.perf_pattern:
                        merge_V = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        if args.predict_deviation:
                            merge_base_V = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_delta_V = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        elif args.predict_value_mask:
                            merge_base_V = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        if 'clebsch' in args.PD_V_type:
                            merge_Phi = torch.zeros((2, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        elif 'stream' in args.PD_V_type:
                            merge_Phi = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        elif 'HHD' in args.PD_V_type:
                            merge_Phi = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_H = torch.zeros((data_dim[0], data_dim[1], data_dim[2]), device = device)
                    if 'diff' in args.perf_pattern:
                        merge_D = torch.zeros((data_dim[0], data_dim[1], data_dim[2]), device = device) if 'scalar' in args.PD_D_type else \
                            torch.zeros((6, data_dim[0], data_dim[1], data_dim[2]), device = device) # (Dxx, Dxy, Dxxz, Dyy, Dyz, Dzz)
                        if 'cholesky' in args.PD_D_type or 'dual' in args.PD_D_type:
                            merge_L =torch.zeros((6, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        if 'spectral' in args.PD_D_type:
                            merge_S =torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_L =torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_U =torch.zeros((9, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        if args.predict_deviation:
                            merge_base_D = torch.zeros((6, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_delta_D = torch.zeros((6, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_base_L = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_delta_L = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                        elif args.predict_value_mask:
                            merge_base_D = torch.zeros((6, data_dim[0], data_dim[1], data_dim[2]), device = device)
                            merge_base_L = torch.zeros((3, data_dim[0], data_dim[1], data_dim[2]), device = device)
                    if args.stochastic:
                        # TODO: TESTING shared uncertainty # 
                        '''if args.separate_DV_value_mask:
                            merge_Sigma_D = torch.zeros((data_dim[0], data_dim[1],data_dim[2]), device = device)
                            merge_Sigma_V = torch.zeros((data_dim[0], data_dim[1],data_dim[2]), device = device)
                        else:
                            merge_Sigma = torch.zeros((data_dim[0], data_dim[1],data_dim[2]), device = device)'''
                        merge_Sigma = torch.zeros((data_dim[0], data_dim[1],data_dim[2]), device = device)

                    for i, batch in enumerate(test_loaders[i_test]):
                        #print('%s of %s' % (i, len(test_loaders[i_test])))

                        start_slc, end_slc = temp_dataset.idx2info[i, 0].astype(int)
                        start_row, end_row = temp_dataset.idx2info[i, 1].astype(int)
                        start_col, end_col = temp_dataset.idx2info[i, 2].astype(int)
                        #start_it,  end_it  = temp_dataset.idx2info[i, 3].astype(int)

                        crop_start_s = 0 if start_slc == 0 else args.boundary_crop_testing[0] 
                        crop_end_s = 0 if end_slc == temp_dataset.slc else args.boundary_crop_testing[0] 
                        crop_start_r = 0 if start_row == 0 else args.boundary_crop_testing[1] 
                        crop_end_r = 0 if end_row == temp_dataset.row else args.boundary_crop_testing[1] 
                        crop_start_c = 0 if start_col == 0 else args.boundary_crop_testing[2] 
                        crop_end_c = 0 if end_col == temp_dataset.col else args.boundary_crop_testing[2] 

                        start_slc, end_slc = start_slc + crop_start_s, end_slc - crop_end_s
                        start_row, end_row = start_row + crop_start_r, end_row - crop_end_r
                        start_col, end_col = start_col + crop_start_c, end_col - crop_end_c
                        patch_start_slc, patch_end_slc = crop_start_s, args.data_dim[0] - crop_end_s
                        patch_start_row, patch_end_row = crop_start_r, args.data_dim[1] - crop_end_r
                        patch_start_col, patch_end_col = crop_start_c, args.data_dim[2] - crop_end_c

                        cnt[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += 1

                        movie4input = Variable(batch['movie4input'].float().to(device)) # (n_batch, nT, S, r, c)
                        # Predict V, D ## Pad V, D patches into full domain #
                        PIANO.input_features = movie4input
                        if args.VesselMasking and args.img_type != 'IXI':
                            PIANO.vessel_mask = Variable(batch['vessel_mask'].unsqueeze(1).float().to(device)) #  # (n_batch, s, r, c) ->(n_batch, 1, s, r, c)
                            PIANO.vessel_mirror_mask = Variable(batch['vessel_mirror_mask'].unsqueeze(1).float().to(device)) #  # (n_batch, s, r, c) ->(n_batch, 1, s, r, c)
                            '''if args.predict_value_mask:
                                if args.separate_DV_value_mask:
                                    PIANO.anomaly_D_mask = Variable(batch['value_mask_D'].unsqueeze(1).float().to(device))
                                    PIANO.anomaly_V_mask = Variable(batch['value_mask_V'].unsqueeze(1).float().to(device))
                                else:
                                    PIANO.anomaly_mask = Variable(batch['value_mask'].unsqueeze(1).float().to(device)) # (n_batch, 1, s, r, c)'''
                            if args.DiffusionMasking:
                                PIANO.diffusion_mask = 1. - PIANO.vessel_mask 
                        if args.stochastic and args.img_type != 'IXI':
                            # TODO: TESTING shared uncertainty # 
                            '''if args.predict_value_mask: 
                                PIANO.sigma_D_mask = 1. - PIANO.anomaly_D_mask
                                PIANO.sigma_V_mask = 1. - PIANO.anomaly_V_mask
                            else:
                                PIANO.sigma_mask = 1. - PIANO.vessel_mask # (n_batch, 1, s, r, c)'''
                            PIANO.sigma_mask = 1. - PIANO.vessel_mask # (n_batch, 1, s, r, c)

                        base_V, base_D, delta_V, delta_D, Sigma = PIANO.get_VD() # n_batch = 1 
                        
                        if args.predict_segment:
                            if args.segment_net_type == 'conc':
                                seg_mask = PIANO.get_segment(threshold = None)[0, 0] # (n_batch, 1, s, r, c)
                            elif args.segment_net_type == 'dev':
                                assert args.predict_deviation
                                base_L, delta_L = PIANO.get_L() # (batch = 1, 3, s, r, c)
                                seg_mask = PIANO.get_segment(threshold = None, physics_deviation = torch.cat([delta_V, delta_L], dim = 1))[0, 0]
                            merge_Seg[start_slc : end_slc, start_row : end_row, start_col : end_col] += seg_mask[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
 
                        if args.predict_value_mask:
                            if args.separate_DV_value_mask:
                                value_mask_D, value_mask_V = PIANO.get_value_mask() # (n_batch=1, 1, s, r, c)
                                value_mask_D, value_mask_V = value_mask_D[0, 0], value_mask_V[0, 0]
                                merge_VM_D[start_slc : end_slc, start_row : end_row, start_col : end_col] += value_mask_D[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                merge_VM_V[start_slc : end_slc, start_row : end_row, start_col : end_col] += value_mask_V[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                            else:
                                value_mask = PIANO.get_value_mask()[0, 0] # (n_batch=1, 1, s, r, c)  
                                value_mask_D, value_mask_V = value_mask, value_mask
                                merge_VM[start_slc : end_slc, start_row : end_row, start_col : end_col] += value_mask[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]

                        if args.predict_deviation:
                            V, D = base_V + delta_V, base_D + delta_D
                            base_V, delta_V, base_D, delta_D = base_V[0], delta_V[0], base_D[0], delta_D[0]
                        elif args.predict_value_mask:
                            if args.img_type != 'IXI':
                                V, D = base_V * PIANO.vessel_mask, base_D * PIANO.vessel_mask
                                base_V, base_D = base_V * PIANO.vessel_mirror_mask, base_D
                            else:
                                if args.separate_DV_value_mask:
                                    V = base_V * value_mask_V[None, None]
                                    D = base_D * value_mask_D[None, None]
                                else:
                                    V, D = base_V * value_mask[None, None], base_D * value_mask[None, None]
                            base_V, base_D = base_V[0], base_D[0]
                        else:
                            V, D = base_V, base_D
                        V, D = V[0] , D[0] # (n_channel, s, r, c)  

                        if 'adv' in args.perf_pattern: 
                            merge_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += V[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                            if args.predict_deviation:
                                merge_base_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += base_V[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                merge_delta_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += delta_V[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                            if args.predict_value_mask:
                                merge_base_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += base_V[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                            if 'stream' in args.PD_V_type or 'clebsch' in args.PD_V_type:
                                Phi = PIANO.get_Phi()[0]
                                merge_Phi[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += Phi[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col] 
                                del Phi
                            elif 'HHD' in args.PD_V_type:
                                Phi = PIANO.get_Phi()[0]
                                merge_Phi[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += Phi[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col] 
                                H = PIANO.get_H()[0]
                                merge_H[start_slc : end_slc, start_row : end_row, start_col : end_col] += H[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col] 
                                del Phi, H

                        if 'diff' in args.perf_pattern: 
                            if 'scalar' in args.PD_D_type:
                                merge_D[start_slc : end_slc, start_row : end_row, start_col : end_col] += D[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                            elif 'full' in args.PD_D_type: # (bacth = 1, 3, s, r, c): Dxx, Dxy, Dyy
                                merge_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += D[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col] 
                                if args.predict_deviation:
                                    merge_base_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += base_D[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                    merge_delta_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += delta_D[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                if args.predict_value_mask:
                                    merge_base_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += base_D[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                if 'cholesky' in args.PD_D_type or 'dual' in args.PD_D_type: # Archieved
                                    L = PIANO.get_L()[0]
                                    merge_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += L[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                    del L
                                if 'spectral' in args.PD_D_type:
                                    base_L, delta_L = PIANO.get_L() # (batch = 1, 3, s, r, c)
                                    if args.predict_deviation:
                                        L = base_L + delta_L
                                        L, base_L, delta_L = L[0], base_L[0], delta_L[0] 
                                        merge_base_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += base_L[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                        merge_delta_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += delta_L[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                    if args.predict_value_mask:
                                        L = base_L * value_mask_D[None, None]
                                        L, base_L = L[0], base_L[0] 
                                        merge_base_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += base_L[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                    else:
                                        L = base_L[0]
                                    
                                    merge_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += L[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                    U = PIANO.get_U()[0]
                                    merge_U[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += U[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                    S = PIANO.get_S()[0]
                                    merge_S[:, start_slc : end_slc, start_row : end_row, start_col : end_col] += S[:, patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                    del L, U, S
                        if args.stochastic:
                            # TODO: TESTING shared uncertainty # 
                            '''if args.separate_DV_value_mask:
                                Sigma_D, Sigma_V = Sigma[0, 0], Sigma[0, 1] # (n_batch = 1, 2, s, r, c) -> (s, r, c)   
                                merge_Sigma_D[start_slc : end_slc, start_row : end_row, start_col : end_col] += Sigma_D[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                merge_Sigma_V[start_slc : end_slc, start_row : end_row, start_col : end_col] += Sigma_V[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                                del Sigma_D, Sigma_V
                            else:'''
                            Sigma = Sigma[0, 0] # (n_batch = 1, 1, s, r, c) -> (s, r, c)   
                            merge_Sigma[start_slc : end_slc, start_row : end_row, start_col : end_col] += Sigma[patch_start_slc : patch_end_slc, patch_start_row : patch_end_row, patch_start_col : patch_end_col]
                            del Sigma 

                    test_sample = temp_dataset.get_full_sample()
                    # (nT, s, r, c)
                    sub_collocation_t, full_movie = test_sample['sub_collocation_t'].float().to(device), test_sample['full_movie'].float().to(device) 
                    sub_collocation_t, full_movie = Variable(sub_collocation_t).to(device), Variable(full_movie).to(device)
                    if 'dirichlet' in temp_dataset.BC or 'cauchy' in temp_dataset.BC:
                        contours = test_sample['movie_BC'] # BC list
                        #print(contours)
                        #contours = Variable(args.contours.to(device)) # (n_batch=1, nT, 6, BC_size, r, c)
                    elif 'source' in temp_dataset.BC:
                        contours = test_sample['movie_dBC']
                        #contours = Variable(args.contours.to(device)) # (n_batch=1, nT, 6, BC_size, s, r, c)'''
                    #args.contours, args.dcontours = None, None

                    ############################################
                    ######## Avg. over patched results #########
                    ############################################ 

                    #############################################
                    ######### Losses over entire domain #########
                    #############################################

                    if args.stochastic: 
                        merge_Sigma[cnt[0] != 0] = merge_Sigma[cnt[0] != 0] / cnt[0][cnt[0] != 0]  # (s, r, c)
                        test_PDE.Sigma = merge_Sigma.unsqueeze(0) # (n_batch = 1, s, r, c)  
                        
                        if args.img_type == 'IXI': 
                            # TODO: TESTING shared uncertainty # 
                            '''if args.separate_DV_value_mask:
                                GT_Sigma_D = 1. - test_sample['value_mask_D'].float().to(device) # (s, r, c) 
                                GT_Sigma_V = 1. - test_sample['value_mask_V'].float().to(device) # (s, r, c) 
                                
                                full_Sigma_D_loss = (loss_SDE_test_criterion(merge_Sigma_D, GT_Sigma_D)).item() 
                                full_Sigma_V_loss = (loss_SDE_test_criterion(merge_Sigma_V, GT_Sigma_V)).item() 
                                test_loss_Sigma_D[i_test] = full_Sigma_D_loss
                                test_loss_Sigma_V[i_test] = full_Sigma_V_loss

                                print('      | SigD {:.9f}'.format(full_Sigma_D_loss))  
                                print('      | SigV {:.9f}'.format(full_Sigma_V_loss))  
                                file.write('\n      | SigD {:.9f}'.format(full_Sigma_D_loss)) 
                                file.write('\n      | SigV {:.9f}'.format(full_Sigma_V_loss)) 
                            else:
                                GT_Sigma =  1. - test_sample['value_mask'].float().to(device) # (s, r, c) 
                                
                                full_Sigma_loss = (loss_SDE_test_criterion(merge_Sigma, GT_Sigma)).item() 
                                test_loss_Sigma_D[i_test] = full_Sigma_loss
                                test_loss_Sigma_V[i_test] = full_Sigma_loss'''
                            GT_Sigma = test_sample['sigma'].float().to(device) # (s, r, c) 
                            
                            full_Sigma_loss = (loss_SDE_test_criterion(merge_Sigma, GT_Sigma)).item() 
                            test_loss_Sigma[i_test] = full_Sigma_loss

                            print('      |  Sig {:.9f}'.format(full_Sigma_loss))  
                            file.write('\n      |  Sig {:.9f}'.format(full_Sigma_loss))  
                    
                    if args.predict_segment:
                        merge_Seg[cnt[0] != 0] = merge_Seg[cnt[0] != 0] / cnt[0][cnt[0] != 0] 
                        GT_Seg = test_sample['lesion_seg'].float().to(device)

                        full_Seg_loss = (loss_segment_criterion(merge_Seg, GT_Seg)).item()
                        test_loss_Seg[i_test] = full_Seg_loss
                        print('      |  Seg {:.9f}'.format(full_Seg_loss))  
                        file.write('\n      |  Seg {:.9f}'.format(full_Seg_loss)) 

                    if args.predict_value_mask and args.img_type == 'IXI':
                        GT_VM_D = test_sample['value_mask_D'].float().to(device)
                        GT_VM_V = test_sample['value_mask_V'].float().to(device)
                        if args.separate_DV_value_mask:
                            merge_VM_D[cnt[0] != 0] = merge_VM_D[cnt[0] != 0] / cnt[0][cnt[0] != 0]
                            merge_VM_V[cnt[0] != 0] = merge_VM_V[cnt[0] != 0] / cnt[0][cnt[0] != 0]
                            
                            full_VM_D_loss = loss_param_criterion(merge_VM_D, GT_VM_D).item()
                            full_VM_V_loss = loss_param_criterion(merge_VM_V, GT_VM_V).item()  
                        else:    
                            merge_VM[cnt[0] != 0] = merge_VM[cnt[0] != 0] / cnt[0][cnt[0] != 0]  
                            #GT_VM = test_sample['value_mask'].float().to(device) 
                            full_VM_D_loss = (loss_param_criterion(merge_VM, GT_VM_D)).item()
                            full_VM_V_loss = (loss_param_criterion(merge_VM, GT_VM_V)).item()
                        
                        test_loss_VM_D[i_test] = full_VM_D_loss 
                        test_loss_VM_V[i_test] = full_VM_V_loss 
                        print('      | VM_D {:.9f}'.format(full_VM_D_loss))  
                        print('      | VM_V {:.9f}'.format(full_VM_V_loss))  
                        file.write('\n      | VM_D {:.9f}'.format(full_VM_D_loss)) 
                        file.write('\n      | VM_V {:.9f}'.format(full_VM_V_loss))  

                    if 'adv' in args.perf_pattern:
                        merge_V[cnt[:3] != 0] = merge_V[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0]  

                        test_PDE.Vlst = get_Vlst(merge_V.unsqueeze(0)) 
                        if 'clebsch' in args.PD_V_type:
                            merge_Phi[cnt[:2] != 0] = merge_Phi[cnt[:2] != 0] / cnt[:2][cnt[:2] != 0]

                        elif 'stream' in args.PD_V_type:
                            merge_Phi[cnt[:3] != 0] = merge_Phi[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0]

                            if args.predict_deviation:
                                merge_base_V[cnt[:3] != 0] = merge_base_V[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0]
                                merge_delta_V[cnt[:3] != 0] = merge_delta_V[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0]

                            if args.predict_value_mask:
                                merge_base_V[cnt[:3] != 0] = merge_base_V[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0] 

                        if args.img_type != "MRP" and args.img_type != "CTP":
                            if args.predict_deviation:
                                GT_orig_V, GT_delta_V = test_sample['orig_V'].float().to(device), test_sample['delta_V'].float().to(device)
                                GT_V = GT_orig_V + GT_delta_V
                                full_V_loss, full_orig_V_loss, full_delta_V_loss = 0., 0., 0.
                                for i in range(3):
                                    full_V_loss += (loss_param_criterion(merge_V[i], GT_V[i])).item()
                                    full_orig_V_loss += (loss_param_criterion(merge_base_V[i], GT_orig_V[i])).item()
                                    full_delta_V_loss += (loss_param_criterion(merge_delta_V[i], GT_delta_V[i])).item()
  
                                test_loss_orig_V[i_test] = full_orig_V_loss 
                                test_loss_delta_V[i_test] = full_delta_V_loss 

                                print('      | barV {:.9f}'.format(full_orig_V_loss)) 
                                file.write('\n      | barV {:.9f}'.format(full_orig_V_loss)) 
                                print('      |   dV {:.9f}'.format(full_delta_V_loss)) 
                                file.write('\n      |   dV {:.9f}'.format(full_delta_V_loss)) 
                            elif args.predict_value_mask:
                                GT_orig_V, GT_V = test_sample['orig_V'].float().to(device), test_sample['V'].float().to(device)
                                full_V_loss, full_orig_V_loss = 0., 0.
                                for i in range(3):
                                    full_V_loss += (loss_param_criterion(merge_V[i], GT_V[i])).item()
                                    full_orig_V_loss += (loss_param_criterion(merge_base_V[i], GT_orig_V[i])).item()
 
                                test_loss_orig_V[i_test] = full_orig_V_loss 

                                print('      | barV {:.9f}'.format(full_orig_V_loss)) 
                                file.write('\n      | barV {:.9f}'.format(full_orig_V_loss)) 
                            else:
                                GT_V = test_sample['V'].float().to(device) # (3, s, r, c)
                                full_V_loss = 0.
                                for i in range(3):
                                    full_V_loss += (loss_param_criterion(merge_V[i], GT_V[i])).item() 

                            test_loss_V[i_test] = full_V_loss  
                            print('      |    V {:.9f}'.format(full_V_loss)) 
                            file.write('\n      |    V {:.9f}'.format(full_V_loss)) 


                    if 'diff' in args.perf_pattern:
                        test_PDE.Dlst = get_Dlst(merge_D.unsqueeze(0))
                        if 'scalar' in args.PD_D_type:
                            merge_D[cnt[0] != 0] = merge_D[cnt[0] != 0] / cnt[0][cnt[0] != 0] 
                        else:
                            merge_D[cnt[:6] != 0] = merge_D[cnt[:6] != 0] / cnt[:6][cnt[:6] != 0] # (6, s, r, c): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz

                            if 'cholesky' in args.PD_D_type or 'dual' in args.PD_D_type:
                                merge_L[cnt[:6] != 0] = merge_L[cnt[:6] != 0] / cnt[:6][cnt[:6] != 0] 

                            if 'spectral' in args.PD_D_type:
                                merge_L[cnt[:3] != 0] = merge_L[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0]
                                merge_U[cnt[:9] != 0] = merge_U[cnt[:9] != 0] / cnt[:9][cnt[:9] != 0] 
                                merge_S[cnt[:3] != 0] = merge_S[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0] 
                            
                                if args.predict_value_mask:
                                    merge_base_D[cnt[:6] != 0] = merge_base_D[cnt[:6] != 0] / cnt[:6][cnt[:6] != 0] # (6, s, r, c): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz 
                                    merge_base_L[cnt[:3] != 0] = merge_base_L[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0] 

                                if args.predict_deviation:
                                    merge_base_D[cnt[:6] != 0] = merge_base_D[cnt[:6] != 0] / cnt[:6][cnt[:6] != 0] # (6, s, r, c): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
                                    merge_delta_D[cnt[:6] != 0] = merge_delta_D[cnt[:6] != 0] / cnt[:6][cnt[:6] != 0] # (6, s, r, c): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
                                    merge_base_L[cnt[:3] != 0] = merge_base_L[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0]
                                    merge_delta_L[cnt[:3] != 0] = merge_delta_L[cnt[:3] != 0] / cnt[:3][cnt[:3] != 0]

                            if args.img_type != "MRP" and args.img_type != "CTP":

                                if args.predict_deviation:
                                    GT_orig_D, GT_delta_D = test_sample['orig_D'].float().to(device), test_sample['delta_D'].float().to(device)
                                    GT_orig_L, GT_delta_L = test_sample['orig_L'].float().to(device), test_sample['delta_L'].float().to(device)
                                    GT_D = GT_orig_D + GT_delta_D
                                    GT_L = GT_orig_L + GT_delta_L 
                                    # Dxx, Dxy, Dxz, Dyy, Dyz, Dzz

                                    full_D_loss = (loss_param_criterion(merge_D[0], GT_D[0]) + loss_param_criterion(merge_D[1], GT_D[1]) * 2 + loss_param_criterion(merge_D[2], GT_D[2]) * 2 + \
                                        loss_param_criterion(merge_D[3], GT_D[4]) + loss_param_criterion(merge_D[4], GT_D[5]) * 2 + loss_param_criterion(merge_D[5], GT_D[8])).item()
                                    full_orig_D_loss = (loss_param_criterion(merge_base_D[0], GT_orig_D[0]) + loss_param_criterion(merge_base_D[1], GT_orig_D[1]) * 2 + loss_param_criterion(merge_base_D[2], GT_orig_D[2]) * 2 + \
                                        loss_param_criterion(merge_base_D[3], GT_orig_D[4]) + loss_param_criterion(merge_base_D[4], GT_orig_D[5]) * 2 + loss_param_criterion(merge_base_D[5], GT_orig_D[8])).item()
                                    full_delta_D_loss = (loss_param_criterion(merge_delta_D[0], GT_delta_D[0]) + loss_param_criterion(merge_delta_D[1], GT_delta_D[1]) * 2 + loss_param_criterion(merge_delta_D[2], GT_delta_D[2]) * 2 + \
                                        loss_param_criterion(merge_delta_D[3], GT_delta_D[4]) + loss_param_criterion(merge_delta_D[4], GT_delta_D[5]) * 2 + loss_param_criterion(merge_delta_D[5], GT_delta_D[8])).item()
                                    
                                    full_L_loss, full_orig_L_loss, full_delta_L_loss = 0., 0., 0.
                                    for i in range(3):
                                        full_L_loss += (loss_param_criterion(merge_L[i], GT_L[i])).item() 
                                        full_orig_L_loss += (loss_param_criterion(merge_base_L[i], GT_orig_L[i])).item() 
                                        full_delta_L_loss += (loss_param_criterion(merge_delta_L[i], GT_delta_L[i])).item() 
                                    test_loss_orig_L[i_test] = full_orig_L_loss 
                                    test_loss_delta_L[i_test] = full_delta_L_loss 
                                         
                                    test_loss_orig_D[i_test] = full_orig_D_loss
                                    test_loss_delta_D[i_test] = full_delta_D_loss  

                                    print('      | barD {:.9f}'.format(full_orig_D_loss)) 
                                    file.write('\n      | barD {:.9f}'.format(full_orig_D_loss)) 
                                    print('      |   dD {:.9f}'.format(full_delta_D_loss)) 
                                    file.write('\n      |   dD {:.9f}'.format(full_delta_D_loss)) 
                                    print('      | barL {:.9f}'.format(full_orig_L_loss)) 
                                    file.write('\n      | barL {:.9f}'.format(full_orig_L_loss)) 
                                    print('      |   dL {:.9f}'.format(full_delta_L_loss)) 
                                    file.write('\n      |   dL {:.9f}'.format(full_delta_L_loss))
                                elif args.predict_value_mask:
                                    GT_orig_D, GT_D = test_sample['orig_D'].float().to(device), test_sample['D'].float().to(device)
                                    GT_orig_L, GT_L = test_sample['orig_L'].float().to(device), test_sample['L'].float().to(device) 

                                    full_D_loss = (loss_param_criterion(merge_D[0], GT_D[0]) + loss_param_criterion(merge_D[1], GT_D[1]) * 2 + loss_param_criterion(merge_D[2], GT_D[2]) * 2 + \
                                        loss_param_criterion(merge_D[3], GT_D[4]) + loss_param_criterion(merge_D[4], GT_D[5]) * 2 + loss_param_criterion(merge_D[5], GT_D[8])).item()
                                    full_orig_D_loss = (loss_param_criterion(merge_base_D[0], GT_orig_D[0]) + loss_param_criterion(merge_base_D[1], GT_orig_D[1]) * 2 + loss_param_criterion(merge_base_D[2], GT_orig_D[2]) * 2 + \
                                        loss_param_criterion(merge_base_D[3], GT_orig_D[4]) + loss_param_criterion(merge_base_D[4], GT_orig_D[5]) * 2 + loss_param_criterion(merge_base_D[5], GT_orig_D[8])).item()
                                    
                                    full_L_loss, full_orig_L_loss = 0., 0.
                                    for i in range(3):
                                        full_L_loss += (loss_param_criterion(merge_L[i], GT_L[i])).item() 
                                        full_orig_L_loss += (loss_param_criterion(merge_base_L[i], GT_orig_L[i])).item() 

                                    test_loss_orig_L[i_test] = full_orig_L_loss 
                                    test_loss_orig_D[i_test] = full_orig_D_loss

                                    print('      | barD {:.9f}'.format(full_orig_D_loss)) 
                                    file.write('\n      | barD {:.9f}'.format(full_orig_D_loss)) 
                                    print('      | barL {:.9f}'.format(full_orig_L_loss)) 
                                    file.write('\n      | barL {:.9f}'.format(full_orig_L_loss)) 
                                else:
                                    GT_D = test_sample['D'].float().to(device) # (9, s, r, c) 
                                    #GT_D = torch.stack([GT_D[0], GT_D[1], GT_D[2], GT_D[4], GT_D[5], GT_D[8]]) # (6, s, r, c): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
                                    full_D_loss = (loss_param_criterion(merge_D[0], GT_D[0]) + loss_param_criterion(merge_D[1], GT_D[1]) * 2 + loss_param_criterion(merge_D[2], GT_D[2]) * 2 + \
                                        loss_param_criterion(merge_D[3], GT_D[4]) + loss_param_criterion(merge_D[4], GT_D[5]) * 2 + loss_param_criterion(merge_D[5], GT_D[8])).item()  

                                    GT_L = test_sample['L'].float().to(device) # (3, s, r, c) 
                                    full_L_loss = 0.
                                    for i in range(3):
                                        full_L_loss += (loss_param_criterion(merge_L[i], GT_L[i])).item() 
                                
                                test_loss_D[i_test] = full_D_loss 
                                test_loss_L[i_test] = full_L_loss  

                                GT_U = test_sample['U'].float().to(device) # (9, s, r, c)
                                full_U_loss = 0.
                                for i in range(9):
                                    full_U_loss += (min(loss_param_criterion(merge_U[i], GT_U[i]), loss_param_criterion(-merge_U[i], GT_U[i]))).item()  
                                test_loss_U[i_test] = full_U_loss    

                                print('      |    D {:.9f}'.format(full_D_loss)) 
                                file.write('\n      |    D {:.9f}'.format(full_D_loss)) 
                                print('      |    L {:.9f}'.format(full_L_loss)) 
                                file.write('\n      |    L {:.9f}'.format(full_L_loss))  
                                print('      |    U {:.9f}'.format(full_U_loss)) 
                                file.write('\n      |    U {:.9f}'.format(full_U_loss)) 

                    ############################################
                    ###### Integrated concentration looses #####
                    ############################################
                    
                    save_movie_fld = make_dir(os.path.join(save_test_fld, 'Movies'))
                    termporal_movie_fld = make_dir(os.path.join(save_test_fld, 'TimeMachines'))

                    test_PDE.perf_pattern = args.perf_pattern 
                    test_PDE.BC = None # TODO # 
                    test_PDE.stochastic = False # NOTE: disable stochastic term(s) when computing concentration loss #

                    pred_full_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, s, r, c)
                    for i_coll in range(1, full_movie.size(0)):
                        pred_full_movie[i_coll] = masked(pred_full_movie[i_coll - 1])
                        #print(i_coll) # NOTE: (nT, n_batch = 1, s, r, c) -> (nT = -1, s, r, c)
                        for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (s, r, c)
                            pred_full_movie[i_coll] = odeint(test_PDE, pred_full_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                        '''# NOTE: Collocation_t for testing is 2 as entire testing image is too large #
                        pred_full_movie_i = pred_full_movie[-1]
                        for i_t in range(args.collocation_len):
                            print(i_t)
                            pred_full_movie_i = odeint(test_PDE, pred_full_movie_i.unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] # (s, r, c)
                        if 'dirichlet' in temp_dataset.BC or 'cauchy' in temp_dataset.BC or 'source' in temp_dataset.BC:
                            pred_full_movie_i = apply_BC_3D(pred_full_movie_i, contours, i_coll, temp_dataset.BC, batched = False) # (s, r, c)'''
                    
                    full_perf_loss = loss_param_criterion(full_movie, pred_full_movie).item()
                    test_loss_perf[i_test] = full_perf_loss 
                    print('      | Perf {:.9f}'.format(full_perf_loss)) 
                    file.write('\n      | Perf {:.9f}'.format(full_perf_loss))

                    pred_full_movie = masked(pred_full_movie.permute(1, 2, 3, 0)) # (nT, s, r, c) -> (s, r, c, nT)
                    save_sitk(pred_full_movie, os.path.join(save_movie_fld, "PD.mha"), origin, spacing, direction) # (time, row, column)
                    save_temporal(pred_full_movie, termporal_movie_fld, origin, spacing, direction, prefix = 'PD', postfix = '.mha')
                    
                    ############################################
                    ########### Computing Uncertainty ##########
                    ############################################

                    if args.compute_uncertainty > 0 and args.stochastic:
                        test_PDE.stochastic = True # NOTE: reset stochastic term(s) #
                        
                        pred_movie_samples = torch.zeros(tuple([args.compute_uncertainty]) + full_movie[:-1].size()).to(device) # NOTE (n_samples, nT-1, s, r, c): exclude t=0 #
                        for i_sample in range(args.compute_uncertainty):
                            #print('Uncertainty test %d/%d' % (i_sample + 1, args.compute_uncertainty))
                            pred_full_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, s, r, c)
                            for i_coll in range(1, full_movie.size(0)):
                                pred_full_movie[i_coll] = masked(pred_full_movie[i_coll - 1])
                                #print(i_coll) # NOTE: (nT, n_batch = 1, s, r, c) -> (nT = -1, s, r, c)
                                for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (s, r, c)
                                    pred_full_movie[i_coll] = odeint(test_PDE, pred_full_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                            pred_movie_samples[i_sample] = pred_full_movie[:-1] 
                        
                        pred_uncertainty = masked(torch.mean(torch.var(pred_movie_samples, dim = 0), dim = 0)) # (s, r, c)
                        save_sitk(pred_uncertainty, os.path.join(save_test_fld, 'Uncertainty.mha'), origin, spacing, direction)
                        

                    ############################################
                    ######### Save all testing results #########
                    ############################################

                    # Check overlapping paddings #
                    if not os.path.join(GT_fld, "Cnt.mha"):
                        save_sitk(cnt[0], os.path.join(GT_fld, "Cnt.mha"), origin, spacing) # (time, row, column)

                    ## Save adv_only, diff_only part prediction ##
                    '''if args.perf_pattern == 'adv_diff':
                        
                        test_PDE.stochastic = False # NOTE: disable stochastic term(s) when computing concentration loss #
                        
                        #args.contours, args.dcontours = None, None # NOTE: should not imposing abs B.C., only apply Neumann
                        # adv movie predict from t0 #
                        test_PDE.perf_pattern = 'adv_only'
                        pred_adv_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, s, r, c)
                        for i_coll in range(1, full_movie.size(0)):
                            pred_adv_movie[i_coll] = masked(pred_adv_movie[i_coll - 1])
                            #print(i_coll) # NOTE: (nT, n_batch = 1, s, r, c) -> (nT = -1, s, r, c)
                            for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (s, r, c)
                                pred_adv_movie[i_coll] = odeint(test_PDE, pred_adv_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                            
                        pred_adv_movie = masked(pred_adv_movie.permute(1, 2, 3, 0)) # (nT, s, r, c) -> (s, r, c, nT)
                        save_sitk(pred_adv_movie, os.path.join(save_movie_fld, "PD_Adv.mha"), origin, spacing) # (time, row, column)
                        save_temporal(pred_adv_movie, termporal_movie_fld, origin, spacing, direction, prefix = 'PD_Adv', postfix = '.mha')
                    
                        # diff movie predict from t0 #
                        test_PDE.perf_pattern = 'diff_only'
                        pred_diff_movie = torch.stack([full_movie[0]] * full_movie.size(0), dim = 0) # (nT, s, r, c)
                        for i_coll in range(1, full_movie.size(0)):
                            pred_diff_movie[i_coll] = masked(pred_diff_movie[i_coll - 1])
                            #print(i_coll) # NOTE: (nT, n_batch = 1, s, r, c) -> (nT = -1, s, r, c)
                            for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (s, r, c)
                                pred_diff_movie[i_coll] = odeint(test_PDE, pred_diff_movie[i_coll].unsqueeze(0), sub_collocation_t, method = args.integ_method, options = args)[-1, 0] 
                            
                        pred_diff_movie = masked(pred_diff_movie.permute(1, 2, 3, 0)) # (nT, s, r, c) -> (s, r, c, nT)
                        save_sitk(pred_diff_movie, os.path.join(save_movie_fld, "PD_Diff.mha"), origin, spacing) # (time, row, column)
                        save_temporal(pred_diff_movie, termporal_movie_fld, origin, spacing, direction, prefix = 'PD_Diff', postfix = '.mha')
                        del pred_adv_movie, pred_diff_movie'''

                    V_path = os.path.join(save_test_fld, "V.mha")
                    D_path = os.path.join(save_test_fld, "D.mha")
                    '''if args.separate_DV_value_mask:
                        Sigma_D_path = os.path.join(save_test_fld, "Sigma_D.mha") 
                        Sigma_V_path = os.path.join(save_test_fld, "Sigma_V.mha") 
                    else:'''
                    Sigma_path = os.path.join(save_test_fld, 'Sigma.mha') # TODO: TESTING shared uncertainty # 
                    
                    if args.stochastic:
                        # TODO: TESTING shared uncertainty # 
                        '''if args.separate_DV_value_mask:
                            Sigma_D_path = os.path.join(save_test_fld, "Sigma_D.mha") 
                            Sigma_V_path = os.path.join(save_test_fld, "Sigma_V.mha") 
                            Sigma_D = masked(merge_Sigma_D) # (s, r, c)
                            Sigma_V = masked(merge_Sigma_V) # (s, r, c)
                            save_sitk(Sigma_D, Sigma_D_path, origin, spacing, direction) 
                            save_sitk(Sigma_V, Sigma_V_path, origin, spacing, direction) 
                        else:'''
                        Sigma_path = os.path.join(save_test_fld, 'Sigma.mha')
                        Sigma = masked(merge_Sigma) # (s, r, c)
                        save_sitk(Sigma, Sigma_path, origin, spacing, direction) 
                        del Sigma

                    if args.predict_segment:
                        merge_Seg = masked(merge_Seg)
                        save_sitk(merge_Seg, os.path.join(save_test_fld, "LesionSeg.mha") , origin, spacing, direction) 
                        del merge_Seg

                    if args.predict_value_mask:
                        if args.separate_DV_value_mask:
                            merge_VM_D = masked(merge_VM_D)
                            merge_VM_V = masked(merge_VM_V)
                            save_sitk(merge_VM_D, os.path.join(save_test_fld, 'ValueMask_D.mha'), origin, spacing, direction)
                            save_sitk(merge_VM_V, os.path.join(save_test_fld, 'ValueMask_V.mha'), origin, spacing, direction)
                            del merge_VM_D, merge_VM_V
                        else:
                            merge_VM = masked(merge_VM) 
                            save_sitk(merge_VM, os.path.join(save_test_fld, "ValueMask.mha"), origin, spacing, direction) 
                            del merge_VM

                    if 'adv' in args.perf_pattern:
                        merge_V = masked(merge_V.permute(1, 2, 3, 0)) # (s, r, c, dim) 
                        save_sitk(merge_V, V_path, origin, spacing, direction)   
    
                        V_measures(merge_V, save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction)
                        if 'stream' in args.PD_V_type or 'clebsch' in args.PD_V_type:
                            save_sitk(masked(merge_Phi.permute(1, 2, 3, 0)), os.path.join(save_test_fld, "Phi.mha"), origin, spacing, direction) 
                            del merge_Phi
                        if args.predict_value_mask:
                            merge_base_V = masked(merge_base_V.permute(1, 2, 3, 0))  
                            save_sitk(merge_base_V, os.path.join(save_test_fld, 'orig_V.mha'), origin , spacing, direction)
                            V_measures(merge_base_V, save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction, prefix = 'orig_') 
                            
                            del merge_base_V
                        if args.predict_deviation:
                            merge_base_V = masked(merge_base_V.permute(1, 2, 3, 0))
                            merge_delta_V = masked(merge_delta_V.permute(1, 2, 3, 0)) 
                            save_sitk(merge_base_V, os.path.join(save_test_fld, 'orig_V.mha'), origin , spacing, direction)
                            V_measures(merge_base_V, save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction, prefix = 'orig_')
                            save_sitk(merge_delta_V, os.path.join(save_test_fld, 'delta_V.mha'), origin , spacing, direction)
                            V_measures(merge_delta_V, save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction, prefix = 'delta_')
                            
                            del merge_base_V, merge_delta_V

                        del merge_V

                    if 'diff' in args.perf_pattern:
                        if 'scalar' in args.PD_D_type:
                            save_sitk(masked(merge_D), D_path, origin, spacing, direction) # (time, row, column)
                        else:
                            merge_D = masked(merge_D.permute(1, 2, 3, 0)) # (s, r, c, 6): order: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
                            merge_D_tensor = torch.stack([merge_D[..., 0], merge_D[..., 1], merge_D[..., 2], \
                                merge_D[..., 1], merge_D[..., 3], merge_D[..., 4], \
                                    merge_D[..., 2], merge_D[..., 4], merge_D[..., 5]], dim = -1) # (s, r, c, 9): order: [Dxx, Dxy, Dxz, Dxy, Dyy, Dyz, Dxz, Dyz, Dzz]
                            save_sitk(merge_D_tensor, D_path, origin, spacing, direction) # (time, row, column)
                            
                            del merge_D_tensor
                            if 'cholesky' in args.PD_D_type:
                                merge_L = masked(merge_L.permute(1, 2, 3, 0)) # (s, r, c, 6): order: [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz]
                                merge_L_tensor = torch.stack([merge_L[..., 0], merge_L[..., 1], merge_L[..., 2], \
                                    merge_L[..., 1], merge_L[..., 3], merge_L[..., 4], \
                                        merge_L[..., 2], merge_L[..., 4], merge_L[..., 5]], dim = -1) # (s, r, c, 9): order: [Lxx, Lxy, Lxz, Lxy, Lyy, Lyz, Lxz, Lyz, Lzz]
                                save_sitk(merge_L_tensor, os.path.join(save_test_fld, "L.mha"), origin, spacing, direction) # (time, row, column)
                                del merge_L, merge_L_tensor
                            elif 'dual' in args.PD_D_type: # (s, r, c, 6): order: [L1, L2, L3, L4, L5, L6]
                                save_sitk(masked(merge_L.permute(1, 2, 3, 0)), os.path.join(save_test_fld, "L.mha"), origin, spacing, direction) 
                                del merge_L
                            elif 'spectral' in args.PD_D_type:
                                merge_L = masked(merge_L.permute(1, 2, 3, 0)) # (s, r, c, 3)
                                merge_U = masked(merge_U.permute(1, 2, 3, 0)) # (s, r, c, 9)
                                
                                save_sitk(merge_L, os.path.join(save_test_fld, "L.mha"), origin, spacing, direction) # (time, row, column)
                                save_sitk(merge_U, os.path.join(save_test_fld, "U.mha"), origin, spacing, direction) # (time, row, column)
                                save_sitk(masked(merge_S.permute(1, 2, 3, 0)), os.path.join(save_test_fld, "S.mha"), origin, spacing, direction) # (time, row, column)
                                
                                D_measures(merge_L, merge_U.view(merge_U.size(0), merge_U.size(1), merge_U.size(2), 3, 3), \
                                    save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction)

                                if args.predict_value_mask:
                                    merge_base_D = masked(merge_base_D.permute(1, 2, 3, 0)) # (s, r, c, 6): order: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
                                    merge_base_D_tensor = torch.stack([merge_base_D[..., 0], merge_base_D[..., 1], merge_base_D[..., 2], \
                                        merge_base_D[..., 1], merge_base_D[..., 3], merge_base_D[..., 4], \
                                            merge_base_D[..., 2], merge_base_D[..., 4], merge_base_D[..., 5]], dim = -1) # (s, r, c, 9): order: [Dxx, Dxy, Dxz, Dxy, Dyy, Dyz, Dxz, Dyz, Dzz]
                                    save_sitk(merge_base_D_tensor, os.path.join(save_test_fld, 'orig_D.mha'), origin, spacing, direction) # (time, row, column)
                                    
                                    merge_base_L = masked(merge_base_L.permute(1, 2, 3, 0)) # (s, r, c, 3)
                                    save_sitk(merge_base_L, os.path.join(save_test_fld, 'ScalarMaps', "orig_L.mha"), origin, spacing, direction) # (time, row, column)  
                                    D_measures(merge_base_L, merge_U.view(merge_U.size(0), merge_U.size(1), merge_U.size(2), 3, 3), \
                                        save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction, prefix = 'orig_') 

                                    del merge_base_D, merge_base_L 

                                if args.predict_deviation:
                                    merge_base_D = masked(merge_base_D.permute(1, 2, 3, 0)) # (s, r, c, 6): order: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
                                    merge_base_D_tensor = torch.stack([merge_base_D[..., 0], merge_base_D[..., 1], merge_base_D[..., 2], \
                                        merge_base_D[..., 1], merge_base_D[..., 3], merge_base_D[..., 4], \
                                            merge_base_D[..., 2], merge_base_D[..., 4], merge_base_D[..., 5]], dim = -1) # (s, r, c, 9): order: [Dxx, Dxy, Dxz, Dxy, Dyy, Dyz, Dxz, Dyz, Dzz]
                                    save_sitk(merge_base_D_tensor, os.path.join(save_test_fld, 'orig_D.mha'), origin, spacing, direction) # (time, row, column)
                                    
                                    merge_delta_D = masked(merge_delta_D.permute(1, 2, 3, 0)) # (s, r, c, 6): order: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
                                    merge_delta_D_tensor = torch.stack([merge_delta_D[..., 0], merge_delta_D[..., 1], merge_delta_D[..., 2], \
                                        merge_delta_D[..., 1], merge_delta_D[..., 3], merge_delta_D[..., 4], \
                                            merge_delta_D[..., 2], merge_delta_D[..., 4], merge_delta_D[..., 5]], dim = -1) # (s, r, c, 9): order: [Dxx, Dxy, Dxz, Dxy, Dyy, Dyz, Dxz, Dyz, Dzz]
                                    save_sitk(merge_delta_D_tensor, os.path.join(save_test_fld, 'delta_D.mha'), origin, spacing, direction) # (time, row, column)

                                    merge_base_L = masked(merge_base_L.permute(1, 2, 3, 0)) # (s, r, c, 3)
                                    merge_delta_L = masked(merge_delta_L.permute(1, 2, 3, 0)) # (s, r, c, 3)

                                    save_sitk(merge_base_L, os.path.join(save_test_fld, "orig_L.mha"), origin, spacing, direction) # (time, row, column)
                                    save_sitk(merge_delta_L, os.path.join(save_test_fld, "delta_L.mha"), origin, spacing, direction) # (time, row, column)

                                    D_measures(merge_base_L, merge_U.view(merge_U.size(0), merge_U.size(1), merge_U.size(2), 3, 3), \
                                        save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction, prefix = 'orig_')
                                    D_measures(merge_delta_L, merge_U.view(merge_U.size(0), merge_U.size(1), merge_U.size(2), 3, 3), \
                                        save_fld = make_dir(os.path.join(save_test_fld, 'ScalarMaps')), origin = origin, spacing = spacing, direction = direction, prefix = 'delta_')

                                    del merge_base_D, merge_delta_D, merge_base_L, merge_delta_L
                                del merge_L, merge_U, merge_S

                        del merge_D

                    #if 'spectral' not in args.PD_D_type and 'diff' in args.perf_pattern:
                    if 'diff' in args.perf_pattern:
                        # Compute PTI measures of D #
                        PTIWriter = WritePTIImage(save_test_fld, origin, spacing, direction, device, to_smooth = args.smooth_when_learn)
                        PTISolver = PTI(save_test_fld, mask_path, 'diff_only', D_path = D_path, D_type = args.PD_D_type, \
                            V_path = V_path, V_type = args.PD_V_type, device = device, EigenRecompute = True)
                        if 'full' in args.PD_D_type:
                            Trace_path = PTIWriter.save(PTISolver.Trace(), 'Trace (PTI).mha')
                            L_path = PTIWriter.save(PTISolver.eva, 'L (PTI).mha')
                            U_path = PTIWriter.save(PTISolver.U(), 'U (PTI).mha')
                            FA_path = PTIWriter.save(PTISolver.FA(), 'FA (PTI).mha')
                            DColorDirection_path = PTIWriter.save(PTISolver.D_Color_Direction(), 'D_Color_Direction (PTI).mha')
                        '''if 'adv' in args.perf_pattern:
                            Abs_V_path = PTIWriter.save(PTISolver.Abs_V(), 'Abs_V.mha') # NOTE: For V color-coding
                            Norm_V_path = PTIWriter.save(PTISolver.Norm_V(), 'Norm_V.mha')''' 


                #writer.add_scalar('testing_loss', test_loss_perf, epoch) 

                if not args.for_test: 

                    test_loss_perf_lst = plot_loss(test_loss_perf_lst, test_loss_perf, label = 'loss_perf')  
                    
                    if args.img_type != "MRP" and args.img_type != "CTP": 

                        test_loss_V_lst = plot_loss(test_loss_V_lst, test_loss_V, label = 'loss_V')   
                        test_loss_D_lst = plot_loss(test_loss_D_lst, test_loss_D, label = 'loss_D')   

                        test_loss_L_lst = plot_loss(test_loss_L_lst, test_loss_L, label = 'loss_L')   
                        test_loss_U_lst = plot_loss(test_loss_U_lst, test_loss_U, label = 'loss_U')  

                        if args.predict_deviation and args.img_type == 'IXI':
                            test_loss_orig_V_lst = plot_loss(test_loss_orig_V_lst, test_loss_orig_V, label = 'loss_orig_V') 
                            test_loss_delta_V_lst = plot_loss(test_loss_delta_V_lst, test_loss_delta_V, label = 'loss_delta_V') 

                            test_loss_orig_D_lst = plot_loss(test_loss_orig_D_lst, test_loss_orig_D, label = 'loss_orig_D') 
                            test_loss_delta_D_lst = plot_loss(test_loss_delta_D_lst, test_loss_delta_D, label = 'loss_delta_D') 

                            test_loss_orig_L_lst = plot_loss(test_loss_orig_L_lst, test_loss_orig_L, label = 'loss_orig_L') 
                            test_loss_delta_L_lst = plot_loss(test_loss_delta_L_lst, test_loss_delta_L, label = 'loss_delta_L')  

                        if args.predict_segment and args.img_type == 'IXI':
                            test_loss_Seg_lst = plot_loss(test_loss_Seg_lst, test_loss_Seg, label = 'loss_Seg')   

                        if args.predict_value_mask and args.img_type == 'IXI': 
                            test_loss_VM_D_lst = plot_loss(test_loss_VM_D_lst, test_loss_VM_D, label = 'loss_VM_D')  
                            test_loss_VM_V_lst = plot_loss(test_loss_VM_V_lst, test_loss_VM_V, label = 'loss_VM_V')  
                            test_loss_orig_V_lst = plot_loss(test_loss_orig_V_lst, test_loss_orig_V, label = 'loss_orig_V') 
                            test_loss_orig_D_lst = plot_loss(test_loss_orig_D_lst, test_loss_orig_D, label = 'loss_orig_D') 
                            test_loss_orig_L_lst = plot_loss(test_loss_orig_L_lst, test_loss_orig_L, label = 'loss_orig_L') 

                        if args.stochastic and args.img_type == 'IXI': 
                            # TODO: TESTING shared uncertainty # 
                            test_loss_Sigma_lst = plot_loss(test_loss_Sigma_lst, test_loss_Sigma, label = 'loss_Sigma')
                            #test_loss_Sigma_D_lst = plot_loss(test_loss_Sigma_D_lst, test_loss_Sigma_D, label = 'loss_Sigma_D')
                            #test_loss_Sigma_V_lst = plot_loss(test_loss_Sigma_V_lst, test_loss_Sigma_V, label = 'loss_Sigma_V')
                else:
                    
                    print('---------------------------------------') 
                    print('Average:')
                    file.write('\n---------------------------------------') 
                    file.write('\nAverage:')

                    # record avg testnig losses 
                    avg_test_loss_perf = np.mean(test_loss_perf[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                    var_test_loss_perf = np.var(test_loss_perf[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                    print('      |    C {:.9f} ({:.9f})'.format(avg_test_loss_perf, var_test_loss_perf)) 
                    file.write('\n      |    C {:.9f} ({:.9f})'.format(avg_test_loss_perf, var_test_loss_perf)) 

                    if args.img_type != "MRP" and args.img_type != "CTP": 
                        avg_test_loss_V = np.mean(test_loss_V[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        var_test_loss_V = np.var(test_loss_V[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        print('      |    V {:.9f} ({:.9f})'.format(avg_test_loss_V, var_test_loss_V)) 
                        file.write('\n      |    V {:.9f} ({:.9f})'.format(avg_test_loss_V, var_test_loss_V))  
                        
                        avg_test_loss_D = np.mean(test_loss_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        var_test_loss_D = np.var(test_loss_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        print('      |    D {:.9f} ({:.9f})'.format(avg_test_loss_D, var_test_loss_D)) 
                        file.write('\n      |    D {:.9f} ({:.9f})'.format(avg_test_loss_D, var_test_loss_D))  
                        
                        avg_test_loss_L = np.mean(test_loss_L[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        var_test_loss_L = np.var(test_loss_L[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        print('      |    L {:.9f} ({:.9f})'.format(avg_test_loss_L, var_test_loss_L)) 
                        file.write('\n      |    L {:.9f} ({:.9f})'.format(avg_test_loss_L, var_test_loss_L))  
                        
                        avg_test_loss_U = np.mean(test_loss_U[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        var_test_loss_U = np.mean(test_loss_U[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                        print('      |    U {:.9f} ({:.9f})'.format(avg_test_loss_U, var_test_loss_U)) 
                        file.write('\n      |    U {:.9f} ({:.9f})'.format(avg_test_loss_U, var_test_loss_U))    

                        if args.predict_segment and args.img_type == 'IXI':
                            avg_test_loss_Seg = np.mean(test_loss_Seg[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            var_test_loss_Seg = np.mean(test_loss_Seg[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            print('      |  Seg {:.9f} ({:.9f})'.format(avg_test_loss_Seg, var_test_loss_Seg)) 
                            file.write('\n      |  Seg {:.9f} ({:.9f})'.format(avg_test_loss_Seg))    

                        if args.predict_value_mask and args.img_type == 'IXI': 
                            avg_test_loss_VM_D = np.mean(test_loss_VM_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            var_test_loss_VM_D = np.mean(test_loss_VM_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            print('      | VM_D {:.9f} ({:.9f})'.format(avg_test_loss_VM_D, var_test_loss_VM_D)) 
                            file.write('\n      | VM_D {:.9f} ({:.9f})'.format(avg_test_loss_VM_D, var_test_loss_VM_D))  
                            
                            avg_test_loss_VM_V = np.mean(test_loss_VM_V[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            var_test_loss_VM_V = np.mean(test_loss_VM_V[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            print('      | VM_V {:.9f} ({:.9f})'.format(avg_test_loss_VM_V, var_test_loss_VM_V)) 
                            file.write('\n      | VM_V {:.9f} ({:.9f})'.format(avg_test_loss_VM_V, var_test_loss_VM_V))  

                        if args.stochastic and args.img_type == 'IXI': 
                            '''if args.separate_DV_value_mask:
                                avg_test_loss_Sigma_D = np.mean(test_loss_Sigma_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                                var_test_loss_Sigma_D = np.mean(test_loss_Sigma_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                                avg_test_loss_Sigma_V = np.mean(test_loss_Sigma_V[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                                var_test_loss_Sigma_V = np.mean(test_loss_Sigma_V[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                                print('      | SigD {:.9f} ({:.9f})'.format(avg_test_loss_Sigma_D, var_test_loss_Sigma_D)) 
                                print('      | SigV {:.9f} ({:.9f})'.format(avg_test_loss_Sigma_V, var_test_loss_Sigma_V)) 
                                file.write('\n      | SigD {:.9f} ({:.9f})'.format(avg_test_loss_Sigma_D, var_test_loss_Sigma_D))   
                                file.write('\n      | SigV {:.9f} ({:.9f})'.format(avg_test_loss_Sigma_V, var_test_loss_Sigma_V))   
                            else:
                                avg_test_loss_Sigma = np.mean(test_loss_Sigma_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                                var_test_loss_Sigma = np.mean(test_loss_Sigma_D[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                                print('      |  Sig {:.9f} ({:.9f})'.format(avg_test_loss_Sigma, var_test_loss_Sigma)) 
                                file.write('\n      |  Sig {:.9f} ({:.9f})'.format(avg_test_loss_Sigma, var_test_loss_Sigma))   '''
                            # TODO: TESTING shared uncertainty # 
                            avg_test_loss_Sigma = np.mean(test_loss_Sigma[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            var_test_loss_Sigma = np.mean(test_loss_Sigma[:-2]) # (n_test_itr = 1, n_test_samples + 2) 
                            print('      |  Sig {:.9f} ({:.9f})'.format(avg_test_loss_Sigma, var_test_loss_Sigma)) 
                            file.write('\n      |  Sig {:.9f} ({:.9f})'.format(avg_test_loss_Sigma, var_test_loss_Sigma))   

                test_PDE.stochastic = args.stochastic # NOTE: reset stochastic term(s) #
                gc.collect() 

            # Change learning rate when model tends to converge #
            if lr_to_change and not args.no_concentration_loss:
                if len(test_loss_perf_lst) > 10 and np.array(test_loss_perf_lst[-5:]).mean() < lr_reduce_criterion * np.array(test_loss_perf_lst[:2]).mean():
                    print('Reduce learning rate')
                    for g in optimizer_PIANO.param_groups:
                        g['lr'] = g['lr'] * (1 - args.lr_reduce_rate)
                    num_lr_reduce += 1
                    lr_reduce_criterion *= 0.5
                    if num_lr_reduce >= args.max_num_lr_reduce:
                        lr_to_change = False
                
            # Extend batch_nT while approaching to GT_V
            if epoch % args.increase_loss_n_collocations_freq == 0 and args.increase_input_n_collocations:
                if DataSet_Training.loss_n_collocations < args.max_loss_n_collocations:
                    DataSet_Training.loss_n_collocations = DataSet_Training.loss_n_collocations + 1 
                    print('Training batch_nT:', str(DataSet_Training.loss_n_collocations))

            time.sleep(5)
            if epoch >= args.n_epochs_total:
                return # stop here for args.for_test

        ####################################################################################
        ####################################################################################
        ####################################################################################
        
        ## Re-set ##

        epoch_loss_perf  = 0.
        epoch_loss_D     = 0. 
        epoch_loss_L     = 0.
        epoch_loss_U     = 0.
        epoch_loss_V     = 0.
        epoch_loss_Phi   = 0.
        epoch_loss_sgl   = 0.
        epoch_loss_grad  = 0.  
        epoch_loss_gauge = 0.
        epoch_loss_Seg   = 0.
        epoch_loss_VM_D  = 0.
        epoch_loss_VM_V  = 0.

        epoch_loss_jbld  = 0.

        epoch_loss_Sigma   = 0.
        #epoch_loss_Sigma_V = 0.
        #epoch_loss_Sigma_D = 0.


        epoch_loss_orig_D  = 0.
        epoch_loss_delta_D = 0.
        epoch_loss_orig_V  = 0.
        epoch_loss_delta_V = 0.
        epoch_loss_orig_L  = 0.
        epoch_loss_delta_L = 0.


        for i, batch in enumerate(train_loader):
            
            ## Re-set for training ##
            PIANO.vessel_mask, PIANO.vessel_mirror_mask = 1., 1.
            PIANO.anomaly_mask = 1.
            PIANO.diffusion_mask = 1.
            PIANO.sigma_mask = 1.

            optimizer_PIANO.zero_grad()

            # 0 down_scale sample #
            sub_collocation_t, movie4input = \
                Variable(batch['sub_collocation_t'][0].float().to(device), requires_grad = True), \
                    Variable(batch['movie4input'].float().to(device), requires_grad = True) # (n_batch, nT, s, r, c)
            if args.VesselMasking and args.img_type != 'IXI':
                PIANO.vessel_mask = Variable(batch['vessel_mask'].unsqueeze(1).float().to(device)) #  # (n_batch, s, r, c) ->(n_batch, 3, s, r, c)
                if args.DiffusionMasking:
                    PIANO.diffusion_mask = 1. - PIANO.vessel_mask

            if args.predict_value_mask and have_GT:
                #if args.separate_DV_value_mask:
                GT_VM_D = Variable(batch['value_mask_D'].float().to(device))
                GT_VM_V = Variable(batch['value_mask_V'].float().to(device))
                #else:
                #    GT_VM = Variable(batch['value_mask'].float().to(device))

            if args.stochastic and have_GT: 
                # TODO: TESTING shared uncertainty # 
                '''if args.separate_DV_value_mask:
                    GT_Sigma_D = Variable(1. - batch['value_mask_D'].float().to(device), requires_grad = True) 
                    GT_Sigma_V = Variable(1. - batch['value_mask_V'].float().to(device), requires_grad = True) 
                else:
                    GT_Sigma = Variable(1. - batch['value_mask'].float().to(device), requires_grad = True) '''
                GT_Sigma = Variable(batch['sigma'].float().to(device), requires_grad = True) 
            
            if args.predict_segment and have_GT: # archived: not working #
                GT_Seg = Variable(batch['lesion_seg'].float().to(device), requires_grad = False)
            
            if args.GT_LU and 'diff' in args.perf_pattern and have_GT:
                GT_U = Variable(batch['U'].float().to(device), requires_grad = True)
                if args.predict_deviation:
                    GT_orig_L = Variable(batch['orig_L'].float().to(device), requires_grad = True)
                    GT_delta_L = Variable(batch['delta_L'].float().to(device), requires_grad = True)
                elif args.predict_value_mask:
                    GT_orig_L = Variable(batch['orig_L'].float().to(device), requires_grad = True)
                    GT_L = Variable(batch['L'].float().to(device), requires_grad = True)
                else:
                    GT_L = Variable(batch['L'].float().to(device), requires_grad = True)

            if args.GT_D and 'diff' in args.perf_pattern and have_GT:
                if args.predict_deviation:
                    GT_orig_D = Variable(batch['orig_D'].float().to(device), requires_grad = True)
                    GT_delta_D = Variable(batch['delta_D'].float().to(device), requires_grad = True)
                elif args.predict_value_mask:
                    GT_orig_D = Variable(batch['orig_D'].float().to(device), requires_grad = True)
                    GT_D = Variable(batch['D'].float().to(device), requires_grad = True)
                else:
                    GT_D = Variable(batch['D'].float().to(device), requires_grad = True)  

            if args.GT_V and 'adv' in args.perf_pattern and have_GT:
                if args.predict_deviation:
                    GT_orig_V = Variable(batch['orig_V'].float().to(device), requires_grad = True)
                    GT_delta_V = Variable(batch['delta_V'].float().to(device), requires_grad = True)
                elif args.predict_value_mask:
                    GT_orig_V = Variable(batch['orig_V'].float().to(device), requires_grad = True)
                    GT_V = Variable(batch['V'].float().to(device), requires_grad = True)
                else:
                    GT_V = Variable(batch['V'].float().to(device), requires_grad = True)
            if args.GT_Phi and 'adv' in args.perf_pattern and have_GT:
                GT_Phi = Variable(batch['Phi'].float().to(device), requires_grad = True)

            if 'dirichlet' in args.BC or 'cauchy' in args.BC:
                contours = batch['movie_BC']
                #contours = Variable(batch['movie_BC'].float().to(device)) # (n_batch, batch_nT, s, r, c)
            elif 'source' in args.BC:
                contours = batch['movie_dBC']
                #contours = Variable(batch['movie_dBC'].float().to(device)) # (n_batch, batch_nT, s, r, c)

            PIANO.perf_pattern = args.perf_pattern
            PIANO.input_features = movie4input
            #print(movie4input.size())

            #print(batch_nT.size())
            print('  Batch {:d}/{:d}'.format(i + 1, len(train_loader)))

            ## Pass Forward ##
            # Physics ~ (n_batch, channels, s, r, c); Sigma ~ (n_batch, 1/2, s, r, c)
            base_V, base_D, delta_V, delta_D, Sigma = PIANO.get_VD()  
            if args.predict_deviation:
                V, D = base_V + delta_V, base_D + delta_D
            else:
                V, D = base_V, base_D

            ## For Losses ##
            if args.gradient_loss and not args.no_concentration_loss: # NOTE #
                loss_grad = grad_loss_function(V, D, batched = True, perf_pattern  = args.perf_pattern) 
                loss_grad.backward(retain_graph = True)
                epoch_loss_grad += loss_grad.item()
                print('      | Grad {:.9f}'.format(loss_grad.item())) 
            
            if 'gauge' in args.PD_V_type and 'adv' in args.perf_pattern:
                Phi = PIANO.get_Phi() # (batch, 3, s, r, c)
                loss_gauge = loss_gauge_criterion(Phi, batched = True)
                loss_gauge.backward(retain_graph = True)
                epoch_loss_gauge += loss_gauge.item()

            if args.predict_value_mask and have_GT:  
                if args.separate_DV_value_mask:
                    value_mask_D, value_mask_V = PIANO.get_value_mask() # (n_batch, 1, s, r, c)
                    
                    loss_VM_D = loss_param_criterion(value_mask_D[:, 0], GT_VM_D) * args.VM_weight
                    loss_VM_V = loss_param_criterion(value_mask_V[:, 0], GT_VM_V) * args.VM_weight
                    
                    loss_VM_D.backward(retain_graph = True)
                    loss_VM_V.backward(retain_graph = True)
                
                else:
                    value_mask = PIANO.get_value_mask() # (n_batch, 1, s, r, c)
                    value_mask_D, value_mask_V = value_mask, value_mask
                    loss_VM_D = loss_param_criterion(value_mask[:, 0], GT_VM_D) * args.VM_weight
                    loss_VM_V = loss_param_criterion(value_mask[:, 0], GT_VM_V) * args.VM_weight
                    #loss_VM = loss_param_criterion(value_mask[:, 0], GT_VM) * args.VM_weight   

                    loss_VM_D.backward(retain_graph = True) 
                    loss_VM_V.backward(retain_graph = True) 
                    
                epoch_loss_VM_D = loss_VM_D.item()
                epoch_loss_VM_V = loss_VM_V.item()
                print('      | VM_D {:.9f}'.format(loss_VM_D.item()))
                print('      | VM_V {:.9f}'.format(loss_VM_V.item()))
            
            if args.GT_D and 'diff' in args.perf_pattern and have_GT:  
                if args.predict_deviation: # NOTE: archived -- no work well
                    base_D = torch.stack([base_D[:, 0], base_D[:, 1], base_D[:, 2], base_D[:, 1], base_D[:, 3], base_D[:, 4], base_D[:, 2], base_D[:, 4], base_D[:, 5]], dim = 1) # (batch, 9, s, r, c)
                    delta_D = torch.stack([delta_D[:, 0], delta_D[:, 1], delta_D[:, 2], delta_D[:, 1], delta_D[:, 3], delta_D[:, 4], delta_D[:, 2], delta_D[:, 4], delta_D[:, 5]], dim = 1) # (batch, 9, s, r, c)
                    
                    loss_D = loss_param_criterion(base_D, GT_orig_D) * args.GT_D_weight + loss_param_criterion(delta_D, GT_delta_D) * (args.GT_D_weight + args.deviation_extra_weight) 
                     
                elif args.predict_value_mask:
                    base_D = torch.stack([base_D[:, 0], base_D[:, 1], base_D[:, 2], base_D[:, 1], base_D[:, 3], base_D[:, 4], base_D[:, 2], base_D[:, 4], base_D[:, 5]], dim = 1) # (batch, 9, s, r, c) 
                    
                    #####################################
                    ## arvhived JBLD LOSS: not working ##
                    loss_D = 0.

                    #if args.jbld_loss:
                    loss_jbld = JBLD_distance(base_D, GT_orig_D) * 0.1 # * args.GT_D_weight # (batch, 9, s, r, c)
                    if args.actual_physics_loss:
                        loss_jbld = loss_jbld + JBLD_distance(base_D * value_mask_D, GT_D) * 0.1 # * args.GT_D_weight 
                    epoch_loss_jbld += loss_jbld.item() 
                    print('      | JBLD {:.9f}'.format(loss_jbld.item()))

                    if args.jbld_loss:
                        loss_D = loss_D + loss_jbld

                    if (not args.jbld_loss) or (not args.jbld_loss_only):
                        loss_D = loss_D + loss_param_criterion(base_D, GT_orig_D) * args.GT_D_weight 
                        if args.actual_physics_loss:
                            loss_D = loss_D + loss_param_criterion(base_D * value_mask_D, GT_D) * args.GT_D_weight
                    #####################################
                else: 
                    full_D = torch.stack([D[:, 0], D[:, 1], D[:, 2], D[:, 1], D[:, 3], D[:, 4], D[:, 2], D[:, 4], D[:, 5]], dim = 1) # (batch, 9, s, r, c) 
                    loss_D = loss_param_criterion(full_D, GT_D) * args.GT_D_weight
                
                epoch_loss_D += loss_D.item()
                print('      |    D {:.9f}'.format(loss_D.item())) 
                if args.no_concentration_loss and not args.GT_LU and not args.GT_V: 
                    loss_D.backward()
                else:
                    loss_D.backward(retain_graph = True)

            if args.GT_LU and 'diff' in args.perf_pattern and have_GT:

                base_L, delta_L = PIANO.get_L() # (batch = 1, 3, s, r, c)

                if args.predict_deviation:
                    loss_L = loss_param_criterion(base_L, GT_orig_L) * args.GT_L_weight + loss_param_criterion(delta_L, GT_delta_L) * (args.GT_L_weight + args.deviation_extra_weight)
                elif args.predict_value_mask:
                    loss_L = loss_param_criterion(base_L, GT_orig_L) * args.GT_L_weight
                    if args.actual_physics_loss:
                        loss_L = loss_L + loss_param_criterion(base_L * value_mask_D, GT_L) * args.GT_L_weight
                else: 
                    loss_L = loss_param_criterion(base_L, GT_L) * args.GT_L_weight

                epoch_loss_L += loss_L.item()
                loss_L.backward(retain_graph = True)
                print('      |    L {:.9f}'.format(loss_L.item()))


                U = PIANO.get_U()
                loss_U = 0.
                for i in range(9): # take min() avoid orientation ambiguity of eigen-vectors #
                    loss_U += min(loss_param_criterion(U[:, i], GT_U[:, i]), loss_param_criterion(-U[:, i], GT_U[:, i])) * args.GT_U_weight 

                if args.no_concentration_loss and not args.GT_V: 
                    loss_U.backward()
                else:
                    loss_U.backward(retain_graph = True)
                epoch_loss_U += loss_U.item()
                print('      |    U {:.9f}'.format(loss_U.item()))

            if args.GT_V and 'adv' in args.perf_pattern and have_GT:  
                if args.predict_deviation:
                    loss_V = loss_param_criterion(base_V, GT_orig_V) * args.GT_V_weight + loss_param_criterion(delta_V, GT_delta_V) * (args.GT_V_weight + args.deviation_extra_weight) 
                elif args.predict_value_mask:
                    loss_V = loss_param_criterion(base_V, GT_orig_V) * args.GT_V_weight
                    if args.actual_physics_loss:
                        loss_V = loss_V + loss_param_criterion(base_V * value_mask_V, GT_V) * args.GT_V_weight
                else:
                    loss_V = loss_param_criterion(V, GT_V) * args.GT_V_weight
                
                epoch_loss_V += loss_V.item()
                loss_V.backward(retain_graph = True)
                print('      |    V {:.9f}'.format(loss_V.item()))
            
            ####################################
            ############# Archieved ############
            ####################################
            if args.GT_Phi and 'adv' in args.perf_pattern and have_GT:
                Phi = PIANO.get_Phi()
                loss_Phi = loss_param_criterion(Phi, GT_Phi) * args.GT_Phi_weight 

                loss_Phi.backward(retain_graph = True)
                epoch_loss_Phi += loss_Phi.item()
                print('      |  Phi {:.9f}'.format(loss_Phi.item()))

            if args.predict_segment and have_GT: 
                if args.segment_net_type == 'conc':
                    seg_mask = PIANO.get_segment(threshold = None) # (n_batch, 1, s, r, c)
                elif args.segment_net_type == 'dev':
                    if not args.GT_LU:
                        base_L, delta_L = PIANO.get_L() # (batch = 1, 2, r, c)
                    seg_mask = PIANO.get_segment(threshold = None, physics_deviation = torch.cat([delta_V, delta_L], dim = 1))
                    
                loss_seg = loss_segment_criterion(seg_mask[:, 0], GT_Seg) * args.seg_weight 
                loss_seg.backward(retain_graph = True)
                epoch_loss_seg = loss_seg.item()  
            ####################################

            if args.stochastic and args.img_type == 'IXI' and have_GT: 
                # TODO: TESTING shared uncertainty # 
                '''if args.separate_DV_value_mask:
                    loss_SDE_D = loss_SDE_criterion(Sigma[:, 0], GT_Sigma_D) * args.SDE_weight 
                    loss_SDE_V = loss_SDE_criterion(Sigma[:, 1], GT_Sigma_V) * args.SDE_weight 
                    loss_SDE = loss_SDE_D + loss_SDE_V
                    epoch_loss_Sigma_D += loss_SDE_D.item()
                    epoch_loss_Sigma_V += loss_SDE_V.item()
                    print('      | StoD {:.9f}'.format(loss_SDE_D.item()))
                    print('      | StoV {:.9f}'.format(loss_SDE_V.item()))
                else:'''
                loss_SDE = loss_SDE_criterion(Sigma[:, 0], GT_Sigma) * args.SDE_weight 
                epoch_loss_Sigma += loss_SDE.item()
                print('      |  Sto {:.9f}'.format(loss_SDE.item()))
                if args.SDE_loss:
                    if args.no_concentration_loss:
                        loss_SDE.backward()
                    else:
                        loss_SDE.backward(retain_graph = True)

            if not args.no_concentration_loss: 
                movie4loss = Variable(batch['movie4loss'].float().to(device), requires_grad = True)

                if args.model_type == 'vae':
                    mu_lst, logvar_lst = PIANO.get_vars() 

                pred_full_movie = torch.stack([movie4loss[:, 0]] * movie4loss.size(1), dim = 1) # (n_batch, loss_n_collocations, s, r, c)
                for i_coll in range(1, movie4loss.size(1)):  # Loop for n_collocations #
                    #print(i_coll+1, '/', movie4loss.size(1).item())
                    pred_full_movie[:, i_coll] = pred_full_movie[:, i_coll - 1]
                    #print(pred_full_movie.size(), sub_collocation_t.size())
                    # (collocation_nt, n_batch, s, r, c) -> (collocation_it = -1, n_batch, s, r, c)
                    for i_sub_coll in range(args.collocation_len): # i_coll + i_sub_coll * sub_coll_nt: (n_batch, s, r, c)
                        #print(i_sub_coll+1, '/', sub_collocation_t.size().item()) 
                        pred_full_movie[:, i_coll] = odeint(PIANO, pred_full_movie[:, i_coll], sub_collocation_t, method = args.integ_method, options = args)[-1] 
                    if 'dirichlet' in args.BC or 'cauchy' in args.BC: # BC list: [[BC0_0, BC0, 1], [BC1_0, BC1_1]], [BC2_0, BC2_1]]: each: ((n_batch), nT, BC_size, rest_dim_remain) 
                        pred_full_movie[:, i_coll] = apply_BC_3D(pred_full_movie[:, i_coll], contours, i_coll, args.BC, batched = True) # (n_batch, s, r, c)
                    else:
                        pred_full_movie[:, i_coll] = pred_full_movie[:, i_coll]
                        
                #pred_full_movie = movie4loss.clone()
                #pred_full_movie = odeint(PIANO, pred_full_movie[:, 0], sub_collocation_t, method = args.integ_method, options = args).permute(1, 0, 2, 3, 4)
                if args.spatial_gradient_loss:
                    loss_spatial_grad = loss_spatial_grad_criterion(pred_full_movie, movie4loss, batched = True)
                    loss_spatial_grad.backward(retain_graph = True)
                    epoch_loss_sgl += loss_spatial_grad.item()
                    print('      | SpGd {:.9f}'.format(loss_spatial_grad.item()))

                loss_perf = perf_loss_function(pred_full_movie, movie4loss, mu_lst, logvar_lst) #* 1e+7 # TODO (batch, collocation_nt, s, r, c)
                
                loss_perf.backward()
                epoch_loss_perf += loss_perf.item()
                print('      | Perf {:.9f}'.format(loss_perf.item()))

            optimizer_PIANO.step()
        
        #print('Reset scheduler')
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        #if not args.no_concentration_loss:
            #writer.add_scalar('training_loss', epoch_loss_perf / len(train_loader), epoch)
        #if args.gradient_loss:
            #writer.add_scalar('gradient_loss', epoch_loss_grad / len(train_loader), epoch) 
        #if args.spatial_gradient_loss:
            #writer.add_scalar('spatial_gradient_loss', epoch_loss_sgl / len(train_loader), epoch)

        if epoch % args.print_freq == 0:
            print('\nEpoch #{:d}'.format(epoch))
            file.write('\nEpoch #{:d}'.format(epoch))
            if not args.no_concentration_loss:
                file.write('\n      | Perf {:.9f}'.format(epoch_loss_perf / len(train_loader)))
                print('      | Perf {:.9f}'.format(epoch_loss_perf / len(train_loader)))
            if args.GT_D and 'diff' in args.perf_pattern and have_GT:
                file.write('\n      |    D {:.9f}'.format(epoch_loss_D / len(train_loader)))
                print('      |    D {:.9f}'.format(epoch_loss_D / len(train_loader))) 
            #if args.GT_D and 'diff' in args.perf_pattern and have_GT and args.jbld_loss: ## TODO: TESTING -- PRINTING ##
                file.write('\n      | JBLD {:.9f}'.format(epoch_loss_jbld / len(train_loader)))
                print('      | JBLD {:.9f}'.format(epoch_loss_jbld / len(train_loader))) 
            if args.GT_LU and 'diff' in args.perf_pattern and have_GT:
                file.write('\n      |    L {:.9f}'.format(epoch_loss_L / len(train_loader)))
                print('      |    L {:.9f}'.format(epoch_loss_L / len(train_loader)))
                file.write('\n      |    U {:.9f}'.format(epoch_loss_U / len(train_loader)))
                print('      |    U {:.9f}'.format(epoch_loss_U / len(train_loader)))
            if args.GT_V and 'adv' in args.perf_pattern and have_GT:
                file.write('\n      |    V {:.9f}'.format(epoch_loss_V / len(train_loader)))
                print('      |    V {:.9f}'.format(epoch_loss_V / len(train_loader)))
            if args.GT_Phi and 'adv' in args.perf_pattern and have_GT:
                file.write('\n      |  Phi {:.9f}'.format(epoch_loss_Phi / len(train_loader)))
                print('      |  Phi {:.9f}'.format(epoch_loss_Phi / len(train_loader)))
            if args.stochastic and args.img_type == 'IXI' and have_GT: # TODO: TESTING shared uncertainty # 
                '''if args.separate_DV_value_mask:
                    file.write('\n      | StoD {:.9f}'.format(epoch_loss_Sigma_D / len(train_loader))) 
                    file.write('\n      | StoV {:.9f}'.format(epoch_loss_Sigma_V / len(train_loader))) 
                    print('      |  StoD {:.9f}'.format(epoch_loss_Sigma_D / len(train_loader))) 
                    print('      |  StoV {:.9f}'.format(epoch_loss_Sigma_V / len(train_loader))) 
                else:'''
                file.write('\n      |  Sto {:.9f}'.format(epoch_loss_Sigma / len(train_loader))) 
                print('      |  Sto {:.9f}'.format(epoch_loss_Sigma / len(train_loader))) 
            if args.predict_value_mask and have_GT:
                #if args.separate_DV_value_mask:
                file.write('\n      | VM_D {:.9f}'.format(epoch_loss_VM_D))
                file.write('\n      | VM_V {:.9f}'.format(epoch_loss_VM_V))
                print('      | VM_D {:.9f}'.format(epoch_loss_VM_D))
                print('      | VM_V {:.9f}'.format(epoch_loss_VM_V))
                '''else:
                    file.write('\n      |   VM {:.9f}'.format(epoch_loss_VM_D))
                    print('      |   VM {:.9f}'.format(epoch_loss_VM_D))  '''
            if args.predict_segment and have_GT:
                file.write('\n      |  Seg {:.9f}'.format(epoch_loss_seg))
                print('      |  Seg {:.9f}'.format(epoch_loss_seg))  
            if args.gradient_loss:
                file.write('\n      | Grad {:.9f}'.format(epoch_loss_grad / len(train_loader)))
                print('      | Grad {:.9f}'.format(epoch_loss_grad / len(train_loader))) 
            if args.spatial_gradient_loss:
                file.write('\n      | SpGd {:.9f}'.format(epoch_loss_sgl / len(train_loader)))
                print('      | SpGd {:.9f}'.format(epoch_loss_sgl / len(train_loader)))
            if 'gauge' in args.PD_V_type and 'adv' in args.perf_pattern:
                file.write('\n      | Gaug {:.9f}'.format(epoch_loss_gauge / len(train_loader)))
                print('      | Gaug {:.9f}'.format(epoch_loss_gauge / len(train_loader)))
            
            grad_PIANO = avg_grad(PIANO.parameters())
            file.write('\n      |  AvG {:.6f}'.format(grad_PIANO)) 
            print('      |  AvG {:.6f}\n'.format(grad_PIANO))
        gc.collect()
        file.close()
    
    end = time.time()
    return

##############################################################################################################################

if __name__ == '__main__':
    raise NotImplementedError