import os, sys, argparse, time
from shutil import copyfile, move
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import dipy
import numpy as np
import scipy.ndimage as ndimage

import torch

import matplotlib
import matplotlib.pyplot as plt

import itk
import SimpleITK as sitk
from dipy.reconst import dti
#from itkwidgets import view
from itk import TubeTK as ttk

from dipy.segment.mask import median_otsu
from scipy.ndimage.filters import gaussian_filter

from utils import make_dir, nda2img
from Preprocess.IXI.itk_utils import *
from Preprocess.IXI.fit_divfree import fit
from Preprocess.prepro_utils import cropping, rm_by_volume, get_mip
from Preprocess.IXI.v_generator import frangi_velocity, VelocityGenerator, nii2mha


#%% Basic settings
parser = argparse.ArgumentParser('IXI Preprocessing')
parser.add_argument('--new_spacing', type = list, default = [1., 1., 1.])
parser.add_argument('--dti_fit_choice', type = str, default = 'OLS', help = {'LS', 'WLS', 'OLS', 'NLLS', 'RESTORE'})
args_prep = parser.parse_args()  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# General setting for DTI fitting #
fwhm = 1.25
gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std


############################################################################################
##########################################  Utils ##########################################
############################################################################################

def resampling(itk_img, new_spacing, save_path):
    return resample(itk_img, new_spacing = new_spacing, save_path = save_path)

def registering(itk_img_paths, fixed_itk_img, save_paths):
    for i in range(len(itk_img_paths)):
        register_images(read_img(itk_img_paths[i]), fixed_itk_img, save_path = save_paths[i])
    return save_paths

def stripping_skull(itk_img1, itk_img2, itk_img3, save_path):
    return extract_brain(itk_img1, itk_img2, itk_img3, save_path = save_path)

def masking_brain(itk_masker, itk_img_paths, mask_itk_img, save_paths):
    for i in range(len(itk_img_paths)):
        masking(itk_masker, read_img(itk_img_paths[i]), mask_itk_img, save_path = save_paths[i])
    return

def cropping_all(baseline_path, to_crop_paths, mask_save_path, cropped_save_paths):
    _, crop_ranges = cropping(sitk.ReadImage(baseline_path), save_path = mask_save_path)
    for i in range(len(cropped_save_paths)):
        cropping(sitk.ReadImage(cropped_save_paths[i]), crop_range_lst = crop_ranges, save_path = cropped_save_paths[i])
    return

def enhancing_vessel(mra_path, itk_img_path1, itk_img_path2, mask_path, vessel_save_path):
    enhance_vessel(mra_path, itk_img_path1, itk_img_path2, mask_path, vessel_save_path)

def segmenting_vessel(vessel_prob_path, vessel_save_path, tol = -0.00002, save_MIP = True):
    vessel_prob_img = sitk.ReadImage(vessel_prob_path)
    origin, spacing, direction = vessel_prob_img.GetOrigin(), vessel_prob_img.GetSpacing(), vessel_prob_img.GetDirection()

    vessel_prob_nda = sitk.GetArrayFromImage(vessel_prob_img).astype(float)
    vessel_prob_nda[vessel_prob_nda <= tol] = 0
    vessel_prob_nda[vessel_prob_nda > 0.] = 1
    vessel_nda = rm_by_volume(vessel_prob_nda, noise_size_tol = 50) # remove noises by connected region volume size #

    nda2img(vessel_nda, origin, spacing, direction, save_path = vessel_save_path)
    if save_MIP:
        nda2img(np.max(abs(vessel_nda), axis = 0)[::-1], save_path = '%s_MIP%s' % (vessel_save_path[:-4], vessel_save_path[-4:]))
    return

def computing_abs_v(binary_vessel_path, velocity_intensity_path, velocity_save_fld):
    '''
    Only get velocity principle direction based on Frangi Hessian SVD 
    (NOTE: More processing steps in ./Preprocess/MRA/v_generator)
    '''
    principle_direction_nda = frangi_velocity(binary_vessel_path, velocity_magnitude = 1., save_fld = velocity_save_fld)
    return

def stacking_dwi(dwi_img_paths, save_path):
    dwi_base_img = sitk.ReadImage(dwi_img_paths[0])
    os.remove(dwi_img_paths[0])
    dwi_base_nda = sitk.GetArrayFromImage(dwi_base_img).astype(float) # (s, r, c)
    origin, spacing, direction = dwi_base_img.GetOrigin(), dwi_base_img.GetSpacing(), dwi_base_img.GetDirection()

    dwi_stack = np.transpose(np.array([dwi_base_nda] * len(dwi_img_paths)), (1, 2, 3, 0)) # (n_DWI, s, r, c) --> (s, r, c, n_DWI)
    for i in range(1, len(dwi_img_paths)):
        img = sitk.ReadImage(dwi_img_paths[i])
        dwi_stack[..., i] = sitk.GetArrayFromImage(sitk.ReadImage(dwi_img_paths[i])).astype(float)
        # NOTE: Delete previous-satage images
        os.remove(dwi_img_paths[i])
    dwi_stack_img = nda2img(dwi_stack, origin, spacing, direction, save_path = save_path)
    return save_path 

def dwi2dti(dti_fit_model, dwi_path, brain_mask_path, save_fld):

    dwi_img = sitk.ReadImage(dwi_path)
    origin, spacing, direction = dwi_img.GetOrigin(), dwi_img.GetSpacing(), dwi_img.GetDirection()
    dwi_nda = sitk.GetArrayFromImage(dwi_img).astype(float)
    for v in range(dwi_nda.shape[-1]):
        dwi_nda[..., v] = gaussian_filter(dwi_nda[..., v], sigma = gauss_std)
    brain_mask_nda = sitk.GetArrayFromImage(sitk.ReadImage(brain_mask_path)).astype(float)

    dti_fit = dti_fit_model.fit(dwi_nda, mask = brain_mask_nda)
    D = dti_fit.quadratic_form * brain_mask_nda[..., None, None] # (shape, 3, 3)
    CO = abs(dti_fit.directions[:, :, :, 0]) * brain_mask_nda[..., None] # (shape, 3)
    FA = dti_fit.fa * brain_mask_nda
    Trace = dti_fit.trace * brain_mask_nda
    evals = dti_fit.model_params[..., :3] * brain_mask_nda[..., None]
    evecs = dti_fit.model_params[..., 3:] * brain_mask_nda[..., None]

    nda2img(D.reshape(D.shape[0], D.shape[1], D.shape[2], 9), origin, spacing, direction, \
        save_path = os.path.join(os.path.dirname(dwi_path), 'DTI.mha'))
    nda2img(CO, origin, spacing, direction, os.path.join(save_fld, 'ColorOrientation.mha'))
    nda2img(FA, origin, spacing, direction, os.path.join(save_fld, 'FA.mha'))
    nda2img(Trace, origin, spacing, direction, os.path.join(save_fld, 'Trace.mha'))
    nda2img(evals, origin, spacing, direction, os.path.join(save_fld, 'L.mha'))
    nda2img(evecs, origin, spacing, direction, os.path.join(save_fld, 'U.mha'))
    return 


############################################################################################
############################################################################################
############################################################################################


class IXI_Processor(object):
    '''
    Refs:
    TubeTK Doc: https://public.kitware.com/Wiki/TubeTK/Documentation
    Veesel extraction: https://github.com/InsightSoftwareConsortium/ITKTubeTK/tree/master/examples/MRA-Head
    Velociity estimation: https://link.springer.com/chapter/10.1007/BFb0056195
        Codes modified from: https://github.com/scikit-image/scikit-image/blob/bd1dc065025775fc591fb93c7bd20d59e2497e9a/skimage/feature/corner.py#L246
    '''
    def __init__(self, args, data_fld, save_fld, case_name, DTI_Fitter, n_DWI = 16):
        
        self.args = args
        self.n_DWI = n_DWI
        self.case_name = case_name 
        self.DTI_Fitter = DTI_Fitter
        self.new_spacing = self.args.new_spacing

        self.register_paths(data_fld, save_fld)
        
    def register_paths(self, data_fld, save_fld):
        self.DWI_paths, self.DWI_save_paths = [], []
        self.save_main_fld = make_dir(os.path.join(save_fld, case_name))
        
        self.T1_path = os.path.join(data_fld, 'IXI-T1/%s-T1.nii.gz' % self.case_name)
        self.T2_path = os.path.join(data_fld, 'IXI-T2/%s-T2.nii.gz' % self.case_name)
        self.MRA_path = os.path.join(data_fld, 'IXI-MRA/%s-MRA.nii.gz' % self.case_name)
        for i in range(self.n_DWI):
            self.DWI_save_paths.append(os.path.join(self.save_main_fld, 'DWI-{:02d}.mha'.format(i)))
            self.DWI_paths.append(os.path.join(data_fld, 'IXI-DTI/{:s}-DTI-{:02d}.nii.gz'.format(self.case_name, i)))

        self.T1_save_path = os.path.join(self.save_main_fld, 'T1.mha')
        self.T2_save_path = os.path.join(self.save_main_fld, 'T2.mha')
        self.MRA_save_path = os.path.join(self.save_main_fld, 'MRA.mha')
        self.DTI_save_path = os.path.join(self.save_main_fld, 'DTI.mha')
        self.DWI_save_path = os.path.join(self.save_main_fld, 'DWI.mha')
        self.Vessel_save_path = os.path.join(self.save_main_fld, 'Vessel.mha')
        self.Mask_save_path = os.path.join(self.save_main_fld, 'BrainMask.mha')

        self.Adv_fld = make_dir(os.path.join(self.save_main_fld, 'AdvectionMaps'))
        self.Diff_fld = make_dir(os.path.join(self.save_main_fld, 'DiffusionMaps'))
        return 


    def preprocess_all(self):
        '''
        Processing steps:
        (1) Resample baseline image (MRA) to be isotropic (1 mm)
        (2) Register all images (DWI, T1, T2) to isotropic MRA
        (3) Skull stripping: compute brain mask based on MRA & T1 & T2
        (4) Apply brain mask for all images
        (5) Crop all images based on cropped brain mask
        (6) Extract vessels
        (7) Estimate velocity (NOTE: Detailed processing steps in ./Preprocess/MRA/v_generator)
        (8) Fit divergence-free V
        (9) Stack DWI, fit DTI, compute D-related scalar maps
        '''
        ################### (1) Resample MRA ###################
        print('   Resample MRA to isotropic')
        MRA_iso = resampling(read_img(self.MRA_path), self.new_spacing, self.MRA_save_path)

        ################### (2) Register all rest images to MRA ###################
        print('   Register images to isotropic MRA')
        registering(itk_img_paths = [self.T1_path, self.T2_path] + self.DWI_paths, fixed_itk_img = MRA_iso, \
            save_paths = [self.T1_save_path, self.T2_save_path] + self.DWI_save_paths)

        ################### (3) Skull-stripping ###################
        print('   Skull strip, compute brain mask')
        BrainMasker, BrainMask = stripping_skull(itk_img1 = read_img(self.MRA_save_path), itk_img2 = read_img(self.T1_save_path), \
            itk_img3 = read_img(self.T2_save_path), save_path = self.Mask_save_path)

        ################### (4) Mask brain region ###################
        print('   Mask brain region for all images')
        masking_brain(itk_masker = BrainMasker, itk_img_paths = [self.T1_save_path, self.T2_save_path, self.MRA_save_path] + self.DWI_save_paths, \
            mask_itk_img = BrainMask, save_paths = [self.T1_save_path, self.T2_save_path, self.MRA_save_path] + self.DWI_save_paths)
        del BrainMasker, BrainMask

        ################### (5) Crop brain region ###################
        print('   Crop brain region for all images')
        cropping_all(baseline_path = self.Mask_save_path, to_crop_paths = [self.T1_save_path, self.T2_save_path, self.MRA_save_path] + self.DWI_save_paths, \
            mask_save_path = self.Mask_save_path, cropped_save_paths = [self.T1_save_path, self.T2_save_path, self.MRA_save_path] + self.DWI_save_paths)

        ################### (6) Extract vessels ###################
        print('   Enhance vessels')
        vess_prob_path = os.path.join(self.save_main_fld, "MRA_VesselEnhances.mha")
        enhancing_vessel(self.MRA_save_path, self.T1_save_path, self.T2_save_path, self.Mask_save_path, vessel_save_path = vess_prob_path)

        print('   Segment vessels')
        segmenting_vessel(vess_prob_path, vessel_save_path = self.Vessel_save_path)

        ################### (7) Estimate velocities ###################
        print('   Estimate velocities')
        '''NOTE Only get absolute principle vessel direction'''
        #compute_abs_v(self.Vessel_save_path, self.MRA_save_path, velocity_save_fld = self.Adv_fld)
        '''NOTE Full version of V generation'''
        generator = VelocityGenerator(self.Vessel_save_path, self.MRA_save_path, self.Adv_fld)
        v_path = generator.get_velocity()

        

        ################### (8) Fit divergence-free V ###################
        print('   Fit divergence-free velocities')
        fit(v_path, device, n_iter = 8000, lr = 1e-3, save_fld = make_dir(os.path.join(os.path.dirname(v_path), 'DivFree')))

        ################### (9) Fit DTI ###################
        print('   Fit DTI')
        stacking_dwi(self.DWI_save_paths, save_path = self.DWI_save_path)
        dwi2dti(self.DTI_Fitter, self.DWI_save_path, self.Mask_save_path, self.Diff_fld)
        return



if __name__ == '__main__':

    data_fld = '/media/peirong/PR/IXI'
    save_fld = make_dir('/media/peirong/PR5/IXI_Processed')

    names_file_path = os.path.join(data_fld, 'IDs.txt')
    t1_fld  = os.path.join(data_fld, 'IXI-T1')
    t2_fld  = os.path.join(data_fld, 'IXI-T2')
    dwi_fld = os.path.join(data_fld, 'IXI-DTI')
    mra_fld = os.path.join(data_fld, 'IXI-MRA')

    #########################################################################
    ##################### Setting up DTI Fitting Model ######################
    #########################################################################

    bvals_path = os.path.join(data_fld, 'bvals.txt')
    bvecs_path = os.path.join(data_fld, 'bvecs.txt')
    bvals_file = open(bvals_path, "r")
    bvecs_file = open(bvecs_path, "r")

    bvals_list = bvals_file.read().split(' ')
    bvals_list = bvals_list
    bvals = np.array(bvals_list).astype(float)
    bvals_file.close()
    print('bvals:', bvals) # (n_gradients = 16, )
    bvecs_lines = bvecs_file.readlines()
    bvecs_x = bvecs_lines[0].split(' ')
    bvecs_y = bvecs_lines[1].split(' ')
    bvecs_z = bvecs_lines[2].split(' ')
    bvecs = np.stack([bvecs_x, bvecs_y, bvecs_z], axis = -1).astype(float)
    bvecs_file.close()
    print('bves:', bvecs) # (n_gradients = 16, 3)

    gtab = dipy.core.gradients.gradient_table(bvals, bvecs)
    DTI_Fitter = dti.TensorModel(gtab, fit_method = args_prep.dti_fit_choice)

    #########################################################################
    #########################################################################

    names_file = open(names_file_path, 'r')
    case_names = names_file.readlines()
    names_file.close()
    print('Total number of cases (have both MRA & DWI):', len(case_names)) 
    #case_names = ['IXI002-Guys-0828'] # TODO

    for i in range(len(case_names)): # TODO
    #for i in range(132, len(case_names)): # TODO
        #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')
        #time.sleep(8)
        case_name = case_names[i].split('\n')[0]
        #print('\nStart processing case %d/%d: %s' % (i+1, len(case_names), case_name))
        processor = IXI_Processor(args_prep, data_fld, save_fld, case_name, DTI_Fitter, n_DWI = bvals.shape[0])
        #processor.preprocess_all()
        if not os.path.isfile(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/ColorOrientation.mha')):
            if  os.path.isfile(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/L.nii')):
                print('\nRenaming case %d/%d: %s' % (i+1, len(case_names), case_name))
                nii2mha(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/L.nii'))
            if  os.path.isfile(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/U.nii')):
                nii2mha(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/U.nii'))
            if  os.path.isfile(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/Trace.nii')):
                nii2mha(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/Trace.nii'))
            if  os.path.isfile(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/ColorOrientation.nii')):
                nii2mha(os.path.join(os.path.dirname(processor.DWI_save_path), 'DiffusionMaps/ColorOrientation.nii'))
            else:
                print('\nStart processing case %d/%d: %s' % (i+1, len(case_names), case_name))
                dwi2dti(processor.DTI_Fitter, processor.DWI_save_path, processor.Mask_save_path, processor.Diff_fld)
        else:
            continue


    # Save cace names
    '''names_file = open(names_file_path, 'w')

    case_names = []
    mra_files = os.listdir(mra_fld)
    for i in range(len(mra_files)):
        case_name = mra_files[i][:-11]
        if os.path.isfile(os.path.join(dwi_fld, '%s-DTI-00.nii.gz' % case_name)):
            if 'IXI197-Guys-0811' in case_name or 'IXI200-Guys-0812' in case_name:
                continue # Bad case: failed vessel segmentation 
            case_names.append(case_name)
    print('Total number of cases (have both MRA & DWI):', len(case_names)) 

    for i in range(len(case_names) - 1):
        names_file.write('%s\n' % case_names[i])
    names_file.write('%s' % case_names[-1])
    names_file.close()'''