import os, sys, argparse, time
from shutil import copyfile, move
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import binary_fill_holes

import torch

import matplotlib
import matplotlib.pyplot as plt

import itk
import SimpleITK as sitk
#from itkwidgets import view
from itk import TubeTK as ttk

from Preprocess.IXI.itk_utils import *
from Preprocess.contour import get_contour
from Preprocess.prepro_utils import cropping
from Preprocess.print_info import write_info
from Preprocess.IXI.IXI_process import segmenting_vessel
from utils import make_dir, nda2img, get_times, img2nda, cutoff_percentile


#%% Basic settings
parser = argparse.ArgumentParser('ISLES2017 MRP Preprocessing')
parser.add_argument('--new_spacing', type = list, default = [1., 1., 1.])
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

def masking_brain(itk_masker, itk_img_paths, mask_itk_img, save_paths):
    for i in range(len(itk_img_paths)):
        masking(itk_masker, read_img(itk_img_paths[i]), mask_itk_img, save_path = save_paths[i])
    return

def rotating(img_path, angle = 10, isBinary = False, save_path = None):
    # TODO: determine option 'order' #
    nda, origin, spacing, direction = img2nda(img_path)
    nda = ndimage.rotate(nda, angle = angle, axes = (2, 1), reshape = False, order = 1) # rotate on row-column plane #
    if isBinary:
        nda = np.rint(nda)
    if not save_path:
        save_path = '%s_rotated(%s).mha' % (img_path[:-4], angle)
    nda2img(nda, origin, spacing, direction, save_path)
    return save_path

def get_mirrored_vessel(vessel_path):
    vessel_nda, origin, spacing, direction = img2nda(vessel_path)
    if np.mean(vessel_nda[:, :, : int(vessel_nda.shape[2] / 2)]) > np.mean(vessel_nda[:, :, int(vessel_nda.shape[2] / 2) + 1 : ]): # Left/right hemisphere comparison
        # flip_type = 'L2R'
        vessel_nda[:, :, int(vessel_nda.shape[2] / 2) + 1 : ] = np.flip(vessel_nda, axis = 2)[:, :, int(vessel_nda.shape[2] / 2) + 1 : ]
    else:
        # flip_type = 'R2L'
        vessel_nda[:, :, : int(vessel_nda.shape[2] / 2)] = np.flip(vessel_nda, axis = 2)[:, :, : int(vessel_nda.shape[2] / 2)]
    nda2img(vessel_nda, origin, spacing, direction, '%s_mirrored.mha' % vessel_path[:-4])
    return vessel_nda, '%s_mirrored.mha' % vessel_path[:-4]


def stacking_pwi(pwi_img_paths, save_path):
    pwi_base_img = sitk.ReadImage(pwi_img_paths[0])
    # NOTE: Delete previous-satage images
    os.remove(pwi_img_paths[0])
    pwi_base_nda = sitk.GetArrayFromImage(pwi_base_img).astype(float) # (s, r, c)
    origin, spacing, direction = pwi_base_img.GetOrigin(), pwi_base_img.GetSpacing(), pwi_base_img.GetDirection()

    pwi_stack = np.transpose(np.array([pwi_base_nda] * len(pwi_img_paths)), (1, 2, 3, 0)) # (n_DWI, s, r, c) --> (s, r, c, n_DWI)
    for i in range(1, len(pwi_img_paths)):
        img = sitk.ReadImage(pwi_img_paths[i])
        pwi_stack[..., i] = sitk.GetArrayFromImage(sitk.ReadImage(pwi_img_paths[i])).astype(float)
        # NOTE: Delete previous-satage images
        os.remove(pwi_img_paths[i])
    pwi_stack_img = nda2img(pwi_stack, origin, spacing, direction, save_path = save_path)
    return save_path

def convert_binary(img_path):
    nda, origin, spacing, direction = img2nda(img_path)
    nda[nda != 0] = 1.
    nda2img(nda, origin, spacing, direction, save_path = img_path)
    return

def mrp_s0(signal_nda, s0_threshold = 0.05):
    '''
    Calculate the MRP bolus arrival time (bat) and corresponding S0: averaged over signals before bat
    return: s0 # (n_slice, n_row, n_column)
    '''
    nT = signal_nda.shape[-1]
    sig_avg = np.zeros(nT)
    for t in range(nT):
        sig_avg[t] = (signal_nda[..., t]).mean()
    ttp = np.argmin(sig_avg)
    flag = True
    bat  = 0
    while flag:
        s0_avg = np.mean(sig_avg[:bat + 1])
        if abs(s0_avg - sig_avg[bat + 1]) / s0_avg < s0_threshold:
            bat += 1
        else:
            flag = False
            bat -= 1
        if bat == signal_nda.shape[-1] - 1:
            flag = False
            bat -= 1
    if bat > ttp:
        bat = ttp
    print('   - Bolus arrival time (start from 0):', bat)
    print('   - Time to peak (start from 0):', ttp)
    s0 = np.mean(signal_nda[..., :bat], axis = 3) # time dimension == 3
    return s0, bat, ttp

def ctc2signal(ctc_nda, k = 1., TE = 0.025):
    mask = ctc_nda[..., 0]
    mask[mask > 0] = 1.
    mrp = np.exp(ctc_nda / (-k/TE)) * -1000
    return ctc_nda * mask


def signal2ctc(signal_nda, mask_nda, s0 = 1., k = 1., TE = 0.025, normalize_ctc = True):
    s0, bat, ttp = mrp_s0(signal_nda, s0_threshold = 0.05)
    signal_nda = signal_nda[..., bat:]
    ctc_nda = - k/TE * np.log(signal_nda / (s0[..., None] + 1e-14) + 1e-14)
    print(ctc_nda.shape)
    if normalize_ctc: 
        ctc_nda /= np.max(abs(ctc_nda))
    #print(np.min(ctc_nda))
    #print(np.where(ctc_nda < 0.01 * np.min(ctc_nda))[0].shape)
    ctc_nda[ctc_nda < 0.01 * np.min(ctc_nda)] = 0. # Exclude log outliers
    #print(np.max(ctc_nda))
    #print(np.where(ctc_nda > 0.05 * np.max(ctc_nda))[0].shape)
    ctc_nda[ctc_nda > 0.05 * np.max(ctc_nda)] = 0. # Exclude log outliers
    if normalize_ctc: 
        ctc_nda /= np.max(abs(ctc_nda)) # Normalize again after cropping #
    return ctc_nda * mask_nda[..., None] 

def stripping_skull(itk_img1, itk_img2, itk_img3, save_path):
    return extract_brain(itk_img1, itk_img2, itk_img3, save_path = save_path)

def masking_brain(itk_masker, itk_img_paths, mask_itk_img, save_paths):
    for i in range(len(itk_img_paths)):
        masking(itk_masker, read_img(itk_img_paths[i]), mask_itk_img, save_path = save_paths[i])
    return

def cropping_all(baseline_path, to_crop_paths, mask_save_path, cropped_save_paths):
    _, [[x0, y0, z0], [x1, y1, z1]] = cropping(sitk.ReadImage(baseline_path), save_path = mask_save_path)
    for i in range(len(cropped_save_paths)): # Drop 1 boundary #
        cropping(sitk.ReadImage(cropped_save_paths[i]), crop_range_lst = [[x0+1, y0+1, z0+1], [x1-1, y1-1, z1-1]], save_path = cropped_save_paths[i])
    cropping(sitk.ReadImage(mask_save_path), crop_range_lst = [[1, 1, 1], [-1, -1, -1]], save_path = mask_save_path) # Drop 1 boundary #
    return

def get_brain_mask(img_path, save_path):
    nda, origin, spacing, direction = img2nda(img_path)
    mask = np.zeros(nda.shape)
    mask[nda > 160] = 1.
    for s in range(len(mask)):
        mask[s] = ndimage.binary_fill_holes(mask[s])
    for r in range(len(mask[0])):
        mask[:, r] = ndimage.binary_fill_holes(mask[:, r])
    for c in range(len(mask[0, 0])):
        mask[:, :, c] = ndimage.binary_fill_holes(mask[:, :, c])
    for s in range(len(mask)):
        mask[s] = ndimage.binary_fill_holes(mask[s])
    for r in range(len(mask[0])):
        mask[:, r] = ndimage.binary_fill_holes(mask[:, r])
    nda2img(mask.astype(float), origin, spacing, direction, save_path)
    return 


############################################################################################
############################################################################################
############################################################################################


class ISLES2017_Processor(object):

    def __init__(self, args, data_fld, save_fld, case_name, for_test = False):
        
        self.args = args
        self.for_test = for_test
        self.case_name = case_name 
        self.new_spacing = self.args.new_spacing

        self.register_paths(data_fld, save_fld)
        if self.for_test:
            self.register_testing_PWI(self.PWI_path)
        else:
            self.register_training_PWI(self.PWI_path)
        get_brain_mask(self.PWI_paths[0], os.path.join(self.data_main_fld, 'BrainMask.mha')) 
    
    def register_training_PWI(self, PWI_path): # Cropped PWI: (s, r, c, T) #
        PWI_nda, _, _, _ = img2nda(PWI_path)
        _, origin, spacing, direction = img2nda(self.ADC_path)
        def save_nda2img(nda, save_path):
            nda2img(nda, origin, spacing, direction, save_path)
            return save_path
        nT = PWI_nda.shape[-1]
        self.PWI_paths = [None] * nT
        self.PWI_save_paths = [None] * nT
        for it in range(nT):
            self.PWI_paths[it] = save_nda2img(np.clip(PWI_nda[..., it], a_min = 0., a_max = None), '{:s}-{:02d}.mha'.format(self.PWI_path[:-4], it))
            self.PWI_save_paths[it] = '{:s}-{:02d}.mha'.format(self.PWI_save_path[:-4], it)
        return

    def register_testing_PWI(self, PWI_path): # Original PWI: (T, s, r, c) #
        PWI_nda, _, _, _ = img2nda(PWI_path)
        _, origin, spacing, direction = img2nda(self.ADC_path)
        def save_nda2img(nda, save_path):
            nda2img(nda, origin, spacing, direction, save_path)
            return save_path
        nT = PWI_nda.shape[0]
        self.PWI_paths = [None] * nT
        self.PWI_save_paths = [None] * nT
        for it in range(nT):
            self.PWI_paths[it] = save_nda2img(np.clip(PWI_nda[it], a_min = 0., a_max = None), '{:s}-{:02d}.mha'.format(self.PWI_path[:-4], it))
            self.PWI_save_paths[it] = '{:s}-{:02d}.mha'.format(self.PWI_save_path[:-4], it)
        return

    def register_paths(self, data_fld, save_fld):
        self.data_main_fld = os.path.join(data_fld, case_name)
        self.save_main_fld = make_dir(os.path.join(save_fld, case_name))

        fileHandle = open(os.path.join(self.data_main_fld, 'prepro_info.txt'),"r")
        lineList = fileHandle.readlines()
        fileHandle.close()
        if len(lineList) > 5:
            self.rotate_degree = float(lineList[-1].split(':')[-1][1:])
        else:
            self.rotate_degree = None

        self.mask_save_path = os.path.join(self.save_main_fld, 'BrainMask.mha')
        self.vessel_save_path = os.path.join(self.save_main_fld, 'Vessel.mha')
        self.vessel_enhanced_save_path = os.path.join(self.save_main_fld, 'VesselEnhanced.mha')

        def condition(file_name):
            if self.for_test:
                return 'rotated' not in file_name and 'cropped' not in files and 'Linear' not in file_name and file_name.endswith('.nii')
            else:
                return 'rotated' not in file_name and 'Linear' not in file_name and file_name.endswith('cropped.nii')
        
        for module in os.listdir(self.data_main_fld):
            if "PWI" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                PWI_fld = os.path.join(self.data_main_fld, module)
                self.PWI_save_fld = make_dir(os.path.join(self.save_main_fld, 'MRP'))
                for files in os.listdir(PWI_fld):
                    if '4DPWI' in files and condition(files):
                        self.PWI_path = os.path.join(PWI_fld, files)
                        self.PWI_save_path = os.path.join(self.PWI_save_fld, 'MRP.mha')
                        self.CTC_save_path = os.path.join(self.PWI_save_fld, 'CTC.mha')
            if "ADC" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                ADC_fld = os.path.join(self.data_main_fld, module)
                self.ADC_save_fld = make_dir(os.path.join(self.save_main_fld, 'ADC'))
                for files in os.listdir(ADC_fld):
                    if 'ADC' in files and condition(files):
                        self.ADC_path = os.path.join(ADC_fld, files)
                        self.ADC_save_path = os.path.join(self.ADC_save_fld, 'ADC.mha')
            if "MTT" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                MTT_fld = os.path.join(self.data_main_fld, module)
                self.MTT_save_fld = make_dir(os.path.join(self.save_main_fld, 'MTT'))
                for files in os.listdir(MTT_fld):
                    if 'MTT' in files and condition(files):
                        self.MTT_path = os.path.join(MTT_fld, files)
                        self.MTT_save_path = os.path.join(self.MTT_save_fld, 'MTT.mha')
            if "CBF" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                CBF_fld = os.path.join(self.data_main_fld, module)
                self.CBF_save_fld = make_dir(os.path.join(self.save_main_fld, 'CBF'))
                for files in os.listdir(CBF_fld):
                    if 'CBF' in files and condition(files):
                        self.CBF_path = os.path.join(CBF_fld, files)
                        self.CBF_save_path = os.path.join(self.CBF_save_fld, 'CBF.mha')
            if "CBV" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                CBV_fld = os.path.join(self.data_main_fld, module)
                self.CBV_save_fld = make_dir(os.path.join(self.save_main_fld, 'CBV'))
                for files in os.listdir(CBV_fld):
                    if 'CBV' in files and condition(files):
                        self.CBV_path = os.path.join(CBV_fld, files)
                        self.CBV_save_path = os.path.join(self.CBV_save_fld, 'CBV.mha')
            if "Tmax" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                Tmax_fld = os.path.join(self.data_main_fld, module)
                self.Tmax_save_fld = make_dir(os.path.join(self.save_main_fld, 'Tmax'))
                for files in os.listdir(Tmax_fld):
                    if 'Tmax' in files and condition(files):
                        self.Tmax_path = os.path.join(Tmax_fld, files)
                        self.Tmax_save_path = os.path.join(self.Tmax_save_fld, 'Tmax.mha')
            if "TTP" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                TTP_fld = os.path.join(self.data_main_fld, module)
                self.TTP_save_fld = make_dir(os.path.join(self.save_main_fld, 'TTP'))
                for files in os.listdir(TTP_fld):
                    if 'TTP' in files and condition(files):
                        self.TTP_path = os.path.join(TTP_fld, files)
                        self.TTP_save_path = os.path.join(self.TTP_save_fld, 'TTP.mha')
            if "OT" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                OT_fld = os.path.join(self.data_main_fld, module)
                self.OT_save_fld = make_dir(os.path.join(self.save_main_fld, 'OT'))
                for files in os.listdir(OT_fld):
                    if 'OT' in files and condition(files):
                        self.OT_path = os.path.join(OT_fld, files)
                        self.OT_save_path = os.path.join(self.OT_save_fld, 'OT.mha')
                        self.OT_conotur_save_path = os.path.join(self.OT_save_fld, 'OT_Contour.mha')
        return 


    def preprocess_all(self):
        '''
        Processing steps:
        (1) Resample baseline image (PWI-00) to be isotropic (1 mm)
        (2) Register all images to isotropic PWI-00
        (3) Compute brain mask
        (4) Convert MRP to CTC
        (5) Segment vessel map
        '''
        ################### (1) Resample PWI-00 ###################
        
        #if self.rotate_degree:
        #    print('   Rotate degree %.2f' % self.rotate_degree)
        #    #rotating(self.PWI_paths[0], angle = self.rotate_degree, isBinary = False, save_path = self.PWI_paths[0])
        #else:
        #    print('   Not need to re-rotate, skip')
        #    return

        print('   Resample PWI-00 to isotropic')
        PWI0_iso = resampling(read_img(self.PWI_paths[0]), self.new_spacing, self.PWI_save_paths[0])

        ################### (2) Register images to isotropic PWI-00 ###################
        print('   Register images to isotropic PWI-00')
        registering(itk_img_paths = [self.ADC_path, self.CBF_path, self.CBV_path, self.MTT_path, self.TTP_path, self.Tmax_path] + self.PWI_paths[1:], \
            fixed_itk_img = PWI0_iso, save_paths = [self.ADC_save_path, self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.TTP_save_path, self.Tmax_save_path] + self.PWI_save_paths[1:])
        for it in range(len(self.PWI_paths)):
            os.remove(self.PWI_paths[it])

        ################### (3) Get binary OT ###################
        if not self.for_test:
            print('   Get binary OT')
            if self.rotate_degree:
                rotating(self.OT_path, angle = self.rotate_degree, isBinary = False, save_path = self.OT_path)
            resampling(read_img(self.OT_path), self.new_spacing, self.OT_save_path)
            convert_binary(self.OT_save_path)
            get_contour(self.OT_save_path, save_path = self.OT_save_path)
            contour, origin, spacing, direction = img2nda(self.OT_save_path)

            # Get OT: Keep only the outer boundary #
            OT = np.zeros(contour.shape)
            for s in range(OT.shape[0]): 
                OT[s] = ndimage.binary_fill_holes(contour[s])
            # Fill holes along boundaries #
            for s in range(OT.shape[0]):
                OT[s, :, 0] = ndimage.binary_fill_holes(OT[s, :, 0])
                OT[s, :, -1] = ndimage.binary_fill_holes(OT[s, :, -1])
            # Re-fill the entire region #
            for s in range(OT.shape[0]): 
                OT[s] = ndimage.binary_fill_holes(OT[s])
            
            nda2img(OT, origin, spacing, direction, save_path = self.OT_save_path)

        ################### (4) Get brain mask ###################
        print('   Compute brain mask')
        get_brain_mask(self.PWI_save_paths[0], self.mask_save_path)

        ################### (5) Mask brain region ###################
        print('   Crop for all images')
        if not self.for_test:
            cropping_all(self.mask_save_path, to_crop_paths = [self.ADC_save_path, self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.TTP_save_path, self.Tmax_save_path, self.OT_save_path] + self.PWI_save_paths, \
                mask_save_path = self.mask_save_path, cropped_save_paths = [self.ADC_save_path, self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.TTP_save_path, self.Tmax_save_path, self.OT_save_path] + self.PWI_save_paths)
        else:
            cropping_all(self.mask_save_path, to_crop_paths = [self.ADC_save_path, self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.TTP_save_path, self.Tmax_save_path] + self.PWI_save_paths, \
                mask_save_path = self.mask_save_path, cropped_save_paths = [self.ADC_save_path, self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.TTP_save_path, self.Tmax_save_path] + self.PWI_save_paths)
        
        # Get cropped OT contour#
        OT_contour = get_contour(self.OT_save_path, save_path = self.OT_conotur_save_path)

        ################### (6) Stack PWI ###################
        print('   Stack PWI')
        stacking_pwi(self.PWI_save_paths, self.PWI_save_path)

        ################### (7) Convert MRP to CTC ###################
        print('   Convert PWI to CTC')
        signal_nda, _, _, _ = img2nda(self.PWI_save_path)
        mask_nda, origin, spacing, direction = img2nda(self.mask_save_path)
        ctc_nda = signal2ctc(signal_nda, mask_nda, s0 = 1., k = 1., TE = 0.025)
        nda2img(ctc_nda, origin, spacing, direction, save_path = self.CTC_save_path)
        bat, ttp, ttd = write_info(self.CTC_save_path, 'Info.txt', from_BAT = True)

        # Flip # 
        ctc_flip_nda = ctc_nda[:, :, ::-1, :]
        CTC_flip_path = os.path.join(os.path.dirname(self.CTC_save_path), 'CTC_flip.mha')
        nda2img(ctc_flip_nda, origin, spacing, direction, CTC_flip_path)

        # from TTP #
        ctc_from_ttp = ctc_nda[..., ttp:]
        CTC_fromTTP_path = os.path.join(os.path.dirname(self.CTC_save_path), 'CTC_fromTTP.mha')
        nda2img(ctc_from_ttp, origin, spacing, direction, CTC_fromTTP_path)
        write_info(CTC_fromTTP_path, 'Info_fromTTP.txt', from_TTP = True)

        #ctc_flip_from_ttp = ctc_flip_nda[..., ttp:]
        #CTC_flip_fromTTP_path = os.path.join(os.path.dirname(self.CTC_save_path), 'CTC_flip_fromTTP.mha')
        #nda2img(ctc_flip_from_ttp, origin, spacing, direction, CTC_flip_fromTTP_path)



        ################### (8) Segment vessel map ###################
        print('   Segment vessel map')
        ctc_ttp_path = os.path.join(os.path.dirname(self.CTC_save_path), 'CTC_TTP.mha')
        nda2img(ctc_nda[..., ttp], origin, spacing, direction, save_path = ctc_ttp_path)
        enhance_vessel(self.CTC_save_path, self.CBF_save_path, self.CBV_save_path, self.mask_save_path, save_path = self.vessel_enhanced_save_path)

        # Normalize enhanced vessel to [0, 1] #
        vessel_enhanced_nda, origin, spacing, direction = img2nda(self.vessel_enhanced_save_path)
        vessel_enhanced_nda += abs(np.min(vessel_enhanced_nda))
        vessel_enhanced_nda *= mask_nda
        vessel_enhanced_nda /= np.max(vessel_enhanced_nda)
        nda2img(vessel_enhanced_nda, origin, spacing, direction, save_path = '%s_normalized.mha' % self.vessel_enhanced_save_path[:-4])

        # Segment vessel #
        segmenting_vessel(self.vessel_enhanced_save_path, self.vessel_save_path, tol = 0., save_MIP = False)
        os.remove(ctc_ttp_path)

        # Get mirroring vessel map in lesion hemisphere from normal hemisphere #
        vessel_nda, origin, spacing, direction = img2nda(self.vessel_save_path)
        mirror_vessel_nda, mirror_vessel_path = get_mirrored_vessel(self.vessel_save_path)

        # Anisotropic-smoothing binary vessel map #
        smoothed_vessel_path = anisotropic_smoothing(mirror_vessel_path, n_iter = 1, diffusion_time = 3.5, anisotropic_lambda = 0.1, enhancement_type = 3, \
            noise_scale = 3, feature_scale = 5, exponent = 3.5)

        smoothed_vessel_nda, origin, spacing, direction = img2nda(smoothed_vessel_path)
        smoothed_vessel_nda[mirror_vessel_nda == 1.] = 1. # Set original vessel region back at 1 #
        nda2img(smoothed_vessel_nda, origin, spacing, direction, '%s_bin%s' % (smoothed_vessel_path[:-4], smoothed_vessel_path[-4:]))


if __name__ == '__main__':

    on_server = False
    if on_server:
        data_fld = '/playpen-raid1/peirong/Data/ISLES2017/ISLES2017_Training'
        save_fld = '/playpen-raid1/peirong/Data/ISLES2017_Processed'
    else: 
        data_fld = '/media/peirong/PR/ISLES2017/ISLES2017_Training'
        #data_fld = '/home/peirong/biag-raid/Data/ISLES2017/ISLES2017_Training'
        save_fld = '/home/peirong/biag-raid/Data/ISLES2017_Processed'

    #data_fld = '/media/peirong/PR/ISLES2017/ISLES2017_Testing'
    #save_fld = make_dir('/media/peirong/PR5/ISLES2017_Testing_Processed')

    names_file_path = os.path.join(data_fld, 'IDs.txt')
    #names_file_path = os.path.join(save_fld, 'IDs.txt')

    #if not os.path.isfile(os.path.join(save_fld, 'IDs.txt')):
    #    copyfile(names_file_path, os.path.join(save_fld, 'IDs.txt'))

    #########################################################################
    #########################################################################

    names_file = open(names_file_path, 'r')
    case_names = names_file.readlines()
    names_file.close()
    print('Total number of cases:', len(case_names)) 
    #case_names = ['training_16'] # TODO
    #case_names = ['test_28'] # TODO

    for i in range(len(case_names)): # TODO
        #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')
        #time.sleep(8)
        case_name = case_names[i].split('\n')[0]
        print('\nStart processing case %d/%d: %s' % (i+1, len(case_names), case_name))
        processor = ISLES2017_Processor(args_prep, data_fld, save_fld, case_name, for_test = 'test' in case_name)
        #processor.preprocess_all()

        # NOTE: process part of steps # 
 

        ################### (3) Get binary OT ###################
        if not processor.for_test:
            print('   Get binary OT')
            OT_contour = get_contour(processor.OT_save_path, save_path = processor.OT_conotur_save_path)
             

 
'''    # Save cace names
    names_file = open(names_file_path, 'w')

    case_names = []
    case_flds = os.listdir(data_fld)
    for i in range(len(case_flds)):
        case_name = case_flds[i]
        if os.path.isdir(os.path.join(data_fld, case_name)):
            case_names.append(case_name)
    print('Total number of cases:', len(case_names)) 

    for i in range(len(case_names) - 1):
        names_file.write('%s\n' % case_names[i])
    names_file.write('%s' % case_names[-1])
    names_file.close()'''