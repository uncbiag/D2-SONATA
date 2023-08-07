import os, sys, argparse, time, shutil
from shutil import copyfile, move
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.ndimage as ndimage

import torch

import matplotlib
import matplotlib.pyplot as plt

import itk
import SimpleITK as sitk
#from itkwidgets import view
from itk import TubeTK as ttk

from Preprocess.IXI.itk_utils import *
from Preprocess.contour import get_contour
from Preprocess.print_info import write_info
from Preprocess.ISLES.ISLES2017_process import *
from Preprocess.IXI.IXI_process import segmenting_vessel
from utils import make_dir, nda2img, get_times, img2nda, cutoff_percentile
from Preprocess.prepro_utils import keep_largest_volumn, rm_by_slice, rm_by_column, rm_by_row, crop_nda, cropping


#%% Basic settings
parser = argparse.ArgumentParser('ISLES2018 CTP Preprocessing')
parser.add_argument('--new_spacing', type = list, default = [1., 1., 1.])
args_prep = parser.parse_args()  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# General setting for DTI fitting #
fwhm = 1.25
gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std


############################################################################################
##########################################  Utils ##########################################
############################################################################################

def ctp_s0(signal_nda, s0_threshold = 0.05):
    '''
    Calculate the CTP bolus arrival time (bat) and corresponding S0: averaged over signals before bat
    return: s0 # (n_slice, n_row, n_column)
    '''
    nT = signal_nda.shape[-1]
    sig_avg = np.zeros(nT)
    for t in range(nT):
        sig_avg[t] = (signal_nda[..., t]).mean()
    ttp = np.argmax(sig_avg)
    ttd = np.argmin(sig_avg[ttp:]) + ttp
    print('Bolus concentration time-to-peak (start from 0):', ttp)
    threshold = s0_threshold * (np.amax(sig_avg) - np.amin(sig_avg)) 
    flag = True
    bat  = 1
    while flag:
        if sig_avg[bat] - sig_avg[bat - 1] >= threshold and sig_avg[bat + 1] > sig_avg[bat]:
            flag = False
        else:
            bat += 1
        if bat == nT:
            flag = False
    print('Bolus arrival time (start from 0):', bat - 1)
    s0 = np.mean(signal_nda[..., :bat], axis = 3) # time dimension == 3
    return s0, bat, ttp, ttd


def signal2ctc(signal_nda, mask_nda, k_ct = 1., s0_threshold = 0.05, normalize_ctc = True):

    s0, bat, ttp, ttd = ctp_s0(signal_nda, s0_threshold)
    signal_nda = signal_nda[..., bat:]
    ctc_nda = k_ct * (signal_nda - s0[..., None])
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
    return ctc_nda * mask_nda[..., None], [bat, ttp, ttd]

def get_brain_mask(img_path, mask_save_path):
    # NOTE img shape: (s, r, c) #
    ctp_nda, origin, spacing, direction = img2nda(img_path)
    ct_nda = ctp_nda # TIme dimension: 0 #
    ct_max = np.max(ct_nda)

    mask = ct_nda.copy()
    mask[mask <= 0.] = 0.
    mask[mask >= 0.05 * ct_max] = 0.
    mask[mask != 0.] = 1.

    mask = rm_by_slice(mask, 500)
    for s in range(mask.shape[0]):
        mask[s] = ndimage.binary_fill_holes(mask[s])
        
    mask = keep_largest_volumn(mask)
    for r in range(mask.shape[1]):
        mask[:, r] = keep_largest_volumn(mask[:, r])
    for c in range(mask.shape[2]):
        mask[:, :, c] = keep_largest_volumn(mask[:, :, c])
    for s in range(mask.shape[0]):
        mask[s] = keep_largest_volumn(mask[s])
    mask = rm_by_slice(mask, 500)
    mask = rm_by_row(mask, 400)
    mask = rm_by_column(mask, 400)

    nda2img(mask,origin, spacing, direction, save_path = mask_save_path)

    return mask 


def crop_n_mask_all(mask_path, to_crop_paths, mask_save_path, cropped_save_paths):
    mask_cropped_img, [[x0, y0, z0], [x1, y1, z1]] = cropping(sitk.ReadImage(mask_path), save_path = mask_save_path)
    mask_cropped_nda = sitk.GetArrayFromImage(mask_cropped_img)
    for i in range(len(cropped_save_paths)):  
        nda, origin, spacing, direction = img2nda(to_crop_paths[i])
        nda = nda[x0:x1, y0:y1, z0:z1] 
        nda = nda * mask_cropped_nda if len(nda.shape) <= 3 else nda * mask_cropped_nda[..., None]
        nda2img(nda, origin, spacing, direction, cropped_save_paths[i])
    return



############################################################################################
############################################################################################
############################################################################################


class ISLES2018_Processor(object):

    def __init__(self, args, data_fld, save_fld, case_name):
        
        self.args = args
        self.case_name = case_name 
        self.new_spacing = self.args.new_spacing

        self.register_paths(data_fld, save_fld)
        self.register_PWI(self.CTP_path)

    
    def register_PWI(self, CTP_path):
        PWI_nda, origin, spacing, direction = img2nda(CTP_path) # NOTE: PWI shape: (nT, s, r, c)
        direction = direction[:3] + direction[4:7] + direction[8:11]
        def save_nda2img(nda, save_path):
            nda2img(nda, origin, spacing, direction, save_path)
            return save_path 
        nT = PWI_nda.shape[0] # NOTE: PWI shape: (nT, s, r, c)
        self.CTP_paths = [None] * nT
        self.CTP_save_paths = [None] * nT
        for it in range(nT): # NOTE: PWI shape: (nT, s, r, c)
            self.CTP_paths[it] = save_nda2img(PWI_nda[it], '{:s}-{:02d}.mha'.format(self.CTP_path[:-4], it))
            self.CTP_save_paths[it] = '{:s}-{:02d}.mha'.format(self.CTP_save_path[:-4], it)
        return

    def register_paths(self, data_fld, save_fld):
        self.data_main_fld = os.path.join(data_fld, case_name)
        self.save_main_fld = make_dir(os.path.join(save_fld, case_name))
        self.mask_save_path = os.path.join(self.save_main_fld, 'BrainMask.mha')
        self.vessel_save_path = os.path.join(self.save_main_fld, 'Vessel.mha')
        self.vessel_enhanced_save_path = os.path.join(self.save_main_fld, 'VesselEnhanced.mha')
        
        for module in os.listdir(self.data_main_fld):
            if "PWI" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                CTP_fld = os.path.join(self.data_main_fld, module)
                self.CTP_save_fld = make_dir(os.path.join(self.save_main_fld, 'CTP'))
                for files in os.listdir(CTP_fld):
                    if '4DPWI' in files and 'masked' not in files and 'cropped' not in files and files.endswith('.nii'):
                        self.CTP_path = os.path.join(CTP_fld, files)
                        self.CTP_save_path = os.path.join(self.CTP_save_fld, 'CTP.mha')
                        self.CTC_save_path = os.path.join(self.CTP_save_fld, 'CTC.mha')
                        self.CTP_diff_save_path = os.path.join(self.CTP_save_fld, 'CTP_Max-Min.mha')
            if "MTT" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                MTT_fld = os.path.join(self.data_main_fld, module)
                self.MTT_save_fld = make_dir(os.path.join(self.save_main_fld, 'MTT'))
                for files in os.listdir(MTT_fld):
                    if 'MTT' in files and 'masked' not in files and 'cropped' not in files and files.endswith('.nii'):
                        self.MTT_path = os.path.join(MTT_fld, files)
                        self.MTT_save_path = os.path.join(self.MTT_save_fld, 'MTT.mha')
            if "CBF" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                CBF_fld = os.path.join(self.data_main_fld, module)
                self.CBF_save_fld = make_dir(os.path.join(self.save_main_fld, 'CBF'))
                for files in os.listdir(CBF_fld):
                    if 'CBF' in files and 'masked' not in files and 'cropped' not in files and files.endswith('.nii'):
                        self.CBF_path = os.path.join(CBF_fld, files)
                        self.CBF_save_path = os.path.join(self.CBF_save_fld, 'CBF.mha')
            if "CBV" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                CBV_fld = os.path.join(self.data_main_fld, module)
                self.CBV_save_fld = make_dir(os.path.join(self.save_main_fld, 'CBV'))
                for files in os.listdir(CBV_fld):
                    if 'CBV' in files and 'masked' not in files and 'cropped' not in files and files.endswith('.nii'):
                        self.CBV_path = os.path.join(CBV_fld, files)
                        self.CBV_save_path = os.path.join(self.CBV_save_fld, 'CBV.mha')
            if "Tmax" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                Tmax_fld = os.path.join(self.data_main_fld, module)
                self.Tmax_save_fld = make_dir(os.path.join(self.save_main_fld, 'Tmax'))
                for files in os.listdir(Tmax_fld):
                    if 'Tmax' in files and 'masked' not in files and 'cropped' not in files and files.endswith('.nii'):
                        self.Tmax_path = os.path.join(Tmax_fld, files)
                        self.Tmax_save_path = os.path.join(self.Tmax_save_fld, 'Tmax.mha')
            if "OT" in module and os.path.isdir(os.path.join(self.data_main_fld, module)):
                OT_fld = os.path.join(self.data_main_fld, module)
                self.OT_save_fld = make_dir(os.path.join(self.save_main_fld, 'OT'))
                for files in os.listdir(OT_fld):
                    if 'OT' in files and 'masked' not in files and 'cropped' not in files and files.endswith('.nii'):
                        self.OT_path = os.path.join(OT_fld, files)
                        self.OT_save_path = os.path.join(self.OT_save_fld, 'OT.mha')
        return 


    def preprocess_all(self):
        '''
        Processing steps:
        (1) Resample baseline image (PWI-00) to be isotropic (1 mm)
        (2) Register all images to isotropic PWI-00
        (3) Compute brain mask
        (4) Convert CTP to CTC
        (5) Segment vessel map
        '''
        ################### (1) Resample PWI-00 ###################
        print('   Resample PWI-00 to isotropic')
        PWI0_iso = resampling(read_img(self.CTP_paths[0]), self.new_spacing, self.CTP_save_paths[0])

        ################### (2) Register images to isotropic PWI-00 ###################
        print('   Register images to isotropic PWI-00')
        registering(itk_img_paths = [self.CBF_path, self.CBV_path, self.MTT_path, self.Tmax_path] + self.CTP_paths[1:], \
            fixed_itk_img = PWI0_iso, save_paths = [self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.Tmax_save_path] + self.CTP_save_paths[1:])
        for it in range(len(self.CTP_paths)):
            os.remove(self.CTP_paths[it])

        ################### (3) Get binary OT ###################
        print('   Get binary OT')
        resampling(read_img(self.OT_path), self.new_spacing, self.OT_save_path)
        convert_binary(self.OT_save_path)
        get_contour(self.OT_save_path, save_path = self.OT_save_path)
        contour, origin, spacing, direction = img2nda(self.OT_save_path)
        # Keep only the outer boundary #
        for s in range(contour.shape[0]): 
            contour[s] = ndimage.binary_fill_holes(contour[s])
        nda2img(contour, origin, spacing, direction, save_path = self.OT_save_path)
        contour = get_contour(self.OT_save_path, save_path = self.OT_save_path)
        nda2img(contour, origin, spacing, direction, save_path = self.OT_save_path)

        ################### (4) Get brain mask ###################
        print('   Compute brain mask')
        mask_nda = get_brain_mask(self.CTP_save_paths[0], self.mask_save_path)

        ################### (5) Stack PWI ###################
        print('   Stack PWI')
        stacking_pwi(self.CTP_save_paths, self.CTP_save_path)
        
        ################### (6) Mask brain region and crop ###################
        print('   Brain mask and crop for all images')
        crop_n_mask_all(self.mask_save_path, 
            to_crop_paths = [self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.Tmax_save_path, self.OT_save_path, self.CTP_save_path], 
            mask_save_path = self.mask_save_path, 
            cropped_save_paths = [self.CBF_save_path, self.CBV_save_path, self.MTT_save_path, self.Tmax_save_path, self.OT_save_path, self.CTP_save_path])
        
        ################### (7) Convert CTP to CTC ###################
        print('   Convert PWI to CTC')
        signal_nda, _, _, _ = img2nda(self.CTP_save_path)
        mask_nda, origin, spacing, direction = img2nda(self.mask_save_path)
        ctc_nda, [bat, ttp, ttd] = signal2ctc(signal_nda, mask_nda, k_ct = 1., s0_threshold = 0.05, normalize_ctc = True)
        nda2img(ctc_nda, origin, spacing, direction, save_path = self.CTC_save_path)
        write_info(self.CTC_save_path, 'Info.txt', from_BAT = False)

        ################### (8) Segment vessel map ###################
        print('   Segment vessel map')
        # CTA ~ Max CTC - Min CTC #
        ctp_nda, _, _, _ = img2nda(self.CTP_save_path)
        avg = np.zeros(ctp_nda.shape[-1])
        for t in range(ctp_nda.shape[-1]):
            avg[t] = (ctp_nda[..., t]).mean()
        ctp_diff = nda2img(ctp_nda[..., np.argmax(avg)] - ctp_nda[..., np.argmin(avg)], origin, spacing, direction, save_path = self.CTP_diff_save_path)
        enhance_vessel(self.CTP_diff_save_path, self.CBF_save_path, self.CBV_save_path, self.mask_save_path, save_path = self.vessel_enhanced_save_path)
        segmenting_vessel(self.vessel_enhanced_save_path, self.vessel_save_path, tol = 0., save_MIP = False)


def process_flag(ctp_path):
    ctp_nda, _, _, _ = img2nda(ctp_path)
    if ctp_nda.shape[1] < 10:
        return False
    else:
        return True


if __name__ == '__main__':

    data_fld = '/media/peirong/PR5/ISLES2018/ISLES2018_Training'
    save_fld = make_dir('/media/peirong/PR5/ISLES2018_Processed')

    names_file_path = os.path.join(data_fld, 'IDs.txt')
    #names_processed_path = os.path.join(save_fld, 'ID.txt')


    #########################################################################
    #########################################################################

    names_file = open(names_file_path, 'r')
    case_names = names_file.readlines()
    case_names.sort()
    names_file.close()
    print('Total number of cases:', len(case_names)) 
    case_names = ['case_82'] # TODO

    for i in range(len(case_names)): # TODO
        #os.popen('sudo sh -c "echo 1 >/proc/sys/vm/drop_caches"')
        #time.sleep(8)
        case_name = case_names[i].split('\n')[0]
        processor = ISLES2018_Processor(args_prep, data_fld, save_fld, case_name)
        if process_flag(processor.CTP_path):
            print('\nStart processing case %d/%d: %s' % (i+1, len(case_names), case_name))
            processor.preprocess_all()
        else:
            print('Too few slices, skip case %d/%d: %s' % (i+1, len(case_names), case_name))
            for ctp_path in processor.CTP_paths:
                os.remove(ctp_path)
            shutil.rmtree(processor.save_main_fld)

        # NOTE: process part of steps #



    # Record all cases #
    '''
    names_file = open(names_file_path, 'w')
    case_names = []
    case_flds = os.listdir(data_fld)
    for i in range(len(case_flds)):
        case_name = case_flds[i]
        if os.path.isdir(os.path.join(data_fld, case_name)):
            case_names.append(case_name)
    print('Total number of cases:', len(case_names)) 

    for i in range(len(case_names)):
        names_file.write('%s\n' % case_names[i])
    names_file.write('%s' % case_names[-1])
    names_file.close()
    copyfile(names_file_path, os.path.join(save_fld, 'ID.txt'))'''