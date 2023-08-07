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
from utils import make_dir, nda2img, get_times, img2nda, cutoff_percentile, copy_file, move_file, copy_files, move_files, remove_paths


#%% Basic settings
parser = argparse.ArgumentParser('ISLES2017 MRP Preprocessing')
parser.add_argument('--new_spacing', type = list, default = [1., 1., 1.])
#parser.add_argument('--new_spacing', type = list, default = [2., 2., 2.])
args_prep = parser.parse_args()  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# General setting for DTI fitting #
fwhm = 1.25
gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std


############################################################################################
##########################################  Utils ##########################################
############################################################################################
 
def rotating(img_path, angle = 10, isBinary = False, save_path = None):
    # TODO: determine option 'order' #
    nda, origin, spacing, direction = img2nda(img_path)
    nda = ndimage.rotate(nda, angle = angle, axes = (2, 1), reshape = False, order = 1) # rotate on row-column plane #
    if isBinary:
        nda = nda.astype(int)
    if not save_path:
        save_path = '%s_rotated(%s).mha' % (img_path[:-4], angle)
    nda2img(nda, origin, spacing, direction, save_path)
    return save_path

def rotate_all(img_paths, angle, isBinary, save_paths):
    for i in range(len(img_paths)):
        rotating(img_paths[i], angle, isBinary, save_paths[i])
    return save_paths
 
 

if __name__ == '__main__':
 
    data_fld = '/home/peirong/biag-raid0/Data/ISLES2017_Processed' 
    save_fld = '/home/peirong/biag-raid1/Data/ISLES2017_Processed' 
 
    case_names = []
    for case_name in os.listdir(data_fld):
        if 'training_' in case_name:
            case_names.append(case_name) 
    
    #########################################################################
 
    #case_names = ['training_13'] # TODO 
 
    print('Total number of cases:', len(case_names))  
    for i in range(len(case_names)): # TODO 
        case_name = case_names[i].split('\n')[0]
        case_data_fld = os.path.join(data_fld, case_name)
        case_save_fld = os.path.join(save_fld, case_name)

        is_rotate = False
        for file_name in os.listdir(case_save_fld):
            if 'rotate' in file_name:
                is_rotate = True
                rotate_degree = float(file_name.split('=')[-1])
                break
        
        if is_rotate: 
            print('\nCase %d/%d: %s - %s(%s)' % (i+1, len(case_names), case_name, is_rotate, str(rotate_degree)))
        else:
            print('\nCase %d/%d: %s - %s' % (i+1, len(case_names), case_name, is_rotate))

        '''if is_rotate: 
            for file_name in os.listdir(case_save_fld):
                if 'npy' in file_name and 'Vessel' in file_name:
                    os.remove(os.path.join(case_save_fld, file_name))
            copy_file(os.path.join(case_data_fld, 'Vessel.mha'), os.path.join(case_save_fld, 'Vessel.mha'))
            copy_file(os.path.join(case_data_fld, 'Vessel_mirrored.mha'), os.path.join(case_save_fld, 'Vessel_mirrored.mha'))
            copy_file(os.path.join(case_data_fld, 'Vessel_mirrored_smoothed.mha'), os.path.join(case_save_fld, 'Vessel_mirrored_smoothed.mha'))
            copy_file(os.path.join(case_data_fld, 'Vessel_mirrored_smoothed_bin.mha'), os.path.join(case_save_fld, 'Vessel_mirrored_smoothed_bin.mha'))
            copy_file(os.path.join(case_data_fld, 'VesselEnhanced.mha'), os.path.join(case_save_fld, 'VesselEnhanced.mha'))
            copy_file(os.path.join(case_data_fld, 'VesselEnhanced_normalized.mha'), os.path.join(case_save_fld, 'VesselEnhanced_normalized.mha'))

            rotating(os.path.join(case_save_fld, 'Vessel.mha'), rotate_degree, True, os.path.join(case_save_fld, 'Vessel.mha'))
            rotating(os.path.join(case_save_fld, 'Vessel_mirrored.mha'), rotate_degree, True, os.path.join(case_save_fld, 'Vessel_mirrored.mha'))
            rotating(os.path.join(case_save_fld, 'Vessel_mirrored_smoothed_bin.mha'), rotate_degree, True, os.path.join(case_save_fld, 'Vessel_mirrored_smoothed_bin.mha'))
            
            rotate_all([os.path.join(case_save_fld, 'Vessel.mha'), os.path.join(case_save_fld, 'Vessel_mirrored.mha'), os.path.join(case_save_fld, 'Vessel_mirrored_smoothed_bin.mha')], \
                rotate_degree, True, [os.path.join(case_save_fld, 'Vessel.mha'), os.path.join(case_save_fld, 'Vessel_mirrored.mha'), os.path.join(case_save_fld, 'Vessel_mirrored_smoothed_bin.mha')])
            rotate_all([os.path.join(case_save_fld, 'Vessel_mirrored_smoothed.mha'), os.path.join(case_save_fld, 'VesselEnhanced.mha'), os.path.join(case_save_fld, 'VesselEnhanced_normalized.mha')],\
                rotate_degree, False, [os.path.join(case_save_fld, 'Vessel_mirrored_smoothed.mha'), os.path.join(case_save_fld, 'VesselEnhanced.mha'), os.path.join(case_save_fld, 'VesselEnhanced_normalized.mha')])
 '''
        # NOTE: process part of steps #  
        
             

 
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