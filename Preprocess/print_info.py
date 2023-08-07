import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

from utils import * 
from Preprocess.prepro_utils import *


def write_info(img_path, info_basename = 'Info.txt', from_BAT = False, from_TTP = False, show_info = True):
    movie_img = sitk.ReadImage(img_path)
    movie_nda = sitk.GetArrayFromImage(movie_img)
    shp = movie_nda.shape

    bat, ttp, ttd = get_times(movie_nda, show_info = show_info)
    bat = 0 if bat >= shp[-1] - 1 else bat
    ttp = 0 if ttp >= shp[-1] - 1 else ttp
    ttd = shp[-1] if ttd < 1 else ttd
    if from_BAT:
        bat = 0
    if from_TTP:
        bat = 0
        ttp = 0

    org = movie_img.GetOrigin()
    spa = movie_img.GetSpacing()
    dir = movie_img.GetDirection()
    
    if show_info:
        print('   - BAT: %s; TTP: %s; TTD: %s' % (bat, ttp, ttd))
        print('   - Data spacing: %.2f, %.2f, %.2f' % (spa[-1], spa[-2], spa[-3]))
        print('   - shp: %d, %d, %d' % (shp[0], shp[1], shp[2]))
    # Save numpy array and image info #
    #np.save('%s.npy' % movie_path[:-4], movie_nda)
    if info_basename:
        file = open(os.path.join(os.path.dirname(img_path), info_basename), 'w')
        file.write('Origin\n%s\n%s\n%s\n' % (org[0], org[1], org[2]))
        file.write('Spacing\n%s\n%s\n%s\n' % (spa[0], spa[1], spa[2]))
        file.write('Direction\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n' % (dir[0], dir[1], dir[2], dir[3], dir[4], dir[5], dir[6], dir[7], dir[8]))
        file.write('BAT\n%s\nTTP\n%s\nTTD\n%s\n' % (bat, ttp, ttd))
        file.write('Shape\n%d\n%d\n%d\n%d' % (shp[0], shp[1], shp[2], shp[3]))
        file.close()
    return bat, ttp, ttd


'''# archived #
    def proceed_all_ISLES(AllFld, Cases, postfix, img_type = 'MRP'):

    AllPaths = [os.path.join(AllFld, patient) for patient in Cases]
    for i in range(len(AllPaths)):
        if not AllPaths[i].startswith('.') and 'training' in AllPaths[i] or 'test' in AllPaths[i] or 'case' in AllPaths[i]:
            print('Now process', os.path.basename(AllPaths[i]))
            if 'MRP' in img_type:
                ctc_path = get_mrp_paths(AllPaths[i], postfix)
            elif 'CTP' in img_type:
                ctc_path = get_ctp_paths(AllPaths[i], postfix)
            write_info(ctc_path, 'Info_fromTTP.txt') # TODO #
    return'''

def proceed_all_ISLES(AllFld, Cases, img_type = 'MRP'):

    AllPaths = [os.path.join(AllFld, patient) for patient in Cases]
    for i in range(len(AllPaths)):
        if not AllPaths[i].startswith('.') and 'training' in AllPaths[i] or 'test' in AllPaths[i] or 'case' in AllPaths[i]:
            print('Now process', os.path.basename(AllPaths[i]))
            if 'MRP' in img_type:
                ctc_path = os.path.join(AllPaths[i], 'MRP', 'CTC_fromTTP.mha') # TODO # 
            elif 'CTP' in img_type:
                pass # TODO #
            write_info(ctc_path, 'Info_fromTTP.txt') # TODO #
    return

def proceed_all_IXI(AllFld, Cases):

    AllPaths = [os.path.join(AllFld, patient.split('\n')[0]) for patient in Cases]
    for i in range(len(AllPaths)):
        movie_fld = os.path.join(AllPaths[i], 'Movies')
        if os.path.isdir(movie_fld):
            print('Now process', os.path.basename(AllPaths[i]))
            movie_path = os.path.join(movie_fld, 'AdvDiff.mha')
            write_info(movie_path, 'Info.txt')
        else:
            print('Skip no-data case', os.path.basename(AllPaths[i]))
    return



##################################################################################

if __name__ == '__main__':

    #############################################
    #############  ISLES 2017: MRP  #############
    #############################################

    ISLES2017 = '/home/peirong/biag-raid-data/ISLES2017_Processed_rotated'
    Patients = os.listdir(ISLES2017)
    Patients.sort()
    #Patients = ['training_1']

    proceed_all_ISLES(ISLES2017, Patients, postfix = '_BS_EvalSpa_TTPtoTTD_Res(1)_norm.nii', img_type = 'MRP')


    #############################################
    #############  ISLES 2018: CTP  #############
    #############################################

    '''ISLES2018 = '/media/peirong/PR5/ISLES2018/ISLES2018_Training/TRAINING'
    Patients = os.listdir(ISLES2018)
    Patients.sort()
    #Patients = ['case_1']

    proceed_all_ISLES(ISLES2018, Patients, postfix = '_BATtoTTD_norm_Res(1).nii', img_type = 'CTP')'''


    #############################################
    ################  IXI: Demo  ################
    #############################################

    IXI = '/media/peirong/PR5/IXI_Processed'

    names_file = open(os.path.join(IXI, 'IDs.txt'), 'r')
    case_names = names_file.readlines()
    names_file.close()
    #case_names = ['IXI002-Guys-0828'] # TODO

    #proceed_all_IXI(IXI, case_names)