import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

from utils import * 
from Datasets.dataset import get_times
from Preprocess.prepro_utils import *


def proceed_all(AllFld, Cases, postfix, percentile_lower = 6., percentile_upper = 80.):

    AllPaths = [os.path.join(AllFld, patient) for patient in Cases]
    for i in range(len(AllPaths)):
        if not AllPaths[i].startswith('.') and 'training' in AllPaths[i]:
            print('Now process', os.path.basename(AllPaths[i]))
            ctc_path = get_paths(AllPaths[i], postfix = postfix)
            ctc_img = sitk.ReadImage(ctc_path)
            spa = ctc_img.GetSpacing()
            print('Org spa: %.2f, %.2f, %.2f' % (spa[-1], spa[-2], spa[-3]))

            '''# Eval spacing #
            eval_img = sitk.Expand(ctc_img, [1, 1, int(np.round(spa[-1] / spa[0]))], sitk.sitkBSpline)
            eval_spa = eval_img.GetSpacing()
            origin = eval_img.GetOrigin()
            direction = eval_img.GetDirection()
            print('New spa: %.2f, %.2f, %.2f' % (eval_spa[-1], eval_spa[-2], eval_spa[-3]))
            sitk.WriteImage(eval_img, '%s_EvalSpa.nii' % ctc_path[:-4]) # NOTE: Peak Time to Last Time # '''
            
            # Cut times #
            ctc_nda = sitk.GetArrayFromImage(eval_img)
            ArrivalTime, PeakTime, BottomTime = get_times(ctc_nda) # (s, r, c, T)
            ctc_nda = ctc_nda[..., PeakTime : BottomTime]
            img = sitk.GetImageFromArray(ctc_nda, isVector = True)
            img.SetOrigin(origin)
            img.SetSpacing(eval_spa)
            img.SetDirection(direction)
            sitk.WriteImage(img, '%s_EvalSpa_BATtoTTD.nii' % ctc_path[:-4])

            # Cut off outliers #
            '''nda_cut = cutoff_percentile(ctc_nda, percentile_lower = percentile_lower, percentile_upper = percentile_upper)
            img_cut = sitk.GetImageFromArray(nda_cut, isVector = True)
            img_cut.SetOrigin(origin)
            img_cut.SetSpacing(eval_spa)
            img_cut.SetDirection(direction)
            sitk.WriteImage(img_cut, '%s_EvalSpa_BATtoTTD_cutoff.nii' % ctc_path[:-4])'''

            # Normalize #
            ctc_max = np.max(abs(ctc_nda))
            nda_norm = ctc_nda / ctc_max #* 10
            img_norm = sitk.GetImageFromArray(nda_norm, isVector = True)
            img_norm.SetOrigin(origin)
            img_norm.SetSpacing(eval_spa)
            img_norm.SetDirection(direction)
            #sitk.WriteImage(img_norm, '%s_EvalSpa_BATtoTTD_cutoff_norm.nii' % ctc_path[:-4])
            sitk.WriteImage(img_norm, '%s_EvalSpa_BATtoTTD_norm.nii' % ctc_path[:-4])

    return



##################################################################################3

if __name__ == '__main__':

    DataFld = '/media/peirong/PR/ISLES2017/ISLES2017_Training'

    Patients = os.listdir(DataFld)
    Patients.sort()
    #Patients = ['training_14']

    proceed_all(DataFld, Patients, postfix = '_BS_EvalSpa.nii')