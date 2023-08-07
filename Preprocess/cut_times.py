import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

from utils import * 
from Datasets.dataset import get_times
from Preprocess.prepro_utils import *



def proceed_all(AllFld, Cases, postfix = '_BS_EvalSpa.nii'):

    AllPaths = [os.path.join(AllFld, patient) for patient in Cases]
    for i in range(len(AllPaths)):
        if not AllPaths[i].startswith('.') and 'training' in AllPaths[i]:
            print('Now process', os.path.basename(AllPaths[i]))
            ctc_path = get_paths(AllPaths[i], postfix = postfix)
            ctc_img = sitk.ReadImage(ctc_path)
            ctc_nda = sitk.GetArrayFromImage(ctc_img)
            ArrivalTime, PeakTime, BottomTime = get_times(ctc_nda) # (s, r, c, T)
            #ctc_nda_cut = ctc_nda[..., PeakTime : BottomTime + 1] # NOTE: Peak TIme to First Drain Time # 
            #ctc_nda_cut = ctc_nda[..., PeakTime : ] # NOTE: Peak Time to Last Time # 
            ctc_nda_cut = ctc_nda[..., ArrivalTime : BottomTime + 1]

            # Normalize #
            ctc_max = np.max(abs(ctc_nda_cut))
            nda_norm = ctc_nda_cut / ctc_max * 10

            ctc_img_cut = sitk.GetImageFromArray(ctc_nda_cut, isVector = True)
            ctc_img_cut.SetOrigin(ctc_img.GetOrigin())
            ctc_img_cut.SetSpacing(ctc_img.GetSpacing())
            ctc_img_cut.SetDirection(ctc_img.GetDirection())
            #sitk.WriteImage(ctc_img_cut, '%s_cutT.nii' % ctc_path[:-4]) # NOTE: Peak Time to First Drain Time # 
            #sitk.WriteImage(ctc_img_cut, '%s_fromPeak.nii' % ctc_path[:-4]) # NOTE: Peak Time to Last Time # 
            sitk.WriteImage(ctc_img_cut, '%s_BATtoTTD.nii' % ctc_path[:-4]) # NOTE: Peak Time to Last Time # 




##################################################################################3

if __name__ == '__main__':

    DataFld = '/media/peirong/PR/ISLES2017/ISLES2017_Training'

    Patients = os.listdir(DataFld)
    Patients.sort()

    proceed_all(DataFld, Patients, postfix = '_BS_EvalSpa.nii')