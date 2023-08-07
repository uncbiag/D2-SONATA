import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

from utils import * 
from Preprocess.prepro_utils import *


def proceed_all(AllFld, Cases, up_scale, postfix = '_BS_EvalSpa_BATtoTTD_norm_Res.nii'):

    AllPaths = [os.path.join(AllFld, patient) for patient in Cases]
    for i in range(len(AllPaths)):
        if not AllPaths[i].startswith('.') and 'training' in AllPaths[i]:
            print('Now process', os.path.basename(AllPaths[i]))
            ctc_path = get_paths(AllPaths[i], postfix)
            print(ctc_path)

            fileHandle = open(os.path.join(os.path.dirname(ctc_path), 'Info_EvalSpa_Res(2).txt'), 'r')
            lineList = fileHandle.readlines()
            fileHandle.close()
            org = [float(lineList[1]), float(lineList[2]), float(lineList[3])]
            spa = [float(lineList[5]), float(lineList[6]), float(lineList[7])]

            ctc_up_time_img = uptiming(ctc_path, up_scale, interp_method = 'Linear', isNumpy = True, origin = org, spacing = spa)
            ctc_nda = sitk.GetArrayFromImage(ctc_up_time_img)
            np.save('%s_UpTime(%s).npy' % (ctc_path[:-4], up_scale), ctc_nda)

            bat, ttp, ttd = get_times(ctc_nda)
            shp = ctc_nda.shape
            print('spa: %.2f, %.2f, %.2f' % (spa[-1], spa[-2], spa[-3]))
            print('shp: %d, %d, %d' % (shp[0], shp[1], shp[2]))
            # Save numpy array and image info #
            #np.save('%s.npy' % ctc_path[:-4], ctc_nda)
            file = open(os.path.join(os.path.dirname(ctc_path), 'Info_EvalSpa_Res(2)_UpTime(%s).txt' % up_scale), 'w')
            file.write('Origin\n%s\n%s\n%s\n' % (org[0], org[1], org[2]))
            file.write('Spacing\n%s\n%s\n%s\n' % (spa[0], spa[1], spa[2]))
            file.write('BAT\n%s\nTTP\n%s\nTTD\n%s\n' % (bat, ttp, ttd))
            file.write('Shape\n%d\n%d\n%d\n%d' % (shp[0], shp[1], shp[2], shp[3]))
            file.close()




##################################################################################3

if __name__ == '__main__':

    DataFld = '/media/peirong/PR/ISLES2017/ISLES2017_Training'

    Patients = os.listdir(DataFld)
    Patients.sort()
    #Patients = ['training_1']

    proceed_all(DataFld, Patients, up_scale = 5, postfix = '_BS_EvalSpa_BATtoTTD_norm_Res(2).npy')