import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from utils import img2nda, nda2img



def rotating(img_path, angle = 10, isBinary = False, for_test = False):
    nda, origin, spacing, direction = img2nda(img_path)
    # TODO: determine option 'order' #
    nda = ndimage.rotate(nda, angle = angle, axes = (2, 1), reshape = False, order = 1) # rotate on row-column plane
    if isBinary:
        nda = np.rint(nda)
    if for_test:
        save_path = '%s_rotated(test, %s).mha' % (img_path[:-4], angle)
    else:
        save_path = '%s_rotated(%s).mha' % (img_path[:-4], angle)
    nda2img(nda, origin, spacing, direction, save_path)
    return save_path



################################################################################
if __name__ == '__main__':
    

    
    # For ISLES-2017 Training
    main_fld = '/media/peirong/PEIRONG/StrokeData/ISLES2017/ISLES2017_Training'
    

    '''10th step: rotate images on row-column plane'''

    for_test = False
    rotate_angle = -16.0 # clock-wise
    CaseFolder = os.path.join(main_fld, 'training_11')

    for module in os.listdir(CaseFolder):
        if "PWI" in module and not module.startswith('.'):
            PWI_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))
            CTC_path = os.path.join(CaseFolder, "%s/CTC_Axial_cropped_filtered_filtered.nii" % (module))
        if "OT" in module and not module.startswith('.'):
            OT_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))
        if "ADC" in module and not module.startswith('.'):
            ADC_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))
        if "MTT" in module and not module.startswith('.'):
            MTT_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))
        if "CBF" in module and not module.startswith('.'):
            CBF_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))
        if "CBV" in module and not module.startswith('.'):
            CBV_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))
        if "Tmax" in module and not module.startswith('.'):
            Tmax_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))
        if "TTP" in module and not module.startswith('.'):
            TTP_path = os.path.join(CaseFolder, "%s/%s_cropped.nii" % (module, module))

    if for_test:
        rotating(ADC_path, angle = rotate_angle, isBinary = False, for_test = True)
    else:
        rotating(ADC_path, angle = rotate_angle, isBinary = False)
        rotating(PWI_path, angle = rotate_angle, isBinary = False)
        rotating(CTC_path, angle = rotate_angle, isBinary = False)
        rotating(OT_path, angle = rotate_angle, isBinary = False)
        rotating(CBF_path, angle = rotate_angle, isBinary = False)
        rotating(CBV_path, angle = rotate_angle, isBinary = False)
        rotating(MTT_path, angle = rotate_angle, isBinary = False)
        rotating(TTP_path, angle = rotate_angle, isBinary = False)
        rotating(Tmax_path, angle = rotate_angle, isBinary = False)
