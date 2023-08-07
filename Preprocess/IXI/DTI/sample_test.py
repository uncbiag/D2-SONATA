import os
import numpy as np
import SimpleITK as sitk
import numpy as np

import dipy
from dipy.reconst import dti
import scipy.ndimage as ndimage

from dti_utils import *


main_fld = '/media/peirong/PR/DTI_Sample'
save_fld = make_dir('/media/peirong/PR/DTI_Sample/DTI_Fit')

dwi_path = os.path.join(main_fld, 'dwi.nhdr')
dwi_img = sitk.ReadImage(dwi_path)
print(dwi_img)
origin = dwi_img.GetOrigin()
spacing = dwi_img.GetSpacing()
direction = dwi_img.GetDirection()
print("Image Size: {0}".format(dwi_img.GetSize()))

bvecs = []
for k in dwi_img.GetMetaDataKeys():
    v = dwi_img.GetMetaData(k)
    print("({0}) = = \"{1}\"".format(k, v))

    '''if 'b-value' in k:
        bval = float(v)
        print('b value == %.1f' % bval)
    if 'gradient' in k:
        print('(%s) == %s' % (k, v))
        gradient = v.split('   ')
        g_x = float(gradient[0])
        g_y = float(gradient[1])
        g_z = float(gradient[2])
        #print(g_x, g_y, g_z)
        bvecs.append([g_x, g_y, g_z])'''

bvecs = np.array(bvecs)
print(bvecs.shape)
gradients = bvecs * bval

dwi_nda = sitk.GetArrayFromImage(dwi_img)
mask_nda = (dwi_nda[..., 0] > 350).astype(float)
for s in range(len(mask_nda)):
    mask_nda[s] = ndimage.binary_fill_holes(mask_nda[s])
mask_img = sitk.GetImageFromArray(mask_nda, isVector = False)
mask_img.SetOrigin(origin)
mask_img.SetSpacing(spacing)
mask_img.SetDirection(direction)
sitk.WriteImage(mask_img, os.path.join(main_fld, 'mask.nii'))


gtab = dipy.core.gradients.GradientTable(gradients)
tensor_model = dti.TensorModel(gtab, 'WLS') # LS, WLS, OLS, NLLS, RESTORE

dti_fit = tensor_model.fit(dwi_nda)

D = dti_fit.quadratic_form * mask_nda[..., None, None] # (shape, 3, 3)
CO = abs(dti_fit.directions[:, :, :, 0]) * mask_nda[..., None] # (shape, 3)
FA = dti_fit.fa * mask_nda
evals = dti_fit.model_params[..., :3] * mask_nda[..., None]
evecs = dti_fit.model_params[..., 3:] * mask_nda[..., None]

print(D.shape)
print(CO.shape)
print(FA.shape)
print(evals.shape)
print(evecs.shape)

case_save_fld = make_dir(os.path.join(main_fld, 'dti test'))

def nda_save_img(nda, save_path, origin, spacing, direction):
    img = sitk.GetImageFromArray(nda, isVector = len(nda.shape) > 3)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    sitk.WriteImage(img, save_path)
    return
nda_save_img(D.reshape(D.shape[0], D.shape[1], D.shape[2], 9), os.path.join(case_save_fld, 'D.nii'), origin, spacing, direction)
nda_save_img(CO, os.path.join(case_save_fld, 'Color_by_orientation.nii'), origin, spacing, direction)
nda_save_img(FA, os.path.join(case_save_fld, 'FA.nii'), origin, spacing, direction)
nda_save_img(evals, os.path.join(case_save_fld, 'L.nii'), origin, spacing, direction)
nda_save_img(evecs, os.path.join(case_save_fld, 'U.nii'), origin, spacing, direction)

