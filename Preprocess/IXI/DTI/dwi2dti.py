import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import SimpleITK as sitk
import numpy as np

import dipy
from dipy.reconst import dti
import scipy.ndimage as ndimage

from utils import make_dir


def stack_dwi(path_list, to_save = True):
    dwi_stack = []
    for i in range(len(path_list)):
        img = sitk.ReadImage(path_list[i])
        dwi_stack.append(sitk.GetArrayFromImage(img).astype(float))
        if i == 0:
            origin = img.GetOrigin()
            spacing = img.GetSpacing()
            direction = img.GetDirection()
            main_fld  = os.path.dirname(path_list[i])
            case_name = os.path.basename(path_list[i])[:-10]
            brain_mask = (dwi_stack[i] > 60).astype(float)
            for s in range(len(brain_mask)):
                brain_mask[s] = ndimage.binary_fill_holes(brain_mask[s])
            print(case_name)
    dwi_stack = np.stack(dwi_stack, axis = -1)
    if to_save:
        dwi_stack_img = sitk.GetImageFromArray(dwi_stack, isVector = True)
        dwi_stack_img.SetOrigin(origin)
        dwi_stack_img.SetSpacing(spacing)
        dwi_stack_img.SetDirection(direction)
        sitk.WriteImage(dwi_stack_img, os.path.join(main_fld, '%s-DWI.nii' % case_name))
    return dwi_stack, brain_mask, case_name, origin, spacing, direction

def nda2img(nda, origin, spacing, direction, save_path):
    img = sitk.GetImageFromArray(nda, isVector = len(nda.shape) > 3)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    sitk.WriteImage(img, save_path)
    return


main_fld = '/media/peirong/PR/IXI'
save_fld = make_dir('/media/peirong/PR/IXI/DTI_Fit')

bvals_path = os.path.join(main_fld, 'bvals.txt')
bvecs_path = os.path.join(main_fld, 'bvecs.txt')

bvals_file = open(bvals_path, "r")
bvecs_file = open(bvecs_path, "r")

bvals_list = bvals_file.read().split(' ')
#bvals_list = [0.] + bvals_list
bvals_list = bvals_list
bvals = np.array(bvals_list).astype(float)
print(bvals) # (n_gradients = 16, )
print(bvals.shape) # (n_gradients = 16, )

bvecs_lines = bvecs_file.readlines()
bvecs_x = bvecs_lines[0].split(' ')
bvecs_y = bvecs_lines[1].split(' ')
bvecs_z = bvecs_lines[2].split(' ')
bvecs = np.stack([bvecs_x, bvecs_y, bvecs_z], axis = -1).astype(float)
print(bvecs) # (n_gradients = 16, 3)
print(bvecs.shape) # (n_gradients = 16, 3)

gradients = bvecs * bvals[:, None]
print(gradients.shape)
print(gradients)

dwi_path_list = [os.path.join(main_fld, 'IXI-DTI/IXI002-Guys-0828-DTI-{:02d}.nii.gz'.format(i)) for i in range(0, 16)]


gtab = dipy.core.gradients.GradientTable(gradients)
tensor_model = dti.TensorModel(gtab, 'OLS') # LS, WLS, OLS, NLLS, RESTORE

dwi, brain_mask, case_name, origin, spacing, direction = stack_dwi(dwi_path_list, to_save = True) # (shape, n_gradients = 16)
print(dwi.shape)
dti_fit = tensor_model.fit(dwi)

D = dti_fit.quadratic_form * brain_mask[..., None, None] # (shape, 3, 3)
CO = abs(dti_fit.directions[:, :, :, 0]) * brain_mask[..., None] # (shape, 3)
FA = dti_fit.fa * brain_mask
evals = dti_fit.model_params[..., :3] * brain_mask[..., None]
evecs = dti_fit.model_params[..., 3:] * brain_mask[..., None]

print(D.shape)
print(CO.shape)
print(FA.shape)
print(evals.shape)
print(evecs.shape)

case_save_fld = make_dir(os.path.join(save_fld, case_name))

nda2img(D.reshape(D.shape[0], D.shape[1], D.shape[2], 9), origin, spacing, direction, os.path.join(case_save_fld, 'DTI.mha'))
nda2img(CO, origin, spacing, direction, os.path.join(case_save_fld, 'Color_by_orientation.mha'))
nda2img(FA, origin, spacing, direction, os.path.join(case_save_fld, 'FA.mha'))
nda2img(evals, origin, spacing, direction, os.path.join(case_save_fld, 'L.mha'))
nda2img(evecs, origin, spacing, direction, os.path.join(case_save_fld, 'U.mha'))
nda2img(brain_mask, origin, spacing, direction, os.path.join(case_save_fld, 'Mask.mha'))






