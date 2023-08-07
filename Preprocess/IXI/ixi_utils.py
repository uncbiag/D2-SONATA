import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import itk
import torch
import numpy as np
import SimpleITK as sitk
from skimage.feature.corner import hessian_matrix
from itertools import combinations_with_replacement

from Preprocess.prepro_utils import crop_nda, resample_img
from utils import gradient_c_numpy, gradient_f_numpy, nda2img, make_dir



'''More details: https://github.com/scikit-image/scikit-image/blob/bd1dc065025775fc591fb93c7bd20d59e2497e9a/skimage/feature/corner.py#L246'''
def symmetric_image(S_elems):
    image = S_elems[0]
    symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim))
    for idx, (row, col) in \
            enumerate(combinations_with_replacement(range(image.ndim), 2)):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image

def symmetric_compute_eigenvalues(S_elems):
    matrices = symmetric_image(S_elems)
    # eigvalsh returns eigenvalues in increasing order. We want decreasing
    eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
    leading_axes = tuple(range(eigs.ndim - 1))
    eigs = np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)
    return eigs


def hessian_and_eig(nda, sigma=1, mode='constant', cval=0, order='rc'):
    hessian_elements = hessian_matrix(nda, sigma=1, mode='constant', cval=0, order='rc')
    # hessian_elements = Hss, Hsr, Hsc, Hrr, Hrc, Hcc
    hessian_elements = [(sigma ** 2) * e for e in hessian_elements]
    hessian_tensor = symmetric_image(hessian_elements)
    # evals: (shape, dim); evecs: (shape, dim, dim)
    evals, evecs = np.linalg.eigh(hessian_tensor) # evals in ascending order
    return hessian_tensor, evals, evecs





#####################################################################
##############################   OLD   ##############################
#####################################################################


def random_sampling(shape, lower_bound, upper_bound):
    '''
    shape: list or tuple
    return range: [lower_bound, upper_bound]
    '''
    return np.random.random_sample(shape) * (upper_bound - lower_bound) + lower_bound


def cutoff_percentile(nda, mask = None, percentile_lower = 0., percentile_upper = 99.8):
    if mask is None:
        mask = nda != nda[0, 0, 0]
    if percentile_upper is None:
        percentile_upper = 100. - percentile_lower
    res = np.copy(nda)
    cut_off_lower = np.percentile(nda[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(nda[mask != 0].ravel(), percentile_upper)
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    return res


def stream_3D(Phi_a, Phi_b, Phi_c, batched = False, delta_lst = [1., 1., 1.]):
    '''
    input: Numpy array - ((batch), s, r, c)
    '''
    dDa = gradient_c_numpy(Phi_a, batched = batched, delta_lst = [1., 1., 1.])
    dDb = gradient_c_numpy(Phi_b, batched = batched, delta_lst = [1., 1., 1.])
    dDc = gradient_c_numpy(Phi_c, batched = batched, delta_lst = [1., 1., 1.]) 
    Va_x, Va_y, Va_z = dDa[..., 0], dDa[..., 1], dDa[..., 2]
    Vb_x, Vb_y, Vb_z = dDb[..., 0], dDb[..., 1], dDb[..., 2]
    Vc_x, Vc_y, Vc_z = dDc[..., 0], dDc[..., 1], dDc[..., 2]
    Vx = Vc_y - Vb_z
    Vy = Va_z - Vc_x
    Vz = Vb_x - Va_y
    return Vx, Vy, Vz


def resampling(file_name, new_spacing = [1.0, 1.0, 1.0], save_fld = None, to_save = True):
    '''
    NOTE: Abandom 1 layer boundary
    '''
    #print('Resampling', file_name)
    img = sitk.ReadImage(file_name)
    res_img = resample_img(img, out_spacing = new_spacing, is_label = False)
    #origin, spacing, direction = res_img.GetOrigin(), res_img.GetSpacing(), res_img.GetDirection()
    #origin = [origin[i] + spacing[i] for i in range(3)]
    #res_nda = sitk.GetArrayFromImage(res_img)[1:-1, 1:-1, 1:-1] # Abandom 1 layer boundary # 
    #res_img = nda2img(res_nda, origin, spacing, direction)

    if to_save and not save_fld or save_fld == os.path.dirname(file_name):
        sitk.WriteImage(res_img, '%s_Res.nii' % file_name[:-4])
    elif to_save: # keep basename if to be saved in another folder
        sitk.WriteImage(res_img, os.path.join(save_fld,'%s.nii' %  os.path.basename(file_name)))
    return res_img


def match_coordinates(img, target_img): # 3D

    nda = sitk.GetArrayFromImage(img)
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    
    target_nda = sitk.GetArrayFromImage(target_img)
    target_origin = target_img.GetOrigin()
    target_spacing = target_img.GetSpacing()

    target_mid_pos   = [int(target_nda.shape[2-i] / 2) for i in range(3)] # Middle position of target image
    # NOTE: Reverse order between Numpy array and SITK image
    target_mid_coord = [target_origin[i] + target_spacing[i] * target_mid_pos[i] for i in range(3)] # Middle coordinate of target image

    # Find nearest coordinate position in to-match-image
    approx_pos    = [np.rint((target_mid_coord[i] - origin[i]) / spacing[i]) for i in range(3)]
    approx_coord  = [origin[i] + approx_pos[i] * spacing[i] for i in range(3)]
    # Compute shift between raw image and target image
    offset_coord  = [approx_coord[i] - target_mid_coord[i] for i in range(3)]
    offset_origin = [origin[i] - offset_coord[i]  for i in range(3)] # Shift origin

    img_offset = sitk.GetImageFromArray(nda)
    img_offset.SetOrigin(offset_origin) # Reset origin
    img_offset.SetSpacing(spacing)
    img_offset.SetDirection(target_img.GetDirection())

    return img_offset, offset_origin


def match_coordinates_all(img_path_lst, target_img_path):
    target_img = sitk.ReadImage(target_img_path)
    img_offset, offset_origin = match_coordinates(sitk.ReadImage(img_path_lst[0]), target_img)
    sitk.WriteImage(img_offset, img_path_lst[0]) # NOTE: Rewrite matched image
    if len(img_path_lst) > 1:
        for i in range(1, len(img_path_lst)):
            img = sitk.ReadImage(img_path_lst[i])
            img_offset = reset_sitk(img, offset_origin, img.GetSpacing(), img.GetDirection())
            sitk.WriteImage(img_offset, img_path_lst[i]) # NOTE: Rewrite matched image
    return offset_origin


def joint_crop(img_paths1, img_paths2, tol = 0, to_rewrite = True):
    '''
    img_paths1: imgs group1 (same size, same origin, same spacing, same direction)
    img_paths2: imgs group2 (same size, same origin, same spacing, same direction)
    NOTE: reverse spacing order between Numpy array and Simple ITK image: array.shape[0] -- image.GetSize()[-1]
    '''
    img_path1 = img_paths1[0]
    img_path2 = img_paths2[0]
    img1 = sitk.ReadImage(img_path1)
    img2 = sitk.ReadImage(img_path2)
    nda1 = sitk.GetArrayFromImage(img1)
    origin1, spacing1, direction1 = img1.GetOrigin(), img1.GetSpacing(), img1.GetDirection() # sitk order
    nda2 = sitk.GetArrayFromImage(img2)
    origin2, spacing2, direction2 = img2.GetOrigin(), img2.GetSpacing(), img2.GetDirection() # sitk order
    assert spacing1 == spacing2 and nda1.shape == nda2.shape

    _, crop_start1, crop_end1 = crop_nda(nda1)
    _, crop_start2, crop_end2 = crop_nda(nda2)
    crop_start = [max(crop_start1[i], crop_start2[i]) for i in range(3)]
    crop_end   = [min(crop_end1[i], crop_end2[i]) for i in range(3)]

    new_img_paths1 = []
    for i in range(len(img_paths1)):
        nda = sitk.GetArrayFromImage(sitk.ReadImage(img_paths1[i]))
        nda = nda[crop_start[0] : crop_end[0], crop_start[1] : crop_end[1], crop_start[2] : crop_end[2]]
        new_path = '%s_cropped.nii' % img_paths1[i][:-4] if not to_rewrite else img_paths1[i]
        nda2img(nda, origin1, spacing1, direction1, new_path)
        new_img_paths1.append(new_path)

    new_img_paths2 = []
    for i in range(len(img_paths2)):
        nda = sitk.GetArrayFromImage(sitk.ReadImage(img_paths2[i]))
        nda = nda[crop_start[0] : crop_end[0], crop_start[1] : crop_end[1], crop_start[2] : crop_end[2]]
        new_path = '%s_cropped.nii' % img_paths2[i][:-4] if not to_rewrite else img_paths2[i]
        nda2img(nda, origin2, spacing2, direction2, new_path)
        new_img_paths2.append(new_path)
    
    return new_img_paths1, new_img_paths2


def merge_sizes(img_paths1, img_paths2):
    '''
    img_paths1: imgs group1 (same size, same origin, same spacing, same direction)
    img_paths2: imgs group2 (same size, same origin, same spacing, same direction)
    NOTE: reverse spacing order between Numpy array and Simple ITK image: array.shape[0] -- image.GetSize()[-1]
    '''
    img_path1 = img_paths1[0]
    img_path2 = img_paths2[0]
    
    img1 = sitk.ReadImage(img_path1)
    img2 = sitk.ReadImage(img_path2)
    
    nda1 = sitk.GetArrayFromImage(img1)
    origin1, spacing1, direction1 = img1.GetOrigin(), img1.GetSpacing(), img1.GetDirection() # sitk order
    
    nda2 = sitk.GetArrayFromImage(img2)
    origin2, spacing2, direction2 = img2.GetOrigin(), img2.GetSpacing(), img2.GetDirection() # sitk order
    assert spacing1 == spacing2
    #print(nda1.shape, nda2.shape)
    
    mid_index1 = [int(nda1.shape[i] / 2) for i in range(3)] # numpy order
    mid_coord1 = img1.TransformIndexToPhysicalPoint([mid_index1[2], mid_index1[1], mid_index1[0]]) # sitk order
    mid_index2 = img2.TransformPhysicalPointToIndex(mid_coord1) # sitk order
    mid_index2 = [mid_index2[2-i] for i in range(3)] # numpy order
    #print(mid_index1)
    #print(mid_index2)

    save_paths1, save_paths2 = [], []
    for i in range(len(img_paths1)):
        nda = sitk.GetArrayFromImage(sitk.ReadImage(img_paths1[i]))
        large_nda = np.zeros([200, 300, 300]) # NOTE shold be large enough to cover all nda
        large_nda[100 - mid_index1[0] : 100 - mid_index1[0] + nda.shape[0], 
                150 - mid_index1[1] : 150 - mid_index1[1] + nda.shape[1], 
                150 - mid_index1[2] : 150 - mid_index1[2] + nda.shape[2]] = nda
        large_img = nda2img(large_nda, [0., 0., 0.], spacing1, direction1)
        sitk.WriteImage(large_img, os.path.join(os.path.dirname(img_paths1[i]), os.path.basename(img_paths1[i])))
        save_paths1.append(os.path.join(os.path.dirname(img_paths1[i]), os.path.basename(img_paths1[i])))
    for i in range(len(img_paths2)):
        nda = sitk.GetArrayFromImage(sitk.ReadImage(img_paths2[i]))
        large_nda = np.zeros([200, 300, 300])
        #print(nda.shape)
        #print(mid_index2)
        large_nda[100 - mid_index2[0] : 100 - mid_index2[0] + nda.shape[0], 
                150 - mid_index2[1] : 150 - mid_index2[1] + nda.shape[1], 
                150 - mid_index2[2] : 150 - mid_index2[2] + nda.shape[2]] = nda
        large_img= nda2img(large_nda, [0., 0., 0.], spacing2, direction2)
        sitk.WriteImage(large_img, os.path.join(os.path.dirname(img_paths2[i]), os.path.basename(img_paths2[i])))
        save_paths2.append(os.path.join(os.path.dirname(img_paths2[i]), os.path.basename(img_paths2[i])))
    
    return save_paths1, save_paths2


def match_size(img, target_img):
    '''
    img: img to-match
    target_img: img to-be-matched
    NOTE: reverse spacing order between Numpy array and Simple ITK image: array.shape[0] -- image.GetSize()[-1]
    '''
    nda = sitk.GetArrayFromImage(img)
    origin, spacing, direction = img.GetOrigin(), img.GetSpacing(), img.GetDirection() # sitk order
    end_coord = [origin[i] + (img.GetSize()[i] - 1) * spacing[2-i] for i in range(3)] # real-world coordinates of end position of to-match-img (sitk order)

    target_nda = sitk.GetArrayFromImage(target_img)
    target_origin, target_spacing, target_direction = target_img.GetOrigin(), target_img.GetSpacing(), target_img.GetDirection()
    assert spacing == target_spacing
    

    pos_origin = [np.rint((origin[i] - target_origin[i]) / spacing[2-i]) for i in range(3)] # position of origin of to-match-img in target img (sitk order)
    #print(pos_origin)
    # If origin of to-match-img not covered by target img: cutoff #
    cutoff_start = np.array([0, 0, 0]).astype(int) # (Numpy order)
    if pos_origin[2] < 0: 
        cutoff_start[0] = int(- pos_origin[2])
        pos_origin[2] = 0
    if pos_origin[1] < 0:
        cutoff_start[1] = int(- pos_origin[1])
        pos_origin[1] = 0
    if pos_origin[0] < 0:
        cutoff_start[2] = int(- pos_origin[0])
        pos_origin[0] = 0
    #print(pos_origin) 
    #print(cutoff_start) 

    pos_end = [np.rint((end_coord[i] - target_origin[i]) / spacing[2-i]) for i in range(3)] # position of end of to-match-img in target img (sitk order)
    #print(pos_end)
    # If end of to-match-img not covered by target img: cutoff #
    cutoff_end = np.array([nda.shape[0], nda.shape[1], nda.shape[2]]).astype(int) # (Numpy order)
    if pos_end[2] > target_nda.shape[0] - 1:
        cutoff_end[0] = int(np.rint(target_nda.shape[0] - pos_end[2] - 1))
        pos_end[2] = target_nda.shape[0] - 1
    if pos_end[1] > target_nda.shape[1] - 1:
        cutoff_end[1] = int(np.rint(target_nda.shape[1] - pos_end[1] - 1))
        pos_end[1] = target_nda.shape[1] - 1
    if pos_end[0] > target_nda.shape[2] - 1:
        cutoff_end[2] = int(np.rint(target_nda.shape[2] - pos_end[0] - 1))
        pos_end[0] = target_nda.shape[2] - 1
    #print(pos_end) 
    #print(cutoff_end) 

    new_nda = np.zeros(target_nda.shape)
    new_nda[int(pos_origin[2]):int(pos_end[2] + 1), \
            int(pos_origin[1]):int(pos_end[1] + 1), \
            int(pos_origin[0]):int(pos_end[0] + 1)] = nda[cutoff_start[0] : cutoff_end[0], cutoff_start[1] : cutoff_end[1], cutoff_start[2] : cutoff_end[2]]

    new_origin = [origin[i] + spacing[i] * (pos_origin[i] + cutoff_start[2-i]) for i in range(3)]

    new_img = sitk.GetImageFromArray(new_nda, isVector = len(new_nda.shape) > 3)
    new_img.SetOrigin(new_origin)
    new_img.SetSpacing(spacing)
    new_img.SetDirection(direction)

    return new_img, pos_origin, pos_end, cutoff_start, cutoff_end


def reset_sitk(img, new_origin, new_spacing, new_direction):
    nda = sitk.GetArrayFromImage(img)
    new_img = sitk.GetImageFromArray(nda, isVector = len(nda.shape) > 3)
    new_img.SetOrigin(new_origin)
    new_img.SetSpacing(new_spacing)
    new_img.SetDirection(new_direction)
    return new_img

