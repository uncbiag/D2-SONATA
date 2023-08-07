import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch 
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from dipy.segment.mask import median_otsu

from utils import * 


def get_mip(nda, show = True, axis = 0, save_path = None):
    nda_mip = np.max(nda, axis = axis)
    # Normlize for RBG visualization #
    nda_mip = (nda_mip - np.min(nda_mip)) / (np.max(nda_mip) - np.min(nda_mip))
    plt.axis('off')
    plt.tight_layout()
    if show:
        plt.imshow(nda_mip)
    if save_path:
        plt.savefig(save_path)
    return nda_mip

def keep_largest_volumn(nda):
    label_im, nb_labels = ndimage.label(nda)
    sizes = ndimage.sum(nda, label_im, range(nb_labels + 1))
    mask = sizes > np.max(sizes) - 1
    nda = mask[label_im].astype(float)
    return nda
    
def rm_by_volume(nda, noise_size_tol):
    label_im, nb_labels = ndimage.label(nda)
    sizes = ndimage.sum(nda, label_im, range(nb_labels + 1))
    mask = sizes > noise_size_tol
    nda = mask[label_im].astype(float)
    return nda

def rm_by_slice(nda, noise_size_tol):
    for s in range(len(nda)):
        label_im, nb_labels = ndimage.label(nda[s])
        sizes = ndimage.sum(nda[s], label_im, range(nb_labels + 1))
        mask = sizes > noise_size_tol
        nda[s] = mask[label_im].astype(float)
    return nda

def rm_by_row(nda, noise_size_tol):
    for r in range(len(nda[0])):
        label_im, nb_labels = ndimage.label(nda[:, r])
        sizes = ndimage.sum(nda[:, r], label_im, range(nb_labels + 1))
        mask = sizes > noise_size_tol
        nda[:, r] = mask[label_im].astype(float)
    return nda

def rm_by_column(nda, noise_size_tol):
    for c in range(len(nda[0, 0])):
        label_im, nb_labels = ndimage.label(nda[:, :, c])
        sizes = ndimage.sum(nda[:, :, c], label_im, range(nb_labels + 1))
        mask = sizes > noise_size_tol
        nda[:, :, c] = mask[label_im].astype(float)
    return nda

def get_mip(nda, show = True, axis = 0, save_path = None):
    nda_mip = np.max(nda, axis = axis)
    # Normlize for RBG visualization #
    nda_mip = (nda_mip - np.min(nda_mip)) / (np.max(nda_mip) - np.min(nda_mip))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(nda_mip)
    if save_path:
        plt.savefig(save_path)
    return nda_mip


def get_mrp_paths(case_fld, postfix = '_BS_EvalSpa_BATtoTTD.nii'):

    for module in os.listdir(case_fld):
        if "PWI" in module and not module.startswith('.'):
            # Use rotated version if available
            is_rotated = False
            for file in os.listdir(os.path.join(case_fld, module)):
                if 'rotated' in file:
                    is_rotated = True
            if is_rotated:
                for file in os.listdir(os.path.join(case_fld, module)):
                    if 'CTC_Axial_cropped_filtered_filtered_rotated' in file and file.endswith(postfix):
                        CTC_path = os.path.join(case_fld, module, file)
            else:
                for file in os.listdir(os.path.join(case_fld, module)):
                    if file.endswith(postfix):
                        CTC_path = os.path.join(case_fld, module, file)
                #CTC_path = os.path.join(case_fld, "%s/CTC_Axial_cropped_filtered_filtered%s" % (module, postfix))
    assert CTC_path
    return CTC_path

def get_ctp_paths(case_fld, postfix = '_BS_EvalSpa_BATtoTTD.nii'):

    for module in os.listdir(case_fld):
        if "PWI" in module and not module.startswith('.'):
            for file in os.listdir(os.path.join(case_fld, module)):
                #print(file)
                if file.endswith(postfix):
                    CTP_path = os.path.join(case_fld, module, file)
    assert CTP_path
    return CTP_path

def get_times(ctc_nda, bat_threshold = 0.05, show_info = True):
    '''
    ctc_nda: (s, r, c, t)
    Calculate the CTP bolus arrival time (bat) and corresponding S0: averaged over signals before bat
    return: s0 # (n_slice, n_row, n_column)
    '''
    nT = ctc_nda.shape[-1]
    ctc_avg = np.zeros(nT)
    for t in range(nT):
        ctc_avg[t] = (ctc_nda[..., t]).mean()
    ttp = np.argmax(ctc_avg)
    ttd = np.argmin(ctc_avg[ttp:]) + ttp
    threshold = bat_threshold * (np.amax(ctc_avg) - np.amin(ctc_avg)) 
    flag = True
    bat  = 1
    while flag:
        if ctc_avg[bat] - ctc_avg[bat - 1] >= threshold and ctc_avg[bat + 1] > ctc_avg[bat]:
            flag = False
        else:
            bat += 1
        if bat == nT - 1:
            flag = False
    
    if show_info:
        print(ctc_nda.shape)
        print('Bolus arrival time (start from 0):', bat - 1)
        print('Bolus concentration time-to-peak (start from 0):', ttp)
        print('Bolus concentration time-to-drain (start from 0):', ttd)
    return bat, ttp, ttd

def mrp_s0(signal_nda, s0_threshold = 0.05):
    '''
    Calculate the MRP bolus arrival time (bat) and corresponding S0: averaged over signals before bat
    return: s0 # (n_slice, n_row, n_column)
    '''
    nT = signal_nda.shape[-1]
    sig_avg = np.zeros(nT)
    for t in range(nT):
        sig_avg[t] = (signal_nda[..., t]).mean()
    ttp = np.argmin(sig_avg)
    ttd = np.argmax(sig_avg[ttp:]) + ttp
    print('Bolus concentration time-to-peak (start from 0):', ttp)
    flag = True
    bat  = 0
    while flag:
        s0_avg = np.mean(sig_avg[:bat + 1])
        if abs(s0_avg - sig_avg[bat + 1]) / s0_avg < s0_threshold:
            bat += 1
        else:
            flag = False
            bat -= 1
        if bat == signal_nda.shape[-1] - 1:
            flag = False
            bat -= 1
    print('Bolus arrival time (start from 0):', bat)
    s0 = np.mean(signal_nda[..., :bat], axis = 3) # time dimension == 3
    
    return s0, bat, ttp, ttd

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

cal_s0 = {
    'CTP': ctp_s0,
    'MRP': mrp_s0
}

def signal2ctc(img_type, signal_nda, k_ct = 1., k_mr = 1., TE = 0.025, TR = 1.55, s0_threshold = 0.05):
    
    if img_type == 'MRP':
        # Convert signal of those voxels that are negative to 1
        signal_nda[signal_nda <= 0] = 1.0
    s0, bat, ttp, ttd = cal_s0[img_type](signal_nda, s0_threshold)
    ctc_nda = np.zeros(signal_nda.shape)
    for t in range(signal_nda.shape[-1]):
        if img_type == 'CTP':
            ctc_nda[..., t] = k_ct * (signal_nda[..., t] - s0)
        if img_type == 'MRP':
            ctc_nda[..., t] = - k_mr/TE * np.log(signal_nda[..., t] / s0)
    # Check computed CTC: should have no NaN value
    if np.any(np.isnan(ctc_nda)):
        raise ValueError('Computed CTC contains NaN value, double check!')
    return ctc_nda, bat, ttp, ttd


def crop_nda(nda, tol = 0, crop_range_lst = None): # nda: volumes
    # Mask of non-black pixels (assuming image has a single channel).
    mask = nda > tol
    dim = len(nda.shape)
    if crop_range_lst is None:
        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)
        # Bounding box of non-black pixels.
        if dim == 2:
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
            return nda[x0 : x1, y0 : y1], [x0, y0], [x1, y1]
        elif dim > 2:
            x0, y0, z0 = coords.min(axis=0)
            x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
            # Check the the bounding box.
            #print('    Cropping Slice [%d, %d)' % (x0, x1))
            #print('    Cropping Row [%d, %d)' % (y0, y1))
            #print('    Cropping Column [%d, %d)' % (z0, z1))
            return nda[x0 : x1, y0 : y1, z0 : z1], [[x0, y0, z0], [x1, y1, z1]]
        else:
            raise NotImplementedError('Unsupported dimension for cropping')
    else:
        if dim == 2:
            [[x0, y0], [x1, y1]] = crop_range_lst
            return nda[x0 : x1, y0 : y1], [x0, y0], [x1, y1]
        elif dim > 2:
            [[x0, y0, z0], [x1, y1, z1]] = crop_range_lst
            return nda[x0 : x1, y0 : y1, z0 : z1], [[x0, y0, z0], [x1, y1, z1]]


def cropping(img, tol = 0, crop_range_lst = None, save_path = None):
    
    '''
    img: sitk readable image
    (if) crop_range_lst: [[x0, y0, z0], [x1, y1, z1]]
    '''

    orig_nda = sitk.GetArrayFromImage(img)
    if len(orig_nda.shape) > 3: # 4D data: last axis (t=0) as time dimension
        nda = orig_nda[..., 0]
    else:
        nda = np.copy(orig_nda)
    
    if crop_range_lst is None:
        
        # Mask of non-black pixels (assuming image has a single channel).
        mask = nda > tol

        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)
        
        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top
        
        # Check the the bounding box #
        #print('    Cropping Slice [%d, %d)' % (x0, x1))
        #print('    Cropping Row [%d, %d)' % (y0, y1))
        #print('    Cropping Column [%d, %d)' % (z0, z1))

    else:
        
        [[x0, y0, z0], [x1, y1, z1]] = crop_range_lst

    cropped_nda = orig_nda[x0 : x1, y0 : y1, z0 : z1]
    '''new_origin = [img.GetOrigin()[0] + img.GetSpacing()[0] * z0,\
        img.GetOrigin()[1] + img.GetSpacing()[1] * y0,\
            img.GetOrigin()[2] + img.GetSpacing()[2] * x0]  # numpy reverse to sitk'''
    cropped_img = sitk.GetImageFromArray(cropped_nda, isVector = len(orig_nda.shape) > 3)
    #cropped_img.SetOrigin(new_origin)
    cropped_img.SetOrigin(img.GetOrigin())
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    if save_path:
        sitk.WriteImage(cropped_img, save_path)

    return cropped_img, [[x0, y0, z0], [x1, y1, z1]] #, new_origin


def resample_img(itk_image, out_spacing = [2.0, 2.0, 2.0], is_label = False):
    
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def extract_brain_region(nda, tol = 0., get_masked_nda = False):
    '''
    tol: tolerance for background value
    '''
    nda[nda < tol] = 0.
    _, brain_mask = median_otsu(nda, median_radius=2, numpass=1, autocrop=False)
    label_im, nb_labels = ndimage.label(brain_mask)
    sizes = ndimage.sum(brain_mask, label_im, range(nb_labels + 1))
    mask = sizes >= np.max(sizes) # NOTE: Select the largest  component 
    brain_mask = mask[label_im].astype(float)
    brain_mask = ndimage.binary_fill_holes(brain_mask).astype(float) # NOTE fill in holes
    # NOTE: Crop all-zero planes #
    brain_mask, crop_start, crop_end = crop_nda(brain_mask, tol = 0.)

    for s in range(brain_mask.shape[0]):
        brain_mask[s] = ndimage.binary_fill_holes(brain_mask[s]) # NOTE fill in holes
        label_im, nb_labels = ndimage.label(brain_mask[s])
        sizes = ndimage.sum(brain_mask[s], label_im, range(nb_labels + 1))
        mask = sizes >= np.max(sizes) # NOTE: Keep only the largest component per slice
        brain_mask[s] = mask[label_im].astype(float)
    brain_mask, crop_start_s, crop_end_s = crop_nda(brain_mask, tol = 0.) # NOTE: Crop unnecessary rows, and columes #
    crop_end = [crop_start[i] + crop_end_s[i] for i in range(3)]
    crop_start = [crop_start[i] + crop_start_s[i] for i in range(3)]

    for r in range(brain_mask.shape[1]):
        brain_mask[:, r] = ndimage.binary_fill_holes(brain_mask[:, r]) # NOTE fill in holes
        label_im, nb_labels = ndimage.label(brain_mask[:, r])
        sizes = ndimage.sum(brain_mask[:, r], label_im, range(nb_labels + 1))
        mask = sizes >= np.max(sizes) # NOTE: Keep only the largest component per slice
        brain_mask[:, r] = mask[label_im].astype(float)
    brain_mask, crop_start_r, crop_end_r = crop_nda(brain_mask, tol = 0.) # NOTE: Crop unnecessary slices, and columes #
    crop_end = [crop_start[i] + crop_end_r[i] for i in range(3)]
    crop_start = [crop_start[i] + crop_start_r[i] for i in range(3)]

    for c in range(brain_mask.shape[2]):
        brain_mask[:, :, c] = ndimage.binary_fill_holes(brain_mask[:, :, c]) # NOTE fill in holes
        label_im, nb_labels = ndimage.label(brain_mask[:, :, c])
        sizes = ndimage.sum(brain_mask[:, :, c], label_im, range(nb_labels + 1))
        mask = sizes >= np.max(sizes) # NOTE: Keep only the largest component per slice
        brain_mask[:, :, c] = mask[label_im].astype(float)
    brain_mask, crop_start_c, crop_end_c = crop_nda(brain_mask, tol = 0.) # NOTE: Crop unnecessary slices, and rows #
    crop_end = [crop_start[i] + crop_end_c[i] for i in range(3)]
    crop_start = [crop_start[i] + crop_start_c[i] for i in range(3)]

    if get_masked_nda:
        return brain_mask, nda * brain_mask
    else:
        return brain_mask, crop_start, crop_end


interpolator = {
    'Linear': sitk.sitkLinear,
    'BSpline': sitk.sitkBSpline,
    'Gaussian': sitk.sitkGaussian,
    'HWS': sitk.sitkHammingWindowedSinc,
    'BWS': sitk.sitkBlackmanWindowedSinc,
    'CWS': sitk.sitkCosineWindowedSinc,
    'WWS': sitk.sitkWelchWindowedSinc,
    'LWS': sitk.sitkLanczosWindowedSinc,
}
def uptiming(img_path, up_scale, interp_method = 'Linear', isNumpy = False, origin = None, spacing = None):
    if not isNumpy:
        img = sitk.ReadImage(img_path)
        nda = sitk.GetArrayFromImage(img) # (s, r, c, t)
    else:
        nda = np.load(img_path)
        print(nda.shape)
    new_nda = []
    for s in range(nda.shape[0]):
        new_nda.append(sitk.GetArrayFromImage(sitk.Expand(sitk.GetImageFromArray(nda[s]), [up_scale, 1, 1], interpolator[interp_method])))
    
    new_img = sitk.GetImageFromArray(np.array(new_nda), isVector = True)
    
    if not isNumpy:
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetDirection(img.GetDirection())
    else:
        new_img.SetOrigin(origin)
        new_img.SetSpacing(spacing)

    #print('Saved up-timed img in %s_UpTime(%s).nii' % (img_path[:-4], up_scale))
    #sitk.WriteImage(new_img, '%s_UpTime(%s).nii' % (img_path[:-4], up_scale))

    #return '%s_UpTime(%s).nii' % (img_path[:-4], up_scale)
    return new_img

    
def cutoff_percentile(image, mask = None, percentile_lower = 0., percentile_upper = 99.8):
    if mask is None:
        mask = image != 0.
    if percentile_upper is None:
        percentile_upper = 100. - percentile_lower
    res = np.copy(image)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower #cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper #cut_off_upper
    return res



class GaussianSmoother(nn.Module):
    '''Fixed layer'''
    def __init__(self, kernel_size = 3, sigma = 1):
        super(GaussianSmoother, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.conv = nn.Conv3d(1, 1, self.kernel_size, stride = 1, padding = int(kernel_size / 2), bias = None)
        self.weights_init()

    def forward(self, X):
        # X: (slc, row, col)
        return self.conv((X.unsqueeze(0).unsqueeze(0)))[0, 0]
    
    def weights_init(self):
        w = np.zeros((self.kernel_size, self.kernel_size, self.kernel_size))
        center = int(self.kernel_size / 2)
        w[center, center, center] = 1.
        k = gaussian_filter(w, sigma = self.sigma)

        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False
           