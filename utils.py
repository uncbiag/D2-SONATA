from itertools import starmap
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shutil import copyfile, move
import torch
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter

'''
Utility functions:

# For spectral decomposition and initializations:
henaff_init_, cayley_init_, creat_diag_, get_L, get_U

# For PDENet initialization, learning and postprocessing:
load_param, load_spectral_param,
round_up, round_down, Upwind, SetBC,
decompose_tensor, save_sitk, stability_threshold, save_params

# For finding nearest PSD:
_getAplus, _getPs, _getPu, IdentityTensor, nearPD

# Others:
make_dir, get_batch_C, gradient_lp, gradient, nda_save_img, avg_grad, get_lr, RunningAverageMeter

'''

def D_measures(L_tensor, U_tensor, save_fld, origin = None, spacing = None, direction = None, prefix = ''):
    '''
    L_tensor: eigenvalues in (NOTE) "DESCENDING" order (last dim)
    U_tensor: corresponding eigenvectors (last_dim)
    '''
    eva1, eva2, eva3 = L_tensor[..., 0], L_tensor[..., 1], L_tensor[..., 2]
    #evec1, evec2, evec3 = U_tensor[..., 0], U_tensor[..., 1], U_tensor[..., 2]
    
    trace = torch.sum(L_tensor, dim = -1)

    save_sitk(trace, os.path.join(save_fld, prefix + 'Trace.mha'), origin = origin, spacing = spacing, direction = direction)
    if 'orig' in prefix or 'delta' in prefix:
        return
        
    fa = torch.sqrt(0.5 * ((eva1 - eva2) ** 2 + \
        (eva2 - eva3) ** 2 + (eva3 - eva1) ** 2) \
            / (torch.sum(L_tensor ** 2, dim = -1) + 1e-14)) # + all_zero # (s, r, c)

    d_color_direction = abs(U_tensor[..., 0]) # (s, r, c, 3)

    cbo_fa = fa[..., None] * d_color_direction 

    save_sitk(fa, os.path.join(save_fld, prefix + 'FA.mha'), origin = origin, spacing = spacing, direction = direction)
    save_sitk(d_color_direction, os.path.join(save_fld, prefix + 'D_Color_Direction.mha'), origin = origin, spacing = spacing, direction = direction)
    save_sitk(cbo_fa, os.path.join(save_fld, prefix + 'D_Color_Direction_FA_masked.mha'), origin = origin, spacing = spacing, direction = direction)
    return

def V_measures(V_tensor, save_fld, origin = None, spacing = None, direction = None, prefix = ''):
    '''
    V_tensor: velocity vector field (last_dim)
    '''
    norm_v = torch.sqrt(torch.sum(V_tensor ** 2, dim = -1)) 
    save_sitk(norm_v, os.path.join(save_fld, prefix + 'Norm_V.mha'), origin = origin, spacing = spacing, direction = direction) 
    save_sitk(abs(V_tensor), os.path.join(save_fld, prefix + 'Abs_V.mha'), origin = origin, spacing = spacing, direction = direction)
    return

def apply_BC_2D(X, BC = None, i_t = 0, BC_type = 'dirichlet', batched = True):
    if 'dirichlet' in BC_type or 'cauchy' in BC_type:
        return set_BC_2D(X, BC, i_t, batched = batched)
    elif 'source' in BC_type:
        return set_dBC_2D(X, BC, i_t, batched = batched)

def apply_BC_3D(X, BC = None, i_t = 0, BC_type = 'dirichlet', batched = True):
    if 'dirichlet' in BC_type or 'cauchy' in BC_type:
        return set_BC_3D(X, BC, i_t, batched = batched)
    elif 'source' in BC_type:
        return set_dBC_3D(X, BC, i_t, batched = batched)
def set_BC_2D(X, BCs, i_t, batched = True): # X: (n_batch, spatial_size); BCs: list (n_batch, BC_size, rest_dim_remain)
    '''if not batched:
        X = X.unsqueeze(0)'''
    # X: (n_batch, r, c)
    # BC: [[BC0_0, BC0, 1], [BC1_0, BC1_1]]: each: ((n_batch), nT, BC_size, rest_dim_remain)
    BC_size = BCs[0][0].size(1)
    if batched:
        X[:, : BC_size] = BCs[0][0][:,i_t]
        X[:, - BC_size :] = BCs[0][1][:,i_t]
        X[:, :, : BC_size] = BCs[1][0][:,i_t]
        X[:, :, - BC_size :] = BCs[1][1][:,i_t]
    else:
        X[: BC_size] = BCs[0][0][i_t]
        X[- BC_size :] = BCs[0][1][i_t]
        X[:, : BC_size] = BCs[1][0][i_t]
        X[:, - BC_size :] = BCs[1][1][i_t]
    del BCs
    return X 
def set_BC_3D(X, BCs, i_t, batched = True): # X: (n_batch, spatial_size); BCs: list: (n_batch, 6, BC_size, rest_dim_remain)
    BC_size = BCs[0][0].size(1)
    #print(BC_size)
    #print(BCs[0][0].size(), BCs[0][1].size())
    #print(BCs[1][0].size(), BCs[1][1].size())
    #print(BCs[2][0].size(), BCs[2][1].size())
    # X: (n_batch, s, r, c)
    # BC: [[BC0_0, BC0, 1], [BC1_0, BC1_1]], [BC2_0, BC2_1]]: each: ((n_batch), nT, BC_size, rest_dim_remain)
    if batched:
        X[:, : BC_size] = BCs[0][0][:, i_t]
        X[:, - BC_size :] = BCs[0][1][:,i_t]
        X[:, :, : BC_size] = BCs[1][0][:,i_t]
        X[:, :, - BC_size :] = BCs[1][1][:,i_t]
        X[:, :, :, : BC_size] = BCs[2][0][:,i_t]
        X[:, :, :, - BC_size :] = BCs[2][1][:,i_t]
    else:
        X[: BC_size] = BCs[0][0][i_t]
        X[- BC_size :] = BCs[0][1][i_t]
        X[:, : BC_size] = BCs[1][0][i_t]
        X[:, - BC_size :] = BCs[1][1][i_t]
        X[:, :, : BC_size] = BCs[2][0][i_t]
        X[:, :, - BC_size :] = BCs[2][1][i_t]
    del BCs
    return X

def set_dBC_2D(X, dBCs, i_t, batched = True): # X: (n_batch, spatial_size); BCs: list (n_batch, BC_size, rest_dim_remain)
    BC_size = dBCs[0][0].size(1)
    X[:, : dBC_size] = dBCs[0][0]
    X[:, - dBC_size :] = dBCs[0][1]
    X[:, :, : dBC_size] = dBCs[1][0]
    X[:, :, - dBC_size :] = dBCs[1][1]
    del dBCs
    return X
def set_dBC_3D(X, dBCs, i_t, batched = True): # X: (n_batch, spatial_size); BCs: (batch, 6, BC_shape, data_dim, dta_dim)
    BC_size = dBCs[0][0].size(1)
    X[:, : dBC_size] = dBCs[0][0]
    X[:, - dBC_size :] = dBCs[0][1]
    X[:, :, : dBC_size] = dBCs[1][0]
    X[:, :, - dBC_size :] = dBCs[1][1]
    X[:, :, :, : dBC_size] = dBCs[2][0]
    X[:, :, :, - dBC_size :] = dBCs[2][1]
    del dBCs
    return X


# Old: For regular size patch # 
''' 
def set_BC_2D(X, BCs): # X: (n_batch, spatial_size); BCs: (batch, 4, BC_shape, data_dim)
    BC_size = BCs.size(2)
    X[:, : BC_size] = BCs[:, 0]
    X[:, - BC_size :] = BCs[:, 1]
    X[:, :, : BC_size] = BCs[:, 2].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    X[:, :, - BC_size :] = BCs[:, 3].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    del BCs
    return X
def set_BC_3D(X, BCs): # X: (n_batch, spatial_size); BCs: (batch, 6, BC_shape, data_dim, dta_dim)
    BC_size = BCs.size(2)
    X[:, : BC_size] = BCs[:, 0]
    X[:, - BC_size :] = BCs[:, 1]
    X[:, :, : BC_size] = BCs[:, 2].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, - BC_size :] = BCs[:, 3].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, :, : BC_size] = BCs[:, 4].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    X[:, :, :, - BC_size :] = BCs[:, 5].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    del BCs
    return X'''

''' X[t] = X[t] + dBC[t] (dBC[t] = BC[t+1] - BC[t]) '''
'''def add_dBC_2D(X, dBCs): # X: (n_batch, spatial_size); BCs: (batch, 4, BC_shape, data_dim)
    BC_size = dBCs.size(2)
    X[:, : BC_size] += dBCs[:, 0]
    X[:, - BC_size :] += dBCs[:, 1]
    X[:, :, : BC_size] += dBCs[:, 2].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    X[:, :, - BC_size :] += dBCs[:, 3].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    del dBCs
    return X
def add_dBC_3D(X, dBCs): # X: (n_batch, spatial_size); BCs: (batch, 6, BC_shape, data_dim, dta_dim)
    BC_size = dBCs.size(2)
    X[:, : BC_size] += dBCs[:, 0]
    X[:, - BC_size :] += dBCs[:, 1]
    X[:, :, : BC_size] += dBCs[:, 2].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, - BC_size :] += dBCs[:, 3].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, :, : BC_size] += dBCs[:, 4].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    X[:, :, :, - BC_size :] += dBCs[:, 5].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    del dBCs
    return X'''


def move_file(img_path, to_move_path):
    move(img_path, to_move_path)
    return to_move_path

def copy_file(img_path, to_copy_path):
    copyfile(img_path, to_copy_path)
    return to_copy_path

def move_files(img_paths, to_move_paths):
    for i in range(len(img_paths)):
        move(img_paths[i], to_move_paths[i])
    return to_move_paths

def copy_files(img_paths, to_copy_paths):
    for i in range(len(img_paths)):
        copy_file(img_paths[i], to_copy_paths[i])
    return to_copy_paths


def remove_paths(paths):
    for i in range(len(paths)):
        if os.path.isfile(paths[i]):
            os.remove(paths[i])
    return

def numpy_percentile(nda, q, bg = 0.):
    """
    Return the q-th percentile of the flattened input tensor's data.
    
    :param nda: Input Numpy array.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :param bg: background values to be ignored.
    :return: Resulting value (scalar).
    """
    return np.percentile(nda[nda != bg].ravel(), q)

def tensor_percentile(t, q, bg = 0.):
    """
    Return the q-th percentile of the flattened input tensor's data.
    
    :param t: Input Torch tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :param bg: background values to be ignored.
    :return: Resulting value (scalar).
    """
    t = t[t != bg].flatten()
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def img2nda(img_path, save_path = None):
    img = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(img)
    if save_path:
        np.save(save_path, nda)
    return nda, img.GetOrigin(), img.GetSpacing(), img.GetDirection()

def nda2img(nda, origin = None, spacing = None, direction = None, save_path = None, isVector = None):
    isVector = isVector if isVector is not None else len(nda.shape) > 3
    img = sitk.GetImageFromArray(nda, isVector = isVector)
    if origin:
        img.SetOrigin(origin)
    if spacing:
        img.SetSpacing(spacing)
    if direction:
        img.SetDirection(direction)
    if save_path:
        sitk.WriteImage(img, save_path)
    return img


def cutoff_percentile(image, mask = None, percentile_lower = 0., percentile_upper = 99.8, lower_val = None, upper_val = None):
    if mask is None:
        mask = image != image[0, 0, 0]
    if percentile_upper is None:
        percentile_upper = 100. - percentile_lower
    res = np.copy(image)
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower) 
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res[(res < cut_off_lower) & (mask != 0)] = 0 if lower_val else cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = 0 if upper_val else cut_off_upper
    return res

def save_sitk(p, filename, origin = None, spacing = None, direction = None, toCut = False, percentile_lower = 0., percentile_upper = 99.8, isVector = None):
    def cutoff_percentile(image, basename, mask = None, percentile_lower = 0., percentile_upper = 99.8):
        if mask is None:
            mask = image != image[0, 0, 0]
        if percentile_upper is None:
            percentile_upper = 100. - percentile_lower
        res = np.copy(image)
        cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
        if 'D' in basename or 'Abs' in basename:
            res[(res > cut_off_upper) & (mask != 0)] = 0. #cut_off_upper
        elif 'V' in basename and 'Abs' not in basename:
            cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
            res[(res < cut_off_lower) & (mask != 0)] = 0. #cut_off_lower
            res[(res > cut_off_upper) & (mask != 0)] = 0. #cut_off_upper
        return res

    if not isinstance(p, (np.ndarray)):
        if p.requires_grad:
            p = p.detach()
        if p.device != 'cpu': 
            p = p.cpu()
        p = p.numpy()

    isVector = isVector if isVector is not None else len(p.shape) > 3

    if toCut: 
        p_cut = cutoff_percentile(p, os.path.basename(filename), percentile_lower = percentile_lower, percentile_upper = percentile_upper)
        p_cut = sitk.GetImageFromArray(p_cut, isVector = len(p.shape) > 3)
        p_cut.SetOrigin(origin)
        p_cut.SetSpacing(spacing)
        sitk.WriteImage(p_cut, '%s_cutted.mha' % filename[:-4])
    p = sitk.GetImageFromArray(p, isVector = len(p.shape) > 3)
    if origin:
        p.SetOrigin(origin)
    if spacing:
        p.SetSpacing(spacing)
    if direction:
        p.SetDirection(direction)
    sitk.WriteImage(p, filename)
    return filename 
     
def get_mip(nda, show = True, axis = 0, save_path = None):
    nda_mip = np.max(nda, axis = axis)
    nda_mip = nda_mip / np.max(nda_mip)
    
    # Normlize for RBG visualization #
    nda_mip = (nda_mip - np.min(nda_mip)) / (np.max(nda_mip) - np.min(nda_mip))
    plt.axis('off')
    plt.tight_layout()
    if show:
        plt.imshow(nda_mip)
    if save_path:
        if show and save_path.endswith('png'):
            plt.savefig(save_path)
        else:
            assert save_path.endswith('nii') or save_path.endswith('mha')
            sitk.GetImageFromArray()
    return nda_mip

def plot_quivers(nda1, nda2, save_path = None):
	nda1 /= 10
	nda2 /= 10
	'''
	nda1, nda2: numpy array (2, r, c), quivers plot
	'''
	X0, X1, dX = -4, 4, .25
	Y0, Y1, dY = -4, 4, .25
	X0, X1, dX = -8, 8, .25
	Y0, Y1, dY = -8, 8, .25
	assert nda1.shape == nda2.shape # (2, r, c)
	# !NOTE!: In grid, X -- 2nd dim, Y -- 1st dim
	X, Y = np.meshgrid(np.arange(X0, X1, dX), np.arange(Y0, Y1, dY))
	#X, Y = np.meshgrid(np.arange(nda1.shape[2]), np.arange(nda1.shape[1]))
	fig = plt.figure(figsize = (20, 10))

	ax1 = fig.add_subplot(1, 2, 1, xticks = [], yticks = []) 
	Q = ax1.quiver(X, Y, nda1[0], nda1[1], np.hypot(nda1[0], nda1[1]), units='x', pivot='tip', width=0.02, scale=1)
	#Q = ax1.quiver(X, Y, nda1[1], nda1[0], np.hypot(nda1[1], nda1[0]), units='x', pivot='tip', width=0.02, scale=1)
	#qk = ax1.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')
	ax1.set_title('GT', color = 'green', fontsize = 26)

	ax2 = fig.add_subplot(1, 2, 2, xticks = [], yticks = [])
	Q = ax2.quiver(X, Y, nda2[0], nda2[1], np.hypot(nda2[0], nda2[1]), units='x', pivot='tip', width=0.02, scale=1)
	#Q = ax2.quiver(X, Y, nda2[1], nda2[0], np.hypot(nda2[1], nda2[0]), units='x', pivot='tip', width=0.02, scale=1)
	#qk = ax2.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')
	ax2.set_title('PD', color = 'blue', fontsize = 26)
	if save_path:
		plt.savefig(save_path)
	plt.close()
	return fig

def plot_imgs(nda1, nda2, save_path = None):
	'''
	nda1, nda2: numpy array (r, c), 2D plot
	'''
	assert nda1.shape == nda2.shape # (r, c)
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.imshow(nda1)
	ax1.set_title('GT', color = 'green', fontsize = 26)
	ax2 = fig.add_subplot(1, 2, 2)
	ax2.imshow(nda2)
	ax2.set_title('PD', color = 'blue', fontsize = 26)
	plt.close()
	return fig

################################################################

def clebsch_3D(Va, Vb, batched = False, delta_lst = [1., 1., 1.]):
    '''
    input: (batch, s, r, c)
    '''
    device = Va.device
    #dVa = gradient_c(Va, batched = batched, delta_lst = delta_lst)
    #dVb = gradient_c(Vb, batched = batched, delta_lst = delta_lst)
    dVa = gradient_c(Va, batched = batched, delta_lst = [1., 1., 1.])
    dVb = gradient_c(Vb, batched = batched, delta_lst = [1., 1., 1.])
    Va_x, Va_y, Va_z = dVa[..., 0], dVa[..., 1], dVa[..., 2]
    Vb_x, Vb_y, Vb_z = dVb[..., 0], dVb[..., 1], dVb[..., 2]
    Vx = Va_y * Vb_z - Va_z * Vb_y
    Vy = Va_z * Vb_x - Va_x * Vb_z
    Vz = Va_x * Vb_y - Va_y * Vb_x
    return Vx, Vy, Vz

def HHD_3D(Phi_a, Phi_b, Phi_c, H, batched = False, delta_lst = [1., 1., 1.]):
    '''
    input: (batch, s, r, c)
    Refs:
    http://www.sci.utah.edu/~hbhatia/pubs/2013_TVCG_survey.pdf
    https://hal.archives-ouvertes.fr/hal-01134194/document
    '''
    device = Phi_a.device 
    dDa = gradient_c(Phi_a, batched = batched, delta_lst = delta_lst)
    dDb = gradient_c(Phi_b, batched = batched, delta_lst = delta_lst)
    dDc = gradient_c(Phi_c, batched = batched, delta_lst = delta_lst) 
    dH  = gradient_c(H, batched = batched, delta_lst = delta_lst)
    Va_x, Va_y, Va_z = dDa[..., 0], dDa[..., 1], dDa[..., 2]
    Vb_x, Vb_y, Vb_z = dDb[..., 0], dDb[..., 1], dDb[..., 2]
    Vc_x, Vc_y, Vc_z = dDc[..., 0], dDc[..., 1], dDc[..., 2]
    Vx = Vc_y - Vb_z + dH[..., 0]
    Vy = Va_z - Vc_x + dH[..., 1]
    Vz = Vb_x - Va_y + dH[..., 2]
    return Vx, Vy, Vz

def stream_3D(Phi_a, Phi_b, Phi_c, batched = False, delta_lst = [1., 1., 1.]):
    '''
    input: (batch, s, r, c)
    '''
    device = Phi_a.device
    dDa = gradient_c(Phi_a, batched = batched, delta_lst = delta_lst)
    dDb = gradient_c(Phi_b, batched = batched, delta_lst = delta_lst)
    dDc = gradient_c(Phi_c, batched = batched, delta_lst = delta_lst) 
    #dDa = gradient_f(Phi_a, batched = batched, delta_lst = delta_lst)
    #dDb = gradient_f(Phi_b, batched = batched, delta_lst = delta_lst)
    #dDc = gradient_f(Phi_c, batched = batched, delta_lst = delta_lst) 
    Va_x, Va_y, Va_z = dDa[..., 0], dDa[..., 1], dDa[..., 2]
    Vb_x, Vb_y, Vb_z = dDb[..., 0], dDb[..., 1], dDb[..., 2]
    Vc_x, Vc_y, Vc_z = dDc[..., 0], dDc[..., 1], dDc[..., 2]
    Vx = Vc_y - Vb_z
    Vy = Va_z - Vc_x
    Vz = Vb_x - Va_y
    return Vx, Vy, Vz

def curl_3D(Fx, Fy, Fz, batched = False, delta_lst = [1., 1., 1.]):
    '''
    input: (batch, s, r, c)
    output: V := (Vx, Vy, Vz) is divergence-free by construction
    '''
    return stream_3D(Fx, Fy, Fz, batched, delta_lst)


def clebsch_2D(Phi, batched = False, delta_lst = [1., 1., 1.]):
    '''
    input: (batch, r, c)
    In 2D, dim_2D == stream_2D (Ref: https://pdfs.semanticscholar.org/c50d/17197599467382c8e91c94ac659fce9fc5ce.pdf)
    '''
    return stream_2D(Phi, batched, delta_lst)

def stream_2D(Phi, batched = False, delta_lst = [1., 1.]):
    '''
    input: Phi as a scalar field in 2D grid: (r, c) or (n_batch, r, c)
    output: V := (Vx, Vy) is divergence-free by construction
    '''
    device = Phi.device
    dD = gradient_c(Phi, batched = batched, delta_lst = delta_lst) 
    Vx = - dD[..., 1]
    Vy = dD[..., 0]
    return Vx, Vy

def curl_2D(Fx, Fy, batched = False, delta_lst = [1., 1.]):
    '''
    input: Fx, Fy as a scalar field in 2D grid: (r, c) or (n_batch, r, c)
    output: V := (Vx, Vy) is divergence-free by construction
    '''
    dFx = gradient_c(Fx, batched = batched, delta_lst = delta_lst) 
    dFy = gradient_c(Fy, batched = batched, delta_lst = delta_lst) 
    return dFy[..., 0] - dFx[..., 1]

def HHD_2D(Phi, H, batched = False, delta_lst = [1., 1.]):
    '''
    input: (batch, s, r, c)
    Refs:
    http://www.sci.utah.edu/~hbhatia/pubs/2013_TVCG_survey.pdf
    https://hal.archives-ouvertes.fr/hal-01134194/document
    '''
    device = Phi.device
    dD  = gradient_c(Phi, batched = batched, delta_lst = delta_lst) 
    dH  = gradient_c(H, batched = batched, delta_lst = delta_lst)
    Vx = - dD[..., 1] + dH[..., 0]
    Vy = dD[..., 0] + dH[..., 1]
    return Vx, Vy

def JBLD_distance(X, Y):
    '''
    Ref: Jensen-Bregman LogDet Divergence (https://ieeexplore.ieee.org/abstract/document/6126523)
    J_ld(X, Y) = log( | (X + Y) / 2 | ) - log( | XY | ) / 2
    '''
    # X, Y: ((n_batch), 4 or 9, (s,) r, c) 
    assert X.size() == Y.size()
    X, Y = X.float(), Y.float()
    if X.size(1) == 4:
        X, Y = X.permute((0, 2, 3, 1)), Y.permute((0, 2, 3, 1)) # (n_batch, 4, r, c) -> (n_batch, r, c, 4) -> (n_batch, r, c, 2, 2)
        X, Y = torch.reshape(X, (X.size(0), X.size(1), X.size(2), 2, 2)), torch.reshape(Y, (Y.size(0), Y.size(1), Y.size(2), 2, 2))
    elif X.size(1) == 9:
        X, Y = X.permute((0, 2, 3, 4, 1)), Y.permute((0, 2, 3, 4, 1)) # (n_batch, 9, s, r, c) -> (n_batch, s, r, c, 9) -> (n_batch, s, r, c, 3, 3)
        X, Y = torch.reshape(X, (X.size(0), X.size(1), X.size(2), X.size(3), 3, 3)), torch.reshape(Y, (Y.size(0), Y.size(1), Y.size(2), Y.size(3), 3, 3))
    else:
        raise ValueError('Matrix dimension not supperted:', X.size())
    
    A = torch.det((X + Y) / 2.)
    B = torch.det(torch.matmul(X, Y))
    #out = (torch.log(A) - torch.log(B) / 2.).mean()
    out = torch.log(A) - torch.log(B) / 2. 
    out = torch.nan_to_num(out, nan = 0., posinf = 0.)  # NOTE: nan/inf: at least one of X, Y equals to 0 --- meaningless
    #if torch.isnan(out):
    #    print('NOTE: JBLD distance == NaN')
    #    out = torch.tensor(0.)
    return out.mean()

def cayley_map(S): # S: ((n_batch), r, c) for 2D or ((n_batch), n_channel = 3, s, r, c) for 3D
    '''expRNN: https://arxiv.org/pdf/1901.08428.pdf'''
    if len(S.size()) == 2 or len(S.size()) == 3: # 2D: batched or not batched #
       # Get skew-symmetric A and do Caylay mapping for approximating Riemann exponential #
       # Doc: https://github.com/uncbiag/PIANOinD/blob/master/Doc/PIANOinD.pdf #
       A = torch.ones(S.size() + tuple([2, 2]), dtype = torch.float, device = S.device)
       B = torch.ones(S.size() + tuple([2, 2]), dtype = torch.float, device = S.device)
       A[..., 0, 1] = S / 2
       A[..., 1, 0] = - S / 2
       B[..., 0, 1] = - S / 2
       B[..., 1, 0] = S / 2
       return torch.matmul(A, torch.inverse(B)) # U = (I + A / 2) * (I - A / 2)^(-1)
       
       #U = torch.ones(tuple([S.size(0), S.size(1), S.size(2), 2, 2]), dtype = torch.float, device = S.device)
       #den =  4 + S ** 2
       #U[..., 0, 0] = (4 - S ** 2) / den
       #U[..., 0, 1] = (4 * S) / den
       #U[..., 1, 0] = - (4 * S) / den
       #U[..., 1, 1] = (4 - S ** 2) / den
       #return U # U = (I + A / 2) * (I - A / 2)^(-1)

    elif len(S.size()) == 4 or len(S.size()) == 5: # 3D: batched or not batched # ((batch), 3, s, r, c)
       # Get skew-symmetric A and do Caylay mapping for approximating Riemann exponential #
       # Doc: https://github.com/uncbiag/PIANOinD/blob/master/Doc/PIANOinD.pdf #
        if len(S.size()) == 5:
            A = torch.ones(tuple([S.size(0), S.size(2), S.size(3), S.size(4), 3, 3]), dtype = torch.float, device = S.device)
            B = torch.ones(tuple([S.size(0), S.size(2), S.size(3), S.size(4), 3, 3]), dtype = torch.float, device = S.device)
        else:
            A = torch.ones(tuple([S.size(1), S.size(2), S.size(3), 3, 3]), dtype = torch.float, device = S.device)
            B = torch.ones(tuple([S.size(1), S.size(2), S.size(3), 3, 3]), dtype = torch.float, device = S.device)
        #print(S.size())
        A[..., 0, 1] = S[..., 0, :, :, :] / 2
        A[..., 0, 2] = S[..., 1, :, :, :] / 2
        A[..., 1, 2] = S[..., 2, :, :, :] / 2
        A[..., 1, 0] = - S[..., 0, :, :, :] / 2
        A[..., 2, 0] = - S[..., 1, :, :, :] / 2
        A[..., 2, 1] = - S[..., 2, :, :, :]/ 2

        B[..., 0, 1] = - S[..., 0, :, :, :] / 2
        B[..., 0, 2] = - S[..., 1, :, :, :] / 2
        B[..., 1, 2] = - S[..., 2, :, :, :] / 2
        B[..., 1, 0] = S[..., 0, :, :, :] / 2
        B[..., 2, 0] = S[..., 1, :, :, :] / 2
        B[..., 2, 1] = S[..., 2, :, :, :] / 2
        return torch.matmul(A, torch.inverse(B))
    else:
        raise ValueError('Not supported dimension for Cayley mapping !')
    

def flatten_U_2D(U, batched  = True): # U: ((batch), r, c, 2, 2) 
    stack_dim = 1 if batched else 0
    return torch.stack([U[..., 0, 0], U[..., 0, 1], \
                        U[..., 1, 0], U[..., 1, 1]], dim = stack_dim)

def flatten_U_3D(U, batched  = True): # U: ((batch), s, r, c, 3, 3) 
    stack_dim = 1 if batched else 0
    return torch.stack([U[..., 0, 0], U[..., 0, 1], U[..., 0, 2], \
                        U[..., 1, 0], U[..., 1, 1], U[..., 1, 2], \
                        U[..., 2, 0], U[..., 2, 1], U[..., 2, 2]], dim = stack_dim)

def construct_spectralD_2D(U, L, batched = True):
    '''
    U: (batch, r, c, 2, 2)
    L: (batch, 2, r, c)
    '''
    if batched:
        L1, L2 = L[:, 0], L[:, 1] # (n_batch, r, c)
        stack_dim = 1
    else:
        L1, L2 = L[0], L[1] # (r, c)
        stack_dim = 0
    Dxx = L1 * U[..., 0, 0] ** 2 + L2 * U[..., 0, 1] ** 2
    Dxy = L1 * U[..., 1, 0] * U[..., 0, 0] + L2 * U[..., 1, 1] * U[..., 0, 1]
    Dyy = L1 * U[..., 1, 0] ** 2 + L2 * U[..., 1, 1] ** 2
    return torch.stack([Dxx, Dxy, Dyy], dim = stack_dim) # ((batch), 3, r, c): (Dxx, Dxy, Dyy)

def construct_spectralD_3D(U, L, batched = True):
    '''
    U: (batch, s, r, c, 3, 3)
    L: (batch, 3, s, r, c)
    '''
    if batched:
        L1, L2, L3 = L[:, 0], L[:, 1], L[:, 2] # (n_batch, s, r, c)
        stack_dim = 1
    else:
        L1, L2, L3 = L[0], L[1], L[2] # (s, r, c)
        stack_dim = 0
    Dxx = U[..., 0, 0] ** 2 * L1 + U[..., 0, 1] ** 2 * L2 + U[..., 0, 2] ** 2 * L3
    Dxy = U[..., 0, 0] * U[..., 1, 0] * L1 + U[..., 0, 1] * U[..., 1, 1] * L2 + U[..., 0, 2] * U[..., 1, 2] * L3
    Dxz = U[..., 0, 0] * U[..., 2, 0] * L1 + U[..., 0, 1] * U[..., 2, 1] * L3 + U[..., 0, 2] * U[..., 2, 2] * L3
    Dyy = U[..., 1, 0] ** 2 * L1 + U[..., 1, 1] ** 2 * L2 + U[..., 1, 2] ** 2 * L3
    Dyz = U[..., 1, 0] * U[..., 2, 0] * L1 + U[..., 1, 1] * U[..., 2, 1] * L2 + U[..., 1, 2] * U[..., 2, 2] * L3
    Dzz = U[..., 2, 0] ** 2 * L1 + U[..., 2, 1] ** 2 * L2 + U[..., 2, 2] ** 2 * L3
    return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = stack_dim) # ((batch), 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)

def construct_choleskyD_2D(Lxx, Lxy, Lyy, batched = True):
    stack_dim = 1 if batched else 0
    Dxx = Lxx ** 2
    Dxy = Lxx * Lxy
    Dyy = Lxy ** 2 + Lyy ** 2
    return torch.stack([Dxx, Dxy, Dyy], dim = stack_dim) # ((batch), 3, r, c): (Dxx, Dxy, Dyy)

def construct_choleskyD_3D(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz, batched = True):
    stack_dim = 1 if batched else 0
    Dxx = Lxx ** 2
    Dxy = Lxx * Lxy
    Dxz = Lxx * Lxz
    Dyy = Lxy ** 2 + Lyy ** 2
    Dyz = Lxy * Lxz + Lyy * Lyz
    Dzz = Lxz ** 2 + Lyz ** 2 + Lzz ** 2
    return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = stack_dim) # ((batch), 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)

def construct_dualD_3D(L1, L2, L3, L4, L5, L6, batched = True):
    stack_dim = 1 if batched else 0
    Dxx = L1 - L2 + L3 - L4 + L5 + L6
    Dxy = L1 - L5
    Dxz = L3 - L6
    Dyy = L1 + L2 - L3 + L4 + L5 - L6
    Dyz = L2 - L4
    Dzz = - L1 + L2 + L3 + L4 - L5 + L6
    return torch.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], dim = stack_dim) # (batch, 6, r, c): (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)



def bilateral_smoothing(img_path, domain_sigma = 1, range_sigma = 8, remask = False):
    '''
    domain_sigma, range_sigma: higher -> smoother
    '''
    img = sitk.ReadImage(img_path)
    filter = sitk.BilateralImageFilter()
    filter.SetDomainSigma(domain_sigma)
    filter.SetRangeSigma(range_sigma)
    new_img = filter.Execute(img)
    if remask:
        nda = sitk.GetArrayFromImage(img)
        new_nda = sitk.GetArrayFromImage(new_img)
        new_nda[nda == 0.] = 0.
        new_img = sitk.GetImageFromArray(new_nda, isVector = len(new_nda.shape) > 3) 
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())
    sitk.WriteImage(new_img, os.path.join(os.path.dirname(img_path), '%s_BS%s' % (os.path.basename(img_path)[:-4], os.path.basename(img_path)[-4:])))
    #sitk.WriteImage(new_img, os.path.join(os.path.dirname(img_path), '%s_BS(%.1f).nii' % (os.path.basename(img_path)[:-4], range_sigma)))
    return os.path.join(os.path.dirname(img_path), '%s_BS%s' % (os.path.basename(img_path)[:-4], os.path.basename(img_path)[-4:]))



def gradient_f(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Upper-boundaries: Backward Difference
    Non-boundaries & Upper-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    device = X.device
    dim = len(X.size()) - 1 if batched else len(X.size())
    #print(batched)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        dX = torch.zeros(X.size(), dtype = torch.float, device = device)
        X = X.permute(1, 0) if batched else X
        dX = dX.permute(1, 0) if batched else dX
        dX[-1] = X[-1] - X[-2] # Backward Difference
        dX[:-1] = X[1:] - X[:-1] # Forward Difference

        dX = dX.permute(1, 0) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = torch.zeros(X.size() + tuple([2]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 0) if batched else X
        dX = dX.permute(1, 2, 3, 0) if batched else dX # put batch to last dim
        dX[-1, :, 0] = X[-1, :] - X[-2, :] # Backward Difference
        dX[:-1, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX = dX.permute(3, 0, 1, 2) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = torch.zeros(X.size() + tuple([3]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 3, 0) if batched else X
        dX = dX.permute(1, 2, 3, 4, 0) if batched else dX
        dX[-1, :, :, 0] = X[-1, :, :] - X[-2, :, :] # Backward Difference
        dX[:-1, :, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, :, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, :, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2] # Backward Difference
        dX[:, :, :-1, 2] = X[:, :, 1:] - X[:, :, :-1] # Forward Difference

        dX = dX.permute(4, 0, 1, 2, 3) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


def gradient_b(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Non-boundaries & Upper-boundaries: Backward Difference
    Lower-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    device = X.device
    dim = len(X.size()) - 1 if batched else len(X.size())
    #print(batched)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        dX = torch.zeros(X.size(), dtype = torch.float, device = device)
        X = X.permute(1, 0) if batched else X
        dX = dX.permute(1, 0) if batched else dX
        dX[1:] = X[1:] - X[:-1] # Backward Difference
        dX[0] = X[1] - X[0] # Forward Difference

        dX = dX.permute(1, 0) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = torch.zeros(X.size() + tuple([2]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 0) if batched else X
        dX = dX.permute(1, 2, 3, 0) if batched else dX # put batch to last dim
        dX[1:, :, 0] = X[1:, :] - X[:-1, :] # Backward Difference
        dX[0, :, 0] = X[1] - X[0] # Forward Difference

        dX[:, 1:, 1] = X[:, 1:] - X[:, :-1] # Backward Difference
        dX[:, 0, 1] = X[:, 1] - X[:, 0] # Forward Difference

        dX = dX.permute(3, 0, 1, 2) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = torch.zeros(X.size() + tuple([3]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 3, 0) if batched else X
        dX = dX.permute(1, 2, 3, 4, 0) if batched else dX
        dX[1:, :, :, 0] = X[1:, :, :] - X[:-1, :, :] # Backward Difference
        dX[0, :, :, 0] = X[1] - X[0] # Forward Difference

        dX[:, 1:, :, 1] = X[:, 1:] - X[:, :-1] # Backward Difference
        dX[:, 0, :, 1] = X[:, 1] - X[:, 0] # Forward Difference

        dX[:, :, 1:, 2] = X[:, :, 1:] - X[:, :, :-1] # Backward Difference
        dX[:, :, 0, 2] = X[:, :, 1] - X[:, :, 0] # Forward Difference

        dX = dX.permute(4, 0, 1, 2, 3) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX
  

def gradient_c(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Non-boundaries: Central Difference
    Upper-boundaries: Backward Difference
    Lower-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    device = X.device
    dim = len(X.size()) - 1 if batched else len(X.size())
    #print(X.size())
    #print(batched)
    #print('dim:', dim)
    if dim == 1:
        #print('dim = 1')
        dX = torch.zeros(X.size(), dtype = torch.float, device = device)
        X = X.permute(1, 0) if batched else X
        dX = dX.permute(1, 0) if batched else dX
        dX[1:-1] = (X[2:] - X[:-2]) / 2 # Central Difference
        dX[0] = X[1] - X[0] # Forward Difference
        dX[-1] = X[-1] - X[-2] # Backward Difference

        dX = dX.permute(1, 0) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = torch.zeros(X.size() + tuple([2]), dtype = torch.float, device = device) 
        X = X.permute(1, 2, 0) if batched else X
        dX = dX.permute(1, 2, 3, 0) if batched else dX # put batch to last dim
        dX[1:-1, :, 0] = (X[2:, :] - X[:-2, :]) / 2
        dX[0, :, 0] = X[1] - X[0]
        dX[-1, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, 1] = (X[:, 2:] - X[:, :-2]) / 2
        dX[:, 0, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, 1] = X[:, -1] - X[:, -2]

        dX = dX.permute(3, 0, 1, 2) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1] 
    elif dim == 3:
        #print('dim = 3')
        dX = torch.zeros(X.size() + tuple([3]), dtype = torch.float, device = device)
        X = X.permute(1, 2, 3, 0) if batched else X
        dX = dX.permute(1, 2, 3, 4, 0) if batched else dX
        dX[1:-1, :, :, 0] = (X[2:, :, :] - X[:-2, :, :]) / 2
        dX[0, :, :, 0] = X[1] - X[0]
        dX[-1, :, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, :, 1] = (X[:, 2:, :] - X[:, :-2, :]) / 2
        dX[:, 0, :, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, :, 1] = X[:, -1] - X[:, -2]
        dX[:, :, 1:-1, 2] = (X[:, :, 2:] - X[:, :, :-2]) / 2
        dX[:, :, 0, 2] = X[:, :, 1] - X[:, :, 0]
        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2]

        dX = dX.permute(4, 0, 1, 2, 3) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


def gradient_c_numpy(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a Numpy array "X" in each direction
    Non-boundaries: Central Difference
    Upper-boundaries: Backward Difference
    Lower-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    dim = len(X.shape) - 1 if batched else len(X.shape)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        X = np.transpose(X, (1, 0)) if batched else X
        dX = np.zeros(X.shapee).astype(float)
        dX[1:-1] = (X[2:] - X[:-2]) / 2 # Central Difference
        dX[0] = X[1] - X[0] # Forward Difference
        dX[-1] = X[-1] - X[-2] # Backward Difference

        dX = np.transpose(X, (1, 0)) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = np.zeros(X.shape + tuple([2])).astype(float)
        X = np.transpose(X, (1, 2, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 0)) if batched else dX # put batch to last dim
        dX[1:-1, :, 0] = (X[2:, :] - X[:-2, :]) / 2
        dX[0, :, 0] = X[1] - X[0]
        dX[-1, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, 1] = (X[:, 2:] - X[:, :-2]) / 2
        dX[:, 0, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, 1] = X[:, -1] - X[:, -2]

        dX = np.transpose(dX, (3, 0, 1, 2)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = np.zeros(X.shape + tuple([3])).astype(float)
        X = np.transpose(X, (1, 2, 3, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 4, 0)) if batched else dX # put batch to last dim
        dX[1:-1, :, :, 0] = (X[2:, :, :] - X[:-2, :, :]) / 2
        dX[0, :, :, 0] = X[1] - X[0]
        dX[-1, :, :, 0] = X[-1] - X[-2]
        dX[:, 1:-1, :, 1] = (X[:, 2:, :] - X[:, :-2, :]) / 2
        dX[:, 0, :, 1] = X[:, 1] - X[:, 0]
        dX[:, -1, :, 1] = X[:, -1] - X[:, -2]
        dX[:, :, 1:-1, 2] = (X[:, :, 2:] - X[:, :, :-2]) / 2
        dX[:, :, 0, 2] = X[:, :, 1] - X[:, :, 0]
        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2]

        dX = np.transpose(dX, (4, 0, 1, 2, 3)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


def gradient_f_numpy(X, batched = False, delta_lst = [1., 1., 1.]):
    '''
    Compute gradient of a torch tensor "X" in each direction
    Upper-boundaries: Backward Difference
    Non-boundaries & Upper-boundaries: Forward Difference
    if X is batched: (n_batch, ...);
    else: (...)
    '''
    dim = len(X.shape) - 1 if batched else len(X.shape)
    #print(dim)
    if dim == 1:
        #print('dim = 1')
        X = np.transpose(X, (1, 0)) if batched else X
        dX = np.zeros(X.shapee).astype(float)
        dX[-1] = X[-1] - X[-2] # Backward Difference
        dX[:-1] = X[1:] - X[:-1] # Forward Difference

        dX = np.transpose(X, (1, 0)) if batched else dX
        dX /= delta_lst[0]
    elif dim == 2:
        #print('dim = 2')
        dX = np.zeros(X.shape + tuple([2])).astype(float)
        X = np.transpose(X, (1, 2, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 0)) if batched else dX # put batch to last dim
        dX[-1, :, 0] = X[-1, :] - X[-2, :] # Backward Difference
        dX[:-1, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX = np.transpose(dX, (3, 0, 1, 2)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
    elif dim == 3:
        #print('dim = 3')
        dX = np.zeros(X.shape + tuple([3])).astype(float)
        X = np.transpose(X, (1, 2, 3, 0)) if batched else X
        dX = np.transpose(dX, (1, 2, 3, 4, 0)) if batched else dX # put batch to last dim
        dX[-1, :, :, 0] = X[-1, :, :] - X[-2, :, :] # Backward Difference
        dX[:-1, :, :, 0] = X[1:] - X[:-1] # Forward Difference

        dX[:, -1, :, 1] = X[:, -1] - X[:, -2] # Backward Difference
        dX[:, :-1, :, 1] = X[:, 1:] - X[:, :-1] # Forward Difference

        dX[:, :, -1, 2] = X[:, :, -1] - X[:, :, -2] # Backward Difference
        dX[:, :, :-1, 2] = X[:, :, 1:] - X[:, :, :-1] # Forward Difference

        dX = np.transpose(dX, (4, 0, 1, 2, 3)) if batched else dX
        dX[..., 0] /= delta_lst[0]
        dX[..., 1] /= delta_lst[1]
        dX[..., 2] /= delta_lst[2]
    return dX


class GaussianSmoother3D(nn.Module):
    '''Fixed layer'''
    def __init__(self, kernel_size = 3, sigma = 1):
        super(GaussianSmoother3D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.conv = nn.Conv3d(1, 1, self.kernel_size, stride = 1, padding = int(kernel_size / 2), bias = None)
        self.weights_init()

    def forward(self, X):
        # X: (slc, row, col)
        return self.conv((X.unsqueeze(0)).unsqueeze(0))
    
    def weights_init(self):
        w = np.zeros((self.kernel_size, self.kernel_size, self.kernel_size))
        center = int(self.kernel_size / 2)
        w[center, center, center] = 1.
        k = gaussian_filter(w, sigma = self.sigma)

        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False
    
class GaussianSmoother2D(nn.Module):
    '''Fixed layer'''
    def __init__(self, kernel_size = 3, sigma = 1):
        super(GaussianSmoother2D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.conv = nn.Conv2d(1, 1, self.kernel_size, stride = 1, padding = int(kernel_size / 2), bias = None)
        self.weights_init()

    def forward(self, X):
        # X: (slc, row, col)
        return self.conv((X.unsqueeze(0)).unsqueeze(0))
    
    def weights_init(self):
        w = np.zeros((self.kernel_size, self.kernel_size))
        center = int(self.kernel_size / 2)
        w[center, center] = 1.
        k = gaussian_filter(w, sigma = self.sigma)

        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False
    


def divergence3D(V, vector_dim = -1, batched = True, data_spacing = [1., 1., 1.]):
    '''
    Get divergence of a 3-component vector field V defined on 3D domain
    '''
    dim = V.size(-1)
    if vector_dim == -1:
        VXx = gradient_c(V[..., 0], batched, data_spacing)[..., 0]
        VYy = gradient_c(V[..., 1], batched, data_spacing)[..., 1]
        VZz = gradient_c(V[..., 2], batched, data_spacing)[..., 2]
    elif batched:
        VXx = gradient_c(V[:, 0], batched, data_spacing)[..., 0]
        VYy = gradient_c(V[:, 1], batched, data_spacing)[..., 1]
        VZz = gradient_c(V[:, 2], batched, data_spacing)[..., 2]
    else:
        VXx = gradient_c(V[0], batched, data_spacing)[..., 0]
        VYy = gradient_c(V[1], batched, data_spacing)[..., 1]
        VZz = gradient_c(V[2], batched, data_spacing)[..., 2]
    return VXx + VYy + VZz


def divergence3D_numpy(V, vector_dim = -1, batched = True, data_spacing = [1., 1., 1.]):
    '''
    Get divergence of a 3-component vector field V defined on 3D domain
    '''
    dim = V.shape[-1]
    if vector_dim == -1: # vector in last dimension #
        VXx = gradient_c_numpy(V[..., 0], batched, data_spacing)[..., 0]
        VYy = gradient_c_numpy(V[..., 1], batched, data_spacing)[..., 1]
        VZz = gradient_c_numpy(V[..., 2], batched, data_spacing)[..., 2]
    elif batched:
        VXx = gradient_c_numpy(V[:, 0], batched, data_spacing)[..., 0]
        VYy = gradient_c_numpy(V[:, 1], batched, data_spacing)[..., 1]
        VZz = gradient_c_numpy(V[:, 2], batched, data_spacing)[..., 2]
    else:  # vector in first dimension #
        VXx = gradient_c_numpy(V[0], batched, data_spacing)[..., 0]
        VYy = gradient_c_numpy(V[1], batched, data_spacing)[..., 1]
        VZz = gradient_c_numpy(V[2], batched, data_spacing)[..., 2]
    return VXx + VYy + VZz

def sqr_norm(x):
    '''
    :return: Spuared L2 norm of input x
    '''
    return (abs(x) ** 2).sum(-1)

def gradient_l2_loss_test(p, mask, weight, batched = False): 
    # NOTE: using Square Loss performs better
    '''p.shape: (slc, row, col)'''
    dp = gradient_c(p, bactched = batched) # (p.size(0)-1, p.size(1)-1, p.size(2)-1, p_dim)
    dp_lp = (abs(dp) ** 2).sum(dim = -1)
    if mask is None:
        return ((dp_lp).view(-1)).mean() * weight
    else:
        return ((mask * dp_lp).view(-1)).mean() * weight

def gradient_l2_loss(p, mask, weight, batched = False):
    # NOTE: using Square Loss performs better
    '''p.shape: (slc, row, col)'''
    dp = gradient_c(p, batched = batched) # (p.size(0)-1, p.size(1)-1, p.size(2)-1, p_dim)
    dp_lp = (abs(dp) ** 2).sum(dim = -1)  # (p.size(0)-1, p.size(1)-1, p.size(2)-1) 
    if mask is None:
        return ((dp_lp).view(-1)).mean() * weight
    else:
        return ((mask * dp_lp).view(-1)).mean() * weight

# Get batched data with size "batch_size"
def get_batch_C(CTC_torch, shuffle = True, batch_size = 1, batch_nT = 2, stride = 1):
    nT = CTC_torch.size(-1)
    if batch_size > nT - batch_nT + 1:
        raise ValueError('Not satisfied: batch_size <= nT - batch_nT + 1, check out !')
    if shuffle: 
        # Randomly select available start t
        start_t = torch.from_numpy(np.random.choice(np.arange(nT - batch_nT + 1, dtype = np.int64), batch_size, replace = False))
    else: 
        # List all available start t in order
        start_t = torch.from_numpy(np.arange(0, nT - batch_nT + 1, stride, dtype = np.int64))
    # CTC_torch: (slc, row, col, t)
    batch_C = []
    for i in range(batch_size):
        batch_C.append(CTC_torch[..., start_t[i] : start_t[i] + batch_nT + 1])
    return torch.stack(batch_C) # (batch_size, slc, row, col, batch_nT)

def avg_grad(params):
    ''''
    params: Torch learnable parameters with gradients
    returns: avg. abs(gradient of all params)
    '''
    grad = 0.
    for p in params:
        #print(p)
        grad = grad + abs(p.grad).sum().item()
    #for p in params:
    #    grad = grad + abs(p.grad).sum().item() if p.grad is not None else grad
    return grad

def avg_grad_named(params):
    ''''
    params: Torch learnable parameters with gradients
    returns: avg. abs(gradient of all params)
    '''
    grad = 0.
    for full_name, p in params:
        p_name = full_name.split('.')[-1]
        if 'D' in p_name or 'V' in p_name or 'L' in p_name or 'A' in p_name or 'a' in p_name:
            if p.grad is not None:
                grad += abs(p.grad).sum().item()
    return grad


def element_tile(element, base_shape):
    '''
    Create an tensor T, with shape: (base_shape + (matrix_size, element_size)),
    where at each location of the base_shape, the element is the same 'element'
    :element: numpy array
    :base_shape: tuple
    '''
    def first2last_axis(a):
        return np.moveaxis(a, 0, -1)
    length = np.prod(np.array(base_shape))
    T = np.repeat(element, length).reshape(element.shape + base_shape) # (element.shape, base_shape)
    for i in range(len(element.shape)):
        T = first2last_axis(T)
    return T # (base_shape, element.shape)

def henaff_init_(A): # A: (slc, row, col, 3, 3)
    size = A.size(-1) // 2
    diag = A.new(size).uniform_(-np.pi, np.pi) * 1e-6 # Get all s_i
    return create_diag_(A, diag)

def cayley_init_(A):
    size = A.size(-1) // 2
    diag = A.new(size).uniform_(0., np.pi / 2.) * 1e-6
    diag = -torch.sqrt((1. - torch.cos(diag))/(1. + torch.cos(diag)))
    return create_diag_(A, diag)
    
def create_diag_(A, diag): # A: (slc, row, col, 3, 3)
    n = A.size(-1)
    diag_z = torch.zeros(n-1)
    diag_z[::2] = diag
    A_init = A.new_zeros(A.size())
    if len(A_init.size()) == 2:
        A_init[:, :] = torch.diag(diag_z, diagonal = 1)
        A_init = A_init - A_init.permute(0, 1, 3, 2)
    elif len(A_init.size()) == 3:
        A_init[:, :, :] = torch.diag(diag_z, diagonal = 1)
        A_init = A_init - A_init.permute(0, 1, 2, 4, 3)
    with torch.no_grad():
        A.copy_(A_init)
        return A

def get_L(L1, L2, L3):
	L = torch.zeros((L1.size(0), L1.size(1), L1.size(2), 3, 3))
	L[..., 0, 0] = L1
	L[..., 1, 1] = L2
	L[..., 2, 2] = L3
	return L
	
'''def get_U(A):
    # Do the retraction #
    A = A.triu(diagonal = 1) # Get upper triangular matrix
    n = A.size(-1)
    Id = torch.eye(n, dtype = A.dtype, device = A.device)
    if len(A.size()) == 2:
        A = A - A.permute(0, 1, 3, 2)
        Id = Id.reshape((1, 1, n, n))
        Id = Id.repeat(A.size(0), A.size(1), 1, 1)
    if len(A.size()) == 3:
        A = A - A.permute(0, 1, 2, 4, 3) # (slc, row, col, 3, 3) # Get skew-symmetric matrix
        Id = Id.reshape((1, 1, 1, n, n))
        Id = Id.repeat(A.size(0), A.size(1), A.size(2), 1, 1)
    return torch.solve(Id - A, Id + A)[0] # (slc, row, col, 3, 3)'''


	
def get_U(A, batched = True):
    ''' Do the retraction, batched version: A: (n_batch, spatial_shape, 2, 2) or (n_batch, spatial_shape, 3, 3) '''
    A = A.triu(diagonal = 1) # Get upper triangular matrix
    n = A.size(-1)
    Id = torch.eye(n, dtype = A.dtype, device = A.device)
    if len(A.size()) == 2:
        A = A - A.permute(0, 1, 3, 2)
        Id = Id.reshape((1, 1, n, n))
        Id = Id.repeat(A.size(0), A.size(1), 1, 1)
    if len(A.size()) == 3:
        A = A - A.permute(0, 1, 2, 4, 3) # (slc, row, col, 3, 3) # Get skew-symmetric matrix
        Id = Id.reshape((1, 1, 1, n, n))
        Id = Id.repeat(A.size(0), A.size(1), A.size(2), 1, 1)
    return torch.solve(Id - A, Id + A)[0] # (slc, row, col, 3, 3)

### Compute nearest PSD ###
def _getAplus(A): # For tensor A: (slc, row, col, n, n)
    eigval, eigvec = np.linalg.eig(A) # (slc, row, col, n), (slc, row, col, n, n)
    diagonal = np.zeros(eigval.shape + tuple([eigval.shape[-1]])) # (slc, row, col, n, n)
    for i in range(diagonal.shape[-1]):
        diagonal[..., i, i] = np.maximum(eigval[..., i], 0)
    return eigvec * diagonal * eigvec.transpose(0, 1, 2, 4, 3)

def _getPs(A, W = None): # For tensor A: (slc, row, col, n, n)
    W05 = W ** .5
    W05_inv = np.linalg.inv(W05)
    return  W05_inv * _getAplus(W05 * A * W05) * W05_inv

    Aret = np.copy(A)
    Aret[W > 0] = W[W > 0]
    return Aret

def _getPu(A, W = None):
    Aret = np.copy(A)
    Aret[W > 0] = W[W > 0]
    return Aret

def IdentityTensor(base_shape, n, isTensor = False):
    '''
    TODO: only support len(base_shape) == 1, 2 or 3
    Create an identity-like tensor I, with shape: (base_shape + (n, n)),
    where I[..., i, i] = 1, I[..., i, j] = 0 (i != j)
    '''
    In = np.identity(n)
    if len(base_shape) == 1:
        I = np.repeat(In, base_shape[0]).reshape(In.shape + base_shape).transpose(2, 0, 1)
    elif len(base_shape) == 2:
        I = np.repeat(In, base_shape[0] * base_shape[1]).reshape(In.shape + base_shape).transpose(2, 3, 0, 1)
    elif len(base_shape) == 3:
        I = np.repeat(In, base_shape[0] * base_shape[1] * base_shape[2]).reshape(In.shape + base_shape).transpose(2, 3, 4, 0, 1)
    else:
        raise ValueError('Only support base_shape with no larger than 3 dimensions, input base_shape agrain !')
    if isTensor:
        I = torch.from_numpy(I.astype(float))
    return I


'''def nearPD(A, nit = 10): # For tensor A: (slc, row, col, n, n)
    n = A.shape[-1]
    In = np.identity(n) 
    # Create identity-like tensor (with last two dimensions as In)
    W = IdentityTensor(tuple([*(A.shape[:-2])]), n, isTensor = False)
    # W: matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W = W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W = W)
    return Yk
'''

def decompose_tensor(tensor, min_diffusivity = - 10000):
    """ Returns eigenvalues and eigenvectors given a diffusion tensor
    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors
    Parameters
    ----------
    tensor : array (..., 3, 3)
        Hermitian matrix representing a diffusion tensor.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.
    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[..., :, j] is associated with
        eigvals[..., j])
    """
    # outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = np.linalg.eigh(tensor)

    # need to sort the eigenvalues and associated eigenvectors
    if eigenvals.ndim == 1:
        # this is a lot faster when dealing with a single voxel
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
    else:
        # temporarily flatten eigenvals and eigenvecs to make sorting easier
        shape = eigenvals.shape[:-1]
        eigenvals = eigenvals.reshape(-1, 3)
        eigenvecs = eigenvecs.reshape(-1, 3, 3)
        size = eigenvals.shape[0]
        order = eigenvals.argsort()[:, ::-1]
        xi, yi = np.ogrid[:size, :3, :3][:2]
        eigenvecs = eigenvecs[xi, yi, order[:, None, :]]
        xi = np.ogrid[:size, :3][0]
        eigenvals = eigenvals[xi, order]
        eigenvecs = eigenvecs.reshape(shape + (3, 3))
        eigenvals = eigenvals.reshape(shape + (3, ))
    eigenvals = eigenvals.clip(min = min_diffusivity)
    # eigenvecs: each vector is columnar
    return eigenvals, eigenvecs

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname
'''
def nda_save_img(nda, origin = None, spacing = None, direction = None, save_path = None, to_smooth = False):
    if len(nda.shape) > 4:
        nda = nda.reshape(*(nda.shape[:3]), -1)
        print('Note: nda to-be-saved has dimension > 4')
    img = sitk.GetImageFromArray(nda, isVector = len(nda.shape) > 3)
    if origin is not None:
        img.SetOrigin(origin)
    if spacing is not None:
        img.SetSpacing(spacing)
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, save_path)
    if to_smooth:
        smoothing(save_path, domain_sigma = 1, range_sigma = 1)
    return img'''

def get_FA(L, isTensor = True):
    # L: (s, r, c, 3)
    if not isTensor:
        L = torch.from_numpy(L)
    eva1, eva2, eva3 = L[..., 0], L[..., 1], L[..., 2]
    return torch.sqrt(0.5 * ((eva1 - eva2) ** 2 + \
        (eva2 - eva3) ** 2 + (eva3 - eva1) ** 2) \
            / (torch.sum(L ** 2, dim = -1)) + 1e-14) # + all_zero 
    

def get_times(ctc_nda, bat_threshold = 0.05):
    '''
    ctc_nda: (s, r, c, t)
    Calculate the CTP bolus arrival time (bat) and corresponding S0: averaged over signals before bat
    return: s0 # (n_slice, n_row, n_column)
    '''
    print(ctc_nda.shape)
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
    print('   - Bolus arrival time (start from 0):', bat - 1)
    print('   - Bolus concentration time-to-peak (start from 0):', ttp)
    print('   - Bolus concentration time-to-drain (start from 0):', ttd)
    return bat, ttp, ttd


def stability_threshold(named_parameters, args):
    for p_name, p in named_parameters:
        if 'D' not in p_name and 'V' not in p_name and 'L' not in p_name and 'A' not in p_name:
            pass
        else:
            if 'D' in p_name:
                if 'scalar' in args.PD_D_type:
                    p.data = torch.clamp(p.data, min = 0.)
                    #p.data = torch.clamp(p.data, min = 0., max = args.thres_D ** (0.5))
                elif args.PD_D_type == 'full' and 'Dxx' in p_name or 'Dyy' in p_name or 'Dzz' in p_name:
                    p.data = torch.clamp(p.data, min = 0.)
                    #p.data = torch.clamp(p.data, min = 0., max = args.thres_D ** (0.5))
                #else:
                    #p.data = torch.clamp(p.data, min = - args.thres_D ** (0.5), max = args.thres_D ** (0.5))
            elif 'L' in p_name:
                p.data = torch.clamp(p.data, min = 0.)
            #elif 'V' in p_name:
                #p.data = torch.clamp(p.data, min = -1 * args.thres_V, max = args.thres_V)
    return 

def save_params(named_parameters, args, save_fld, setting_info, brain_mask, sitk_origin, sitk_spacing, device, isCut = False):
    # Save variable value & gradient
    if brain_mask is None:
        brain_mask = 1.
    for p_name, p in named_parameters:
        if args.PD_D_type == 'constant':
            print('                           Paramter %s: %.6f' % (p_name, p.item()))
        else:
            if args.is_resume:
                param_name = os.path.join(save_fld, '%s (%s, resumed).mha' % (p_name, setting_info))
            else:
                param_name = os.path.join(save_fld, '%s (%s).mha' % (p_name, setting_info))
            if 'A' in p_name:
                dim = len(p.size()) - 2
                if dim == 2:
                    p = p.view(p.size()[0], p.size()[1], -1) # TODO: so far, 2D demo data does not have mask
                elif dim == 3:
                    p = p.view(p.size()[0], p.size()[1], p.size()[2],  -1)
                if brain_mask.mean() != 1.:
                    p = p * (brain_mask.expand(p.size(-1), -1, -1, -1).permute(1, 2, 3, 0)) # Repeat 3D brain_mask to (slc, row, col, 9)
                save_sitk(p, param_name, sitk_origin, sitk_spacing, isCut = isCut, percentile_lower = 0., percentile_upper = 99.8, device = device, isVector = True, RequireGrad = True) 
            else:
                p = p * brain_mask
                save_sitk(p, param_name, sitk_origin, sitk_spacing, isCut = isCut, percentile_lower = 0., percentile_upper = 99.8, device = device, isVector = False) 
            
            if 'V' in p_name:
                save_sitk(abs(p), os.path.join(save_fld, 'Abs - %s (%s).mha' % (p_name, setting_info)), sitk_origin, sitk_spacing, isCut = isCut, \
                    percentile_lower = 0., percentile_upper = 99.8, device = device, isVector = False)
    del p_name, p
    return


def round_up(X):
    return torch.round(X + 0.5 - 1e-7).int()

def round_down(X):
    return torch.round(X - 0.5 + 1e-7).int()
   
'''class Upwind_old(object):
    # Backward if > 0, forward if <= 0 #
    # NOTE: work for batched !!! i.e., V should be of the shape (n_batch, ...) #
    def __init__(self, FDSolver, U):
        self.FDSolver = FDSolver
        self.U = U
        self.dim = len(self.U.size()) - 1
        self.I = torch.ones(self.U.size(), dtype = torch.float, device = U.device)

    def dX(self, FGx):
        dXf, dXb = self.FDSolver.dXf(self.U), self.FDSolver.dXb(self.U)
        Xflag = (FGx > 0).float()
        return dXf * (self.I - Xflag) + dXb * Xflag

    def dY(self, FGy):
        dYf, dYb = self.FDSolver.dYf(self.U), self.FDSolver.dYb(self.U)
        Yflag = (FGy > 0).float()
        return dYf * (self.I - Yflag) + dYb * Yflag

    def dZ(self, FGz):
        dZf, dZb = self.FDSolver.dZf(self.U), self.FDSolver.dZb(self.U)
        Zflag = (FGz > 0).float()
        return dZf * (self.I - Zflag) + dZb * Zflag'''


class Upwind(object):
    '''
    Backward if > 0, forward if <= 0
    '''
    def __init__(self, U, data_spacing = [1., 1, 1.], batched = True):
        self.U = U # (s, r, c)
        self.batched = batched
        self.data_spacing = data_spacing
        self.dim = len(self.U.size()) - 1 if batched else len(self.U.size())
        self.I = torch.ones(self.U.size(), dtype = torch.float, device = U.device)

    def dX(self, FGx):
        dXf = gradient_f(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 0]
        dXb = gradient_b(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 0]
        Xflag = (FGx > 0).float()
        return dXf * (self.I - Xflag) + dXb * Xflag

    def dY(self, FGy):
        dYf = gradient_f(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 1]
        dYb = gradient_b(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 1]
        Yflag = (FGy > 0).float()
        return dYf * (self.I - Yflag) + dYb * Yflag

    def dZ(self, FGz):
        dZf = gradient_f(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 2]
        dZb = gradient_b(self.U, batched = self.batched, delta_lst = self.data_spacing)[..., 2]
        Zflag = (FGz > 0).float()
        return dZf * (self.I - Zflag) + dZb * Zflag

def load_R(device, requires_grad = True):
    return nn.Parameter(torch.ones((3), dtype = torch.float, device = device, requires_grad = requires_grad))

def load_D(D_type, param_name, data_dim, device, requires_grad = True, D_init = 1e-3):
    '''Load a single parameter, return as torch.tensor.to(device)'''
    if 'cholesky' in D_type:
        return nn.Parameter(torch.ones(data_dim, dtype = torch.float, device = device, requires_grad = requires_grad) * D_init)
    else:
        return nn.Parameter(torch.ones(data_dim, dtype = torch.float, device = device, requires_grad = requires_grad) * D_init) # 1e-3
        #if 'Dxx' in param_name or 'Dyy' in param_name or 'Dzz' in param_name:
        #    return nn.Parameter(torch.ones(data_dim, device = device, requires_grad = requires_grad) * 1e-14)
        #else:
        #    return nn.Parameter(torch.ones(data_dim, device = device, requires_grad = requires_grad) * 1e-14)

def load_V(param_name, data_dim, device, requires_grad = True, V_init = 0.):
    '''Load a single parameter, return as torch.tensor.to(device)'''
    if param_name in {'Va', 'Vb', 'Vc', 'Da', 'Db', 'Dc', 'H'}:
        return nn.Parameter(torch.randn(data_dim, dtype = torch.float, device = device, requires_grad = requires_grad) * V_init)
    else:
        #return nn.Parameter(torch.zeros(data_dim, dtype = torch.float, device = device, requires_grad = requires_grad))
        return nn.Parameter(torch.ones(data_dim, dtype = torch.float, device = device, requires_grad = requires_grad) * V_init)

def load_spectral_param(param_name, data_dim, device, requires_grad = True):
    if 'A' in param_name:
        if len(data_dim) == 2:
            return nn.Parameter(torch.empty(tuple(data_dim) + (2, 2), dtype = torch.float, device = device, requires_grad = requires_grad))
        elif len(data_dim) == 3:
            return nn.Parameter(torch.empty(tuple(data_dim) + (3, 3), dtype = torch.float, device = device, requires_grad = requires_grad))
    else:
        return nn.Parameter(torch.ones(data_dim, device = device, dtype = torch.float, requires_grad = requires_grad) * 1e-6)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val