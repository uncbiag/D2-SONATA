import os, sys, argparse, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import itk
import torch
import numpy as np
import SimpleITK as sitk
from shutil import copyfile, rmtree
from scipy.ndimage import gaussian_filter
from skimage.filters.ridges import frangi
from skimage.morphology import skeletonize

from Preprocess.IXI.fit_divfree import fit
from Preprocess.IXI.ixi_utils import hessian_and_eig
from Preprocess.IXI.itk_utils import anisotropic_smoothing

#from Preprocess.IXI.mra_utils import *
#from Preprocess.IXI.smoothers import diffuser, diffuser_numpy

from utils import make_dir, nda2img, numpy_percentile, remove_paths, divergence3D_numpy


#%% Basic settings
parser = argparse.ArgumentParser('Divergence-free Velocity Fields Generation')
parser.add_argument('--velocity_magnitude', type = float, default = 10.)

# For divergence-free V fitting #
parser.add_argument('-divfree_fit_lr', type = float, default = 1e-3)
parser.add_argument('--n_iter_divfree_fit', type = int, default = 2000)

# For vessel thresholding and anisotropy smoothing #
parser.add_argument('--vessel_percentile', type = float, default = 99, help = 'Percentile above which be treated as vessel regions')
parser.add_argument('--n_iter_vessel_smoothing', type = int, default = 1)
''' Parameters setting: https://insightsoftwareconsortium.github.io/ITKAnisotropicDiffusionLBR/ '''
parser.add_argument('--vessel_diffusion_time', type = float, default = 1.5, help = 'Range: [0.5, 50]') # 3.5
parser.add_argument('--vessel_anisotropic_lambda', type = float, default = 0.05, help = 'Range: [0.0001, 0.5]') # 0.1
parser.add_argument('--vessel_enhancement_type', type = int, default = 2, help = \
    '0: EED (Edge Enhancing Diffusion); 1: cEED (Conservative Edge Enhancing Diffusion); 2: CED (Coherence Enhancing Diffusion); \
     3: cCED (Conservative Coherence Enhancing Diffusion); 4: Isotropic Diffusion')
parser.add_argument('--vessel_noise_scale', type = float, default = 3, help = 'Range: [0.5, 5]')
parser.add_argument('--vessel_feature_scale', type = float, default = 5, help = 'Range: [2, 6]')
parser.add_argument('--vessel_exponent', type = float, default = 3.5, help = 'Range: [1, 6]')

# For v anisotropy smoothing #
parser.add_argument('--n_iter_v_smoothing', type = int, default = 1)
parser.add_argument('--use_gaussian_smoothed_anisotropy', type = bool, default = True)
''' Parameters setting: https://insightsoftwareconsortium.github.io/ITKAnisotropicDiffusionLBR/ '''
parser.add_argument('--v_diffusion_time', type = float, default = 2.5, help = 'Range: [0.5, 50]') # 3.5
parser.add_argument('--v_anisotropic_lambda', type = float, default = 0.05, help = 'Range: [0.0001, 0.5]') # 0.1
parser.add_argument('--v_enhancement_type', type = int, default = 2, help = \
    '0: EED (Edge Enhancing Diffusion); 1: cEED (Conservative Edge Enhancing Diffusion); 2: CED (Coherence Enhancing Diffusion); \
     3: cCED (Conservative Coherence Enhancing Diffusion); 4: Isotropic Diffusion')
parser.add_argument('--v_noise_scale', type = float, default = 3, help = 'Range: [0.5, 5]')
parser.add_argument('--v_feature_scale', type = float, default = 5, help = 'Range: [2, 6]')
parser.add_argument('--v_exponent', type = float, default = 3.5, help = 'Range: [1, 6]')

args_gen = parser.parse_args()  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


############################################################################################
##########################################  Utils ##########################################
############################################################################################


def nii2mha(img_path, new_basename = None):
    img = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(img)
    filename = os.path.join(os.path.dirname(img_path), '%s.mha' % new_basename) if new_basename else '%s.mha' % img_path[:-4] 
    nda2img(nda, img.GetOrigin(), img.GetSpacing(), img.GetDirection(), '%s.mha' % img_path[:-4])
    os.remove(img_path)


def frangi_velocity(vessel_path, velocity_magnitude = 1., save_fld = None):
    '''
    vessel: binary vessel map
    Compute Hessian, get principle direction as velocity (single case use)
    ''' 
    save_fld = os.path.join(os.path.dirname(vessel_path)) if save_fld is None else save_fld
    vessel_img = sitk.ReadImage(vessel_path)
    origin, spacing, direction = vessel_img.GetOrigin(), vessel_img.GetSpacing(), vessel_img.GetDirection()
    vessel_nda = sitk.GetArrayFromImage(vessel_img)

    #vessel_nda /= np.max(abs(vessel_nda)) # Re-ranging to [0, 1]
    hessian_nda, evals, evecs = hessian_and_eig(vessel_nda) 

    # NOTE Sanity check #
    # Smallest abs(eigenvalue) -- vessel singular direction
    # Other two eigenvalues should be approx. same, and have same sign --> form the orthogonal base of vessel direction
    #nda2img(evals * vessel_nda[..., None], origin, spacing, direction, os.path.join(save_fld, 'L.mha'))
    #nda2img(evecs[..., 0] * vessel_nda[..., None], origin, spacing, direction, os.path.join(save_fld, 'U1.mha'))
    #nda2img(evecs[..., 1] * vessel_nda[..., None], origin, spacing, direction, os.path.join(save_fld, 'U2.mha'))
    #nda2img(evecs[..., 2] * vessel_nda[..., None], origin, spacing, direction, os.path.join(save_fld, 'U3.mha'))

    ## resort evals according to abs ##
    abs_evals = abs(evals)
    # temporarily flatten eigenvals and eigenvecs to make sorting easier
    shape = abs_evals.shape[:-1]
    eigenvals = abs_evals.reshape(-1, 3)
    eigenvecs = evecs.reshape(-1, 3, 3)
    size = eigenvals.shape[0]
    order = eigenvals.argsort() # Ascending order #
    xi, yi = np.ogrid[:size, :3, :3][:2]
    eigenvecs = eigenvecs[xi, yi, order[:, None, :]]
    xi = np.ogrid[:size, :3][0]
    eigenvals = eigenvals[xi, order]
    eigenvals = eigenvals.reshape(shape + (3, ) ) * vessel_nda[..., None] # Ascending #
    eigenvecs = eigenvecs.reshape(shape + (3, 3)) * vessel_nda[..., None, None]

    vessel_cross_area = eigenvals[..., 1] * eigenvals[..., 2] * np.pi
    #nda2img(vessel_cross_area, origin, spacing, direction, os.path.join(save_fld, 'VesselCrossArea.mha'))
    #nda2img(np.max(vessel_cross_area, axis = 0)[::-1], save_path = os.path.join(save_fld, 'VesselCrossArea_MIP.mha'), isVector = True)

    # NOTE: vessel direction is determined by eigenvector corresponding to the smallest absolute eigenvalue #
    v = eigenvecs[..., 0]  # V is essentially eigenvector: norm_V == 1
    # Re-range to 1 * args.velocity_magnitude
    v =  v * velocity_magnitude / np.max(abs(v)) # Multiply the magnitude #

    #nda2img(v, origin, spacing, direction, os.path.join(save_fld, 'PrincipleDirection.mha'))
    nda2img(abs(v), origin, spacing, direction, os.path.join(save_fld, 'AbsPrincipleDirection.mha')) # For color-by-orientation #
    nda2img(np.max(abs(v), axis = 0)[::-1], save_path = os.path.join(save_fld, 'AbsPrincipleDirection_MIP.mha'), isVector = True)
    #nda2img(np.linalg.norm(v, axis = -1), origin, spacing, direction, os.path.join(save_fld, 'Norm_V.nii'))
    #self.nda2img(divergence3D_numpy(v, vector_dim = -1, batched = False), os.path.join(save_fld, '%s-div_V.nii' % self.case_name))

    return abs(v)


def nearest_nonzero_idx(nda, idx):
    tmp = nda[idx[0], idx[1], idx[2]]
    nda[idx[0], idx[1], idx[2]] = 0
    s, r, c = np.nonzero(nda)
    nda[idx[0], idx[1], idx[2]] = tmp
    nearest_idx = ((s - idx[0]) ** 2 + (r - idx[1]) ** 2 + (c - idx[2]) ** 2).argmin()
    return [s[nearest_idx], r[nearest_idx], c[nearest_idx]]


def get_direction(nda, center_idx):
    '''
    Find nearest non-zero neighbot
    Compute gradient: if self higher (MRA intensity),    return direction self -> neighbor;
                      if self lower (MRA intensity), return direction neighbor -> self;
    '''
    center_val = nda[center_idx[0], center_idx[1], center_idx[2]].astype(float)
    neighb_idx = nearest_nonzero_idx(nda, center_idx)
    neighb_val = nda[neighb_idx[0], neighb_idx[1], neighb_idx[2]].astype(float)
    #print(center_val, neighb_val)
    
    return - (center_val - neighb_val) / abs(center_val - neighb_val) * np.array([center_idx[0] - neighb_idx[0],\
                                              center_idx[1] - neighb_idx[1], \
                                              center_idx[2] - neighb_idx[2]])


def nearest_centerline_idx(centerline_nda, idx):
    s, r, c = np.nonzero(centerline_nda)
    nearest_idx = ((s - idx[0]) ** 2 + (r - idx[1]) ** 2 + (c - idx[2]) ** 2).argmin()
    return [s[nearest_idx], r[nearest_idx], c[nearest_idx]]




############################################################################################
############################################################################################


class VelocityGenerator(object):
    '''
    Refs:
    Centerline extraction: https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology
    Frangi vesselness measure: https://link.springer.com/chapter/10.1007/BFb0056195
    Frangi codes: https://github.com/scikit-image/scikit-image/blob/bd1dc065025775fc591fb93c7bd20d59e2497e9a/skimage/feature/corner.py#L246
    '''

    def __init__(self, vessel_path, mra_path, save_fld):
        
        self.save_fld = save_fld
        self.mra_path = mra_path
        self.vessel_path = vessel_path
        vessel_img = sitk.ReadImage(vessel_path)
        self.vessel_nda = sitk.GetArrayFromImage(vessel_img).astype(float)
        self.mra_nda = sitk.GetArrayFromImage(sitk.ReadImage(mra_path)).astype(float)
        self.origin, self.spacing, self.direction = vessel_img.GetOrigin(), vessel_img.GetSpacing(), vessel_img.GetDirection()

        '''ImageDimension = 3
        PixelType = itk.ctype('float')
        ImageType = itk.Image[PixelType, ImageDimension]
        self.img_reader = itk.ImageFileReader[ImageType].New()
        self.img_writer = itk.ImageFileWriter[ImageType].New()'''

    def get_velocity(self):
        '''
        Processing steps:
        (1) Fraingi-Hessian SVD --> determine absolaute principle vessel direction
        (2) Extract centerline from vessel region
        (3) Infer centerline direction sign by MRA value at centerline: high --> (sign -1 ==> -1; sign 0, 1 ==> 1) 
        (4) For all points in vessel, set direction sign same as nearest centerline point
        (5) V = principle vessel direction * direction sign * MRA intensity
        (6) Anisotropic smoothing on V
        '''
        ################### (1) Fraingi-Hessian SVD ###################
        print('   -- Get absolute principle vessel direction')
        abs_principle_direction = frangi_velocity(self.vessel_path, save_fld = self.save_fld)

        ################### (2) Extract vessel centerline ###################
        print('   -- Extract vessel centerline')
        vessel_skel_bin = skeletonize(self.vessel_nda)
        self.nda2img(vessel_skel_bin,  os.path.join(self.save_fld, 'VesselSkeleton.mha'))
        nda2img(np.max(vessel_skel_bin[:, ::-1], axis = 0), save_path =  os.path.join(self.save_fld, 'VesselSkeleton_MIP.mha'), isVector = True)

        ################### (3) Infer centerline direction sign ###################
        print('   -- Infer centerline direction sign')
        vessel_skel_mra = self.mra_nda * vessel_skel_bin
        s, r, c = np.nonzero(vessel_skel_bin)
        n_centerline_pts = len(s)
        vessel_skel_direction_sign_nda = np.zeros(vessel_skel_bin.shape + tuple([3])) # {-1, 1}
        for i_pts  in range(n_centerline_pts):
            i_s, i_r, i_c = s[i_pts], r[i_pts], c[i_pts]
            vessel_skel_direction_sign_nda[i_s, i_r, i_c] = get_direction(vessel_skel_mra, [i_s, i_r, i_c])
        vessel_skel_direction_sign_nda[vessel_skel_direction_sign_nda == 0.] = 1. # NOTE Set no-direction to reserved sign 

        ################### (4) Set direction sign for all vessel points (same as the nearest centerline point) ###################
        print('   -- Set direction sign for all vessel points') # TODO: expedite
        s, r, c = np.nonzero(self.vessel_nda)
        n_vessel_pts = len(s)
        vessel_direction_sign_nda = np.zeros(vessel_skel_direction_sign_nda.shape)
        for i_pts in range(n_vessel_pts):
            i_s, i_r, i_c = s[i_pts], r[i_pts], c[i_pts]
            centerline_idx = nearest_centerline_idx(vessel_skel_bin, [i_s, i_r, i_c]) # Search for nearest centerline point 
            vessel_direction_sign_nda[i_s, i_r, i_c] = vessel_skel_direction_sign_nda[centerline_idx[0], centerline_idx[1], centerline_idx[2]]
        v_direction = vessel_direction_sign_nda * abs_principle_direction
        self.nda2img(v_direction, os.path.join(self.save_fld, 'PrincipleDirection.mha'))

        ################### (5) Get V with direction magnitude & direction sign & intensity ###################
        print('   -- Get V with direction magnitude & direction sign & intensit')
        v = v_direction * self.mra_nda[..., None]
        v /= np.max(abs(v)) # Re-normalize to [-1, 1]
        self.nda2img(v, os.path.join(self.save_fld, 'V.mha'))
        
        ################### (6) Anisotropic smoothing on V ###################
        print('   -- Anisotropic smoothing on V')
        vx_path = self.nda2img(v[..., 0], os.path.join(self.save_fld, 'Vx.mha'))
        vy_path = self.nda2img(v[..., 1], os.path.join(self.save_fld, 'Vy.mha'))
        vz_path = self.nda2img(v[..., 2], os.path.join(self.save_fld, 'Vz.mha'))

        smoothed_vx_path = anisotropic_smoothing(vx_path, n_iter = 1, diffusion_time = 0.8, \
            anisotropic_lambda = 0.05, enhancement_type = 3, noise_scale = 0.15, feature_scale = 3, exponent = 2)
        smoothed_vy_path = anisotropic_smoothing(vy_path, n_iter = 1, diffusion_time = 0.8, \
            anisotropic_lambda = 0.05, enhancement_type = 3, noise_scale = 0.15, feature_scale = 3, exponent = 2)
        smoothed_vz_path = anisotropic_smoothing(vz_path, n_iter = 1, diffusion_time = 0.8, \
            anisotropic_lambda = 0.05, enhancement_type = 3, noise_scale = 0.15, feature_scale = 3, exponent = 2)
        smoothed_vx_nda = sitk.GetArrayFromImage(sitk.ReadImage(smoothed_vx_path))
        smoothed_vy_nda = sitk.GetArrayFromImage(sitk.ReadImage(smoothed_vy_path))
        smoothed_vz_nda = sitk.GetArrayFromImage(sitk.ReadImage(smoothed_vz_path))
        smoothed_v_nda = np.stack([smoothed_vx_nda, smoothed_vy_nda, smoothed_vz_nda], axis = -1)

        smoothed_v_nda /= np.max(smoothed_v_nda) # Re-normalize to [-1, 1]
        self.nda2img(smoothed_v_nda, os.path.join(self.save_fld, 'V_smoothed.mha'))
        self.nda2img(abs(smoothed_v_nda), os.path.join(self.save_fld, 'Abs_V_smoothed.mha'))
        self.nda2img(np.linalg.norm(smoothed_v_nda, axis = -1), os.path.join(self.save_fld, 'Norm_V_smoothed.mha'))
        os.remove(vx_path)
        os.remove(vy_path)
        os.remove(vz_path)
        os.remove(smoothed_vx_path)
        os.remove(smoothed_vy_path)
        os.remove(smoothed_vz_path)

        return os.path.join(self.save_fld, 'V_smoothed.mha')


    '''def fit_divfree_v(self, v_path):
        print('    Fitting divergence-free V')
        fit_v_path = fit(v_path, device, self.args.n_iter_divfree_fit, lr = self.args.divfree_fit_lr)
        return fit_v_path'''

    def nda2img(self, nda, save_path):
        nda2img(nda, self.origin, self.spacing, self.direction, save_path)
        return save_path



if __name__ == '__main__':

    # NOTE raise NotImplementedError('v_generator not used any more, see IXI_process.py')

    main_fld = '/media/peirong/PR/IXI'
    processed_fld = '/media/peirong/PR5/IXI_Processed'
    
    names_file = open(os.path.join(main_fld, 'IDs.txt'), 'r')
    case_names = names_file.readlines()
    names_file.close()
    #case_names = ['IXI002-Guys-0828'] # TODO

    for i in range(len(case_names)):
        case_name = case_names[i].split('\n')[0]
        print('\nStart processing case NO.%d (of %d): %s' % (i, len(case_names), case_name))
        case_fld = os.path.join(processed_fld, case_name)

        #mra_path = os.path.join(processed_fld, case_name, 'MRA.mha')
        #vessel_path = os.path.join(processed_fld, case_name, 'Vessel.mha')
        #generator = VelocityGenerator(vessel_path, mra_path, make_dir(os.path.join(case_fld, 'AdvectionMaps')))
        #generator.get_velocity()
        #fit_v_path = generator.fit_divfree_v(v_path)

        if os.path.isfile(os.path.join(case_fld, 'DTI.nii')):
            #os.rename(os.path.join(case_fld, 'AdvectionMaps/DivFree/norm_V.mha'),\
            #    os.path.join(case_fld, 'AdvectionMaps/DivFree/Norm_V.mha'))
            nii2mha(os.path.join(case_fld, 'DTI.nii'))
            #os.remove(os.path.join(case_fld, 'AdvectionMaps/VesselCrossArea_MIP.mha'))
    gc.collect()

