import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import numpy as np
import SimpleITK as sitk

from utils import *
from Datasets.demo_dataset import PIGenerator
from Learning.Modules.AdvDiffPDE import PIANO_Skeleton


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#%% Basic settings
parser = argparse.ArgumentParser('3D demo prep')
parser.add_argument('--case', type = str, default = 'PI_demo')
parser.add_argument('--adjoint', type = bool, default = False)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--is_resume', type=bool, default = False)
parser.add_argument('--integ_method', type = str, default = 'dopri5', choices=['dopri5', 'adams', 'rk4', 'euler'])

# For ground truth types
parser.add_argument('--GT_perf_pattern', type = str, default = 'adv_diff', choices = ['mix', 'circle_adv_only', 'adv_diff', 'adv_only', 'diff_only'])

parser.add_argument('--t0', type = float, default = 0.0) # the beginning time point
parser.add_argument('--dt', type = float, default = 0.01) # Real dt between scanned images
parser.add_argument('--nT', type = int, default = 100) # 200: data_dim = 128, 100: data_dim = 64

parser.add_argument('--U0_pattern', type = str, default = 'gaussian', choices = ['tophat_corner', 'tophat_top', 'tophat_center', 'gaussian'])
parser.add_argument('--U0_magnitude', type = float, default = 1., help = 'Initial max value for tophat-type U0') 

parser.add_argument('--GT_D_type', type = str, default = 'full', choices = ['scalar', 'full'])
#parser.add_argument('--diffusivity', type = dict, default = {'D': 0.1, 'Dxx': 0.2, 'Dxy': 0.1, 'Dyy': 0.06})
parser.add_argument('--diffusion_magnitude', type = float, default = 100)

parser.add_argument('--GT_V_type', type = str, default = 'vector', choices = ['vector', 'vector_div_free'])
parser.add_argument('--velocity_magnitude', type = float, default = 5.)


parser.add_argument('--time_stepsize', type = float, default = 1e-2, help = 'Time step for integration, \
    if is not None, the assigned value should be able to divide args.time_spacing')
parser.add_argument('--BC', type = str, default = 'neumann', choices = [None, 'neumann'])

args_demo = parser.parse_args()  

def main(args, save_fld, D_path, V_path, U0_path, Mask_path):

    Generator = PIGenerator(args, D_path, V_path, U0_path, Mask_path, save_fld = save_fld)
    GT_u = Generator.get_U(PIANO_Skeleton, RecordSum = True)
    
        



##############################################################################################################################
if __name__ == '__main__':

    main_dir = '/media/peirong/PR5/PIDemo'
    save_dir = make_dir(os.path.join(main_dir, 'Movies', args_demo.GT_perf_pattern))

    V_path = os.path.join(main_dir, 'V/V_BS.nii')
    D_path = os.path.join(main_dir, 'D/DTIBrain_cropped.nii')
    U0_path = os.path.join(main_dir, 'D/ADC_cropped.nii')
    Mask_path = os.path.join(main_dir, 'Mask_cropped.nii')

    main(args_demo, save_dir, D_path, V_path, U0_path, Mask_path)