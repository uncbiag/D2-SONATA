import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.ndimage as ndimage

import torch

from utils import *
from Preprocess.IXI.v_generator import frangi_velocity
from Preprocess.IXI.itk_utils import *
from Preprocess.IXI.IXI_process import segmenting_vessel

import itk
import SimpleITK as sitk
from itk import TubeTK as ttk 
 

main_fld = '/home/peirong/biag-raid1/Results/ISLES2017_Results/adv_diff/[Vess_enhanced_normalied]-[D-0.05,V-1]-nCol[5]_Res(1000)_[Conc]_Vess_[1]/1/training_13/ScalarMaps'
main_fld = '/home/peirong/biag-raid1/Results/ISLES2017_Results/adv_diff/NotRotated/[Vess_enhanced_normalied]-[D-0.05,V-1]-nCol[5]_Res(1000)_[Conc]_Vess_[1]/1/training_13/ScalarMaps'
CbO_path = os.path.join(main_fld, 'D_Color_Direction (PTI).mha')
CbO_smo_path = bilateral_smoothing(CbO_path, domain_sigma = 0.8, range_sigma = 0.5) 
FA_path = os.path.join(main_fld, 'FA.mha')
FA_smo_path = bilateral_smoothing(FA_path, domain_sigma = 0.8, range_sigma = 0.5) 
Tr_path = os.path.join(main_fld, 'Trace.mha')
Tr_smo_path = bilateral_smoothing(Tr_path, domain_sigma = 0.8, range_sigma = 0.5) 

#main_fld = '/home/peirong/biag-raid1/Results/IXI_Results/adv_diff/LargeD-JointLoss/Forward-Stream/L1-nCol[5]_[NoConc]_Vess_[GT_V(10]_[GT_LU(10, 10)]_[98]-vector_div_free_stream_full_spectral - cauchy/3980/IXI002-Guys-0828/ScalarMaps'
#CbO_path = os.path.join(main_fld, 'D_Color_Direction (PTI).mha')
#FA_path = os.path.join(main_fld, 'FA.mha')
#Tr_path = os.path.join(main_fld, 'Trace.mha')

CbO, o, s, d = img2nda(CbO_smo_path) # (s, r, c, 3)
FA, _, _, _  = img2nda(FA_smo_path)  # (s, r, c)
Tr, _, _, _  = img2nda(Tr_smo_path)  # (s, r, c)

save_path = os.path.join(main_fld, 'CbOxFA.mha') 

nda2img(CbO * FA[..., None], o, s, d, save_path = save_path)
bilateral_smoothing(save_path, domain_sigma = 0.8, range_sigma = 0.5) 