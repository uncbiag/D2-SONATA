import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
 
from utils import gradient_c, make_dir


def get_Phi(D1, D2, pattern_type = '0'):
    if pattern_type is '0':
        return np.sin(D1) + np.cos(D2)

def get_U(D1, D2, pattern_type = '0'):
    if pattern_type is '0':
        return np.sin(D2)

def get_V(D1, D2, pattern_type = '0'):
    if pattern_type is '0':
        return np.cos(D1)


class HHDDataset2D(Dataset):
    '''
    Create V via general HHD: V = curl(Phi) + div(H)
    Ref: 
    [1] http://www.sci.utah.edu/~hbhatia/pubs/2013_TVCG_survey.pdf
    [2] https://hal.archives-ouvertes.fr/hal-01134194/document
    '''

    def __init__(self, PhiFunc, HFunc, XGrid = [0, 32, 1.], YGrid = [0, 32, 1.], SaveFolder = None):

        self.Phi = PhiFunc(XGrid, YGrid),
        self.H = HFunc(XGrid, YGrid)
        self.GridSize = GridSize
        self.SaveFolder = SaveFolder
    


#################################################################################
################################# Basic settings ################################
#################################################################################

pattern_type = '0'
save_fld = '/media/peirong/PR5/V_2d_demo/Data2'

#################################################################################
################################ Create train_V #################################
#################################################################################

X0, X1, dX = -16, 16, 0.25
Y0, Y1, dY = -16, 16, 0.25
#X0, X1, dX = -5, 5, .05
#Y0, Y1, dY = -5, 5, .05

nX = np.floor((X1 - X0) / dX)
nY = np.floor((Y1 - Y0) / dY)

# !NOTE!: In grid, X -- 2nd dim, Y -- 1st dim
print('(%d, %d)' % (nY, nX))

D2, D1 = np.meshgrid(np.arange(X0, X1, dX), np.arange(Y0, Y1, dY))

Phi = get_Phi(D1, D2, pattern_type)
dPhi = gradient_c(torch.from_numpy(Phi), batched = False, delta_X = [dY, dX]).numpy() 
U, V = - dPhi[..., 1], dPhi[..., 0] 
fig1, ax1 = plt.subplots()
Q = ax1.quiver(D1, D2, U*1, V*1, np.hypot(U*1, V*1), units='x', pivot='tip', width=0.02, scale=1)
fig1.savefig(os.path.join(make_dir(os.path.join(save_fld, 'Stream')), 'Train_V.png'))
sitk.WriteImage(sitk.GetImageFromArray(Phi), os.path.join(make_dir(os.path.join(save_fld, 'Stream')), 'Train_Phi.nii')) # (D1, D2)
sitk.WriteImage(sitk.GetImageFromArray(np.array([U, V])), os.path.join(make_dir(os.path.join(save_fld, 'Stream')), 'Train_V.nii')) # (2, D1, D2)


U = get_U(D1, D2, pattern_type)
V = get_V(D1, D2, pattern_type)
fig1, ax1 = plt.subplots()
Q = ax1.quiver(D1, D2, U*1, V*1, np.hypot(U*1, V*1), units='x', pivot='tip', width=0.02, scale=1)
fig1.savefig(os.path.join(make_dir(os.path.join(save_fld, 'Vector')), 'Train_V.png'))
sitk.WriteImage(sitk.GetImageFromArray(np.array([U, V])), os.path.join(make_dir(os.path.join(save_fld, 'Vector')), 'Train_V.nii')) # (2, D1, D2)


#################################################################################
################################ Create test_V ##################################
#################################################################################


X0, X1, dX = -4, 4, .25
Y0, Y1, dY = -4, 4, .25
nX = np.floor((X1 - X0) / dX)
nY = np.floor((Y1 - Y0) / dY)
print('(%d, %d)' % (nX, nY))
D2, D1 = np.meshgrid(np.arange(X0, X1, dX), np.arange(Y0, Y1, dY))

Phi = get_Phi(D1, D2, pattern_type)
dPhi = gradient_c(torch.from_numpy(Phi), batched = False, delta_X = [dY, dX]).numpy() 
U, V = - dPhi[..., 1], dPhi[..., 0] 
fig1, ax1 = plt.subplots()
Q = ax1.quiver(D1, D2, U*1, V*1, np.hypot(U*1, V*1), units='x', pivot='tip', width=0.02, scale=1)
fig1.savefig(os.path.join(make_dir(os.path.join(save_fld, 'Stream')), 'Test_V.png'))
sitk.WriteImage(sitk.GetImageFromArray(Phi), os.path.join(make_dir(os.path.join(save_fld, 'Stream')), 'Test_Phi.nii')) # (D1, D2)
sitk.WriteImage(sitk.GetImageFromArray(np.array([U, V])), os.path.join(make_dir(os.path.join(save_fld, 'Stream')), 'Test_V.nii')) # (2, D1, D2)


U = get_U(D1, D2, pattern_type)
V = get_V(D1, D2, pattern_type)
fig1, ax1 = plt.subplots()
Q = ax1.quiver(D1, D2, U*1, V*1, np.hypot(U*1, V*1), units='x', pivot='tip', width=0.02, scale=1)
fig1.savefig(os.path.join(make_dir(os.path.join(save_fld, 'Vector')), 'Test_V.png'))
sitk.WriteImage(sitk.GetImageFromArray(np.array([U, V])), os.path.join(make_dir(os.path.join(save_fld, 'Vector')), 'Test_V.nii')) # (2, D1, D2)