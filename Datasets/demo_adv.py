import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from utils import make_dir

def gaussian(data_grid, value, device = 'cpu'):
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        U0 = np.exp(np.array([[(- X[i] ** 2 - Y[j] ** 2)/0.2 for j in range(len(Y))] for i in range(len(X))])) * value
    return U0

save_fld = '/media/peirong/PR5/demo/datasets/diff_only/heat'

value = 1.
nT = 100
x0, y0 = -1, -1
dx, dy = 0.05, 0.05
nX, nY = 64, 64
X = x0 + np.arange(nX) * dx
Y = y0 + np.arange(nY) * dy
U0 = gaussian([X, Y], value)
save_fld = make_dir('/media/peirong/PR5/demo/datasets/adv_only/dataset/0')
save_path = os.path.join(save_fld, 'Perf.nii')

U = [U0]
for it in range(1, nT):
    X, Y = X - 0.008, Y - 0.015
    U.append(gaussian([X, Y], value))

sitk.WriteImage(sitk.GetImageFromArray(np.array(U)), save_path)



