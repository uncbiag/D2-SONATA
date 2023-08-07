import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt

from utils import make_dir


save_fld = '/media/peirong/PR5/demo/datasets/diff_only/heat'


Nx, Ny = [200, 200]
C0 = torch.rand(Nx, Ny, dtype=torch.double).reshape(1, 1, Nx, Ny)
C = C0
laplacian = torch.tensor([[[[0, 1., 0],
                           [1, -4, 1],
                           [0, 1, 0]]]], dtype=torch.double)
BC = nn.ReplicationPad2d(1)
movie = [C0[0, 0].numpy()] # (200, 200)

for i in range(300):
    C = .01 * nn.functional.conv2d(BC(C), laplacian) + C
    movie.append(C[0, 0].numpy())

sitk.WriteImage(sitk.GetImageFromArray(np.array(movie), isVector = False), os.path.join(save_fld, 'Heat.nii'))


#plt.figure(figsize=(8,4))
#plt.subplot(121)
#plt.imshow(C0.numpy()[0,0,:,:], cmap='inferno')
#plt.subplot(122)
#plt.imshow(C.numpy()[0,0,:,:], cmap='inferno')

