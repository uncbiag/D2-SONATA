import torch
import numpy as np


'''
Available Patterns:
gaussian
'''

def gaussian(data_grid, center, radius, strength, device = 'cpu'): 
    #print('strength:', strength) 
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        value_mask = torch.exp(torch.tensor([[(- ((X[i]- center[0]) / radius[0]) ** 2 - ((Y[j]- center[1]) / radius[1]) ** 2) for j in range(len(Y))] for i in range(len(X))], \
            dtype = torch.float, device = device))
    elif len(data_grid) == 3:
        X, Y, Z = data_grid[0], data_grid[1], data_grid[2]
        value_mask = torch.exp(torch.tensor([[[(- ((X[i]- center[0]) / radius[0]) ** 2 - ((Y[j]- center[1]) / radius[1]) ** 2 - ((Z[k]- center[2]) / radius[2]) ** 2) \
            for k in range(len(Z))] for j in range(len(Y))] for i in range(len(X))], \
            dtype = torch.float, device = device))
    else:
        raise ValueError('Unsupported dimension:', len(data_grid))
    value_mask = 1 - value_mask * strength # strength > 0: anomaly < 1; strenth < 0: anomaly > 1; strength = 0: normal.
    return value_mask


def gaussian_resize(data_dim, ctr_lst = None, r_lst = None, strength = 1., device = 'cpu'):  
    if len(data_dim) == 3: 
        # referenece grid #
        init_loc = [-16, -16, -16] # [-20, -20] 
        ref_d = 0.5
        ref_dim = 64
        
        # real grid #
        nX, nY, nZ = data_dim[0], data_dim[1], data_dim[2]
        dX = ref_d * ref_dim / nX 
        dY = ref_d * ref_dim / nY 
        dZ = ref_d * ref_dim / nZ 

        X = init_loc[0] + dX * np.arange(nX)
        Y = init_loc[1] + dY * np.arange(nY)
        Z = init_loc[2] + dZ * np.arange(nZ)

        if ctr_lst is None:
            mask_x0 = init_loc[0] + (np.random.random_sample() * 0.5 + 0.2) * nX * dX
            mask_y0 = init_loc[1] + (np.random.random_sample() * 0.5 + 0.2) * nY * dY
            mask_z0 = init_loc[0] + (np.random.random_sample() * 0.5 + 0.2) * nX * dX
            center = [mask_x0, mask_y0, mask_z0]
            
            mask_rx = (np.random.random_sample() * 0.3 + 0.1) * nX * dX
            mask_ry = (np.random.random_sample() * 0.3 + 0.1) * nY * dY 
            mask_rz = (np.random.random_sample() * 0.3 + 0.1) * nZ * dZ
            radius = [mask_rx, mask_ry, mask_rz]
        else:
            center = ctr_lst
            radius = r_lst
        
        #print('Applied value mask: center = [%.1f, %.1f, %.1f], radius = [%.1f, %.1f, %.1f]' % (mask_x0, mask_y0, mask_z0, mask_rx, mask_ry, mask_rz))

        lesion_mask = torch.exp(torch.tensor([[[(- ((X[i]- center[0]) / radius[0]) ** 2 - ((Y[j]- center[1]) / radius[1]) ** 2 - ((Z[k]- center[2]) / radius[2]) ** 2) \
            for k in range(len(Z))] for j in range(len(Y))] for i in range(len(X))], \
            dtype = torch.float, device = device))
    else:
        raise ValueError('Unsupported dimension:', len(data_dim))
    
    value_mask = 1 - lesion_mask * strength # strength > 0: anomaly < 1; strenth < 0: anomaly > 1; strength = 0: normal.
    #value_mask = torch.where(value_mask < 0.8, value_mask, torch.ones_like(value_mask)) # value larger than 0.8 --> normal
    
    return value_mask, center, radius