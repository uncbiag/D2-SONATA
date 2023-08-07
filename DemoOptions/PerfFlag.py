import numpy as np
import torch

'''
Available Patterns:
adv_diff
adv_only
diff_only
mix
circle_adv_only
'''

# Basic Functions
def adv_diff(data_dim, D_type, V_type, device):
    isPerf = {}
    if V_type is 'constant' or V_type is 'scalar':
        isPerf.update(isV = torch.ones(data_dim, dtype = torch.float, device = device))
    else:
        isPerf.update(isVx = torch.ones(data_dim, dtype = torch.float, device = device), \
            isVy = torch.ones(data_dim, dtype = torch.float, device = device))
    if D_type is 'constant' or D_type is 'scalar':
        isPerf.update(isD = torch.ones(data_dim, dtype = torch.float, device = device))
    elif D_type is 'diag':
        isPerf.update(isDxx = torch.ones(data_dim, dtype = torch.float, device = device))
        isPerf.update(isDyy = torch.ones(data_dim, dtype = torch.float, device = device))
    elif 'full' in D_type:
        isPerf.update(isDxx = torch.ones(data_dim, dtype = torch.float, device = device))
        isPerf.update(isDyy = torch.ones(data_dim, dtype = torch.float, device = device))
        isPerf.update(isDxy = torch.ones(data_dim, dtype = torch.float, device = device))
    return isPerf
    
def adv_only(data_dim, D_type, V_type, device):
    isPerf = {}
    if V_type is 'constant' or V_type is 'scalar':
        isPerf.update(isV = torch.ones(data_dim, dtype = torch.float, device = device))
    else:
        isPerf.update(isVx = torch.ones(data_dim, dtype = torch.float, device = device), \
            isVy = torch.ones(data_dim, dtype = torch.float, device = device))
    return isPerf

def diff_only(data_dim, D_type, V_type, device):
    isPerf = {}
    if D_type is 'constant' or D_type is 'scalar':
        isPerf.update(isD = torch.ones(data_dim, dtype = torch.float, device = device))
    elif D_type is 'diag':
        isPerf.update(isDxx = torch.ones(data_dim, dtype = torch.float, device = device))
        isPerf.update(isDyy = torch.ones(data_dim, dtype = torch.float, device = device))
    elif 'full' in D_type:
        isPerf.update(isDxx = torch.ones(data_dim, dtype = torch.float, device = device))
        isPerf.update(isDyy = torch.ones(data_dim, dtype = torch.float, device = device))
        isPerf.update(isDxy = torch.ones(data_dim, dtype = torch.float, device = device))
    return isPerf


# Extended Functions
def mix(data_dim, D_type, V_type, device):
    isPerf = adv_diff(data_dim, D_type, V_type, device)
    for key in isPerf:
        if 'V' in key:
            #pass
            isPerf[key][...] = 1.
            isPerf[key][:int(np.rint(data_dim[0] * 2 / 4)), :int(np.rint(data_dim[1] * 2 / 4))] = 0.
        else:
            isPerf[key][...] = 0.
            isPerf[key][:int(np.rint(data_dim[0] * 3 / 5)), :int(np.rint(data_dim[1] * 3 / 5))] = 1.
    return isPerf



def circle_adv_only(data_dim, D_type, V_type, device):
    isPerf = adv_only(data_dim, D_type, V_type, device)
    for key in isPerf:
        isPerf[key] *= 0
    unit = round(data_dim[0] / 15) # cut into 15 units

    Lx = unit * 5
    Ly = unit * 8
    corner = unit * 2

    x0 = unit * 3
    x1 = x0 + corner
    x2 = x1 + Lx
    x3 = x2 + corner

    y0 = unit * 1
    y1 = y0 + corner
    y2 = y1 + Ly
    y3 = y2 + corner
    
    for i in range(corner):
        x_top = x0 + i
        x_bottom = x3 - i
        y_left = y0 + i
        y_right = y3 - i
        value = ((corner - i) / (corner)) ** 3
        isPerf['isVx'][x0+1 : x3, y0 + i] = - value
        isPerf['isVx'][x0 : x3-1, y3 - (i+1)] = value
        isPerf['isVy'][x0 + i, y0 : y3-1] = value
        isPerf['isVy'][x3 - (i+1), y0+1 : y3] = - value
    #    isPerf['isVx'][x0 + i : x3 - i, y0 + i] = - value
    #    isPerf['isVx'][x0 + i : x3 - i, y3 - (i + 1)] =  value
    #    isPerf['isVy'][x0 + i, y0 + i : y3 - i] =  value
    #    isPerf['isVy'][x3 - (i + 1), y0 + i : y3 - i] = - value

    return isPerf



def rect_adv_only(data_dim, D_type, V_type, device):
    isPerf = adv_only(data_dim, D_type, V_type, device)
    for key in isPerf:
        isPerf[key] *= 0
    unit = round(data_dim[0] / 15) # cut into 15 units
    #isPerf['isVx'][:, unit * 6 : unit * 8] = 1.
    isPerf['isVy'][unit * 3: unit * 5, unit : unit * 12 - 1] = 1.
    return isPerf