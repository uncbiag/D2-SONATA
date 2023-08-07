import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter

from utils import make_dir, gradient_c, divergence3D_numpy, gradient_c_numpy, divergence3D, tensor_percentile, numpy_percentile, gradient_f_numpy



def K(x, percentile = 100):
    '''
    Noise calculator
    Ref: Perona, P., Malik, J.: Scale-space and edge detection using anisotropic diffusion. 
    IEEE Transactions on Pattern Analysis and Machine Intelligence 12(7), 629–639 (1990). 
    https://doi.org/10.1109/34.56205
    '''
    if not isinstance(x, (np.ndarray)):
        return numpy_percentile(x, percentile)
    else:
        return tensor_percentile(x, percentile)

def sqr_norm(x):
    '''
    :return: Spuared L2 norm of input x
    '''
    return (abs(x) ** 2).sum(-1)

def ani_func(x, k, ani_type = '1'):
    '''
    Compute anisotropic coefficient
    '''
    if ani_type is 'exp':
        return torch.exp(- x / k ** 2)
    elif ani_type is 'frac':
        return 1 / (1 + x / k ** 2)
    else:
        raise NotImplementedError('Assigned anisotropic filter type not supported!')


def set_neumann(X):
    dim = len(X.shape)
    if dim == 1:
        return np.pad(X[1:-1], 1, 'edge')
    elif dim == 2:
        return np.pad(X[1:-1, 1:-1], 1, 'edge')
    elif dim == 3:
        return np.pad(X[1:-1, 1:-1, 1:-1], 1, 'edge')
    else:
        raise NotImplementedError

def diffuser(X, diff_coeff, dt = 0.01, nt = 10, isVector = True):
    '''
    X: (s, r, c, (vector_dim))
    diff_coeff: (s, r, c)
    delta_X = div(diff_coeff * \ndabla X)
    '''
    if isVector:
        delta_X = torch.zeros_like(X)
        for it in range(nt):
            for dim in range(3):
                delta_X[..., dim] = divergence3D(diff_coeff[..., None] * gradient_c(X[..., dim], batched = False))
            X += delta_X * dt
    else:
        for it in range(nt):
            delta_X = divergence3D(diff_coeff[..., None] * gradient_c(X, batched = False))
            X += delta_X * dt
    return X

def diffuser_numpy(X, diff_coeff, dt = 0.01, nt = 1, VectorToSmooth = True, VectorCoeff = True, data_psacing = [1., 1., 1.]):
    '''
    X: ((s), r, c, (vector_dim))
    diff_coeff: ((s), r, c)
    delta_X = div(diff_coeff * \ndabla X)
    '''
    if VectorToSmooth:
        dimension = len(X.shape) - 1
        if not VectorCoeff:
            diff_coeff = np.transpose(np.array([diff_coeff] * dimension), \
                                      tuple([i+1 for i in range(dimension)] + [0]))
        delta_X = np.zeros(X.shape)
        for it in range(nt):
            for axis in range(dimension):
                # Neumann B.C.
                X[..., axis] = set_neumann(X[..., axis])
                delta_X[..., axis] = divergence3D_numpy(diff_coeff[..., axis][..., None] \
                                * gradient_f_numpy(X[..., axis], batched = False, delta_lst = data_psacing), batched = False, data_spacing = data_psacing)
                #delta_X[..., axis] = divergence3D_numpy(diff_coeff[..., axis][..., None] \
                #                * np.transpose(np.array(np.gradient(X[..., axis])), \
                #                               tuple([i+1 for i in range(dimension)] + [0])))
            X += delta_X * dt
    else:
        dimension = len(X.shape)
        diff_coeff = np.array([diff_coeff] * dimension)
        for it in range(nt):
            # Neumann B.C
            for axis in range(dimension):
                X[..., axis] = set_neumann(X[..., axis])
            delta_X = divergence3D_numpy(diff_coeff * gradient_f_numpy(X, batched = False, delta_lst = data_psacing), batched = False, data_spacing = data_psacing)
            #print(np.mean(delta_X))
            X += delta_X * dt
    return X 


class Diffuser(nn.Module): # Diffuser skeleton given diffusion coefficients #
    '''
    Refs: 
    [1] Perona, P., Malik, J.: Scale-space and edge detection using anisotropic diffusion. 
        IEEE Transactions on Pattern Analysis and Machine Intelligence 12(7), 629–639 (1990). 
        https://doi.org/10.1109/34.56205
    [2] Chourmouzios, T., Maria, P.: On the choice of the parameters for anisotropic diffusion in image processing. 
        Pattern Recognition (2012). http://www.commsp.ee.ic.ac.uk/~ctsiotsi/pubs/AD_Tsiotsios.pdf 
    '''
    def __init__(self, args):
        super(Diffuser, self).__init__()
        self.args = args
        self.dt = args.dt
        self.nt = args.nt 
    
    def diff_gradient(self, X):
        pass

    def diffusing(self, X, diff_coeff, dt = 0.01, nt = 10, isVector = True):
        '''
        X: (s, r, c, (vector_dim))
        diff_coeff: (s, r, c)
        delta_X = div(diff_coeff * \ndabla X)
        '''
        if isVector:
            delta_X = torch.zeros_like(X)
            for it in range(nt):
                for axis in range(3):
                    delta_X[..., axis] = divergence3D(diff_coeff[..., None] * gradient_c(X[..., axis], batched = False))
                X += delta_X * dt
        else:
            for it in range(nt):
                delta_X = divergence3D(diff_coeff * gradient_c_numpy(X, batched = False))
                X += delta_X * dt
        return X



class GaussianSmoother(nn.Module):
    '''
    Fixed gaussian smoothing layer
    '''
    def __init__(self, kernel_size = 3, sigma = 1, device = 'cpu'):
        super(GaussianSmoother, self).__init__()
        self.device = device
        self.conv = nn.Conv3d(1, 1, kernel_size, stride = 1, set_neumann = int(kernel_size / 2), bias = None)
        self.weights_init(kernel_size, sigma)

    def forward(self, X):
        '''
        return conv-ed X: (s, l, c)
        '''
        return self.conv((X.unsqueeze(0)).unsqueeze(0))[0, 0]
    
    def weights_init(self, kernel_size, sigma):
        w = np.zeros((kernel_size, kernel_size, kernel_size))
        center = int(kernel_size / 2)
        w[center, center, center] = 1
        k = gaussian_filter(w, sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.to(self.device)



if __name__ == '__main__':

    main_fld = make_dir('/home/peirong/Desktop/SmoothersTest')
    

        
