import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn as nn
from utils import gradient_l2_loss, gradient_c, get_FA


class GradientLoss(nn.Module):
    def __init__(self, weight):
        super(GradientLoss, self).__init__()
        self.w = weight
    def forward(self, x_lst, batched = True):
        gl = 0.
        for i in range(len(x_lst)):
            gl += gradient_l2_loss(x_lst[i], None, self.w, batched = batched)
        return gl

class SpatialGradientLoss(nn.Module):
    def __init__(self, weight):
        super(SpatialGradientLoss, self).__init__()
        self.w = weight
    def forward(self, gt, pd, batched = True):
        sgl = 0.
        for it in range(gt.size(1)):
            sgl += gradient_l2_loss(gt[:, it] - pd[:, it], None, self.w, batched = batched)
        return sgl
        

class LaplacianLoss2D(nn.Module):
    ''' Push tensor's Laplacian to 0 '''
    def __init__(self, weight):
        super(LaplacianLoss2D, self).__init__()
        self.w = weight

    def forward(self, H, batched = True):
        ''' H: (r, c) '''
        dH = gradient_c(H, batched = batched) # (r, c, dim)
        div_dH = gradient_c(dH[..., 0], batched = batched)[..., 0] + \
            gradient_c(dH[..., 1], batched = batched)[..., 1]
        return (div_dH ** 2).mean() * self.w

class LaplacianLoss3D(nn.Module):
    ''' Push tensor's Laplacian to 0 '''
    def __init__(self, weight):
        super(LaplacianLoss3D, self).__init__()
        self.w = weight

    def forward(self, H, batched = True):
        ''' H: (s, r, c) '''
        dH = gradient_c(H, batched = batched) # (s, r, c, dim)
        div_dH = gradient_c(dH[..., 0], batched = batched)[..., 0] + \
            gradient_c(dH[..., 1], batched = batched)[..., 1] + \
                gradient_c(dH[..., 2], batched = batched)[..., 2] 
        return (div_dH ** 2).mean() * self.w

class DivergenceLoss3D(nn.Module):
    ''' Push vector's divergence to 0 '''
    def __init__(self, weight):
        super(DivergenceLoss3D, self).__init__()
        self.w = weight

    def forward(self, Phi, batched = True):
        ''' Phi: (n_batch, 3, s, r, c) '''
        if batched:
            d1_1 = gradient_c(Phi[:, 0], batched = True)[..., 0] # (n_batch, s, r, c)
            d2_2 = gradient_c(Phi[:, 1], batched = True)[..., 1] # (n_batch, s, r, c)
            d3_3 = gradient_c(Phi[:, 2], batched = True)[..., 2] # (n_batch, s, r, c)
        else:
            d1_1 = gradient_c(Phi[0], batched = False)[..., 0] # (s, r, c)
            d2_2 = gradient_c(Phi[1], batched = False)[..., 1] # (s, r, c)
            d3_3 = gradient_c(Phi[2], batched = False)[..., 2] # (s, r, c)
        div_Phi = d1_1 + d2_2 + d3_3
        return (div_Phi ** 2).mean() * self.w

class ScalarFrobenius(nn.Module):
    ''' Regularize scalar D '''
    def __init__(self, weight):
        super(ScalarFrobenius, self).__init__()
        self.w = weight

    def forward(self, D, batched = True):
        ''' D: (n_batch, (s), r, c): D '''
        return (D ** 2).mean() * self.w

class TensorFrobenius2D(nn.Module):
    ''' Regularize tensor D '''
    def __init__(self, weight):
        super(TensorFrobenius2D, self).__init__()
        self.w = weight

    def forward(self, D, batched = True):
        ''' D: (n_batch, 3, r, c): Dxx, Dxy, Dyy'''
        FL = D[:, 0] ** 2 + D[:, 1] ** 2 * 2 + D[:, 2] ** 2
        return FL.mean() * self.w

class TensorFrobenius3D(nn.Module):
    ''' Regularize tensor D '''
    def __init__(self, weight):
        super(TensorFrobenius3D, self).__init__()
        self.w = weight

    def forward(self, D, batched = True, vessel_mask = None):
        ''' D: (n_batch, 6, r, c): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz'''
        FL = D[:, 0] ** 2 + D[:, 1] ** 2 * 2 + D[:, 2] ** 2 * 2 + D[:, 3] ** 2 + D[:, 4] ** 2 * 2 + D[:, 5] ** 2
        if vessel_mask is not None:
            FL *= vessel_mask
        #FL = D[:, 1] ** 2 * 2 + D[:, 2] ** 2 * 2 + D[:, 4] ** 2 * 2 # non-diagonal penalty
        return FL.mean() * self.w


class DualAnisotropyLoss3D(nn.Module):
    ''' 
    Regularize dual basis constructed tensor D as isotropic
    Symmetric diffusion tensor based on Dual basis (https://www.sciencedirect.com/science/article/pii/S1361841502000531) 
    '''
    def __init__(self, weight):
        super(DualAnisotropyLoss3D, self).__init__()
        self.w = weight

    def forward(self, L, batched = True):
        ''' 
        Dual basis (3D) L: (n_batch, 6, s, r, c): L1, L2, L3, L4, L5, L6 
        '''
        avg = L.mean(dim = 1)
        den = 5 * (L ** 2).sum(dim = 1) ** 2 / 6. 
        den[den == 0] = 1e-14
        num = (L[:, 0] - avg) ** 2 + (L[:, 1] - avg) ** 2 + (L[:, 2] - avg) ** 2 + (L[:, 3] - avg) ** 2 + (L[:, 4] - avg) ** 2 + (L[:, 5] - avg) ** 2
        return (num / den).sqrt().mean() * self.w

class SpectralColorOrientLoss2D(nn.Module):
    ''' 
    Regularize tensor D's major orientation as spatially smooth
    For spectral basis: Spectral constructed diffusion tensor (https://github.com/Lezcano/expRNN)
    '''
    def __init__(self, weight):
        super(SpectralColorOrientLoss2D, self).__init__()
        self.w = weight

    def forward(self, L, U, batched = True):
        ''' 
        Spectral basis (2D) 
        L: (n_batch, 2, r, c): L1, L2
        U: (n_batch, 4, r, c): u1_1, u2_1, u1_2, u2_2
        '''
        '''_, indices = torch.sort(L, dim = 1, descending = True)
        indices = indices.repeat(1, 2, 1, 1).view(U.size(0), 2, 2, U.size(-2), U.size(-1))
        major_eigenvec = (U.view(indices.size())).gather(dim = 2, index = indices)[:, :, 0] # (n_batch, 2, 2, r, c) -> (n_batch, 2, (major_eigen), r, c)
        #major_eigenvec = U[:, :, -1]
        return (gradient_l2_loss(major_eigenvec[:, 0], None, self.w, batched = batched) + \
            gradient_l2_loss(major_eigenvec[:, 1], None, self.w, batched = batched)).mean() * self.w'''
        cl = 0.
        for i in range(U.size(1)):
            cl += gradient_l2_loss(U[:, i], None, self.w, batched = batched)
        return cl.mean() * self.w


class SpectralAnisotropyLoss2D(nn.Module):
    ''' 
    Regularize tensor D as isotropic
    For spectral basis: Spectral constructed diffusion tensor (https://github.com/Lezcano/expRNN)
    '''
    def __init__(self, weight):
        super(SpectralAnisotropyLoss2D, self).__init__()
        self.w = weight

    def forward(self, L, batched = True):
        ''' 
        Spectral basis (2D) L: (n_batch, 2, r, c): L1, L2 (eigen-values)
        '''
        #avg = L.mean(dim = 1)
        den = (L ** 2).sum(dim = 1) ** 2 * 2. 
        #den = avg ** 2 * 2. 
        den[den < 1e-14] = 1e-14
        num = (L[:, 0] - L[:, 1]) ** 2
        fa = (num / den).sqrt()
        return fa.mean() * self.w
        #return fa[fa <= 1.].mean() * self.w

        
class SpectralAnisotropyLoss3D(nn.Module):
    ''' 
    Regularize tensor D as isotropic
    For spectral basis: Spectral constructed diffusion tensor (https://github.com/Lezcano/expRNN)
    '''
    def __init__(self, weight):
        super(SpectralAnisotropyLoss3D, self).__init__()
        self.w = weight

    def forward(self, L, batched = True):
        ''' 
        Spectral basis (3D) L: (n_batch, 3, s, r, c): L1, L2, L3 (eigen-values)
        '''
        den = torch.sum(L ** 2, dim = 1)
        den[den < 1e-14] = 1e-14
        eva1, eva2, eva3 = L[:, 0], L[:, 1], L[:, 2]

        '''FA = 0.5 * ((eva1 - eva2) ** 2 + \
            (eva2 - eva3) ** 2 + (eva3 - eva1) ** 2) \
                / den'''
        FA = torch.sqrt(0.5 * ((eva1 - eva2) ** 2 + 
            (eva2 - eva3) ** 2 + (eva3 - eva1) ** 2) 
                / den)  # NOTE: Must Sqrt
        mask = (FA > 0.3).float() # TODO
        return (FA * mask).mean() * self.w
        #return FA.mean() * self.w


class TensorAnisotropyLoss3D(nn.Module):
    ''' 
    Regularize 3x3 tensor D as isotropic
    '''
    def __init__(self, weight):
        super(TensorAnisotropyLoss3D, self).__init__()
        self.w = weight

    def forward(self, D, batched = True):
        ''' 
        Tensor D: (n_batch, 6, s, r, c): Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
        '''
        TA = D[:, 1] ** 2 + D[:, 2] ** 2 + D[:, 4] ** 2
        return TA.mean() * self.w


class ColorOrientL2Loss3D(nn.Module):
    ''' 
    Regularize tensor D's major orientation as spatially smooth
    For spectral basis: Spectral constructed diffusion tensor (https://github.com/Lezcano/expRNN)
    '''
    def __init__(self, weight):
        super(ColorOrientL2Loss3D, self).__init__()
        self.w = weight
        self.criterion = nn.MSELoss()

    def forward(self, L, U, GT_CO, batched = True):
        '''
        GT_CO: (batch, 3, s, r, c)
        '''
        CO = 0.
        # For max eigenvector U_max
        _, indices = torch.sort(L, dim = 1, descending = True)
        indices = indices.repeat(1, 3, 1, 1, 1).view(U.size(0), 3, 3, U.size(-3), U.size(-2), U.size(-1))
        major_eigenvec = (U.view(indices.size())).gather(dim = 2, index = indices)[:, :, 0] # (n_batch, 3, 3, s, r, c) -> (n_batch, 3, (major_eigen), s, r, c)
        abs_major_eigenvec = abs(major_eigenvec) # (batch, 3, s, r, c)

        return self.criterion(abs_major_eigenvec, GT_CO) * self.w


class SpectralColorOrientLoss3D(nn.Module):
    ''' 
    Regularize tensor D's major orientation as spatially smooth
    For spectral basis: Spectral constructed diffusion tensor (https://github.com/Lezcano/expRNN)
    '''
    def __init__(self, weight):
        super(SpectralColorOrientLoss3D, self).__init__()
        self.w = weight

    def forward(self, L, U, batched = True):
        ''' 
        Spectral basis (3D) 
        L: (n_batch, 3, s, r, c): L1, L2, L3
        U: (n_batch, 9, s, r, c): U1_1, U2_1, U3_1, U1_2, U2_2, U3_2, U1_3, U2_3, U3_3
        '''
        CO = 0.

        # For all U1, U2, U3
        '''U = abs(U)
        for i in range(3):
            evec1, evec2, evec3 =U[:, i], U[:, i+3], U[:, i+6]
            den = evec1 + evec2 + evec3
            den[den < 1e-14] = 1e-14
            CO += torch.sqrt(0.5 * ((evec1 - evec2) ** 2 + \
                (evec2 - evec3) ** 2 + (evec3 - evec1) ** 2) / den) # NOTE: Must Sqrt'''

        # For max eigenvector U_max
        _, indices = torch.sort(L, dim = 1, descending = True)
        indices = indices.repeat(1, 3, 1, 1, 1).view(U.size(0), 3, 3, U.size(-3), U.size(-2), U.size(-1))
        major_eigenvec = (U.view(indices.size())).gather(dim = 2, index = indices)[:, :, 0] # (n_batch, 3, 3, s, r, c) -> (n_batch, 3, (major_eigen), s, r, c)
        abs_major_eigenvec = abs(major_eigenvec)

        den = torch.sum(abs_major_eigenvec ** 2, dim = 1)
        den[den < 1e-14] = 1e-14
        evec1, evec2, evec3 = abs_major_eigenvec[:, 0], abs_major_eigenvec[:, 1], abs_major_eigenvec[:, 2]

        CO = torch.sqrt(0.5 * ((evec1 - evec2) ** 2 + \
            (evec2 - evec3) ** 2 + (evec3 - evec1) ** 2) / den)
        #mask = (CO > 0.6).float() # TODO
        #CO = CO * mask

        CO_grad = 0.
        #for i in range(3):
        #    CO_grad += gradient_l2_loss(major_eigenvec[:, i], None, self.w, batched = batched)
        for i in range(9):
            CO_grad += gradient_l2_loss(U[:, i], None, self.w, batched = batched)

        return (CO + CO_grad).mean() * self.w

        #return (gradient_l2_loss(major_eigenvec[:, 0], None, self.w, batched = batched) + \
        #    gradient_l2_loss(major_eigenvec[:, 1], None, self.w, batched = batched) + \
        #        gradient_l2_loss(major_eigenvec[:, 2], None, self.w, batched = batched)).mean() * self.w
        '''gl = 0.
        for i in range(3):
            gl += (gradient_l2_loss(L[:, i], None, self.w, batched = batched) + \
                gradient_l2_loss(U[:, 3 * i], None, self.w, batched = batched) + 
                gradient_l2_loss(U[:, 3 * i + 1], None, self.w, batched = batched) + 
                gradient_l2_loss(U[:, 3 * i + 2], None, self.w, batched = batched)).mean()
        return gl * self.w'''