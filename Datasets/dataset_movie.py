import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
from torch.autograd import Variable

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from torch.utils.data import Dataset
from scipy.ndimage.morphology import binary_closing, binary_fill_holes

from utils import * 

######################################################################
##############################  Utils  ###############################
######################################################################

#neumann_2D = torch.nn.ReplicationPad2d(1)
#neumann_3D = torch.nn.ReplicationPad3d(1)
#def neumann_BC_2D(X): # X: (batch, r, c)
#    return neumann_2D(X[:, 1:-1, 1:-1].unsqueeze(1))[:, 0]
#def neumann_BC_3D(X): # X: (batch, r, c)
#    return neumann_3D(X[:, 1:-1, 1:-1, 1:-1].unsqueeze(1))[:, 0]

def extract_BC_2D(X, BC_size, to_variable = True): # X: (nT, r, c)
    BC = X.clone()
    BC_r1, BC_r2 = Variable(BC[:, : BC_size]), Variable(BC[:, - BC_size :]) # (nT, BC_size, n_col)
    BC_c1, BC_c2 = Variable(BC[:, :, : BC_size]), Variable(BC[:, :, - BC_size :]) # (nT, n_row, BC_size)
    return [[BC_r1, BC_r2], [BC_c1, BC_c2]] # List
    # NOTE: r == c
    #BC_c1, BC_c2 = BC[:, :, : BC_size].permute(0, 2, 1), BC[:, :, - BC_size :].permute(0, 2, 1) # (nT, r, BC_size) -> (nT, BC_size, c)
    #return torch.stack([BC_r1, BC_r2, BC_c1, BC_c2], dim = 1) # (nT, 4, BC_size, c) # NOTE: r == c
def extract_BC_3D(X, BC_size, to_variable = True): # X: (nT, s, r, c)
    BC = X.clone()
    BC_s1, BC_s2 = Variable(BC[:, : BC_size]), Variable(BC[:, - BC_size :]) # (nT, BC_size, n_row, n_col)
    BC_r1, BC_r2 = Variable(BC[:, :, : BC_size]), Variable(BC[:, :, - BC_size :]) # (nT, n_slice, BC_size, n_col)
    BC_c1, BC_c2 = Variable(BC[:, :, :, : BC_size]), Variable(BC[:, :, :, - BC_size :]) # (nT, n_slice, n_row, BC_size)
    return [[BC_s1, BC_s2], [BC_r1, BC_r2], [BC_c1, BC_c2]] # List
    # NOTE: r == c
    #BC_r1, BC_r2 = BC[:, :, : BC_size].permute(0, 2, 1, 3), BC[:, :, - BC_size :].permute(0, 2, 1, 3) # (nT, s, BC_size, c) -> (nT, BC_size, s, c) # NOTE: s=r=c
    #BC_c1, BC_c2 = BC[:, :, :, : BC_size].permute(0, 3, 1, 2), BC[:, :, :, - BC_size :].permute(0, 3, 1, 2) # (nT, s, r, BC_size) -> (nT, BC_size, s, r) # NOTE: s=r=c
    #return torch.stack([BC_s1, BC_s2, BC_r1, BC_r2, BC_c1, BC_c2], dim = 1) # (nT, 6, BC_size, r, c)

''' dBC[t] = BC[t+1] - BC[t] '''
def extract_dBC_2D(X, BC_size): # X: (nT, r, c)
    BC = X.clone()
    dBC = torch.zeros_like(X)
    BC[:, BC_size : - BC_size, BC_size : - BC_size] = 0
    dBC[:-1] = BC[1:] - BC[:-1] 
    dBC_r1, dBC_r2 = dBC[:, : BC_size], dBC[:, - BC_size :]  # (nT, BC_size, n_col)
    dBC_c1, dBC_c2 = dBC[:, :, : BC_size], dBC[:, :, - BC_size :] # (nT, n_row, BC_size)
    return [[dBC_r1, dBC_r2], [dBC_c1, dBC_c2]] # List
    # NOTE: r == c
    #dBC_c1, dBC_c2 = dBC[:, :, : BC_size].permute(0, 2, 1), dBC[:, :, - BC_size :].permute(0, 2, 1) # (nT, r, BC_size) -> (nT, BC_size, c)
    #return torch.stack([dBC_r1, dBC_r2, dBC_c1, dBC_c2], dim = 1) # (nT, 4, BC_size, c)
def extract_dBC_3D(X, BC_size): # X: (nT, s, r, c)
    BC = X.clone()
    dBC = torch.zeros_like(X)
    BC[:, BC_size : - BC_size, BC_size : - BC_size, BC_size : - BC_size] = 0
    dBC[:-1] = BC[1:] - BC[:-1]
    dBC_s1, dBC_s2 = dBC[:, : BC_size], dBC[:, - BC_size :] # (nT, BC_size, n_row, n_col)
    dBC_r1, dBC_r2 = dBC[:, :, : BC_size], dBC[:, :, - BC_size :] # (nT, n_slice, BC_size, n_col)
    dBC_c1, dBC_c2 = dBC[:, :, :, : BC_size], dBC[:, :, :, - BC_size :] # (nT, n_slice, n_row, BC_size)
    return [[dBC_s1, dBC_s2], [dBC_r1, dBC_r2], [dBC_c1, dBC_c2]] # List
    # NOTE: r == c
    #dBC_r1, dBC_r2 = dBC[:, :, : BC_size].permute(0, 2, 1, 3), dBC[:, :, - BC_size :].permute(0, 2, 1, 3) # (nT, s, BC_size, c) -> (nT, BC_size, s, c) # NOTE: s=r=c
    #dBC_c1, dBC_c2 = dBC[:, :, :, : BC_size].permute(0, 3, 1, 2), dBC[:, :, :, - BC_size :].permute(0, 3, 1, 2) # (nT, s, r, BC_size) -> (nT, BC_size, s, r) # NOTE: s=r=c
    #return torch.stack([dBC_s1, dBC_s2, dBC_r1, dBC_r2, dBC_c1, dBC_c2], dim = 1) # (nT, 6, BC_size, r, c)

######################################################################
################################  2D  ################################
######################################################################

class MovieDataset_2D(Dataset):
    '''
    For testing
    len(dataset) = avail_row * avail_col
    time points are randomly selected for each sample
    ''' 
    def __init__(self, args, loss_time_frame, patch_size, spatial_stride, MoviePath, PerfFlagPath, DPath, LPath, UPath, SPath, VPath):

        self.args = args
        self.BC = args.BC
        self.patch_size = patch_size 
        self.loss_time_frame = loss_time_frame

        self.DPath, self.LPath, self.UPath, self.SPath, self.VPath = DPath, LPath, UPath, SPath, VPath
        if self.DPath:
            self.D = sitk.GetArrayFromImage(sitk.ReadImage(self.DPath['D']))
            #self.Dxx = sitk.GetArrayFromImage(sitk.ReadImage(self.DPath['Dxx']))
            #self.Dxy = sitk.GetArrayFromImage(sitk.ReadImage(self.DPath['Dxy']))
            #self.Dyy = sitk.GetArrayFromImage(sitk.ReadImage(self.DPath['Dyy']))
        if self.LPath:
            self.L1 = sitk.GetArrayFromImage(sitk.ReadImage(self.LPath['L1']))
            self.L2 = sitk.GetArrayFromImage(sitk.ReadImage(self.LPath['L2']))
        if self.UPath:
            self.U = sitk.GetArrayFromImage(sitk.ReadImage(self.UPath['U']))
        if self.SPath:
            self.S = sitk.GetArrayFromImage(sitk.ReadImage(self.SPath['S']))
        if self.VPath:
            self.Vx = sitk.GetArrayFromImage(sitk.ReadImage(self.VPath['Vx']))
            self.Vy = sitk.GetArrayFromImage(sitk.ReadImage(self.VPath['Vy']))

        self.PerfFlagPath = PerfFlagPath
        if self.PerfFlagPath:
            self.isV = sitk.GetArrayFromImage(sitk.ReadImage(self.PerfFlagPath['V']))
            self.isD = sitk.GetArrayFromImage(sitk.ReadImage(self.PerfFlagPath['D']))
        self.movie_img  = sitk.ReadImage(MoviePath)
        self.movie_nda = sitk.GetArrayFromImage(self.movie_img) # (nT, r, c)
        self.data_dim = [self.movie_nda.shape[1], self.movie_nda.shape[2]] # (nT, r, c)
        self.origin, self.spacing, self.direction = self.movie_img.GetOrigin(), self.movie_img.GetSpacing(), self.movie_img.GetDirection()
        self.data_spacing = [self.spacing[1], self.spacing[0]]
        self.nT       = self.movie_nda.shape[0]

        self.sub_collocation_t = torch.from_numpy(np.arange(self.args.sub_collocation_nt) * self.args.dt) 
        self.input_time_frame = min(args.input_time_frame, self.nT) # input all available time points for predicting V, D # NOTE: TBD during training

        self.stride_row, self.stride_col = spatial_stride[0], spatial_stride[1]
        self.row, self.col = self.movie_nda.shape[1], self.movie_nda.shape[2]
        self.patch_row = self.row if patch_size[0] < 1 else self.patch_size[0]
        self.patch_col = self.col if patch_size[1] < 1 else self.patch_size[1]
        assert self.patch_row <= self.row and self.patch_col <= self.col
        #self.avail_it = int(self.nT - self.batch_nT + 1)
        self.avail_row = int(np.floor((self.row - self.patch_row) / self.stride_row) + 1)
        if ((self.avail_row - 1) * self.stride_row) + self.patch_row < self.row: 
            self.avail_row += 1
        self.avail_col = int(np.floor((self.col - self.patch_col) / self.stride_col) + 1)
        if ((self.avail_col - 1) * self.stride_col) + self.patch_col < self.col: 
            self.avail_col += 1
        #print('len(dataset):', int(self.avail_row * self.avail_col))
        #print('Available start it:', self.avail_it)
        print('Available start row:', self.avail_row)
        print('Available start col:', self.avail_col)

    def __len__(self):  
        #return int(self.avail_row * self.avail_col * self.avail_it)
        return int(self.avail_row * self.avail_col)
    
    @property
    def idx2info(self):
        #info_nda = np.zeros((int(self.avail_row * self.avail_col * self.avail_it), 3, 2)) 
        info_nda = np.zeros((int(self.avail_row * self.avail_col), 2, 2)) 
        for idx in range(info_nda.shape[0]):
            #i_row, i_col, i_t = np.unravel_index(idx, (self.avail_row, self.avail_col, self.avail_it))
            i_row, i_col = np.unravel_index(idx, (self.avail_row, self.avail_col))
            if i_row == self.avail_row - 1:
                i_row = self.row - self.patch_row
            else:
                i_row = self.stride_row * i_row
            if i_col == self.avail_col - 1:
                i_col = self.col - self.patch_col
            else:
                i_col = self.stride_col * i_col
            #info_nda[idx] = np.array([[i_row, i_row + self.patch_row], [i_col, i_col + self.patch_col], [i_t, i_t + self.batch_nT]])
            info_nda[idx] = np.array([[i_row, i_row + self.patch_row], [i_col, i_col + self.patch_col]])
        return info_nda # (n_samples, corres. [[start_r, end_r], [start_c, end_c]])

    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}
        start_row, end_row = self.idx2info[idx, 0].astype(int)
        start_col, end_col = self.idx2info[idx, 1].astype(int)
        #start_it,  end_it  = self.idx2info[idx, 2].astype(int)

        patch_movie_nT = torch.from_numpy(self.movie_nda[:, start_row : end_row, start_col : end_col])[: self.input_time_frame] # TODO #
        #if 'dirichlet' in self.BC: # return list of BC # 
        #    sample['movie_BC'] = extract_BC_2D(patch_movie_nT[start_it : end_it], self.args.dirichlet_width)
        #elif 'cauchy' in self.BC:
        #    sample['movie_BC'] = extract_BC_2D(patch_movie_nT[start_it : end_it], 1)
        #elif 'source' in self.BC: # return list of dBC # 
        #    sample['movie_dBC'] = extract_dBC_2D(patch_movie_nT[start_it : end_it], self.args.source_width)

        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    patch_movie_nT = neumann_BC_2D(patch_movie_nT)
        sample['u_4input'] = patch_movie_nT
        #sample['u_batch_nT'] = patch_movie_nT[start_it : end_it] # (batch_nT, r, c)
        #sample['t'] = torch.from_numpy(np.arange(self.nT) * self.args.dt)
        if self.DPath:
            sample['Dxx'] = torch.from_numpy(self.D[0, start_row : end_row, start_col : end_col])
            sample['Dxy'] = torch.from_numpy(self.D[1, start_row : end_row, start_col : end_col])
            sample['Dyy'] = torch.from_numpy(self.D[2, start_row : end_row, start_col : end_col])
        if self.LPath:
            sample['L1'] = torch.from_numpy(self.L1[start_row : end_row, start_col : end_col])
            sample['L2'] = torch.from_numpy(self.L2[start_row : end_row, start_col : end_col])
        if self.SPath:
            sample['S'] = torch.from_numpy(self.S[start_row : end_row, start_col : end_col])
        if self.UPath:
            sample['Uxx'] = torch.from_numpy(self.U[0, start_row : end_row, start_col : end_col])
            sample['Uxy'] = torch.from_numpy(self.U[1, start_row : end_row, start_col : end_col])
            sample['Uyx'] = torch.from_numpy(self.U[2, start_row : end_row, start_col : end_col])
            sample['Uyy'] = torch.from_numpy(self.U[3, start_row : end_row, start_col : end_col])
        if self.VPath:
            sample['Vx'] = torch.from_numpy(self.Vx[start_row : end_row, start_col : end_col])
            sample['Vy'] = torch.from_numpy(self.Vy[start_row : end_row, start_col : end_col])
        if self.PerfFlagPath:
            sample['isV'] = torch.from_numpy(self.isV[start_row : end_row, start_col : end_col])
            sample['isD'] = torch.from_numpy(self.isD[start_row : end_row, start_col : end_col]) 
        sample['sub_collocation_t'] = self.sub_collocation_t
        
        return sample

    @property
    def get_full_sample(self):
        sample = {}
        full_movie = torch.from_numpy(self.movie_nda) # (nT, r, c)
        if 'dirichlet' in self.BC:
            sample['BC'] = extract_BC_2D(full_movie, self.args.dirichlet_width) 
        elif 'cauchy' in self.BC:
            sample['BC'] = extract_BC_2D(full_movie, 1) # (batch = 1, nT, r, c)
        elif 'source' in self.BC:
            sample['dBC'] = extract_dBC_2D(full_movie, self.args.source_width) # (batch = 1, nT, r, c)

        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    full_movie = neumann_BC_2D(full_movie)
        sample['u_nT'] = full_movie # ( nT, r, c)
        sample['T'] = torch.from_numpy(np.arange(self.nT) * self.args.dt) # (nT)
        sample['sub_collocation_t'] = self.sub_collocation_t

        if self.DPath:
            sample['Dxx'] = torch.from_numpy(self.D[0])
            sample['Dxy'] = torch.from_numpy(self.D[1])
            sample['Dyy'] = torch.from_numpy(self.D[2])
        if self.LPath:
            sample['L1'] = torch.from_numpy(self.L1)
            sample['L2'] = torch.from_numpy(self.L2)
        if self.UPath:
            sample['Uxx'] = torch.from_numpy(self.U[0])
            sample['Uxy'] = torch.from_numpy(self.U[1])
            sample['Uyx'] = torch.from_numpy(self.U[2])
            sample['Uyy'] = torch.from_numpy(self.U[3])
        if self.SPath:
            sample['S'] = torch.from_numpy(self.S)
        if self.VPath:
            sample['Vx'] = torch.from_numpy(self.Vx)
            sample['Vy'] = torch.from_numpy(self.Vy)
        
        return sample


class MoviesDataset_2D(Dataset):
    '''
    len(dataset) = len(cases)
    time points are randomly selected for each sample
    '''
    def __init__(self, args, loss_time_frame, patch_size, MoviePaths, PerfFlagPaths, DPaths, LPaths, UPaths, SPaths, VPaths):

        self.args = args
        self.BC = args.BC
        self.MoviePaths = MoviePaths
        self.patch_size = patch_size
        self.PerfFlagPaths = PerfFlagPaths
        self.DPaths, self.VPaths, self.LPaths, self.UPaths = DPaths, VPaths, LPaths, UPaths
        self.input_time_frame = min(args.input_time_frame, self.min_max_nT) # num. of time points for predicting V, D # NOTE: TBD during training
        self.loss_time_frame   = min(loss_time_frame, self.input_time_frame) # Assure batch_nT <= input_time_frame
        self.sub_collocation_t = torch.from_numpy(np.arange(self.args.sub_collocation_nt) * self.args.dt)
        
    def __len__(self):  
        return len(self.MoviePaths)

    @property
    def min_max_nT(self):
        min_max = 1000
        for i_movie in range(len(self.MoviePaths)):
            min_max = min(sitk.GetArrayFromImage(sitk.ReadImage(self.MoviePaths[i_movie])).shape[0], min_max)
        return min_max

    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}
        movie_img = sitk.ReadImage(self.MoviePaths[idx])
        movie_nda = sitk.GetArrayFromImage(movie_img)#[: self.input_time_frame] # (nT, r, c)

        avg = movie_nda.mean()

        '''start_row = np.random.randint(0, movie_nda.shape[1] - self.patch_size[0] + 1)
        start_col = np.random.randint(0, movie_nda.shape[2] - self.patch_size[1] + 1)
        start_it  = np.random.randint(0, movie_nda.shape[0] - self.loss_time_frame + 1)
        end_row, end_col, end_it = start_row + self.patch_size[0], start_col + self.patch_size[1], start_it + self.loss_time_frame'''

        start_row = np.random.randint(0, movie_nda.shape[1] - self.patch_size[0] + 1)
        start_col = np.random.randint(0, movie_nda.shape[2] - self.patch_size[1] + 1)
        start_it  = np.random.randint(0, movie_nda.shape[0] - self.input_time_frame + 1) # NOTE: select within input_time_frame
        end_row, end_col, end_it = start_row + self.patch_size[0], start_col + self.patch_size[1], start_it + self.input_time_frame 
        '''while True:
            start_row = np.random.randint(0, movie_nda.shape[1] - self.patch_size[0] + 1)
            start_col = np.random.randint(0, movie_nda.shape[2] - self.patch_size[1] + 1)
            start_it  = np.random.randint(0, movie_nda.shape[0] - self.input_time_frame + 1) # NOTE: select within input_time_frame
            end_row, end_col, end_it = start_row + self.patch_size[0], start_col + self.patch_size[1], start_it + self.input_time_frame 
            if movie_nda[start_it, start_row : end_row, start_col : end_col].mean() > 0.5 * avg: # NOTE #
                break # Criterion: batch_u_t0 should contain eanough mass #
            time.sleep(0.05)'''
        patch_movie_nT = torch.from_numpy(movie_nda[start_it : start_it + self.input_time_frame, start_row : end_row, start_col : end_col]) # TODO
        
        start_loss_t = np.random.randint(0, patch_movie_nT.shape[0] - self.loss_time_frame + 1)
        end_loss_t   = start_loss_t + self.loss_time_frame

        # For checking #
        #print('input', start_it, start_it + self.input_time_frame)
        #print('loss', start_loss_t, end_loss_t)

        if 'dirichlet' in self.BC:
            sample['movie_BC'] = extract_BC_2D(patch_movie_nT[start_loss_t : end_loss_t], self.args.dirichlet_width)
        elif 'cauchy' in self.BC:
            sample['movie_BC'] = extract_BC_2D(patch_movie_nT[start_loss_t : end_loss_t], 1)
        elif 'source' in self.BC:
            sample['movie_dBC'] = extract_dBC_2D(patch_movie_nT[start_loss_t : end_loss_t], self.args.source_width)

        # For V, D prediction # 
        #sample['u_nT'] = patch_movie_nT
        
        # For integration #
        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    patch_movie_nT = neumann_BC_2D(patch_movie_nT[:, 0]) # Set Neumann on t0 for input for integration
        sample['u_4input'] = patch_movie_nT # (input_time_frame, r, c)
        sample['u_4loss'] = patch_movie_nT[start_loss_t : end_loss_t] # (loss_time_frame, r, c)
        #sample['t'] = torch.from_numpy(np.arange(self.batch_nT) * self.args.dt)
        #sample['T'] = torch.from_numpy(np.arange(self.input_time_frame) * self.args.dt) # TODO
        sample['sub_collocation_t'] = self.sub_collocation_t

        if self.DPaths:
            D = sitk.GetArrayFromImage(sitk.ReadImage(self.DPaths[idx]['D']))
            #Dxx = sitk.GetArrayFromImage(sitk.ReadImage(self.DPaths[idx]['Dxx']))
            #Dxy = sitk.GetArrayFromImage(sitk.ReadImage(self.DPaths[idx]['Dxy']))
            #Dyy = sitk.GetArrayFromImage(sitk.ReadImage(self.DPaths[idx]['Dyy']))
            sample['Dxx'] = torch.from_numpy(D[0, start_row : end_row, start_col : end_col])
            sample['Dxy'] = torch.from_numpy(D[1, start_row : end_row, start_col : end_col]) 
            sample['Dyy'] = torch.from_numpy(D[2, start_row : end_row, start_col : end_col]) 

        if self.LPaths:
            L1 = sitk.GetArrayFromImage(sitk.ReadImage(self.LPaths[idx]['L1']))
            L2 = sitk.GetArrayFromImage(sitk.ReadImage(self.LPaths[idx]['L2']))
            sample['L1'] = torch.from_numpy(L1[start_row : end_row, start_col : end_col]) 
            sample['L2'] = torch.from_numpy(L2[start_row : end_row, start_col : end_col]) 

        if self.SPaths:
            S = sitk.GetArrayFromImage(sitk.ReadImage(self.SPaths[idx]['S']))
            sample['S'] = torch.from_numpy(S[start_row : end_row, start_col : end_col]) 
            
        if self.UPaths:
            U = sitk.GetArrayFromImage(sitk.ReadImage(self.UPaths[idx]['U']))
            sample['Uxx'] = torch.from_numpy(U[0, start_row : end_row, start_col : end_col])
            sample['Uxy'] = torch.from_numpy(U[1, start_row : end_row, start_col : end_col]) 
            sample['Uyx'] = torch.from_numpy(U[2, start_row : end_row, start_col : end_col]) 
            sample['Uyy'] = torch.from_numpy(U[3, start_row : end_row, start_col : end_col]) 

        if self.VPaths:
            Vx = sitk.GetArrayFromImage(sitk.ReadImage(self.VPaths[idx]['Vx']))
            Vy = sitk.GetArrayFromImage(sitk.ReadImage(self.VPaths[idx]['Vy']))
            sample['Vx'] = torch.from_numpy(Vx[start_row : end_row, start_col : end_col]) 
            sample['Vy'] = torch.from_numpy(Vy[start_row : end_row, start_col : end_col]) 

        if self.PerfFlagPaths:
            isV = sitk.GetArrayFromImage(sitk.ReadImage(self.PerfFlagPaths[idx]['V']))
            isD = sitk.GetArrayFromImage(sitk.ReadImage(self.PerfFlagPaths[idx]['D']))
            sample['isV'] = torch.from_numpy(isV[start_row : end_row, start_col : end_col])
            sample['isD'] = torch.from_numpy(isD[start_row : end_row, start_col : end_col]) 

        sample['origin'] = movie_img.GetOrigin()
        sample['spacing'] = movie_img.GetSpacing()
        sample['direction'] = movie_img.GetDirection()

        #print('Domain avg.: %.3f' % movie_nda.mean())
        #print('\nbatch_t     : [%d, %d]' % (start_it, start_it + self.batch_nT))
        #print('t_0     avg.: %.3f' % patch_movie_batch_nT[0].mean())
        #print('Patch_t avg.: %.3f' % patch_movie_batch_nT.mean())
        
        return sample


######################################################################
################################  3D  ################################
######################################################################

class MovieDataset_3D(Dataset):
    '''
    len(dataset) = avail_slc * avail_row * avail_col
    time points are randomly selected for each sample
    ''' 
    def __init__(self, args, batch_nT, patch_size, spatial_stride, MoviePath, MaskPath):

        self.args       = args
        self.BC         = args.BC
        self.patch_size = patch_size

        self.movie_img  = sitk.ReadImage(MoviePath)
        self.origin, self.spacing, self.direction = self.movie_img.GetOrigin(), self.movie_img.GetSpacing(), self.movie_img.GetDirection()
        self.data_spacing = [self.spacing[2], self.spacing[1], self.spacing[0]] # inverse relation between sitk img and numpy array

        movie_nda = sitk.GetArrayFromImage(self.movie_img) # (s, r, c, nT) 
        self.movie_nda = np.transpose(movie_nda, (3, 0, 1, 2)) # (s, r, c, nT) -> (nT, s, r, c)
        self.nT = self.movie_nda.shape[0]
        self.input_time_frame = min(args.input_time_frame, self.nT) # time points for predicting V, D # NOTE: TBD during training
        self.batch_nT = min(batch_nT, self.input_time_frame)  

        self.MaskPath = MaskPath
        self.mask_nda = self.get_mask()

        self.stride_slc, self.stride_row, self.stride_col = spatial_stride[0], spatial_stride[1], spatial_stride[2]
        self.slc, self.row, self.col = self.movie_nda.shape[1], self.movie_nda.shape[2], self.movie_nda.shape[3]
        self.data_dim = [self.slc, self.row, self.col]
        self.patch_slc = self.slc if patch_size[0] < 1 else patch_size[0]
        self.patch_row = self.row if patch_size[1] < 1 else patch_size[1]
        self.patch_col = self.col if patch_size[2] < 1 else patch_size[2]
        assert self.patch_slc <= self.slc and self.patch_row <= self.row and self.patch_col <= self.col and self.batch_nT <= self.nT
        self.avail_it = int(self.nT - self.batch_nT + 1)
        self.avail_slc = int(np.floor((self.slc - self.patch_slc) / self.stride_slc) + 1)
        if ((self.avail_slc - 1) * self.stride_slc) + self.patch_slc < self.slc: 
            self.avail_slc += 1
        self.avail_row = int(np.floor((self.row - self.patch_row) / self.stride_row) + 1)
        if ((self.avail_row - 1) * self.stride_row) + self.patch_row < self.row: 
            self.avail_row += 1
        self.avail_col = int(np.floor((self.col - self.patch_col) / self.stride_col) + 1)
        if ((self.avail_col - 1) * self.stride_col) + self.patch_col < self.col: 
            self.avail_col += 1
        #print('len(dataset):', int(self.avail_slc * self.avail_row * self.avail_col * self.avail_col))
        print('Available start  it:', self.avail_it)
        print('Available start slc:', self.avail_slc)
        print('Available start row:', self.avail_row)
        print('Available start col:', self.avail_col)

    def get_mask(self):
        if self.MaskPath is None or not os.path.isfile(self.MaskPath):
            mask = np.ones(self.movie_nda[0].shape)
            non_brain = np.where(abs(self.movie_nda[0]) <= 1e-3) # TODO
            mask[non_brain] = 0
            for s in range(len(mask)):
                mask[s] = ndimage.binary_fill_holes(mask[s])
            mask_img = sitk.GetImageFromArray(mask, isVector = False)
            mask_img.SetOrigin(self.origin)
            mask_img.SetSpacing(self.spacing)
            mask_img.SetDirection(self.direction)
            sitk.WriteImage(mask_img, self.MaskPath)
        else:
            mask = sitk.GetArrayFromImage(sitk.ReadImage(self.MaskPath))

        return mask.astype(np.float32) # (s, r, c)

    def __len__(self):  
        return int(self.avail_slc * self.avail_row * self.avail_col * self.avail_it)

    @property
    def idx2info(self):
        info_nda = np.zeros((int(self.avail_slc * self.avail_row * self.avail_col * self.avail_it), 4, 2)) 
        for idx in range(info_nda.shape[0]):
            i_slc, i_row, i_col, i_t = np.unravel_index(idx, (self.avail_slc, self.avail_row, self.avail_col, self.avail_it))
            if i_slc == self.avail_slc - 1:
                i_slc = self.slc - self.patch_slc
            else:
                i_slc = self.stride_slc * i_slc
            if i_row == self.avail_row - 1:
                i_row = self.row - self.patch_row
            else:
                i_row = self.stride_row * i_row
            if i_col == self.avail_col - 1:
                i_col = self.col - self.patch_col
            else:
                i_col = self.stride_col * i_col
            info_nda[idx] = np.array([[i_slc, i_slc + self.patch_slc], [i_row, i_row + self.patch_row], [i_col, i_col + self.patch_col], [i_t, i_t + self.batch_nT]])
        return info_nda # (n_samples, corres. [[start_s, end_s], [start_r, end_r], [start_c, end_c], [start_it, end_it]])

    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}
        start_slc, end_slc = self.idx2info[idx, 0].astype(int)
        start_row, end_row = self.idx2info[idx, 1].astype(int)
        start_col, end_col = self.idx2info[idx, 2].astype(int)
        start_it,  end_it  = self.idx2info[idx, 3].astype(int)
        
        patch_movie_nT = torch.from_numpy(self.movie_nda[:, start_slc : end_slc, start_row : end_row, start_col : end_col])
        if 'dirichlet' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT[start_it : end_it], self.args.dirichlet_width)
        elif 'cauchy' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT[start_it : end_it], 1)
        elif 'source' in self.BC:
            sample['movie_dBC'] = extract_dBC_3D(patch_movie_nT[start_it : end_it], self.args.source_width)

        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    patch_movie_nT = neumann_BC_3D(patch_movie_nT)
        sample['u_nT'] = patch_movie_nT[: self.input_time_frame]
        sample['u_batch_nT'] = patch_movie_nT[start_it : end_it] # (batch_nT, r, c)
        sample['t'] = torch.from_numpy(np.arange(self.batch_nT) * self.args.dt)
        sample['sampe_mask'] = torch.from_numpy(self.mask_nda[start_slc : end_slc, start_row : end_row, start_col : end_col])

        del start_slc, end_slc, start_row, end_row, start_col, end_col, start_it, end_it, patch_movie_nT
        return sample

    @property
    def get_full_sample(self):
        sample = {}
        full_movie = torch.from_numpy(self.movie_nda) # (nT, s, r, c)
        if 'dirichlet' in self.BC:
            sample['BC'] = extract_BC_3D(full_movie, self.args.dirichlet_width) # (batch = 1, nT, s, r, c)
        elif 'cauchy' in self.BC:
            sample['BC'] = extract_BC_3D(full_movie, 1) # (batch = 1, nT, r, c)
        elif 'source' in self.BC:
            sample['dBC'] = extract_dBC_3D(full_movie, self.args.source_width) # (batch = 1, nT, s, r, c)

        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    full_movie = neumann_BC_3D(full_movie)
        sample['u_nT'] = full_movie # (nT, s, r, c)
        sample['T'] = torch.from_numpy(np.arange(self.nT) * self.args.dt) # (nT)
        sample['mask'] = torch.from_numpy(self.mask_nda)
        del full_movie
        return sample



class MoviesDataset_3D(Dataset):
    '''
    len(dataset) = len(cases)
    time points and spatial patches are randomly selected for each sample
    '''
    def __init__(self, args, batch_nT, patch_size, MoviePaths, VesselPaths = None): # TODO: add mask paths

        self.args       = args
        self.BC         = args.BC
        self.MoviePaths = MoviePaths
        self.VesselPaths  = VesselPaths
        self.patch_size = patch_size
        self.batch_nT   = min(batch_nT, self.min_max_nT)
        self.input_time_frame = min(args.input_time_frame, self.min_max_nT) # input the entire available time frame # NOTE: TBD during training

    def __len__(self):  
        return len(self.MoviePaths)

    @property
    def all_movies(self):
        movies, vessel_masks, infos = [], [], []
        for i_movie in range(len(self.MoviePaths)):
            movie_img = sitk.ReadImage(self.MoviePaths[i_movie])
            movies.append(np.transpose(sitk.GetArrayFromImage(movie_img), (3, 0, 1, 2))) # (s, r, c, nT) -> (nT, s, r, c)
            if self.MaskPaths:
                mask_img = sitk.ReadImage(self.VesselPaths[i_movie])
                vessel_masks.append(sitk.GetArrayFromImage(mask_img)) # (s, r, c)
            infos.append({'origin': movie_img.GetOrigin(), 'spacing': movie_img.GetSpacing(), 'direction': movie_img.GetDirection()})
        return {'movies': movies, 'vessel_masks': vessel_masks, 'infos': infos}

    @property
    def min_max_nT(self):
        min_max = 1000
        for i_movie in range(len(self.MoviePaths)):
            min_max = min(sitk.GetArrayFromImage(sitk.ReadImage(self.MoviePaths[i_movie])).shape[-1], min_max)
        return min_max

    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}

        movie_img = sitk.ReadImage(self.MoviePaths[idx])
        movie_nda = self.all_movies['movies'][idx][:self.input_time_frame] # (nT, s, r, c)
        vessel_mask_nda = self.all_movies['vessel_masks'][idx] # (s, r, c)
        start_slc = np.random.randint(0, movie_nda.shape[1] - self.patch_size[0] + 1)
        start_row = np.random.randint(0, movie_nda.shape[2] - self.patch_size[1] + 1)
        start_col = np.random.randint(0, movie_nda.shape[3] - self.patch_size[2] + 1)
        start_it  = np.random.randint(0, movie_nda.shape[0] - self.batch_nT + 1)
        end_slc, end_row, end_col, end_it = \
            start_slc + self.patch_size[0], start_row + self.patch_size[1], start_col + self.patch_size[2], start_it + self.batch_nT

        patch_movie_nT = torch.from_numpy(movie_nda[:, start_slc : end_slc, start_row : end_row, start_col : end_col])
        if 'dirichlet' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT[start_it : end_it], self.args.dirichlet_width)
        elif 'cauchy' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT[start_it : end_it], 1)
        elif 'source' in self.BC:
            sample['movie_dBC'] = extract_dBC_3D(patch_movie_nT[start_it : end_it], self.args.source_width)

        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    patch_movie_nT = neumann_BC_3D(patch_movie_nT)
        sample['u_nT'] = patch_movie_nT
        
        sample['u_batch_nT'] = patch_movie_nT[start_it : end_it] # (batch_nT, s, r, c)
        sample['t'] = torch.from_numpy(np.arange(self.batch_nT) * self.args.dt)

        sample['origin'] = movie_img.GetOrigin()
        sample['spacing'] = movie_img.GetSpacing()
        sample['direction'] = movie_img.GetDirection()
        sample['nT'] = movie_nda.shape[0]
        if self.VesselPaths:
            sample['vessel_mask'] =  torch.from_numpy(vessel_mask_nda[start_slc : end_slc, start_row : end_row, start_col : end_col])
        del start_slc, end_slc, start_row, end_row, start_col, end_col, start_it, end_it, patch_movie_nT, movie_nda, movie_img
        return sample


#############################################################################################################


def get_patch_times(ctc_nda, bat_threshold = 0.05):
    '''
    NOTE: ctc_nda: (t, s, r, c)
    '''
    nT = ctc_nda.shape[0]
    ctc_avg = np.zeros(nT)
    for t in range(nT):
        ctc_avg[t] = (ctc_nda[t]).mean()
    ttp = np.argmax(ctc_avg)
    ttd = np.argmin(ctc_avg[ttp:]) + ttp
    threshold = bat_threshold * (np.amax(ctc_avg) - np.amin(ctc_avg)) 
    flag = True
    bat  = 1
    while flag:
        if ctc_avg[bat] - ctc_avg[bat - 1] >= threshold and ctc_avg[bat + 1] > ctc_avg[bat]:
            flag = False
        else:
            bat += 1
        if bat == nT - 1:
            flag = False
    return bat, ttp, ttd


#############################################################################################################
#############################################################################################################


class MovieDataset_3D_SmartTiming(Dataset):
    '''
    For testing dataset
    len(dataset) = avail_slc * avail_row * avail_col
    time points are randomly selected for each sample
    ''' 
    def __init__(self, args, input_n_collocations, patch_size, spatial_stride, TestPath, CaseName, MoviePath, \
        MaskPath = None, VesselPath = None, VesselMirrorPath = None, ValueMaskPath = None, InfoPath = None, DPath = None, 
        DCoPath = None, LPath = None, UPath = None, VPath = None, PhiPath = None, device = 'cpu'):

        self.args       = args
        self.BC         = args.BC
        self.device     = device
        self.patch_size = patch_size

        if ValueMaskPath is None:
            self.CaseName = CaseName
        elif isinstance(ValueMaskPath, str):
            self.CaseName = CaseName + '(Lesion)'
        else:
            self.CaseName = CaseName + '(Sep-Lesion)'
        
        self.TestPath   = TestPath
        self.MoviePath  = MoviePath
        self.MaskPath   = MaskPath
        self.VesselPath = VesselPath
        self.VesselMirrorPath = VesselMirrorPath
        self.DPath      = DPath 
        self.DCoPath    = DCoPath
        self.LPath      = LPath
        self.UPath      = UPath
        self.VPath      = VPath
        self.PhiPath    = PhiPath
        self.ValueMaskPath   = ValueMaskPath


        #self.movie_img  = sitk.ReadImage(MoviePath)
        #self.origin, self.spacing = self.movie_img.GetOrigin(), self.movie_img.GetSpacing(), self.movie_img.GetDirection()
        #self.data_spacing = [self.spacing[2], self.spacing[1], self.spacing[0]] # inverse relation between sitk img and numpy array

        #movie_nda = sitk.GetArrayFromImage(self.movie_img) # (s, r, c, nT) 
        movie_nda = np.load(MoviePath)
        fileHandle = open(InfoPath,"r")
        lineList = fileHandle.readlines()
        fileHandle.close()
        
        if 'rotated' in MoviePath:
            # For ISLES_Processed_rotated #
            self.origin = [float(lineList[3]), float(lineList[4]), float(lineList[5])]
            self.spacing = [float(lineList[7]), float(lineList[8]), float(lineList[9])]
            self.direction = [float(lineList[11]), float(lineList[12]), float(lineList[13]), 
                            float(lineList[14]), float(lineList[15]), float(lineList[16]),
                                float(lineList[17]), float(lineList[18]), float(lineList[19])]
            BAT, TTP, TTD = int(lineList[21]), int(lineList[23]), int(lineList[25]) # '''
        else:
            # For ISLES_Processed and IXI #
            self.origin = [float(lineList[1]), float(lineList[2]), float(lineList[3])]
            self.spacing = [float(lineList[5]), float(lineList[6]), float(lineList[7])]
            self.direction = [float(lineList[9]), float(lineList[10]), float(lineList[11]), 
                                float(lineList[12]), float(lineList[13]), float(lineList[14]),
                                    float(lineList[15]), float(lineList[16]), float(lineList[17])]
            BAT, TTP, TTD = int(lineList[19]), int(lineList[21]), int(lineList[23]) # ''' 

        self.data_spacing = [self.spacing[2], self.spacing[1], self.spacing[0]]
        
        self.movie_nda = np.transpose(movie_nda, (3, 0, 1, 2)) # (s, r, c, nT) -> (nT, s, r, c)
        self.movie_nda = self.movie_nda[TTP:TTD+1] # NOTE: take global TTP - TTD as entire time frame for testing # TODO
        #self.movie_nda = self.movie_nda[:TTD+1] # NOTE: take global BAT - TTD as entire time frame for testing # TODO
        #self.movie_nda = self.movie_nda[TTP:] # NOTE: take global TTP - Last time as entire time frame for testing # TODO
        self.N_collocations = self.movie_nda.shape[0]
        self.input_n_collocations = min(input_n_collocations, self.N_collocations)  # For V, D prediction input # 
        self.sub_collocation_t = torch.from_numpy(np.arange(self.args.sub_collocation_nt) * self.args.dt).to(self.device) # time intervals between 2 collocation time points #
        print('Input_n_collocations:', self.input_n_collocations)

        self.mask_nda = self.get_mask()

        self.stride_slc, self.stride_row, self.stride_col = spatial_stride[0], spatial_stride[1], spatial_stride[2]
        self.slc, self.row, self.col = self.movie_nda.shape[1], self.movie_nda.shape[2], self.movie_nda.shape[3]
        self.data_dim = [self.slc, self.row, self.col]
        self.patch_slc = self.slc if patch_size[0] < 1 else patch_size[0]
        self.patch_row = self.row if patch_size[1] < 1 else patch_size[1]
        self.patch_col = self.col if patch_size[2] < 1 else patch_size[2]
        assert self.patch_slc <= self.slc and self.patch_row <= self.row and self.patch_col <= self.col
        self.avail_slc = int(np.floor((self.slc - self.patch_slc) / self.stride_slc) + 1)
        if ((self.avail_slc - 1) * self.stride_slc) + self.patch_slc < self.slc: 
            self.avail_slc += 1
        self.avail_row = int(np.floor((self.row - self.patch_row) / self.stride_row) + 1)
        if ((self.avail_row - 1) * self.stride_row) + self.patch_row < self.row: 
            self.avail_row += 1
        self.avail_col = int(np.floor((self.col - self.patch_col) / self.stride_col) + 1)
        if ((self.avail_col - 1) * self.stride_col) + self.patch_col < self.col: 
            self.avail_col += 1
        print('\nlen(dataset):', int( self.avail_slc * self.avail_row * self.avail_col))
        print('Available start slc:', self.avail_slc)
        print('Available start row:', self.avail_row)
        print('Available start col:', self.avail_col)

    def get_mask(self):
        if self.MaskPath is None or not os.path.isfile(self.MaskPath):
            mask = np.ones(self.movie_nda[0].shape)
            non_brain = np.where(abs(self.movie_nda[0]) < 1e-3) # TODO
            mask[non_brain] = 0
            for s in range(len(mask)):
                mask[s] = ndimage.binary_fill_holes(mask[s])
            mask_img = sitk.GetImageFromArray(mask, isVector = False)
            mask_img.SetOrigin(self.origin)
            mask_img.SetSpacing(self.spacing)
            mask_img.SetDirection(self.direction)
            mask_img_path = '%s.nii' % self.MaskPath[:-4]
            print('Write mask %s' % mask_img_path)
            sitk.WriteImage(mask_img, mask_img_path)
        else:
            #mask = sitk.GetArrayFromImage(sitk.ReadImage(self.MaskPath))
            mask = np.load(self.MaskPath)

        return mask.astype(np.float32) # (s, r, c)

    def __len__(self):  
        return int(self.avail_slc * self.avail_row * self.avail_col)

    @property
    def idx2info(self):
        info_nda = np.zeros((int(self.avail_slc * self.avail_row * self.avail_col), 3, 2)) 
        for idx in range(info_nda.shape[0]):
            i_slc, i_row, i_col = np.unravel_index(idx, (self.avail_slc, self.avail_row, self.avail_col))
            if i_slc == self.avail_slc - 1:
                i_slc = self.slc - self.patch_slc
            else:
                i_slc = self.stride_slc * i_slc
            if i_row == self.avail_row - 1:
                i_row = self.row - self.patch_row
            else:
                i_row = self.stride_row * i_row
            if i_col == self.avail_col - 1:
                i_col = self.col - self.patch_col
            else:
                i_col = self.stride_col * i_col
            info_nda[idx] = np.array([[i_slc, i_slc + self.patch_slc], [i_row, i_row + self.patch_row], [i_col, i_col + self.patch_col]])
        return info_nda # (n_samples, corres. [[start_s, end_s], [start_r, end_r], [start_c, end_c]])


    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}
        start_slc, end_slc = self.idx2info[idx, 0].astype(int)
        start_row, end_row = self.idx2info[idx, 1].astype(int)
        start_col, end_col = self.idx2info[idx, 2].astype(int)
        
        patch_movie_nT = torch.from_numpy(self.movie_nda[:, start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
        bat, ttp, ttd = get_patch_times(patch_movie_nT) # (nt, s, r, c)
        #print(bat, ttp, ttd)
        if ttp + self.input_n_collocations < patch_movie_nT.shape[0]:
        #if bat + self.input_n_collocations < patch_movie_nT.shape[0]:
            t0 = ttp # TODO
            t1 = ttp + self.input_n_collocations 
            #t0 = 0# bat
            #t1 = self.input_n_collocations 
        else:
            t1 = patch_movie_nT.shape[0]
            t0 = t1 - self.input_n_collocations
        patch_movie_nT = patch_movie_nT[t0 : t1] # Restict in patch-wise effective time frame # 

        if 'dirichlet' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT, self.args.dirichlet_width)
        elif 'cauchy' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT, 1) 
        elif 'source' in self.BC:
            sample['movie_dBC'] = extract_dBC_3D(patch_movie_nT, self.args.source_width)

        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    patch_movie_nT = neumann_BC_3D(patch_movie_nT)
        sample['movie4input'] = patch_movie_nT  # (batch_nT, s, r, c)
        sample['sub_collocation_t'] = self.sub_collocation_t # (batch_nT, )
        #sample['Mask'] = torch.from_numpy(self.mask_nda[:, start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)  # (s, r, c)
        if self.VesselPath:
            sample['vessel_mask'] = torch.from_numpy(np.load(self.VesselPath)[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device) # (s, r, c)
            sample['vessel_mirror_mask'] = torch.from_numpy(np.load(self.VesselMirrorPath)[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device) # (s, r, c)
        if self.args.stochastic or self.args.predict_value_mask:  
            if self.args.stochastic and self.args.separate_DV_value_mask and self.args.img_type != 'IXI':
                pass
                #raise ValueError('Currently not support: stochastic + separate_DV_value_mask for real MRP data testing') # TODO: to figure out #
            else:
                #if self.args.separate_DV_value_mask:
                if self.ValueMaskPath is not None: # True if case is w/ lesion
                    if isinstance(self.ValueMaskPath, str):
                        value_mask = torch.from_numpy(np.load(self.ValueMaskPath)[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device) # (s, r, c) 
                        sample['value_mask_D'] = value_mask
                        sample['value_mask_V'] = value_mask
                    else:
                        sample['value_mask_D'] = torch.from_numpy(np.load(self.ValueMaskPath['D'])[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device) # (s, r, c) 
                        sample['value_mask_V'] = torch.from_numpy(np.load(self.ValueMaskPath['V'])[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device) # (s, r, c) 
                else: # normal case w/o lesion
                    sample['value_mask_D'] = torch.ones((self.patch_slc, self.patch_row, self.patch_col)).to(self.device) 
                    sample['value_mask_V'] = torch.ones((self.patch_slc, self.patch_row, self.patch_col)).to(self.device) 
                if self.args.stochastic: # TODO: TESTING shared uncertainty # 
                    sample['sigma'] = 1. - sample['value_mask_V'] if (sample['value_mask_D'] - sample['value_mask_V']).mean() > 0. else 1. - sample['value_mask_D']
                '''else:
                    if self.ValueMaskPath is not None: # True if case is w/ lesion
                        sample['value_mask'] = torch.from_numpy(np.load(self.ValueMaskPath)[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device) # (s, r, c) 
                    else: # normal case w/o lesion
                        sample['value_mask'] = torch.ones((self.patch_slc, self.patch_row, self.patch_col)).to(self.device) 
                    if self.args.stochastic: # TODO: TESTING shared uncertainty # 
                        sample['sigma'] = 1. - sample['value_mask']'''

        return sample

    
    def get_full_sample(self):
        sample = {}
        full_movie = torch.from_numpy(self.movie_nda).to(self.device) # (nT, s, r, c)
        if 'dirichlet' in self.BC:
            sample['movie_BC'] = extract_BC_3D(full_movie, self.args.dirichlet_width)# # (batch = 1, nT, s, r, c)
        elif 'cauchy' in self.BC:
            sample['movie_BC'] = extract_BC_3D(full_movie, 1)# # (batch = 1, nT, s, r, c) 
        elif 'source' in self.BC:
            sample['movie_dBC'] = extract_dBC_3D(full_movie, self.args.source_width)# # (batch = 1, nT, s, r, c)
        
        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    full_movie = neumann_BC_3D(full_movie) 
        sample['full_movie'] = full_movie # (nT, s, r, c)
        sample['sub_collocation_t'] = self.sub_collocation_t # (nT)
        #sample['Mask'] = torch.from_numpy(self.mask_nda).to(self.device) # (s, r, c)
        if self.VesselPath:
            sample['vessel_mask'] = torch.from_numpy(np.load(self.VesselPath)).to(self.device) # (s, r, c) 
            sample['vessel_mirror_mask'] = torch.from_numpy(np.load(self.VesselMirrorPath)).to(self.device) # (s, r, c) 
        
        if self.DPath:
            if self.args.predict_deviation:
                if self.TestPath['orig_D'] is not None: # lesion cases
                    sample['orig_D'] = torch.from_numpy(np.load(self.TestPath['orig_D'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['delta_D'] = torch.from_numpy(np.load(self.TestPath['delta_D'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                else: # normal cases
                    sample['orig_D'] = torch.from_numpy(np.load(self.DPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['delta_D'] = torch.zeros((tuple([9]) + full_movie[0].size())).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
            elif self.args.predict_value_mask:
                if self.TestPath['orig_D'] is not None: # lesion cases
                    sample['orig_D'] = torch.from_numpy(np.load(self.TestPath['orig_D'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['D'] = torch.from_numpy(np.load(self.DPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                else: # normal cases
                    sample['orig_D'] = torch.from_numpy(np.load(self.DPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['D'] = torch.from_numpy(np.load(self.DPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
            else:
                sample['D'] = torch.from_numpy(np.load(self.DPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
        if self.DCoPath:
            sample['D_CO'] = torch.from_numpy(np.load(self.DCoPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
        if self.LPath:
            if self.args.predict_deviation:
                if self.TestPath['orig_L'] is not None:
                    sample['orig_L'] = torch.from_numpy(np.load(self.TestPath['orig_L'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['delta_L'] = torch.from_numpy(np.load(self.TestPath['delta_L'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                else:
                    sample['orig_L'] = torch.from_numpy(np.load(self.LPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['delta_L'] = torch.zeros((tuple([3]) + full_movie[0].size())).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
            elif self.args.predict_value_mask:
                if self.TestPath['orig_L'] is not None:
                    sample['orig_L'] = torch.from_numpy(np.load(self.TestPath['orig_L'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['L']= torch.from_numpy(np.load(self.LPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                else:
                    sample['orig_L'] = torch.from_numpy(np.load(self.LPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['L']= torch.from_numpy(np.load(self.LPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
            else:
                sample['L']= torch.from_numpy(np.load(self.LPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
        if self.UPath:
            sample['U'] = torch.from_numpy(np.load(self.UPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (4, s, r, c)
        if self.VPath:
            if self.args.predict_deviation:
                if self.TestPath['orig_V'] is not None:
                    sample['orig_V'] = torch.from_numpy(np.load(self.TestPath['orig_V'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['delta_V'] = torch.from_numpy(np.load(self.TestPath['delta_V'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                else:
                    sample['orig_V'] = torch.from_numpy(np.load(self.VPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['delta_V'] = torch.zeros((tuple([3]) + full_movie[0].size())).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
            if self.args.predict_value_mask:
                if self.TestPath['orig_V'] is not None:
                    sample['orig_V'] = torch.from_numpy(np.load(self.TestPath['orig_V'])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['V'] = torch.from_numpy(np.load(self.VPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                else:
                    sample['orig_V'] = torch.from_numpy(np.load(self.VPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['V'] = torch.from_numpy(np.load(self.VPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
            else:
                sample['V'] = torch.from_numpy(np.load(self.VPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
        if self.PhiPath:
            sample['Phi'] = torch.from_numpy(np.load(self.PhiPath)).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
        if self.args.stochastic or self.args.predict_value_mask:  
            if self.args.stochastic and self.args.separate_DV_value_mask and self.args.img_type != 'IXI':
                pass
                #raise ValueError('Currently not support: stochastic + separate_DV_value_mask for real MRP data testing') # TODO: to figure out #
            else:
                #if self.args.separate_DV_value_mask:
                if self.ValueMaskPath is not None: # True if case is w/ lesion
                    if isinstance(self.ValueMaskPath, str):
                        value_mask = torch.from_numpy(np.load(self.ValueMaskPath)).to(self.device) # (s, r, c)
                        sample['value_mask_D'] = value_mask
                        sample['value_mask_V'] = value_mask
                    else:
                        sample['value_mask_D'] = torch.from_numpy(np.load(self.ValueMaskPath['D'])).to(self.device)
                        sample['value_mask_V'] = torch.from_numpy(np.load(self.ValueMaskPath['V'])).to(self.device)
                else: # normal case w/o lesion
                    sample['value_mask_D'] = torch.ones(full_movie[0].size()).to(self.device)
                    sample['value_mask_V'] = torch.ones(full_movie[0].size()).to(self.device)
                if self.args.stochastic: # TODO: TESTING shared uncertainty # 
                    sample['sigma'] = 1. - sample['value_mask_V'] if (sample['value_mask_D'] - sample['value_mask_V']).mean() > 0. else 1. - sample['value_mask_D']
                '''else:
                    if self.ValueMaskPath is not None: # True if case is w/ lesion
                        sample['value_mask'] = torch.from_numpy(np.load(self.ValueMaskPath)).to(self.device) # (s, r, c)
                    else: # normal case w/o lesion
                        sample['value_mask'] = torch.ones(full_movie[0].size()).to(self.device)
                    if self.args.stochastic: # TODO: TESTING shared uncertainty #
                        sample['sigma'] = 1. - sample['value_mask']'''
        if self.args.predict_segment:  
            if self.TestPath['lesion_seg'] is not None: # lesion cases
                sample['lesion_seg'] = torch.from_numpy(np.load(self.TestPath['lesion_seg'])).to(self.device) # (s, r, c)
            else: # normal cases
                sample['lesion_seg'] = torch.zeros(full_movie[0].size()).to(self.device)

        del full_movie
        return sample



class MoviesDataset_3D_SmartTiming(Dataset):
    '''
    sample including the full time frame --> smart selecting training time-frames
    len(dataset) = len(cases)
    time points and spatial patches are randomly selected for each sample
    '''
    def __init__(self, args, input_n_collocations, loss_n_collocations, patch_size, MoviePaths, VesselPaths = None, VesselMirrorPaths = None, ValueMaskPaths = None, LesionSegPaths = None, \
        InfoPaths = None, DPaths = None, DCoPaths = None, LPaths = None, UPaths = None, VPaths = None, PhiPaths = None, deviation_path_dict = {}, device = 'cpu'): # TODO: add mask paths

        self.args        = args
        self.BC          = args.BC
        self.device      = device
        self.MoviePaths  = MoviePaths
        self.InfoPaths   = InfoPaths
        self.DPaths      = DPaths 
        self.DCoPaths    = DCoPaths
        self.LPaths      = LPaths
        self.UPaths      = UPaths
        self.VPaths      = VPaths
        self.PhiPaths    = PhiPaths
        self.VesselPaths = VesselPaths        
        self.VesselMirrorPaths = VesselMirrorPaths
        self.ValueMaskPaths  = ValueMaskPaths
        self.LesionSegPaths = LesionSegPaths
 
        self.dev_paths = deviation_path_dict

        self.patch_size = patch_size
        self.min_max_nT = self.all_movies['min_max_nT']
        #self.min_max_nT = 9
        self.input_n_collocations  = min(input_n_collocations, self.min_max_nT) # For V, D prediction input # 
        self.loss_n_collocations  = min(input_n_collocations, loss_n_collocations) # For integration and perf_loss computation # 
        self.sub_collocation_t = torch.from_numpy(np.arange(self.args.sub_collocation_nt) * self.args.dt).to(self.device) # time intervals between 2 collocation time points #
        # Full image too large for ode: set collocation be 2 #
        #self.sub_collocation_t = torch.from_numpy(np.arange(2) * self.args.dt).to(self.device) # time intervals between 2 collocation time points #
        print('\nInput_n_collocations:', self.input_n_collocations)
        print('Collocations for loss:', self.loss_n_collocations)

    def __len__(self):  
        return len(self.MoviePaths)

    @property
    def all_movies(self):
        temp_min_max_nT = 1000
        infos = []
        for i_movie in range(len(self.MoviePaths)): 
            fileHandle = open(self.InfoPaths[i_movie],"r")
            lineList = fileHandle.readlines()
            fileHandle.close() 
            '''# For ISLES_Processed_rotated #
            origin = [float(lineList[3]), float(lineList[4]), float(lineList[5])]
            spacing = [float(lineList[7]), float(lineList[8]), float(lineList[9])]
            direction = [float(lineList[11]), float(lineList[12]), float(lineList[13]), 
                            float(lineList[14]), float(lineList[15]), float(lineList[16]),
                                float(lineList[17]), float(lineList[18]), float(lineList[19])]
            BAT, TTP, TTD = int(lineList[21]), int(lineList[23]), int(lineList[25]) # '''
            # For ISLES_Processed and IXI #
            origin = [float(lineList[1]), float(lineList[2]), float(lineList[3])]
            spacing = [float(lineList[5]), float(lineList[6]), float(lineList[7])]
            direction = [float(lineList[9]), float(lineList[10]), float(lineList[11]), 
                                float(lineList[12]), float(lineList[13]), float(lineList[14]),
                                    float(lineList[15]), float(lineList[16]), float(lineList[17])]
            BAT, TTP, TTD = int(lineList[19]), int(lineList[21]), int(lineList[23]) # '''
            
            nT = int(lineList[-1])
            temp_min_max_nT = min([temp_min_max_nT, nT])
            infos.append({'origin': origin, 'spacing': spacing, 'BAT': BAT, 'TTP': TTP, 'TTD': TTD, 'nT': nT})
            del lineList, origin, spacing, BAT, TTP, TTD
        return {'infos': infos, 'min_max_nT': temp_min_max_nT}

    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}
        #movie_nda = np.transpose(np.load(self.MoviePaths[idx]), (3, 0, 1, 2)) # (s, r, c, nT) -> (nT, s, r, c)
        TTP = int(self.all_movies['infos'][idx]['TTP'])
        #print(self.MoviePaths[idx])
        movie_nda = np.transpose(np.load(self.MoviePaths[idx]), (3, 0, 1, 2))[TTP:] # (s, r, c, nT) -> (nT, s, r, c)  # TODO NOTE: from global TTP #
        start_slc = np.random.randint(0, movie_nda.shape[1] - self.patch_size[0] + 1)
        start_row = np.random.randint(0, movie_nda.shape[2] - self.patch_size[1] + 1)
        start_col = np.random.randint(0, movie_nda.shape[3] - self.patch_size[2] + 1)
        end_slc, end_row, end_col = start_slc + self.patch_size[0], start_row + self.patch_size[1], start_col + self.patch_size[2]
        # Smart-selecting the most effective time frame: start from patch-based TTP
        patch_movie_nT = torch.from_numpy(movie_nda[:, start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)

        bat, ttp, ttd = get_patch_times(patch_movie_nT) # (nt, s, r, c)
        # TODO: For checking
        #print('[%s:%s, %s:%s, %s:%s]' % (start_slc, end_slc, start_row, end_row, start_col, end_col))
        #print('BAT~TTP~TTD: %s-%s-%s' % (bat, ttp, ttd)) 
        #print('    TTP~TTD: %s~%s' % (ttp, ttd)) 
        #print('    BAT~TTD: %s~%s' % (bat, ttd)) 
        ttd = self.input_n_collocations if ttd < self.input_n_collocations else ttd
        #ttp = ttd - self.input_n_collocations if ttp > ttd - self.input_n_collocations else ttp
        ttp = 0 # TODO (Now all start selecting from global TTP)
        start_input_t = np.random.randint(ttp, ttd - self.input_n_collocations + 1) 
        #print(start_input_t)
        if start_input_t + self.input_n_collocations < patch_movie_nT.shape[0]:
        #if ttp + self.input_n_collocations < patch_movie_nT.shape[0]:
        #if bat + self.input_n_collocations < patch_movie_nT.shape[0]:
            t0 = start_input_t
            t1 = t0 + self.input_n_collocations 
            #t0 = ttp
            #t1 = ttp + self.input_n_collocations
        else:
            t1 = patch_movie_nT.shape[0]
            t0 = t1 - self.input_n_collocations
        #print('Input t0~t1: %s~%s' % (t0, t1))
        patch_movie_nT = patch_movie_nT[t0 : t1] # Restict in patch-wise effective time frame # 

        #start_it  = np.random.randint(0, patch_movie_nT.shape[0] - self.loss_n_collocations + 1) 
        #end_it = start_it + self.loss_n_collocations
        start_it  = np.random.randint(patch_movie_nT.shape[0] - self.loss_n_collocations + 1) 
        end_it = start_it + self.loss_n_collocations
        #print('Loss  t0~t1: %s~%s (in time patch)' % (start_it, end_it)) # NOTE: time index within patch_movie_nT.nT

        if 'dirichlet' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT[start_it : end_it], self.args.dirichlet_width)
        elif 'cauchy' in self.BC:
            sample['movie_BC'] = extract_BC_3D(patch_movie_nT[start_it : end_it], 1) 
        elif 'source' in self.BC:
            sample['movie_dBC'] = extract_dBC_3D(patch_movie_nT[start_it : end_it], self.args.source_width)

        #if self.BC == 'neumann' or 'cauchy' in self.BC:
        #    patch_movie_nT = neumann_BC_3D(patch_movie_nT)
        sample['movie4input'] = patch_movie_nT
        sample['movie4loss'] = patch_movie_nT[start_it : end_it] # (batch_nT, s, r, c)
        sample['sub_collocation_t'] = self.sub_collocation_t 

        if self.DPaths:
            if self.args.predict_deviation:
                if self.dev_paths['orig_D'][idx] is not None: # lesion cases
                    orig_D = torch.from_numpy(np.load(self.dev_paths['orig_D'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['orig_D'] = orig_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
                    delta_D = torch.from_numpy(np.load(self.dev_paths['delta_D'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['delta_D'] = delta_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
                else: # normal cases
                    orig_D = torch.from_numpy(np.load(self.DPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['orig_D'] = orig_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
                    sample['delta_D'] = torch.zeros((9, self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device)
            elif self.args.predict_value_mask:
                if self.dev_paths['orig_D'][idx] is not None: # lesion cases
                    orig_D = torch.from_numpy(np.load(self.dev_paths['orig_D'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['orig_D'] = orig_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col]  
                    D = torch.from_numpy(np.load(self.DPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['D'] = D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
                else: # normal cases
                    orig_D = torch.from_numpy(np.load(self.DPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                    sample['orig_D'] = orig_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col]  
                    sample['D'] = orig_D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
            else:
                D = torch.from_numpy(np.load(self.DPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 9) -> (9, s, r, c)
                sample['D'] = D[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
        if self.DCoPaths:
            D_CO = torch.from_numpy(np.load(self.DCoPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
            sample['D_CO'] = D_CO[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
        if self.LPaths:
            if self.args.predict_deviation:
                if self.dev_paths['orig_L'][idx] is not None: # lesion cases
                    orig_L = torch.from_numpy(np.load(self.dev_paths['orig_L'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_L'] = orig_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                    delta_L = torch.from_numpy(np.load(self.dev_paths['delta_L'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['delta_L'] = delta_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                else: # normal cases
                    orig_L = torch.from_numpy(np.load(self.LPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_L'] = orig_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
                    sample['delta_L'] = torch.zeros((3, self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device)
            elif self.args.predict_value_mask:
                if self.dev_paths['orig_L'][idx] is not None: # lesion cases
                    orig_L = torch.from_numpy(np.load(self.dev_paths['orig_L'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_L'] = orig_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col] 
                    L = torch.from_numpy(np.load(self.LPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['L'] = L[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                else: # normal cases
                    orig_L = torch.from_numpy(np.load(self.LPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_L'] = orig_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col]  
                    sample['L'] = orig_L[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
            else:
                L = torch.from_numpy(np.load(self.LPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                sample['L'] = L[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
        if self.UPaths:
            U = torch.from_numpy(np.load(self.UPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
            sample['U'] = U[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
        if self.VPaths:
            if self.args.predict_deviation:
                if self.dev_paths['orig_V'][idx] is not None: # lesion cases
                    orig_V = torch.from_numpy(np.load(self.dev_paths['orig_V'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_V'] = orig_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                    delta_V = torch.from_numpy(np.load(self.dev_paths['delta_V'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['delta_V'] = delta_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                else:
                    orig_V = torch.from_numpy(np.load(self.VPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_V'] = orig_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                    sample['delta_V'] = torch.zeros((3, self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device)
            elif self.args.predict_value_mask:
                if self.dev_paths['orig_V'][idx] is not None: # lesion cases
                    orig_V = torch.from_numpy(np.load(self.dev_paths['orig_V'][idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_V'] = orig_V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                    V = torch.from_numpy(np.load(self.VPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['V'] = V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                else:
                    V = torch.from_numpy(np.load(self.VPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                    sample['orig_V'] = V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
                    sample['V'] = V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
            else:
                V = torch.from_numpy(np.load(self.VPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
                sample['V'] = V[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
        if self.PhiPaths:
            Phi = torch.from_numpy(np.load(self.PhiPaths[idx])).permute(3, 0, 1, 2).to(self.device) # (s, r, c, 3) -> (3, s, r, c)
            sample['Phi'] = Phi[:, start_slc : end_slc, start_row : end_row, start_col : end_col]
        if self.VesselPaths:
            #vessel_mask_nda = sitk.GetArrayFromImage(sitk.ReadImage(self.VesselPaths[idx])) # (s, r, c)
            vessel_mask_nda = np.load(self.VesselPaths[idx]) # (s, r, c)
            sample['vessel_mask'] = torch.from_numpy(vessel_mask_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
            vessel_mirror_mask_nda = np.load(self.VesselMirrorPaths[idx]) # (s, r, c)
            sample['vessel_mirror_mask'] = torch.from_numpy(vessel_mirror_mask_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
        if self.ValueMaskPaths is not None: # True if args.stochastic
            #if self.args.separate_DV_value_mask:
                #if self.ValueMaskPaths[idx] is not None:
            if isinstance(self.ValueMaskPaths[idx], str):
                value_mask_nda = np.load(self.ValueMaskPaths[idx]) # (s, r, c)
                sample['value_mask_D'] = torch.from_numpy(value_mask_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
                sample['value_mask_V'] = torch.from_numpy(value_mask_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
            elif self.ValueMaskPaths[idx] is not None:
                value_mask_D_nda = np.load(self.ValueMaskPaths[idx]['D']) # (s, r, c)
                value_mask_V_nda = np.load(self.ValueMaskPaths[idx]['V']) # (s, r, c)
                sample['value_mask_D'] = torch.from_numpy(value_mask_D_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
                sample['value_mask_V'] = torch.from_numpy(value_mask_V_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
            else:
                sample['value_mask_D'] = torch.ones((self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device) 
                sample['value_mask_V'] = torch.ones((self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device)
            if self.args.stochastic: # TODO: TESTING shared uncertainty # 
                sample['sigma'] = 1. - sample['value_mask_V'] if (sample['value_mask_D'] - sample['value_mask_V']).mean() > 0. else 1. - sample['value_mask_D']
            '''else:
                if self.ValueMaskPaths[idx] is not None:
                    value_mask_nda = np.load(self.ValueMaskPaths[idx]) # (s, r, c)
                    sample['value_mask'] = torch.from_numpy(value_mask_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device)
                else:
                    sample['value_mask'] = torch.ones((self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device) 
                if self.args.stochastic: # TODO: TESTING shared uncertainty #
                    sample['sigma'] = 1. - sample['value_mask']'''
        elif self.args.stochastic:
            sample['sigma'] = torch.zeros((self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device)
        
        if self.args.predict_segment:  
            if self.LesionSegPaths[idx] is not None: # lesion cases
                lesion_seg_nda = np.load(self.LesionSegPaths[idx])
                sample['lesion_seg'] = torch.from_numpy(lesion_seg_nda[start_slc : end_slc, start_row : end_row, start_col : end_col]).to(self.device) 
            else: # normal cases
                sample['lesion_seg'] = torch.zeros((self.patch_size[0], self.patch_size[1], self.patch_size[2])).to(self.device)

        #sample['origin'] = self.all_movies['infos'][idx]['origin']
        #sample['spacing'] = self.all_movies['infos'][idx]['spacing']
        return sample