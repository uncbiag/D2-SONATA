import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

#from learning.utils import make_dir

# For Vector Dataset: 2D, 3D vector field dataset #

param_name_3d = 'V'
param_name_2d = 'V'

'''
3D vector
'''

class VTrain3DPatchDataset(Dataset):

    def __init__(self, args, stride, patch_size, DataFolder, Patients, RandomCrop = False, ToTensor = True):

        self.args         = args
        self.stride       = stride
        self.patch_size   = patch_size
        self.DataFolder   = DataFolder
        self.Patients     = Patients
        self.PatientPaths = [os.path.join(self.DataFolder, patient) for patient in self.Patients]
        self.RandomCrop   = RandomCrop
        self.ToTensor     = ToTensor

    def __getitem__(self, idx):
        sample = {}
        V = sitk.ReadImage(os.path.join(self.PatientPaths[idx], '%s.nii' % param_name_3d))
        Mask = sitk.ReadImage(os.path.join(self.PatientPaths[idx], 'Mask.nii'))
        V_nda = sitk.GetArrayFromImage(V).transpose((3, 0, 1, 2)) # (s, r, c, 3) -> (3, s, r, c)
        Mask_nda = sitk.GetArrayFromImage(Mask)

        sample['Origin'] = V.GetOrigin()
        sample['Spacing'] = V.GetSpacing()
        sample['Direction'] = V.GetDirection()

        if self.RandomCrop:
            OrigDepth, OrigHeight, OrigWidth = Mask_nda.shape
            CropDepth, CropHeight, CropWidth = (self.patch_size[0], self.patch_size[1], self.patch_size[2])
            if CropDepth > OrigDepth: OrigDepth = CropDepth
            if CropHeight > OrigHeight: OrigHeight = CropHeight
            if CropWidth > OrigWidth: OrigWidth = CropWidth
            
            temp_V = np.zeros((3, OrigDepth, OrigHeight, OrigWidth))
            temp_V[:, :Mask_nda.shape[0], :Mask_nda.shape[1], :Mask_nda.shape[2]] = V_nda
            temp_mask = np.zeros((OrigDepth, OrigHeight, OrigWidth))
            temp_mask[:Mask_nda.shape[0], :Mask_nda.shape[1], :Mask_nda.shape[2]] = Mask_nda

            if CropDepth  == OrigDepth:  OrigDepth  += 1
            if CropHeight == OrigHeight: OrigHeight += 1
            if CropWidth  == OrigWidth:  OrigWidth  += 1
            
            #print(Mask_nda.shape)
            axial     = np.random.randint(0, OrigDepth - CropDepth)
            coronal   = np.random.randint(0, OrigHeight - CropHeight)
            sagittal  = np.random.randint(0, OrigWidth - CropWidth)
            temp_V    = temp_V[:, axial : axial + CropDepth, coronal : coronal + CropHeight, sagittal : sagittal + CropWidth] # (3, s, r, c)
            temp_mask = temp_mask[axial : axial + CropDepth, coronal : coronal + CropHeight, sagittal : sagittal + CropWidth] # (s, r, c)
        else:
            temp_V    = temp_V
            temp_mask = temp_mask
        if self.ToTensor:
            sample['Data'] = torch.from_numpy(temp_V)
            sample['Mask'] = torch.from_numpy(temp_mask)
        else:
            sample['Data'] = temp_V
            sample['Mask'] = temp_mask
        return sample

    def __len__(self):  
        return len(self.Patients)
            


class VTest3DPatchDataset(Dataset):
    '''
    Only support single case
    '''
    def __init__(self, args, stride, patch_size, PatientFld, ToTensor = True):
        self.args     = args
        self.ToTensor = ToTensor
        self.V = sitk.ReadImage(os.path.join(PatientFld, '%s.nii' % param_name_3d))
        self.Mask = sitk.ReadImage(os.path.join(PatientFld, 'Mask.nii'))
        self.V_nda = sitk.GetArrayFromImage(self.V).transpose((3, 0, 1, 2)) # (s, r, c, 3) -> (3, s, r, c)
        self.Mask_nda = sitk.GetArrayFromImage(self.Mask)
        self.data_dim = self.Mask_nda.shape[0], self.Mask_nda.shape[1], self.Mask_nda.shape[2] 
        self.Origin, self.Spacing, self.Direction = self.V.GetOrigin(), self.V.GetSpacing(), self.V.GetDirection()
    
        self.stride_slc, self.stride_row, self.stride_col = stride[0], stride[1], stride[2]
        self.slc, self.row, self.col = self.Mask_nda.shape
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
        print(self.avail_slc, self.avail_row, self.avail_col)
    
    @property
    def idx2coord(self):
        coord_dict = np.zeros((self.avail_slc * self.avail_row * self.avail_col, 3))
        for idx in range(coord_dict.shape[0]):
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
            coord_dict[idx] = np.array([i_slc, i_row, i_col])
        return coord_dict


    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}
        i_slc, i_row, i_col = self.idx2coord[idx].astype(int)
        #print(i_slc, i_row, i_col)
        temp_V = self.V_nda[:, i_slc : i_slc + self.patch_slc, i_row : i_row + self.patch_row, i_col: i_col + self.patch_col] # (3, s, r, c)
        temp_mask = self.Mask_nda[i_slc : i_slc + self.patch_slc, i_row : i_row + self.patch_row, i_col: i_col + self.patch_col] # (s, r, c)
        if self.ToTensor:
            sample['start_slc'] = torch.from_numpy(np.array(i_slc))
            sample['start_row'] = torch.from_numpy(np.array(i_row))
            sample['start_col'] = torch.from_numpy(np.array(i_col))
            sample['Data'] = torch.from_numpy(temp_V)
            sample['Mask'] = torch.from_numpy(temp_mask)
        else:
            sample['start_slc'] = np.array(i_slc)
            sample['start_row'] = np.array(i_row)
            sample['start_col'] = np.array(i_col)
            sample['Data'] = temp_V
            sample['Mask'] = temp_mask
        return sample

    def __len__(self):  
        '''
        Available number of patches in one case
        '''
        return self.avail_slc * self.avail_row * self.avail_col



'''
2D vector
'''

class VTrain2DPatchDataset(Dataset):

    def __init__(self, args, stride, patch_size, DataFolder, Patients, RandomCrop = False, ToTensor = True):

        self.args         = args
        self.stride       = stride
        self.patch_size   = patch_size
        self.DataFolder   = DataFolder
        self.Patients     = Patients
        self.PatientPaths = [os.path.join(self.DataFolder, patient) for patient in self.Patients]
        self.RandomCrop   = RandomCrop
        self.ToTensor     = ToTensor

    def __getitem__(self, idx):
        sample = {}
        V = sitk.ReadImage(os.path.join(self.PatientPaths[idx], '%s.nii' % param_name_2d))
        V_nda = sitk.GetArrayFromImage(V) # (2, r, c)

        if self.RandomCrop:
            OrigHeight, OrigWidth = Mask_nda.shape
            CropHeight, CropWidth = (self.patch_size[0], self.patch_size[1])
            if CropHeight > OrigHeight: OrigHeight = CropHeight
            if CropWidth > OrigWidth: OrigWidth = CropWidth
            
            temp_V = np.zeros((2, OrigHeight, OrigWidth))
            temp_V[:, :V_nda.shape[1], :V_nda.shape[2]] = V_nda

            if CropHeight == OrigHeight: OrigHeight += 1
            if CropWidth  == OrigWidth:  OrigWidth  += 1
            
            coronal   = np.random.randint(0, OrigHeight - CropHeight)
            sagittal  = np.random.randint(0, OrigWidth - CropWidth)
            temp_V    = temp_V[:, coronal : coronal + CropHeight, sagittal : sagittal + CropWidth] # (2, r, c)
        else:
            temp_V    = temp_V
        if self.ToTensor:
            sample['Data'] = torch.from_numpy(temp_V)
        else:
            sample['Data'] = temp_V
        return sample

    def __len__(self):  
        return len(self.Patients)
            

class VTest2DPatchDataset(Dataset):
    '''
    Only support single case
    '''
    def __init__(self, args, stride, patch_size, DataPath, ToTensor = True):
        self.args     = args
        self.ToTensor = ToTensor
        self.V = sitk.ReadImage(DataPath)
        # TODO
        self.V_nda = sitk.GetArrayFromImage(self.V) * 10000 # (2, r, c) # 1000
        self.data_dim = self.V_nda.shape[1], self.V_nda.shape[2]
        self.Origin, self.Spacing, self.Direction = self.V.GetOrigin(), self.V.GetSpacing(), self.V.GetDirection()
    
        self.stride_row, self.stride_col = stride[0], stride[1]
        self.row, self.col = self.V_nda.shape[1], self.V_nda.shape[2]
        self.patch_row = self.row if patch_size[0] < 1 else patch_size[0]
        self.patch_col = self.col if patch_size[1] < 1 else patch_size[1]
        assert self.patch_row <= self.row and self.patch_col <= self.col
        self.avail_row = int(np.floor((self.row - self.patch_row) / self.stride_row) + 1)
        if ((self.avail_row - 1) * self.stride_row) + self.patch_row < self.row: 
            self.avail_row += 1
        self.avail_col = int(np.floor((self.col - self.patch_col) / self.stride_col) + 1)
        if ((self.avail_col - 1) * self.stride_col) + self.patch_col < self.col: 
            self.avail_col += 1
        print(self.avail_row, self.avail_col)
    
    @property
    def idx2coord(self):
        coord_dict = np.zeros((self.avail_row * self.avail_col, 2))
        for idx in range(coord_dict.shape[0]):
            i_row, i_col = np.unravel_index(idx, (self.avail_row, self.avail_col))
            if i_row == self.avail_row - 1:
                i_row = self.row - self.patch_row
            else:
                i_row = self.stride_row * i_row
            if i_col == self.avail_col - 1:
                i_col = self.col - self.patch_col
            else:
                i_col = self.stride_col * i_col
            coord_dict[idx] = np.array([i_row, i_col])
        return coord_dict


    def __getitem__(self, idx):
        '''
        idx: determine patch 
        '''
        sample   = {}
        i_row, i_col = self.idx2coord[idx].astype(int)
        temp_V = self.V_nda[:, i_row : i_row + self.patch_row, i_col: i_col + self.patch_col] # (2, r, c)
        if self.ToTensor:
            sample['start_row'] = torch.from_numpy(np.array(i_row))
            sample['start_col'] = torch.from_numpy(np.array(i_col))
            sample['Data'] = torch.from_numpy(temp_V)
        else:
            sample['start_row'] = np.array(i_row)
            sample['start_col'] = np.array(i_col)
            sample['Data'] = temp_V
        return sample

    def __len__(self):  
        '''
        Available number of patches in one case
        '''
        return self.avail_row * self.avail_col