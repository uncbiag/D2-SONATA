import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from torch.utils.data import Dataset
from scipy.ndimage.morphology import binary_closing, binary_fill_holes

from utils import save_sitk, make_dir, get_times
from DemoOptions.PerfFlag import adv_diff, adv_only, diff_only
from Preprocess.contour import get_contour


### Read in data info (For OneTimeFit.py, VoxelFit.py) ###
def read_ctc(CTC_path, dt, toTensor = True, isSave = True, device = 'cpu'):
    main_fld = os.path.dirname(CTC_path)
    CTC_img = sitk.ReadImage(CTC_path)
    CTC_data = sitk.GetArrayFromImage(CTC_img)
    origin = CTC_img.GetOrigin()
    spacing = CTC_img.GetSpacing()
    direction = CTC_img.GetDirection()
    BAT, TTP = get_times(CTC_data)
    GT_nT = CTC_data[..., TTP:].shape[-1]
    CTC_data = CTC_data[..., TTP : TTP + GT_nT] # Only use time after TTP
    T = np.arange(GT_nT) * dt
    data_dim = CTC_data.shape[:-1] # (slc, row, col)
    nT = CTC_data.shape[-1]
    data_spacing = [spacing[2], spacing[1], spacing[0]]  # Note: need to reverse beween nda and sitk img
    for t in range(nT):
        #print('Avg. value at t {:03d}: {:.6f}'.format(t, np.nanmean(np.where(CTC_data[..., t] != 0, CTC_data[..., t], np.nan))))
        print('Avg. value at t {:03d}: {:.6f}'.format(t + TTP, np.nanmean(np.where(CTC_data[..., t] != 0, CTC_data[..., t], np.nan))))

    mask = np.zeros(CTC_data[..., 0].shape)
    brain = np.where(CTC_data[..., 0] != 0)
    mask[brain] = 1
    for s in range(len(mask)):
        mask[s] = ndimage.binary_fill_holes(mask[s])
    if isSave:
        save_fld = make_dir(os.path.join(main_fld, '0'))
        nda_save_img(mask.astype(np.float32), origin, spacing, direction, save_path = \
            os.path.join(save_fld, 'BrainMask.nii'))
        # Save temporal CTC
        nda_save_img(CTC_data.astype(np.float32), origin, spacing, direction, save_path = \
            os.path.join(save_fld, 'AxialPerf.nii'))
        temporal_CTC = np.transpose(CTC_data, (3, 1, 2, 0))
        nda_save_img(temporal_CTC.astype(np.float32), origin, spacing, direction, save_path = \
            os.path.join(save_fld, 'TemporalPerf.nii'))
    if toTensor:
        CTC_data = torch.from_numpy(CTC_data).type(torch.float)
        mask = torch.from_numpy(mask).type(torch.float)
        T = torch.from_numpy(T).type(torch.float)
    CTC_data = CTC_data.to(device)
    mask = mask.to(device)
    T = T.to(device)
    return CTC_data, mask, T, origin, spacing, direction, data_dim, data_spacing, nT




def get_perf_flag(vessel_file, brain_mask_file, data_dim, perf_pattern, D_type, V_type, origin, spacing, device, isSave = True, save_fld = None): # has sign
    
    def flag_V(vessel, brain_mask, type, dim):
        flag = {}
        vessel *= brain_mask
        if type is 'constant' or 'scalar' in type:
            flag = {'isV': vessel}
        else:
            flag = {'isVx': vessel, 'isVy': vessel}
            if dim == 3:
                flag.update({'isVz': vessel})
        return flag

    def flag_D(vessel, brain_mask, type, dim):
        flag = {}
        non_vessel = brain_mask # diff for the entire brain region
        #non_vessel = abs(vessel - 1) * brain_mask # diff for only non-vessel region
        if type is 'constant' or 'scalar' in type:
            flag = {'isD': non_vessel}
        elif 'diag' in type:
            flag = {'isDxx': non_vessel, 'isDyy': non_vessel}
            if dim == 3:
                flag.update({'isDzz': non_vessel})
        elif 'full' in type:
            flag = {'isDxx': non_vessel, 'isDyy': non_vessel, 'isDxy': non_vessel}
            if dim == 3:
                flag.update({'isDxz': non_vessel, 'isDyz': non_vessel, 'isDzz': non_vessel})
        return flag

    dim = len(data_dim)
    vessel_img = sitk.ReadImage(vessel_file)
    brain_mask_img = sitk.ReadImage(brain_mask_file)
    vessel = torch.tensor(sitk.GetArrayFromImage(vessel_img), dtype = torch.float, device = device)
    brain_mask = torch.tensor(sitk.GetArrayFromImage(brain_mask_img), dtype = torch.float, device = device)
    perf_flag = {}
    if 'adv_only' in perf_pattern:
        perf_flag.update(flag_V(vessel, brain_mask, V_type, dim))
    elif 'diff_only' in perf_pattern:
        perf_flag.update(flag_D(vessel, brain_mask, D_type, dim))
    else:
        perf_flag.update(flag_V(vessel, brain_mask, V_type, dim))
        perf_flag.update(flag_D(vessel, brain_mask, D_type, dim))
    if isSave:
        for key in perf_flag:
            filename = os.path.join(save_fld, '%s.nii' % key)
            save_sitk(perf_flag[key], filename, origin, spacing, device = device, isVector = False, isNumpy = False)
            
    return perf_flag



class TimeSeriesDataset(Dataset):
    
    def __init__(self, args, SliceNumber, SaveFld, DataPath, MaskPath, ContourPath, BCPath, StartTime = 'PT', EndTime = None, TimeInterval = 10, TimeSpacing = 1., ToTensor = True, ToSave = True):
        '''
        Args:
        DataFolder (string): path/to/TimeSeriesData
        RandomCrop (int): crop time interval
        ToTensor (boolen): whether to convert to torch.Tensor
		'''
        self.args = args
        self.SliceNumber  = SliceNumber
        self.SaveFld      = SaveFld
        self.DataPath     = DataPath
        self.MaskPath     = MaskPath
        self.ContourPath  = ContourPath
        self.BCPath       = BCPath
        self.img          = sitk.ReadImage(self.DataPath)
        self.nda          = sitk.GetArrayFromImage(self.img) # (slice, row, column, time)
        self.ArrivalTime, self.PeakTime, self.BottomTime  = get_times(self.nda)
        if StartTime is 'PT':
            self.StartTime = self.PeakTime
        elif StartTime is 'AT':
            self.StartTime = self.ArrivalTime
        if EndTime is 'PT':
            self.EndTime = self.PeakTime
        else:
            self.EndTime = self.BottomTime # Discrad last 5 time points
        # TODO
        if args.case is 'PWI':
            self.StartTime = 0
            self.EndTime = self.nda.shape[-1] - 1
        if self.EndTime - self.StartTime + 1 < TimeInterval:
            raise ValueError('Inputed TimeInterval is larger than availble time points, reset them !!!')
        elif TimeInterval < 1:
            #self.TimeInterval = min(self.EndTime - self.StartTime + 1, 25) # 15: max number of batch time points
            self.TimeInterval = self.EndTime - self.StartTime + 1
        else:
            self.TimeInterval = TimeInterval
        self.TimeSpacing  = TimeSpacing
        self.ToTensor     = ToTensor
        self.ToSave       = ToSave
        if self.SliceNumber is not None: # is 2D version
            self.nda      = self.nda[SliceNumber] # (row, column, time)
        self.brain_mask   = self.get_brain_mask(self.PeakTime)
        self.nda          = np.repeat(self.brain_mask[..., np.newaxis], self.nda.shape[-1], axis = len(self.nda.shape) - 1) * self.nda
        self.nda          = self.nda.astype(np.float32) 
        self.contour      = self.get_contour()
        self.BC           = self.get_BC()
        if args.case is 'PWI' and args.GT_perf_pattern is 'adv_diff' and not self.args.noisy:
            self.nda_adv  = sitk.GetArrayFromImage(sitk.ReadImage('%s (Adv_Only).nii' % self.DataPath[:-4]))
            self.nda_adv  = np.repeat(self.brain_mask[..., np.newaxis], self.nda_adv.shape[-1], axis = len(self.nda_adv.shape) - 1) * self.nda_adv
            self.nda_adv  = self.nda_adv.astype(np.float32) 
            self.nda_diff = sitk.GetArrayFromImage(sitk.ReadImage('%s (Diff_Only).nii' % self.DataPath[:-4]))
            self.nda_diff = np.repeat(self.brain_mask[..., np.newaxis], self.nda_diff.shape[-1], axis = len(self.nda_diff.shape) - 1) * self.nda_diff
            self.nda_diff = self.nda_diff.astype(np.float32) 

        self.nT           = self.EndTime - self.StartTime + 1 # total number of useful time points
        self.time         = np.arange(self.nT) * self.TimeSpacing # (real_time, )
        self.train_t      = self.time[:self.TimeInterval]
        self.test_t       = self.time[:self.nT]
        self.info = '%s~%s' % (self.StartTime, self.EndTime)
        self.data_dim = self.nda.shape[0], self.nda.shape[1], self.nda.shape[2] 
    
    def get_BC(self): # 4D
        if self.BCPath is not None:
            BC = sitk.GetArrayFromImage(sitk.ReadImage(self.BCPath))
            #if self.ToTensor:
            #    BC = torch.tensor(BC, dtype = torch.float).expand(1, -1, -1, -1)
        else:
            BC = None
        return BC
                
    def get_contour(self): # 3D
        if self.ContourPath is not None:
            if os.path.isfile(self.ContourPath):
                contour = sitk.GetArrayFromImage(sitk.ReadImage(self.ContourPath))
            else:
                contour = get_contour(self.MaskPath)
                #contour_img = sitk.GetImageFromArray(contour, isVector = False)
                #contour_img.SetOrigin(self.get_sitk_info['origin'])
                #contour_img.SetSpacing(self.get_sitk_info['spacing'])
                #contour_img.SetDirection(self.get_sitk_info['direction'])
                #sitk.WriteImage(contour_img, self.ContourPath)
            if self.ToTensor:
                contour = torch.tensor(contour, dtype = torch.float)
        else:
            contour = None
        return contour
    
    def get_brain_mask(self, select_time):
        if select_time < 0 :
            select_time = self.PeakTime
        if self.MaskPath is None or not os.path.isfile(self.MaskPath):
            msk = np.ones(self.nda[..., select_time].shape)
            non_brain = np.where(abs(self.nda[..., select_time]) < 0.01) # TODO
            msk[non_brain] = 0
            for s in range(len(msk)):
                msk[s] = ndimage.binary_fill_holes(msk[s])
            mask_img = sitk.GetImageFromArray(msk, isVector = False)
            mask_img.SetOrigin(self.get_sitk_info['origin'])
            mask_img.SetSpacing(self.get_sitk_info['spacing'])
            mask_img.SetDirection(self.get_sitk_info['direction'])
            sitk.WriteImage(mask_img, self.MaskPath)
        else:
            msk = sitk.GetArrayFromImage(sitk.ReadImage(self.MaskPath))

        if self.SliceNumber is not None: # is 2D version
            msk = msk[self.SliceNumber]
        return msk.astype(np.float32) 

    @property
    def avg_value(self):
        return np.nanmean(np.where(self.nda[..., self.StartTime :]!= 0, self.nda[..., self.StartTime :], np.nan))

    @property
    def get_test_data(self):
        '''Return the entire time series dataset (of full time length) as testing data'''
        sample = {}
        C0     = self.nda[..., self.StartTime] # (slice, row, column)
        C0     = np.expand_dims(C0, axis = 0) # (n_batch = 1, slice, row, column)
        C      = np.stack([self.nda[..., self.StartTime + i] for i in range(self.nT)], axis = 0) # (time_dim = nT, (slice,) row, colmn)
        if self.ToSave:
            save_fld = make_dir(os.path.join(self.SaveFld, '0'))
        for t in range(len(C)):
            print('Avg. non-zero value at time NO.%2d: %.6f' % (t + self.StartTime, np.nanmean(np.where(C[t]!= 0, C[t], np.nan))))
        if self.SliceNumber is None:
            C_axial = np.transpose(C, (1, 2, 3, 0)) # (time, slice, row, column) -> (slice, row, column, time)
            C_temporal = np.transpose(C, (0, 2, 3, 1)) # (time, slice, tow, column) -> (time, row, column, slice)
            if self.ToSave:
                save_sitk(C_axial, self.axial_name, self.get_sitk_info['origin'], self.get_sitk_info['spacing'], device = 'cpu', isVector = True, isNumpy = True)
                save_sitk(C_temporal, self.temporal_name, self.get_sitk_info['origin'], self.get_sitk_info['spacing'], device = 'cpu', isVector = True, isNumpy = True)
        else:
            if self.ToSave:
                name = os.path.join(save_fld, 'Perf_Reg_GT (%s).nii' % self.info)
                save_sitk(C, name, self.get_sitk_info['origin'], self.get_sitk_info['spacing'], device = 'cpu', isVector = False, isNumpy = True)
        
        C  = np.expand_dims(C, axis = 0) # (n_batch = 1, time_dim, slice, row, column)
        if self.BC is not None:
            BC = self.BC[..., self.StartTime + 1 : self.StartTime + self.nT]
            BC = np.expand_dims(BC, axis = 0)
        
        ###############################################################
        if self.args.case is 'PWI' and self.args.GT_perf_pattern is 'adv_diff' and not self.args.noisy:
            C_adv = np.stack([self.nda_adv[..., self.StartTime + i] for i in range(self.nT)], axis = 0) # (time_dim = nT, (slice,) row, colmn)
            C_adv = np.expand_dims(C_adv, axis = 0)
            C_diff = np.stack([self.nda_diff[..., self.StartTime + i] for i in range(self.nT)], axis = 0) # (time_dim = nT, (slice,) row, colmn)
            C_diff = np.expand_dims(C_diff, axis = 0)
        ###############################################################

        if self.ToTensor:
            sample['C0'] = torch.from_numpy(C0)
            sample['C']  = torch.from_numpy(C)
            sample['t']  = torch.from_numpy(self.test_t)
            if self.BC is not None:
                sample['BC'] = torch.from_numpy(BC)
            if self.args.case is 'PWI' and self.args.GT_perf_pattern is 'adv_diff' and not self.args.noisy:
                sample['C_adv'] = torch.from_numpy(C_adv)
                sample['C_diff'] = torch.from_numpy(C_diff)
        else:
            sample['C0'] = C0
            sample['C']  = C
            sample['t']  = self.test_t
            if self.BC is not None:
                sample['BC'] = BC
            if self.args.case is 'PWI' and self.args.GT_perf_pattern is 'adv_diff' and not self.args.noisy:
                sample['C_adv'] = C_adv
                sample['C_diff'] = C_diff
        return sample
    
    @property
    def axial_name(self):
        return os.path.join(os.path.join(self.SaveFld, '0'), 'AxialPerf_Reg_GT (%s).nii' % self.info)
    @property
    def temporal_name(self):
        return os.path.join(os.path.join(self.SaveFld, '0'), 'TemporalPerf_Reg_GT (%s).nii' % self.info)

    @property
    def get_sitk_info(self):
        return {'origin': self.img.GetOrigin(), 'spacing': self.img.GetSpacing(), 'direction': self.img.GetDirection()}
    
    @property
    def data_spacing(self):
        spacing = self.get_sitk_info['spacing']
        return torch.tensor([spacing[2], spacing[1], spacing[0]], dtype = torch.float)
    
    def __getitem__(self, idx):
        sample   = {}
        idx     += self.StartTime
        C0       = self.nda[..., idx] # (slice, row, column)
        if self.BC is not None:
            # BC imposement start from 1 (not 0) time point
            BC   = self.BC[..., self.StartTime + 1 : self.StartTime + self.TimeInterval] # (s, r, c, t)
        C = np.stack([self.nda[..., idx + i] for i in range(self.TimeInterval)], axis = 0) # (time, slice, row, column)
        if self.ToTensor:
            sample['C0'] = torch.from_numpy(C0)
            sample['C']  = torch.from_numpy(C)
            sample['t']  = torch.from_numpy(self.train_t)
            if self.BC is not None:
                sample['BC'] = torch.from_numpy(BC)
        else:
            sample['C0'] = C0
            sample['C']  = C
            sample['t']  = self.train_t
            if self.BC is not None:
                sample['BC'] = BC
        return sample
        
    def __len__(self):  
        return self.nT - self.TimeInterval + 1
