import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import SimpleITK as sitk

from prepro_utils import *

'''
For CTP image: 
Select out skull regions -> skull stripping -> brain region extraction
'''



def proceed_all(AllFld, Cases, postfix = '.nii'):

    AllPaths = [os.path.join(AllFld, patient) for patient in Cases]
    for i in range(len(AllPaths)):
        if not AllPaths[i].startswith('.') and 'case' in AllPaths[i]:
            print('Now process', os.path.basename(AllPaths[i]))
            ctp_path = get_ctp_paths(AllPaths[i], postfix)

            signal_img = sitk.ReadImage(ctp_path)
            signal_nda = sitk.GetArrayFromImage(signal_img) # (slice, row, column, time)
            org = signal_img.GetOrigin()
            spa = signal_img.GetSpacing()
            shp = signal_nda.shape

            ctc_axial, bat, ttp, ttd = signal2ctc('CTP', signal_nda)
            print('BAT-TTP-TTD: %s-%s-%s' % (bat, ttp, ttd))
            ctc_axial = ctc_axial[..., bat : ttd + 1]
            ctc_axial = cutoff_percentile(ctc_axial, mask = None, percentile_lower = 1., percentile_upper = 99)
            ctc_max = np.max(abs(ctc_axial))
            ctc_axial_norm = ctc_axial / ctc_max * 10

            file = open(os.path.join(os.path.dirname(ctp_path), 'info.txt'), 'w')
            file.write('Origin\n%s\n%s\n%s\n' % (org[0], org[1], org[2]))
            file.write('Spacing\n%s\n%s\n%s\n' % (spa[0], spa[1], spa[2]))
            file.write('BAT\n%s\nTTP\n%s\nTTD\n%s\n' % (0, ttp-bat, ctc_axial.shape[-1]-1))
            file.write('Shape\n%d\n%d\n%d\n%d' % (shp[0], shp[1], shp[2], shp[3]))
            file.close()
            
            # Save CTC
            ctc_axial = sitk.GetImageFromArray(ctc_axial, isVector = True)
            ctc_axial.SetOrigin(signal_img.GetOrigin())
            ctc_axial.SetSpacing(signal_img.GetSpacing())
            ctc_axial.SetDirection(signal_img.GetDirection())
            print('Computed CTC-Axial image saved as:', os.path.join(os.path.dirname(ctp_path), 'CTC_Axial_BATtoTTD.nii'))
            sitk.WriteImage(ctc_axial, os.path.join(os.path.dirname(ctp_path), 'CTC_Axial_BATtoTTD.nii'))

            ctc_axial_norm = sitk.GetImageFromArray(ctc_axial_norm, isVector = True)
            ctc_axial_norm.SetOrigin(signal_img.GetOrigin())
            ctc_axial_norm.SetSpacing(signal_img.GetSpacing())
            ctc_axial_norm.SetDirection(signal_img.GetDirection())
            print('Computed CTC-Axial image saved as:', os.path.join(os.path.dirname(ctp_path), 'CTC_Axial_BATtoTTD_norm.nii'))
            sitk.WriteImage(ctc_axial_norm, os.path.join(os.path.dirname(ctp_path), 'CTC_Axial_BATtoTTD_norm.nii'))


            '''ctc_temporal = sitk.GetImageFromArray(np.transpose(ctc_nda, (3, 1, 2, 0)), isVector = True)
            ctc_temporal.SetOrigin(signal_img.GetOrigin())
            ctc_temporal.SetSpacing(signal_img.GetSpacing())
            ctc_temporal.SetDirection(signal_img.GetDirection())
            print('Computed CTC-Temporal image saved as:', os.path.join(os.path.dirname(signal_path), 'CTC_Temporal_cropped.nii'))
            sitk.WriteImage(ctc_temporal, os.path.join(os.path.dirname(signal_path), 'CTC_Temporal_cropped.nii'))'''




       

################################################################################
if __name__ == '__main__':

    ISLES2018 = '/media/peirong/PR5/ISLES2018/ISLES2018_Training/TRAINING'
    Patients = os.listdir(ISLES2018)
    Patients.sort()
    #Patients = ['case_82']

    proceed_all(ISLES2018, Patients, postfix = '_masked_cropped.nii')


