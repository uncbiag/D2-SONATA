import os
import cv2
import matplotlib
import matplotlib.image
import numpy as np
import SimpleITK as sitk

def get_contour(msk_path, contour_thickness = 1, save_path = None):
    # Get BrainMask as slices of 2D images
    msk_fld = os.path.dirname(msk_path)
    msk = sitk.ReadImage(msk_path)
    origin = msk.GetOrigin()
    spacing = msk.GetSpacing()
    direction = msk.GetDirection()
    msk = sitk.GetArrayFromImage(msk)
    for s in range(msk.shape[0]):
        a = msk[s]
        matplotlib.image.imsave(os.path.join(msk_fld, "slc_%s.png" % s), a, cmap = 'gray')
    
    # Find contour for each slice of 2D brain mask image
    for s in range(msk.shape[0]):
        img = cv2.imread(os.path.join(msk_fld, "slc_%s.png" % s))
        img = 255 - img

        # draw gray box around image to detect edge buildings
        h,w = img.shape[:2]
        #cv2.rectangle(img,(0,0),(w-1,h-1), (50,50,50),1)

        # convert image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define color ranges
        low_yellow = (0,28,0)
        high_yellow = (27,255,255)

        low_gray = (0,0,0)
        high_gray = (179,255,233)

        # create masks
        yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow )
        gray_mask = cv2.inRange(hsv, low_gray, high_gray)

        # combine masks
        combined_mask = cv2.bitwise_or(yellow_mask, gray_mask)
        kernel = np.ones((3,3), dtype=np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE,kernel)

        # findcontours
        contours, hier = cv2.findContours(combined_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#, cv2.RETR_EXTERNAL

        # draw the outline of all contours
        for cnt in contours:
            cv2.drawContours(img,[cnt], 0, (0,255,0), thickness = contour_thickness)

        # save 2D result
        cv2.imwrite(os.path.join(msk_fld, 'contour_%s.png' % s), img)

    # Save 3D contour in .nii
    Contours = []
    for s in range(msk.shape[0]):
        msk = cv2.imread(os.path.join(msk_fld, 'contour_%s.png' % s), cv2.IMREAD_GRAYSCALE)
        msk[msk != 149] = 1
        msk[msk != 1] = 0
        msk.astype(float)
        Contours.append(msk)
        matplotlib.image.imsave(os.path.join(msk_fld, "bool_contour_%s.png" % s), msk, cmap = 'gray')
    Contours = np.array(Contours)
    Contours = abs(1 - Contours)

    Contours_img = sitk.GetImageFromArray(Contours)
    Contours_img.SetOrigin(origin)
    Contours_img.SetSpacing(spacing)
    Contours_img.SetDirection(direction)
    if save_path:
        sitk.WriteImage(Contours_img, save_path)
    else:
        sitk.WriteImage(Contours_img, os.path.join(msk_fld, 'Contour_%s.nii' % os.path.basename(msk_path)[:-4]))

    # Delete useless pngs
    for file in os.listdir(msk_fld):
        if file.startswith('slc_') or file.startswith('contour_') or file.startswith('bool_contour_'):
            os.remove(os.path.join(msk_fld, file)) 
    
    return Contours



################################################################################
if __name__ == '__main__':

    img_path = '/media/peirong/PEIRONG/StrokeData/ISLES2017/ISLES2017_Training/training_14/VSD.Brain.XX.O.OT.128055/Mirror_VSD.Brain.XX.O.OT.128055_cropped.nii'
    contour_thickness = 1
    get_contour(img_path, contour_thickness)