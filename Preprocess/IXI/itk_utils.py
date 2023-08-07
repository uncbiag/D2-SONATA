import os

import numpy as np

import itk
from itk import TubeTK as ttk
#from itkwidgets import view

'''TubeTK Doc: https://public.kitware.com/Wiki/TubeTK/Documentation'''



# Baseline images assisting with brain mask extraction #
N = 8
BaseLineFld = '/media/peirong/PR/TubeTK_Data'
readerList = ["003", "010", "026", "034", "045", "056", "063", "071"]

# General setting for itk image reader #
ImageType = itk.Image[itk.F, 3]
ReaderType = itk.ImageFileReader[ImageType]
def read_img(img_path):
    reader = ReaderType.New(FileName = img_path)
    reader.Update()
    return reader.GetOutput() 

def resample(itk_img, new_spacing = [1., 1., 1.], save_path = None):
    res = ttk.ResampleImage.New(Input = itk_img) 
    res.SetSpacing(new_spacing) #res.SetMakeHighResIso(True)
    res.Update()
    iso_img = res.GetOutput()
    if save_path:
        itk.imwrite(iso_img, save_path)
    return iso_img

def register_images(moving_img, fixed_img, save_path = None):
    imReg = ttk.RegisterImages[ImageType].New()  # standard protocol for within patient within visit registration
    imReg.SetFixedImage(fixed_img)
    imReg.SetMovingImage(moving_img)
    imReg.SetReportProgress(False) # True: print register process 
    imReg.SetExpectedOffsetMagnitude(40)
    imReg.SetExpectedRotationMagnitude(0.01)
    imReg.SetExpectedScaleMagnitude(0.01)
    imReg.SetRigidMaxIterations(500)
    imReg.SetRigidSamplingRatio(0.1)
    imReg.SetRegistration("RIGID")
    imReg.SetMetric("MATTES_MI_METRIC")
    imReg.Update() 
    # NOTE: interpolation arg to be chosen
    registered_moving_img = imReg.GetFinalMovingImage("LINEAR_INTERPOLATION")
    if save_path:
        itk.imwrite(registered_moving_img, save_path)
    return registered_moving_img


def extract_brain(itk_img1, itk_img2, itk_img3, save_path = None):
    '''
    Three same-subject registered images for more robust results
    '''
    imBase = []
    imBaseB = []
    for i in range(0, N):
        name = os.path.join(BaseLineFld, "Normal"+readerList[i]+"-FLASH.mha")
        nameB = os.path.join(BaseLineFld, "Normal"+readerList[i]+"-FLASH-Brain.mha")
        imBase.append(read_img(name))
        imBaseB.append(read_img(nameB))
    
    imMath = ttk.ImageMath.New(Input = itk_img2)
    imMath.Blur(2)
    itk_img_Blur = imMath.GetOutput()

    regB = []
    regBB = []
    #print('Start')
    for i in range(0,N):
        #print(i)
        imMath.SetInput(imBase[i])
        imMath.Blur(2)
        imBaseBlur = imMath.GetOutput()
        regBTo1 = ttk.RegisterImages[ImageType].New(FixedImage = itk_img_Blur, MovingImage = imBaseBlur)
        regBTo1.SetReportProgress(False) # True: print register process 
        regBTo1.SetExpectedOffsetMagnitude(40)
        regBTo1.SetExpectedRotationMagnitude(0.01)
        regBTo1.SetExpectedScaleMagnitude(0.1)
        regBTo1.SetRigidMaxIterations(500)
        regBTo1.SetAffineMaxIterations(500)
        regBTo1.SetRigidSamplingRatio(0.1)
        regBTo1.SetAffineSamplingRatio(0.1)
        regBTo1.SetInitialMethodEnum("INIT_WITH_IMAGE_CENTERS")
        regBTo1.SetRegistration("PIPELINE_AFFINE")
        regBTo1.SetMetric("MATTES_MI_METRIC")
        #regBTo1.SetMetric("NORMALIZED_CORRELATION_METRIC") - Really slow!
        #regBTo1.SetMetric("MEAN_SQUARED_ERROR_METRIC")
        regBTo1.Update()
        img = regBTo1.ResampleImage("LINEAR", imBase[i])
        regB.append(img)
        img = regBTo1.ResampleImage("LINEAR", imBaseB[i])
        regBB.append(img)
    
    regBBT = []
    for i in range(0,N):
        imMath = ttk.ImageMath[ImageType,ImageType].New(Input = regBB[i])
        imMath.Threshold(0, 1, 0, 1)
        img = imMath.GetOutput()
        if i==0:
            imMathSum = ttk.ImageMath[ImageType,ImageType].New(img)
            imMathSum.AddImages(img, 1.0/N, 0)
            sumBBT = imMathSum.GetOutput()
        else:
            imMathSum = ttk.ImageMath[ImageType,ImageType].New(sumBBT)
            imMathSum.AddImages(img, 1, 1.0/N)
            sumBBT = imMathSum.GetOutput()
    #view(sumBBT)

    insideMath = ttk.ImageMath[ImageType,ImageType].New(Input = sumBBT)
    insideMath.Threshold(1, 1, 1, 0)
    insideMath.Erode(5, 1, 0)
    brainInside = insideMath.GetOutput()

    outsideMath = ttk.ImageMath[ImageType,ImageType].New( Input = sumBBT )
    outsideMath.Threshold(0, 0, 1, 0)
    outsideMath.Erode(10, 1, 0)
    brainOutsideAll = outsideMath.GetOutput()
    outsideMath.Erode(20, 1, 0)
    outsideMath.AddImages(brainOutsideAll, -1, 1)
    brainOutside = outsideMath.GetOutput()

    outsideMath.AddImages(brainInside, 1, 2)
    brainCombinedMask = outsideMath.GetOutputUChar()

    outsideMath.AddImages(itk_img2, 512, 1)
    brainCombinedMaskView = outsideMath.GetOutput()
    #view(brainCombinedMaskView) # Plotting #

    LabelMapType = itk.Image[itk.UC,3]
    segmenter = ttk.SegmentConnectedComponentsUsingParzenPDFs[ImageType,LabelMapType].New()
    segmenter.SetFeatureImage(itk_img1)
    segmenter.AddFeatureImage(itk_img2)
    segmenter.AddFeatureImage(itk_img3)
    segmenter.SetInputLabelMap( brainCombinedMask )
    segmenter.SetObjectId(2)
    segmenter.AddObjectId(1)
    segmenter.SetVoidId(0)
    segmenter.SetErodeDilateRadius(5)
    segmenter.Update()
    segmenter.ClassifyImages()
    brainCombinedMaskClassified = segmenter.GetOutputLabelMap()

    cast = itk.CastImageFilter[LabelMapType, ImageType].New()
    cast.SetInput(brainCombinedMaskClassified)
    cast.Update()
    brainMaskF = cast.GetOutput()

    brainMath = ttk.ImageMath[ImageType,ImageType].New(Input = brainMaskF)
    brainMath.Threshold(2, 2, 1, 0)
    brainMath.Dilate(2, 1, 0)
    brainMask = brainMath.GetOutput()
    
    if save_path:
        itk.imwrite(brainMask, save_path)
    #view(brain)

    '''brainMath.SetInput(itk_img1) # 
    brainMath.ReplaceValuesOutsideMaskRange( brainMask, 1, 1, 0)
    brain = brainMath.GetOutput()'''

    return brainMath, brainMask


def masking(masker, itk_img, mask_img, save_path = None):
    '''
    Jointly use with extract_brain()
    '''
    masker.SetInput(itk_img)
    masker.ReplaceValuesOutsideMaskRange(mask_img, 1, 1, 0 )
    masked_img = masker.GetOutput()
    if save_path is not None:
        itk.imwrite(masked_img, save_path)
    return masked_img


def enhance_vessel(mra_path, itk_path1, itk_path2, mask_path, save_path = None):
    '''
    Three same-subject registered images for more robust results
    First image: MRA image for vessel enhance implementation
    '''
    mra_img = read_img(mra_path)
    itk_img2 = read_img(itk_path1)
    itk_img3 = read_img(itk_path2)

    imMath = ttk.ImageMath[ImageType,ImageType].New()
    imMath.SetInput(mra_img)
    imMath.Blur(1.0)
    imBlur = imMath.GetOutput()
    imBlurArray = itk.GetArrayViewFromImage(imBlur)

    numSeeds = 10
    seedCoverage = 20
    seedCoord = np.zeros([numSeeds,3])
    for i in range(numSeeds):
        seedCoord[i] = np.unravel_index(np.argmax(imBlurArray, axis = None), imBlurArray.shape)
        indx = [int(seedCoord[i][0]),int(seedCoord[i][1]),int(seedCoord[i][2])]
        minX = max(indx[0]-seedCoverage,0)
        maxX = max(indx[0]+seedCoverage,imBlurArray.shape[0])
        minY = max(indx[1]-seedCoverage,0)
        maxY = max(indx[1]+seedCoverage,imBlurArray.shape[1])
        minZ = max(indx[2]-seedCoverage,0)
        maxZ = max(indx[2]+seedCoverage,imBlurArray.shape[2])
        imBlurArray[minX:maxX,minY:maxY,minZ:maxZ]=0
        indx.reverse()
        seedCoord[:][i] = mra_img.TransformIndexToPhysicalPoint(indx)
    #print(seedCoord)

    # Manually extract a few vessels to form an image-specific training set
    vSeg = ttk.SegmentTubes[ImageType].New()
    vSeg.SetInput(mra_img)
    vSeg.SetVerbose(True)
    vSeg.SetMinRoundness(0.1)
    vSeg.SetMinCurvature(0.001)
    vSeg.SetRadiusInObjectSpace( 1 )
    for i in range(numSeeds):
        #print("**** Processing seed " + str(i) + " : " + str(seedCoord[i]))
        vSeg.ExtractTubeInObjectSpace( seedCoord[i], i )
    tubeMaskImage = vSeg.GetTubeMaskImage()
    #view(tubeMaskImage)

    LabelMapType = itk.Image[itk.UC,3]
    trMask = ttk.ComputeTrainingMask[ImageType,LabelMapType].New()
    trMask.SetInput(tubeMaskImage)
    trMask.SetGap(3)
    #trMask.SetObjectWidth(1)
    trMask.SetNotObjectWidth(1)
    trMask.Update()
    fgMask = trMask.GetOutput()

    enhancer = ttk.EnhanceTubesUsingDiscriminantAnalysis[ImageType,LabelMapType].New()
    enhancer.SetInput(mra_img)
    enhancer.AddInput(itk_img2)
    enhancer.AddInput(itk_img3)
    enhancer.SetLabelMap(fgMask)
    enhancer.SetRidgeId(255)
    enhancer.SetBackgroundId(127)
    enhancer.SetUnknownId(0)
    enhancer.SetTrainClassifier(True)
    enhancer.SetUseIntensityOnly(True)
    enhancer.SetScales([0.3333, 1, 2.5])
    enhancer.Update()
    enhancer.ClassifyImages()
    #view(enhancer.GetClassProbabilityImage(0))

    vess_prob = itk.SubtractImageFilter(Input1 = enhancer.GetClassProbabilityImage(0), Input2 = enhancer.GetClassProbabilityImage(1))
    brainMask = itk.imread(mask_path, itk.F)
    vess_prob_brain = itk.MultiplyImageFilter(Input1 = vess_prob, Input2 = brainMask)
    if save_path:
        #itk.imwrite(vess_prob, os.path.join(save_fld, "VesselEnhanced_Brain.mha"), compression = True)
        itk.imwrite(vess_prob_brain, save_path, compression=True)

    return vess_prob_brain


ImageDimension = 3
PixelType = itk.ctype('float')
ImageType = itk.Image[PixelType, ImageDimension]
img_reader = itk.ImageFileReader[ImageType].New()
img_writer = itk.ImageFileWriter[ImageType].New()
def anisotropic_smoothing(img_path, n_iter, diffusion_time = 3.5, anisotropic_lambda = 0.1, enhancement_type = 3, \
    noise_scale = 3, feature_scale = 5, exponent = 3.5):
    smoothed_paths = []
    img_reader.SetFileName(img_path)
    itk_img = img_reader.GetOutput()
    for i_iter in range(n_iter):
        smoother = itk.CoherenceEnhancingDiffusionImageFilter.New(itk_img)
        smoother.SetDiffusionTime(diffusion_time)
        smoother.SetLambda(anisotropic_lambda)
        smoother.SetEnhancement(enhancement_type)
        smoother.SetNoiseScale(noise_scale)
        smoother.SetFeatureScale(feature_scale)
        smoother.SetExponent(exponent)
        smoother.Update()
        itk_img = smoother.GetOutput()
        '''# For checking #
        smoothed_path = '%s_smoothed-%d.nii' % (img_path[:-4], (i_iter))
        img_writer.SetFileName(smoothed_path)
        img_writer.SetInput(itk_img)
        img_writer.Update()
        smoothed_paths.append(smoothed_path)'''
    smoothed_path = '%s_smoothed%s' % (img_path[:-4], img_path[-4:])
    img_writer.SetFileName(smoothed_path)
    img_writer.SetInput(itk_img)
    img_writer.Update()
    return smoothed_path