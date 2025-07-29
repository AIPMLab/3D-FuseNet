import numpy as np
import SimpleITK as sitk
import torch
from radiomics import firstorder, glcm, gldm, glrlm, glszm, ngtdm, shape


def extract_features(input_image, label_image):
    # input_image: flair_transv, t1ce_transv, t1_transv, t2_transv
    # label_image: Tumor Core, Whole Tumor, Enhance Tumor
    label_image = torch.where(
        torch.sigmoid(label_image) > 0.5, torch.tensor(1.), torch.tensor(0.))
    getter = [["whole_tumor", input_image[0], label_image[1]],
              ["tumor_core", input_image[1], label_image[0]],
              ["enhance_tumor", input_image[1], label_image[2]]]

    all_features = dict()
    for region_name, image, label in getter:
        lbl_img = sitk.GetImageFromArray(label)
        acq_img = sitk.GetImageFromArray(image)
        # Get First order features
        firstorderfeatures = firstorder.RadiomicsFirstOrder(acq_img, lbl_img)
        firstorderfeatures.enableAllFeatures(
        )  # On the feature class level, all features are disabled by default
        firstorderfeatures.execute()
        for (key, val) in firstorderfeatures.featureValues.items():
            all_features[region_name + '_' + key] = val

        # Get Shape features
        shapefeatures = shape.RadiomicsShape(acq_img, lbl_img)
        shapefeatures.enableAllFeatures()
        shapefeatures.execute()
        for (key, val) in shapefeatures.featureValues.items():
            all_features[region_name + '_' + key] = val

        # Get Gray Level Co-occurrence Matrix (GLCM) Features
        glcmfeatures = glcm.RadiomicsGLCM(acq_img, lbl_img)
        glcmfeatures.enableAllFeatures()
        glcmfeatures.execute()
        for (key, val) in glcmfeatures.featureValues.items():
            all_features[region_name + '_' + key] = val

        # Get Gray Level Size Zone Matrix (GLSZM) Features
        glszmfeatures = glszm.RadiomicsGLSZM(acq_img, lbl_img)
        glszmfeatures.enableAllFeatures()
        glszmfeatures.execute()
        for (key, val) in glszmfeatures.featureValues.items():
            all_features[region_name + '_' + key] = val

        # Get Gray Level Run Length Matrix (GLRLM) Features
        glrlmfeatures = glrlm.RadiomicsGLRLM(acq_img, lbl_img)
        glrlmfeatures.enableAllFeatures()
        glrlmfeatures.execute()
        for (key, val) in glrlmfeatures.featureValues.items():
            all_features[region_name + '_' + key] = val

        # Get Neighbouring Gray Tone Difference Matrix (NGTDM) Features
        ngtdmfeatures = ngtdm.RadiomicsNGTDM(acq_img, lbl_img)
        ngtdmfeatures.enableAllFeatures()
        ngtdmfeatures.execute()
        for (key, val) in ngtdmfeatures.featureValues.items():
            all_features[region_name + '_' + key] = val

        # Get Gray Level Dependence Matrix (GLDM) Features
        gldmfeatures = gldm.RadiomicsGLDM(
            acq_img,
            lbl_img,
        )
        gldmfeatures.enableAllFeatures()
        gldmfeatures.execute()
        for (key, val) in gldmfeatures.featureValues.items():
            all_features[region_name + '_' + key] = val

    return all_features


if __name__ == "__main__":
    model = torch.load(r"D:\vit_reg\combination_exp_8\model\best_epoch.pth").cpu()
    image = torch.Tensor(
        torch.load(
            r"D:\vit_reg\SegFormer3D\data\brats2021_seg\BraTS2020_Training_Data\BraTS20_Training_001\BraTS20_Training_001_modalities.pt"
        )[np.newaxis, ...])
    segment, _ = model(image, torch.randn((1, 4)), torch.rand((1, 16)))
    features = extract_features(image[0], segment[0])
    print(list(features))
    print(len(features))
    for a, b in features.items():
        print(b)
