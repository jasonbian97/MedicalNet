import SimpleITK as sitk
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
import shutil
import numpy as np
import copy
import sklearn.model_selection

ANN_ROOT = "../cache/NiiAnn"
BiMask_ROOT = "../cache/NiiBiMask"
DST_PATH = "../cache/NiiAnnotationRegularized"

if os.path.exists(DST_PATH):
    shutil.rmtree(DST_PATH)
    os.makedirs(DST_PATH)
else:
    os.makedirs(DST_PATH)

import nibabel as nib
file_list = os.listdir(ANN_ROOT)
name_list = [s.split("_")[0]+"_"+s.split("_")[1] for s in file_list]

for name in name_list:
    print("process:",name)
    pmask = os.path.join(BiMask_ROOT,"{}_VOI.nii.gz".format(name))
    pann = os.path.join(ANN_ROOT,"{}_VOI_labelled.nii.gz".format(name))
    output_filename = os.path.join(DST_PATH,"{}_VOI_labelled.nii.gz".format(name))

    mask = nib.load(pmask)
    mask_data = mask.get_fdata()
    ann = nib.load(pann)
    ann_data = ann.get_fdata()

    new_ann = np.copy(ann_data)

    arteri_ind = np.where((mask_data-(ann_data>0)) == 1)
    new_ann[arteri_ind] = 1

    new_ann_img = new_ann.astype(np.uint8)
    print("contains labels: ", np.unique(new_ann))
    new_img = nib.Nifti1Image(new_ann_img, ann.affine)
    nib.save(new_img, output_filename)


