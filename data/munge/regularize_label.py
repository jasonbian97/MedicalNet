import SimpleITK as sitk
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
import shutil
import numpy as np
import copy
import sklearn.model_selection

ROOT = "../cache/NiiAnnotation"
DST_PATH = "../cache/NiiAnnotationRegularized"

if os.path.exists(DST_PATH):
    shutil.rmtree(DST_PATH)
    os.makedirs(DST_PATH)
else:
    os.makedirs(DST_PATH)

import nibabel as nib
file_list = os.listdir(ROOT)

for name in file_list:
    print("process:",name)
    input_filename = os.path.join(ROOT,name)
    output_filename = os.path.join(DST_PATH,name)
    img = nib.load(input_filename)
    img_data = img.get_fdata()
    img_data = np.maximum(img_data-1,0)
    img_data = img_data.astype(np.uint8)
    print("contains labels: ", np.unique(img_data))
    new_img = nib.Nifti1Image(img_data, img.affine)
    nib.save(new_img, output_filename)