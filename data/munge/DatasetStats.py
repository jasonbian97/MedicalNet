import SimpleITK as sitk
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
import shutil
import numpy as np
import copy
import sklearn.model_selection
import nibabel as nib
from matplotlib import pyplot as plt

img_list = os.listdir("../cache/NiiBiMask")
sizeMat = np.zeros((len(img_list),3))
for i,name in enumerate(img_list):
    print("process:",name)
    input_filename = os.path.join("../cache/NiiBiMask",name)
    img = nib.load(input_filename)
    sizeMat[i,:] = img.shape

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.hist(sizeMat[:,0])
ax1.set_title("Width")
ax2.hist(sizeMat[:,1])
ax2.set_title("Height")
ax3.hist(sizeMat[:,2])
ax3.set_title("Depth")
plt.savefig("../../results/images/dataset_stats_size.png",dpi=300)

plt.show()
print("mean size:",sizeMat.mean(axis=0))

