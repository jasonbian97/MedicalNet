import SimpleITK as sitk
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
import shutil

def cvtmhd2nii(inputImageFileName, outputImageFileName):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(inputImageFileName)
    image = reader.Execute()

    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)

ROOT = "../raw/DiederikMHD/CT"
DST_PATH = "../cache/NiiCT"

if os.path.exists(DST_PATH):
    shutil.rmtree(DST_PATH)
    os.makedirs(DST_PATH)
else:
    os.makedirs(DST_PATH)

for root, dirs, files in os.walk(ROOT, topdown=False):
   for name in files:
      if len(name.split("."))==2 and name.split(".")[1] == "mhd":
        inputImageFileName = os.path.join(root, name)
        print(inputImageFileName)
        outputImageFileName = os.path.join(DST_PATH,name.split(".")[0]+".nii.gz")
        try:
            cvtmhd2nii(inputImageFileName,outputImageFileName)
        except:
            print("wrong: ",inputImageFileName)



