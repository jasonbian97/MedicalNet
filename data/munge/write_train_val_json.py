import SimpleITK as sitk
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
import shutil
import numpy as np
import copy
import sklearn.model_selection
import random
import json

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

BiMask_prefix = "data/raw/NiiBiMask"
Ann_prefix = "data/raw/NiiAnnReg"
CT_prefix = "data/raw/NiiCT"
split_rate = 0.3

BiMask_list = os.listdir("../raw/NiiBiMask")
base_name_list = [s.split("_")[0]+"_"+s.split("_")[1] for s in BiMask_list]

subjects = np.unique([s.split("_")[0] for s in base_name_list])
val_subjects = random.sample(list(subjects),k = int(len(subjects)*split_rate))
train_subjects = [s for s in subjects if s not in val_subjects]
print("training subjects = ", train_subjects)
print("val subjects = ",val_subjects)

jdict = {"train":
             {"subjects":train_subjects,
              "BiMask_list":[os.path.join(BiMask_prefix,s+"_VOI.nii.gz") for s in base_name_list if s.split("_")[0] in train_subjects],
              "Ann_list":[os.path.join(Ann_prefix,s+"_VOI_labelled.nii.gz") for s in base_name_list if s.split("_")[0] in train_subjects],
              "CT_list":[os.path.join(CT_prefix,s+".nii.gz") for s in base_name_list if s.split("_")[0] in train_subjects]
              },
         "val":
             {"subjects":val_subjects,
              "BiMask_list":[os.path.join(BiMask_prefix,s+"_VOI.nii.gz") for s in base_name_list if s.split("_")[0] in val_subjects],
              "Ann_list":[os.path.join(Ann_prefix,s+"_VOI_labelled.nii.gz") for s in base_name_list if s.split("_")[0] in val_subjects],
              "CT_list":[os.path.join(CT_prefix,s+".nii.gz") for s in base_name_list if s.split("_")[0] in val_subjects]
              }
         }

with open('../raw/test_train_split.json','w') as fp:
    fp.write( json.dumps(jdict,indent=4) )





