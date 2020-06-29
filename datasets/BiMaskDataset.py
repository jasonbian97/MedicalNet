import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import json
from scipy.ndimage import rotate

class TriPairDataset(Dataset):

    def __init__(self, dataset_info, sets, mode = "train", option = "mask-only"):

        with open(dataset_info["train_val_split"], 'r') as fp:
            dataset_dict = json.load(fp)

        # if mode == "train":
        #     with open(dataset_info["fp_train"], 'r') as f:
        #         self.img_list = [line.strip() for line in f]
        #     print("training {} datas".format(len(self.img_list)))
        # else:
        #     with open(dataset_info["fp_val"], 'r') as f:
        #         self.img_list = [line.strip() for line in f]
        #     print("validating {} datas".format(len(self.img_list)))
        if mode == "train":
            self.dataset_dict = dataset_dict["train"]
            print("number of training images = ", len(self.dataset_dict["Ann_list"]))
            print("training subjects: ",self.dataset_dict["subjects"])
        elif mode == "val":
            self.dataset_dict = dataset_dict["val"]
            print("number of val images = ", len(self.dataset_dict["Ann_list"]))
            print("val subjects: ", self.dataset_dict["subjects"])
        else:
            print("wrong mode! from TriPairDataset dataset ")
            exit()

        self.option = option
        # self.root_dir = root_dir
        self.input_D = sets.input_W # intentionally mismatch
        self.input_H = sets.input_H
        self.input_W = sets.input_D
        self.phase = mode

    def __len__(self):
        return len(self.dataset_dict["Ann_list"])

    def __getitem__(self, idx):
        # load annotation image in all conditions
        ann_name = self.dataset_dict["Ann_list"][idx]
        assert os.path.isfile(ann_name)
        ann_img = nibabel.load(ann_name)

        if self.phase == "train":

            if self.option == "CT-only":
                img_name = self.dataset_dict["CT_list"][idx]
            elif self.option == "mask-only":
                img_name = self.dataset_dict["BiMask_list"][idx]
            else:
                print("wrong dataset option!")

            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None

            # data processing
            img_array, ann_array = self.__training_data_process__(img, ann_img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            ann_array = self.__nii2tensorarray__(ann_array)

            assert img_array.shape == ann_array.shape, "img shape:{} is not equal to ann shape:{}".format(
                img_array.shape, ann_array.shape)
            return img_array, ann_array

        else: # "val" or "test"
            # read image
            if self.option == "CT-only":
                img_name = self.dataset_dict["CT_list"][idx]
            elif self.option == "mask-only":
                img_name = self.dataset_dict["BiMask_list"][idx]
            else:
                print("wrong dataset option!")

            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None
            # data processing
            if self.option == "CT-only":
                img_array = self.__testing_data_process__(img,normalization = True)
            else:
                img_array = self.__testing_data_process__(img, normalization=False)

            ann_array = self.__testing_data_process__(ann_img, normalization = False)
            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            ann_array = self.__nii2tensorarray__(ann_array)

            return img_array, ann_array, ann_img.get_data(), ann_img.affine

    def __training_data_process__(self, data, label):
        # crop data according net input size
        data = data.get_data()
        label = label.get_data()

        # drop out the invalid range
        # data, label = self.__drop_invalid_range__(data, label)

        # crop data
        data, label = self.__crop_data__(data, label)
        # rotate
        data, label = self.__rotate_aug__(data, label)
        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)

        # normalization datas
        if self.option == "CT-only":
            data = self.__itensity_normalize_one_volume__(data)

        return data, label

    def __testing_data_process__(self, data, normalization = False):
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        if normalization:
            data = self.__itensity_normalize_one_volume__(data)

        return data

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
        return new_data

    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = self.__random_center_crop__(data, label)

        return data, label

    def __random_center_crop__(self, data, label):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label > 0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __rotate_aug__(self, image, segmentation):
        rz_aroundz = np.random.uniform(low=-30.0, high=30.0)
        rz_aroundxy = np.random.uniform(low=-15.0, high=15.0)
        rd = random.randint(0, 2)

        if rd == 0:
            image = rotate(image, rz_aroundz, axes=(0, 1), reshape=False, order=1, mode='nearest', cval=0.0, prefilter=True)
            segmentation = rotate(segmentation, rz_aroundz, axes=(0, 1), reshape=False, order=0, mode='nearest', cval=0.0,
                                  prefilter=True)
        elif rd == 1:
            image = rotate(image, rz_aroundxy, axes=(1, 2), reshape=False, order=1, mode='nearest', cval=0.0, prefilter=True)
            segmentation = rotate(segmentation, rz_aroundxy, axes=(1, 2), reshape=False, order=0, mode='nearest', cval=0.0,
                                  prefilter=True)
        elif rd == 2:
            image = rotate(image, rz_aroundxy, axes=(0, 2), reshape=False, order=1, mode='nearest', cval=0.0, prefilter=True)
            segmentation = rotate(segmentation, rz_aroundxy, axes=(0, 2), reshape=False, order=0, mode='nearest', cval=0.0,
                                  prefilter=True)

        return image, segmentation

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    # def __itensity_normalize_one_volume__(self, volume):
    #     """
    #     normalize the itensity of an nd volume based on the mean and std
    #     inputs:
    #         volume: the input nd volume
    #     outputs:
    #         out: the normalized nd volume
    #     """
    #
    #     pixels = volume
    #     mean = pixels.mean()
    #     std = pixels.std()
    #     out = (volume - mean) / std
    #     return out

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out
