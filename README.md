Diederik and Ignas:
Original Annotations: 
0-background
2-ascending
3-arch
4-descending
5-ascending root
6-descending root

```shell script
# DEPRECATED
In data/munge/regularize_annotation.py, we handle the label gap, the misiing "1" in this case. 
0-background
1-ascending
2-arch
3-descending
4-ascending root
5-descending root
```
We labeled the ateries as "1", so that we fill up the missing "1" in the original annotation file. By label here,
we take the compare the difference between the complete mask(including the arteries part) file and the original 
annotation file(without arteries). So the labels are:

0-background
1-arteries
2-ascending
3-arch
4-descending
5-ascending root
6-descending root


## Stats of datasets

mean size: [201.84883721 253.84883721 444.09302326]
class distribution =   [0.94458897 0.00113149 0.01155425 0.00562306 0.02999065 0.00517903
 0.00193256]
weights = 1./perc = [  1.05866153, 883.79145444,  86.54825135, 177.83917639,
        33.34373066, 193.08631133, 517.44818692]

```shell script
# run original file
python test.py --gpu_id 0 --resume_path trails/models/resnet_50_epoch_110_batch_0.pth.tar --img_list data/val.txt --input_D 56 --input_H 448 --input_W 448 --model_depth 50 --n_seg_classes 2
# test on BiMask
python test.py --gpu_id 0 --resume_path trails/models/resnet_18_epoch_20_batch_0.pth.tar 
--input_D 450 --input_H 250 --input_W 200 --model_depth 18 --n_seg_classes 6 
--dataset "BiMask" --dataset_info "config/dataset_info.json"
```
