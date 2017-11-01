# pytorch Realtime Multi-Person Pose Estimation

This is a pytroch version of Realtime Multi-Person Pose Estimation, origin code is here https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

## Introduction

Code for reproducing CVPR 2017 Oral paper using pytorch

## Results

TODO

## Contents
1.[preprocessing](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing)

2.[training](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training)

## Require
[Pytorch](http://pytorch.org/)

## Instructions
[Mytransforms.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/Mytransforms.py): some transformer.

transformer the image, mask, keypoints and center points, together.

[CocoFolder.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/CocoFolder.py): to read data for network.

It will generate the PAFs vector and heatmap when get the image.

The PAFs vector's format as follow:

```
POSE_COCO_PAIRS = {
	{3,  4},
	{4,  5},
	{6,  7},
	{7,  8},
	{9,  10},
	{10, 11},
	{12, 13},
	{13, 14},
	{1,  2},
	{2,  9},
	{2,  12},
	{2,  3},
	{2,  6},
	{3,  17},
	{6,  18},
	{1,  16},
	{1,  15},
	{16, 17},
	{15, 18},
}
```
Where each index is the key value corresponding to each part in [POSE_COCO_BODY_PARTS](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing/README.md)

[BasicTool.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/BasicTool.py): some common functions, such as adjust learning rate, read configuration and etc.

[visualize_input.ipynb](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/visualize_input.ipynb): the script to vierfy the validaity of preprocessing and generating heatmap and vectors. It shows some examples.

## Training steps
- Download the data set, annotations and [COCO official toolbox](https://github.com/cocodataset/cocoapi)
- Go to the "preprocessing" folder `cd preprocessing`.
- Generate json file and masks `python generate_json_mask,py`.
- Go to the "training" folder `cd ../training`.
- Set the train parameters in "config.yml".
- Set the train data dir , train mask dir, train json filepath and val data dir, val mask dir, val json filepath. 
- Train the model `sh train.sh`.
## Citation
Please cite the paper in your publocations if it helps your research:
	
	@inProceedings{cao2017realtime,
		title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields}},
		author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		year = {2017}
		}
