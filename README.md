# pytorch Realtime Multi-Person Pose Estimation

This is a pytroch version of Realtime Multi-Person Pose Estimation, origin code is here https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

## Introduction

Code for reproducing CVPR 2017 Oral paper using pytorch

## Contents
1.[preprocessing](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing)

2.[training](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training)

## Require
[Pytorch](http://pytorch.org/)

## Training steps
- Download the data set, annotations and [COCO official toolbox](https://github.com/cocodataset/cocoapi)
- Go to the "preprocessing" folder `cd preprocessing`.
- Generate json file and masks `python generate_json_mask,py`.
- Generate training data `python generate_data.py`.
- Go to the "training" folder `cd ../training`.
- Set the train parameters in "config.yml".
- Set the train data dir and val data dir. 
- Train the model `sh train.sh`.
## Citation
Please cite the paper in your publocations if it helps your research:
	
	@inProceedings{cao2017realtime,
		title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields}},
		author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		year = {2017}
		}
