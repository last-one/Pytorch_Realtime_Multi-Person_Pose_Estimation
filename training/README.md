# Usage

These script are used to train the networks.

[Mytransforms.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training/Mytransforms.py): some transformer.

Because the heatmap, mask are generated offline, so they need be transformed with images. Besides the keypoints and center also need be transformed together.

[CocoFolder.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training/CocoFolder.py): to read data for network.

It will generate the PAFs vector when get the image.

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

[pose_estimation.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training/pose_estimation.py): the structure of networks.

The first 10 layers equals to VGG-19, so if set pretrained as True, it will be initialized by the VGG-19. And the stage is 6. The first stage has 5 layers (3 3*3conv + 2 1*1conv) and the remainder stages have 7 layers (5 3*3conv + 2 1*1conv).

TODO: the stage is adjustable.
