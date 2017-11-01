# Usage

These script are used to train the networks.

[pose_estimation.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training/pose_estimation.py): the structure of networks.

The first 10 layers equals to VGG-19, so if set pretrained as True, it will be initialized by the VGG-19. And the stage is 6. The first stage has 5 layers (3 3x3conv + 2 1x1conv) and the remainder stages have 7 layers (5 3x3conv + 2 1x1conv).

TODO: the stage is adjustable.

[config.yml](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training/config.py): some training parameters.

[train_pose.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/training/train.py): the script for training.
