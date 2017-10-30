# Usage

These scripts are used to deal with COCO dataset and generate the data for traning.

[generate_json_mask.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing/generate_json_mask.py): to generate the json file and mask.

It need three arguments: the path of COCO's annotation, the savepath for json file and mask.

Json file is used to save filename, person_center and keypoints. It's format as follow:
```
[{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [], "keypoints": []}]},
{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [], "keypoints": []}]},
...
{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [], "keypoints": []}]}]
```

Mask is used as weight for heatmap and vectors. It's a binary mask. When the annotation is missing at an image location, it's zero, otherwise, is one.

[generate_data.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing/generate_data.py): to generate the data for training.

I generate the heatmap offline, because when the image is resized, cropped and rotated, the heatmap could change with img and the value will not change. But the vector will changed when rotated. So the vector is generated online. It's different from origin code, which generates the heatmap and vector online.

For an image, the script will generate a numpy.ndarray, img_labels. It's format as follow:

the numpy.ndarray is height * width * (3 + 1 + num_point + 1). The first three channels are for image, the next channel is for mask, the next num_point channels are for keypoints' heatmap and the last channel is also for heatmap, it corresponds background.

The value of background heatmap is calculated as img_labels[h,w,i] = 1.0 = max(img[h,w,4: num_point])
