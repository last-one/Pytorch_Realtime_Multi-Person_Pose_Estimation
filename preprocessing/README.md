# Usage

These scripts are used to deal with COCO dataset and generate the data for traning.

[generate_json_mask.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing/generate_json_mask.py): to generate the json file and mask.

It need four arguments: the path of COCO's annotation, the savepath for json file, mask and filelist.

Json file is used to save filename, person_center and keypoints. It's format as follow:
```
[{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [], "keypoints": []}]},
{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [], "keypoints": []}]},
...
{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [], "keypoints": []}]}]
```
The body part order of the COCO (17 given part and 1 calculated part (neck)) keypoints as follow:

```
POSE_COCO_BODY_PARTS {
	{0,  "background"},
	{1,  "nose"},
	{2,  "neck"},
	{3,  "Rshoulder"},
	{4,  "Relbow"},
	{5,  "Rwrist"},
	{6,  "Lshoulder"},
	{7,  "Lelbow"},
	{8,  "Lwrist"},
	{9,  "Rhip"},
	{10, "Rknee"},
	{11, "Rankle"},
	{12, "Lhip"},
	{13, "Lknee"},
	{14, "Lankle"},
	{15, "Reye"},
	{16, "Leye"},
	{17, "Rear"},
	{18, "Lear"},
}
```

Mask is used as weight for heatmap and vectors. It's a binary mask. When the annotation is missing at an image location, it's zero, otherwise, is one.

[generate_data.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing/generate_data.py): to generate the data for training.

I generate the heatmap offline, because when the image is resized, cropped and rotated, the heatmap could change with img. But the value of vector has different when rotated. So the vector is generated online. It's different from origin code, which generates the heatmap and vector online.

For an image, the script will generate a numpy.ndarray, img_labels. It's format as follow:

img_labels.shape: height * width * (3 + 1 + num_point + 1).

The first three channels are for image, the next channel is for mask, the next num_point channels are for keypoints' heatmap and the last channel is also for heatmap, it corresponds background.

The value of background heatmap is calculated as img_labels[h,w,i] = 1.0 = max(img[h,w,4: num_point])
