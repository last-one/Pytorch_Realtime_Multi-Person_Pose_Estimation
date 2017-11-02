# Usage

These scripts are used to deal with COCO dataset and generate the data for traning.

[generate_json_mask.py](https://github.com/last-one/pytorch_realtime_multi-person_pose_estimation/blob/master/preprocessing/generate_json_mask.py): to generate the json file and mask.

It need four arguments: the path of COCO's annotation, the savepath for json file, mask and filelist.

Json file is used to save filename, person_center and keypoints. It's format as follow:
```
[{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [x, y], "keypoints": [x1, y1, v1, x2, y2, v2, ..., x18, y18, v18]}]},
{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [x, y], "keypoints": [x1, y1, v1, x2, y2, v2, ..., x18, y18, v18]}]},
...
{"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [x, y], "keypoints": [x1, y1, v1, x2, y2, v2, ..., x18, y18, v18]}]}]
```
where, v = 0 means labelled but unvisuable, v = 1 means labelled and visuable, v = 2 means missed.

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
