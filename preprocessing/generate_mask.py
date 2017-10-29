import os
import cv2
import numpy as np
from pycocotools.coco import COCO

ann_path = sys.argv[1]
mask_dir = sys.argv[2]

coco = COCO(ann_path)
ids = list(coco.imgs.keys())

for i, img_id in enumerate(ids):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    img_anns = coco.loadAnns(ann_ids)

    height = coco.imgs[img_id]['height']
    width = coco.imgs[img_id]['width']
    name = coco.imgs[img_id]['file_name']

    mask_all = np.zeros((h, w), dtype=np.uint8)
    mask_miss = np.zeros((h,w), dtype=np.uint8)
    flag = 0
    for p in img_anns:
        if p['iscrowd'] == 1:
            mask_crowd = coco.annToMask(p)
            temp = np.bitwise_and(mask_all, mask_crowd)
            mask_crowd = mask_crowd - temp
            flag += 1
            continue
        else:
            mask = coco.annToMask(p)

        mask_all = np.bitwise_or(mask, mask_all)
    
        if p['num_keypoints'] <= 0:
            mask_miss = np.bitwise_or(mask, mask_miss)

    if flag < 1:
        mask_miss = np.logical_not(mask_miss)
    elif flag == 1:
        mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
        mask_all = np.bitwise_or(mask_all, mask_crowd)
    else:
        raise Exception('crowd segments > 1')

    np.save(os.path.join(mask_dir, name.split('.')[0] + '.npy'), mask_miss)
    if i % 1000 == 0:
        print "Processed {} of {}".format(i, len(ids))

print "done!"


