import os
import sys
from pycocotools.coco import COCO

ann_path = sys.argv[1]
json_dir = sys.argv[2]
mask_dir = sys.argv[3]

coco = COCO(ann_path)
ids = list(coco.imgs.keys())
lists = []

fp = open(sys.argv[4], 'w')

for i, img_id in enumerate(ids):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    img_anns = coco.loadAnns(ann_ids)

    numPeople = len(img_anns)
    name = coco.imgs[img_id]['file_name']
    height = coco.imgs[img_id]['height']
    width = coco.imgs[img_id]['width']

    persons = []
    person_centers = []

    for p in range(numPeople):

        if img_anns[p]['num_keypoints'] < 5 or img_anns[p]['area'] < 32 * 32:
            continue
        kpt = img_anns[p]['keypoints']
        dic = dict()

        person_center = [img_anns[p]['bbox'][0] + img_anns[p]['bbox'][2] / 2.0, img_anns[p]['bbox'][1] + img_anns[p]['bbox'][3] / 2.0]

        # skip this person if the distance to exiting person is too small
        flag = 0
        for pc in person_centers:
            dis = math.sqrt((person_center[0] - pc[0]) * (person_center[0] - pc[0]) + (person_center[1] - pc[1]) * (person_center[1] - pc[1]))
            if dis < pc[2] * 0.3:
                flag = 1;
                break
        if flag == 1:
            continue
        dic['objpos'] = person_center
        dic['bbox'] = img_anns[p]['bbox']
        dic['segment_area'] = img_anns[p]['area']
        dic['num_keypoints'] = img_anns[p]['num_keypoints']
        dic['keypoints'] = np.zeros((17, 3))
        for part in range(17):
            dic['keypoints'][part][0] = kpt[part * 3]
            dic['keypoints'][part][1] = kpt[part * 3 + 1]
            if kpt[part * 3 + 2] == 2:
                dic['keypoints'][part][2] = 1
            elif kpt[part * 3 + 2] == 1:
                dic['keypoints'][part][2] = 0
            else:
                dic['keypoints'][part][2] = 2

        persons.append(dic)
        person_centers.append(np.append(person_center, max(img_anns[p]['bbox'][2], img_anns[p]['bbox'][3])))

    if len(person) > 0:
        fp.write(name + '\n')
        info = dict()
        info['filename'] = name
        info['info'] = []
        cnt = 1
        for k in person:
            dic = dict()
            dic['pos'] = person[k]['objpos']
            dic['keypoints'] = person[k]['keypoints']
            info['info'].append(dic)
        lists.append(info)
        
        mask_all = np.zeros((height, width), dtype=np.uint8)
        mask_miss = np.zeros((height, width), dtype=np.uint8)
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

fp.close()
print 'write json file'

fp = open(json_dir, 'w')
fp.write(json.dumps(lists))
fp.close()

print 'done!'
