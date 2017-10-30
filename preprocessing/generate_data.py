import os
import cv2
import json
import numpy as np
import math

"""num_point: the number of key_points
   
   A image is organised as a np.ndarray: [h, w, (3 + 1 + (num_point + 1) )], 
   
   3 is the channel of image, 1 is the channel of mask_miss,
   
   (num_point + 1) is the number of key_ponts and background.

   the result includes img, mask and heatmap. Because the rotation will change the part affinity field vector, the vector has to be generated online.

"""

def read_file(filename):
    """
        filename: JSON file

        return: two list: img_path list and corresponding key_points list
    """
    fp = open(filename)
    data = json.load(fp)
    img_path = []
    kpts = []

    for info in data:
        img_path.append(info['filename'])
        kpt = []
        lists = info['info']
        for x in lists:
            kpt.append(x['keypoints'])
        kpts.append(kpt)
    fp.close()

    return img_path, kpts

def read_mask(mask_path):

    return np.load(mask_path)

def generate_heatmap(heat, kpts, num_point, sigma):

    height, width, _ = heat.shape

    for h in range(height):
        for w in range(width):
            for i in range(1, num_point):
                if kpts[3 * i + 2] <= 1:
                    continue
                x = kpts[3 * i]
                y = kpts[3 * i + 1]
                dis = ((x - w) * (x - w) + (h - y) * (h - y)) / 2.0 / sigma / sigma
                if dis > 4.6052: # ln(100)
                    continue
                heat[h][w][i] += math.exp(-dis)
                if heat[h][w][i] > 1:
                    heat[h][w][i] = 1

class GenerateData(object):

    def __init__(self, json_file, img_path, mask_path, num_point, save_path):

        self.img_list, self.kpt_list  = read_file(json_file)
        self.img_path = img_path
        self.mask_path = mask_path
        self.num_point = num_point
        self.save_path = save_path
        self.sigma = 7.0

    def _generate(self):

        cnt = 0
        length = len(self.img_list)
        assert len(self.img_list) == len(self.kpt_list)

        for info in zip(self.img_list, self.kpt_list):
            name = info[0].split('/')[-1].split('.')[0]

            img = np.array(cv2.imread(os.path.join(self.img_path, info[0])))
            img = img[:,:,::-1]
            img.dtype=np.float32

            height, width, _ = img.shape
            mask = read_mask(os.path.join(self.mask_path, name + '.npy')).reshape((height, width, 1))
            mask.dtype=np.float32

            kpts = info[1]
            heat = np.zeros((height, width, self.num_point), dtype=np.float32)
            for kpt in kpts:
                generate_heatmap(heat, kpt, self.num_point, self.sigma)

            heat[:,:,0] = 1.0 - np.max(heat[:,:,1:self.num_point], axis=2)

            img_labels = np.concatenate((img, mask, heat), axis=2)
            np.save(os.path.joint(self.save_path, name), img_labels)
            cnt += 1
            if cnt % 1000 == 0:
                print "Processed {} of {}".format(cnt, length)
            if cnt == 20:
                break

if __name__ == '__main__':

    gen = GenerateData('/home/hypan/data/coco/json/train2014.json', '/home/hypan/data/coco/train2014', '/home/hypan/data/coco/mask/train2014', 19, '/home/hypan/data/coco/train_data/train2014')
    gen._generate()
