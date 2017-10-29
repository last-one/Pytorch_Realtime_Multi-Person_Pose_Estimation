import os
import cv2
import json
import numpy as np
import math

"""num_point: the number of key_points
   
   A image is organised as a np.ndarray: [h, w, (3 + 1 + num_vector * 2 + (num_point + 1) )], 
   
   3 is the channel of image, 1 is the channel of mask_miss,
   
   num_vector * 2 is twice the number of vectors,
   
   (num_point + 1) is the number of key_ponts and background.

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
        img_path.append(info['file_name'])
        kpt = []
        key_points = info['keypoints']
        for x in key_points:
           kpt.append(key_pints[x])
        kpts.append(kpt)
    fp.close()

    return img_path, kpts

def read_mask(mask_path):

    return np.load(mask_path)

def generate_vector(vector, cnt, kpts, vec_pair, theta):

    height, width, channel = cnt.shape

    for h in range(height):
        for w in range(width):
            for i in range(channel):
                a = vec_pair[0][i]
                b = vec_pair[1][i]
                if kpts[3 * a + 2] <= 1 and kpts[3 * b + 2] <= 1:
                    continue
                ax = kpts[3 * a]
                ay = kpts[3 * a + 1]
                bx = kpts[3 * b]
                by = kpts[3 * b + 1]
                
                bax = bx - ax
                bay = by - ay
                norm_ba = math.sqrt(1.0 * bax * bax + bay * bay)
                bax /= norm_ba
                bay /= norm_ba

                px = w - ax
                py = h - ay

                ba_p = bax * px + bay * py
                vba_p = abs(bay * px - bax * py)
                if ba_p <= norm_ba and vba_p <= theta:
                    vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                    vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                    cnt[h][w][i] += 1

def generate_heatmap(heat, kpts, num_point, sigma):

    height, width, _ = heat.shape

    for h in range(height):
        for w in range(width):
            for i in range(num_point):
                if kpts[3 * i + 2] <= 1:
                    continue
                x = kpts[3 * i]
                y = kpts[3 * i + 1]
                dis = ((x - w) * (x - w) + (h - y) * (h - y)) / 2.0 / sigma / sigma
                if dis > 4.6052: # ln(100)
                    continue
                heat[h][w][i] += math.exp(-dis)
                # comment openpose
                # if heat[h][w][i] > 1:
                    # heat[h][w][i] = 1

class GenerateData(object):

    def __init__(self, img_file, mask_path, num_point, save_path):

        self.img_list, self.kpt_list  = read_file(img_file)
        self.mask_path = mask_path
        self.num_point = num_point
        self.vec_pair = [[2,3,5,6,8,9,11,12,0,1,1,1,1,2,5,0,0,15,14],[3,4,6,7,9,10,12,13,1,8,11,2,5,16,17,15,14,16,17]] # different from openpose
        self.sigma = 7.0
        self.theta = 1.0

    def _generate(self):

        for info in zip(self.img_list, self.mask_path, self.kpt_list):
            name = info[0].split('/')[-1].split('.')[0]
            img = cv2.imread(info[0])
            img.dtype=np.float32
            height, width, _ = img.shape
            mask = read_mask(info[1]).reshape((height, width, 1))
            mask.dtype=np.float32
            kpts = info[2]
            heat = np.zeros((height, width, self.num_point), dtype=np.float32)
            vector = np.zeros((height, width, len(self.vectors[0]) * 2), dtype=np.float32)
            cnt = np.zeros((height, width, len(self.vectors[0])), dtype=np.int32)
            for kpt in kpts:
                generate_heatmap(heat, kpt, self.num_point, self.sigma)
                generate_vector(vector, cnt, kpt, self.vec_pair, self.theta)
            img_labels = np.concatenate((img, mask, heat, vector), axis=2)
            np.save(os.path.joint(save_path, name), img_labels)
