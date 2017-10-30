import torch
import torch.utils.data as data
import numpy as np
import shutil
import time
import random
import os
from PIL import Image
import json
import cv2
import Mytransforms

def read_data_file(file_dir):

    lists = []
    with open(file_dir, 'r') as fp:
        line = fp.readline()
        while line:
            path = line.strip()
            lists.append(path)
            line = fp.readline()

    return lists

def read_json_file(file_dir):
    """
        filename: JSON file

        return: one list: key_points list
    """
    fp = open(filename)
    data = json.load(fp)
    kpts = []

    for info in data:
        img_path.append(info['file_name'])
        kpt = []
        key_points = info['keypoints']
        for x in key_points:
           kpt.append(key_pints[x])
        kpts.append(kpt)
    fp.close()

    return kpts

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
                if ba_p >= 0 and ba_p <= norm_ba and vba_p <= theta:
                    vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                    vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                    cnt[h][w][i] += 1

class CocoFolder(data.Dataset):

    def __init__(self, file_dir, num_points, stride, transformer=None):

        self.info_list = read_data_file(file_dir[0])
        self.kpt_list = read_json_file(file_dir[1])
        self.stride = stride
        self.num_points = num_points
        self.transformer = transformer
        self.vec_pair = [[2,3,5,6,8,9,11,12,0,1,1,1,1,2,5,0,0,15,14],[3,4,6,7,9,10,12,13,1,8,11,2,5,16,17,15,14,16,17]] # different from openpose
        self.theta = 1.0

    def __getitem__(self, index):

        info_path = self.info_list[index]
        info = np.load(info_path)
        kpt = self.kpt_list[index]

        info, kpt = self.transformer(info, kpt)

        height, width, _ = info.shape
        img = info[:,:,:3]
        img = img.transpose((2, 0, 1))

        mask = info[:,:,3:4]
        mask = cv2.resize(mask, (height / self.stride, width / self.stride))
        mask = mask.transpose((2, 0, 1))

        heatmap = info[:,:,4:]
        heatmap = cv2.resize(heatmap, (height / self.stride, width / self.stride))
        heatmap = heatmap.transpose((2, 0, 1))
        heatmap = heatmap * mask

        vecmap = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair) * 2), dtype=np.float32)
        cnt = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair)), dtype=np.float32)
        vecmap = generate_vector(vector, cnt, kpt, self.vec_pair, self.theta)
        vecmap = vecmap * mask

        img = Mytransforms.to_tensor(img)
        mask = Mytransforms.to_tensor(mask)
        heatmap = Mytransforms.to_tensor(heatmap)
        vecmap = Mytransforms.to_tensor(vecmap)

        return img, heatmap, vecmap, mask

    def __len__(self):

        return len(self.info_list)
