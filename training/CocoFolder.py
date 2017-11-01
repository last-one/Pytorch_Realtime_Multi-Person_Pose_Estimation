import torch
import torch.utils.data as data
import numpy as np
import shutil
import time
import random
import os
import math
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

        return: two list: key_points list and centers list
    """
    fp = open(file_dir)
    data = json.load(fp)
    kpts = []
    centers = []

    for info in data:
        kpt = []
        center = []
        lists = info['info']
        for x in lists:
           kpt.append(x['keypoints'])
           center.append(x['pos'])
        kpts.append(kpt)
        centers.append(center)
    fp.close()

    return kpts, centers

def generate_vector(vector, cnt, kpts, vec_pair, stride, theta):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i]
            b = vec_pair[1][i]
            if kpts[j][a][2] > 1 or kpts[j][b][2] > 1:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay)
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1

    return vector

class CocoFolder(data.Dataset):

    def __init__(self, file_dir, stride, transformer=None):

        self.info_list = read_data_file(file_dir[0])
        self.kpt_list, self.center_list = read_json_file(file_dir[1])
        self.stride = stride
        # self.num_points = num_points
        self.transformer = transformer
        self.vec_pair = [[2,3,5,6,8,9,11,12,0,1,1,1,1,2,5,0,0,15,14],[3,4,6,7,9,10,12,13,1,8,11,2,5,16,17,15,14,17,16]] # different from openpose
        self.theta = 1.0

    def __getitem__(self, index):

        info_path = self.info_list[index]
        info = np.load(info_path)['data']
        kpt = self.kpt_list[index]
        center = self.center_list[index]

        img = info[:,:,:3]
        mask = info[:,:,3:4]
        heatmap = info[:,:,4:]

        if self.transformer is not None:
            img, heatmap, mask, kpt, center = self.transformer(img, heatmap, mask, kpt, center)
        else:
            height, width, _ = img.shape
            img = cv2.resize(img, (368, 368))
            heatmap = cv2.resize(heatmap, (368, 368))
            mask = cv2.resize(mask, (368, 368))
            w_scale = 368.0 / width
            h_scale = 368.0 / height
            num = len(kpt)
            length = len(kpt[0])
            for i in range(num):
                for j in range(length):
                    kpt[i][j][0] *= w_scale
                    kpt[i][j][1] *= h_scale
                center[i][0] *= w_scale
                center[i][1] *= h_scale

        height, width, _ = img.shape

        mask = cv2.resize(mask, (width / self.stride, height / self.stride)).reshape((height / self.stride, width / self.stride, 1))

        heatmap = cv2.resize(heatmap, (width / self.stride, height / self.stride))
        heatmap = heatmap * mask

        vecmap = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0]) * 2), dtype=np.float32)
        cnt = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0])), dtype=np.int32)

        vecmap = generate_vector(vecmap, cnt, kpt, self.vec_pair, self.stride, self.theta)
        vecmap = vecmap * mask

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [128.0, 128.0, 128.0])
        mask = Mytransforms.to_tensor(mask)
        heatmap = Mytransforms.to_tensor(heatmap)
        vecmap = Mytransforms.to_tensor(vecmap)

        return img, heatmap, vecmap, mask

    def __len__(self):

        return len(self.info_list)
