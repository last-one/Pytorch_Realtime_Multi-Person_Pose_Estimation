import os
import cv2
import json
import numpy as np

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

def generate_vector():

    pass

def generate_heatmap():

    pass


class GenerateData(object):

    def __init__(self, img_file, mask_path, num_point, save_path):

        self.img_list, self.kpt_list  = read_file(img_file)
        self.mask_path = mask_path
        self.num_point = num_point
        self.vectors = [[2,3,5,6,8,9,11,12,0,1,1,1,1,2,5,0,0,15,14],[3,4,6,7,9,10,12,13,1,8,11,2,5,16,17,15,14,16,17]] # different from openpose
        self.sigma = 7.0

    def _generate(self):

        for info in zip(self.img_list, self.mask_path, self.kpt_list):
            name = info[0].split('/')[-1].split('.')[0]
            img = cv2.imread(info[0])
            height, width, _ = img.shape
            mask = read_mask(info[1])
            kpts = info[2]
            heat = np.zeros((height, width, self.num_points), dtype=np.float32)
            vector = np.zeros((height, width, len(self.vectors[0])), dtype=np.float32)
            for kpt in kpts:
                generate_heatmap(heat, kpt, self.sigma)
                generate_vector(vector, kpt, self.vectors, self.sigma)
            img_labels = np.concatenate((img, mask, heat, vector), axis=2)
            np.save(os.path.joint(save_path, name), img_labels)
