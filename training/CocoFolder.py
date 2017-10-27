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

    fp = open(file_dir, 'r')
    lines = fp.readlines()
    fp.close()

    lists = []

    for line in lines:
        fp = open(line.strip())
        data = json.load(fp)
        kpts = []
        for key in data:
            if 'human' not in data:
                continue
            kpts.append(data[key]['keypoints'])
        lists.append(kpts)

    return lists

def pil_loader(path):

    img = Image.open(path)
    return img.convert('RGB')

def generate_heatmap(kpts, num_points, height, width, tx, ty, scale, stride):

    new_height = height / stride
    new_width = width / stride
    
    heat_map = np.zeros((num_points + 1, new_height, new_width), dtype=np.float32)
    sigma = 7.0

    for p in range(len(kpts)):
        kpt = kpts[p]
        for i in range(num_points):
            if kpt[3 * i + 2] != 1:
                continue
            x = (kpt[3 * i] - tx) * scale
            y = (kpt[3 * i + 1] - ty) * scale
            heat_cnt = np.zeros((new_height, new_width), dtype=np.int32)
            heat_re = np.zeros((new_height, new_width), dtype=np.float32)
            for j in range(height):
                for k in range(width):
                    score = (j - x) * (j  -x) + (k - y) * (k - y)
                    score = score / 2.0 / sigma / sigma
                    if score > 4.6052:
                        continue
                    xx = int(round((j + 0.5) / stride - 0.5))
                    yy = int(round((k + 0.5) / stride - 0.5))
                    heat_re[xx][yy] += math.exp(-score)
                    heat_cnt[xx][yy] += 1
            heat_re /= heat_cnt
            heat_map[i] += heat_re

    return heat_map

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def normalize(tensor, mean, std):

    if not in _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def crop(img, i, j, h, w):

    if not (_is_pil_image(img): or isinstance(img, np.ndarray))
        raise TypeError('img should be PIL Image or numpy.ndarray. Got {}'.format(type(img)))

    if _is_pil_image(img):
        return img.crop((j, i, j + w, i + h))
    else:
        return img[:,i: i + h, j: j + w]

def resize(img, size, interpolation=Image.BILINEAR):

    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    if not (_is_pil_image(img) or isinstance(img, np.ndarray)):
        raise TypeError('img should be PIL Image or numpy.ndarray. Got {}'.format(type(img)))

    if _is_pil_image(img):

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)
    else:
        if isinstance(size, int):
            h, w, _ = img.shape
            if (w <= h and w == size) or(h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * w / h)
                return cv2.resize(img, (oh, ow))
            else:
                oh = size
                ow = int(size * w / h)
                return cv2.resize(img, (size, size))
        else:
            return cv2.resize(img, (size[0], size[1]))

def resized_crop(img, i, j, h, w, size):
    """Crop the given PIL.Image or np.ndarray and resize it to desired size.

    Notably used in RandomResizedCrop.

    Args:
        img (PIL.Image or np.ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
    Returns:
        PIL.Image: Cropped image.
    """
    assert (_is_pil_image(img) or isinstance(img) == np.ndarray), 'img should be PIL Image or np.ndarray'
    img = crop(img, i, j, h, w)
    img = resize(img, size)
    return img

class RandomResizedCrop(object):
    """Crop the given PIL.Image and np.ndarray to random size and aspect ratio.

    A crop of random size of (0.5 to 1.1) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size):
        self.size = (size, size)

    @staticmethod
    def get_params(img):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.5, 1.1) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, heat, vec, weights):
        """
        Args:
            img (PIL.Image): Image to be flipped.
            heat (numpy.ndarray):
            vec (numpy.ndarray):
            weights (numpy.ndarray):

        Returns:
            PIL.Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img)

        img = resized_crop(img, i, j, h, w, self.size)
        heat = resized_crop(heat, i, j, h, w, self.size)
        vec = resized_crop(vec, i, j, h, w, self.size)
        weights = resized_crop(weights, i, j, h, w, self.size)

        return img, heat, vec, weights

class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, resize=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.resize = resize

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for center crop.

        Args:
            img (PIL.Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        w, h = img.size
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.resize is not None:
            img = resize(img, self.resize)
        i, j, h, w = self.get_params(img, self.size)
        return crop(img, i, j, h, w)

class CocoFolder(data.Dataset):

    def __init__(self, file_dir, num_points, randomcrop=True, crop_size=None, resize=None):

        if crop_size is None:
            raise TypeError('randomcrop must be given the crop_size and resize')

        if randomcrop:
            self.crop = RandomResizedCrop(crop_size)
        else:
            self.crop = CenterCrop(crop_size, resize)

        self.img_list = read_data_file(file_dir[0])
        self.heat_list = read_data_file(file_dir[1])
        self.vec_list = read_data_file(file_dir[1])
        self.weights_list = read_data_file(file_dir[1])

        self.num_points = num_points
        self.loader = pil_loader
        self.transform = transform

    def __getitem(self, index):

        img_path = self.img_list[index]
        heat_path = self.heat_list[index]
        vec_path = self.vec_list[index]
        weights_path = self.weights_list[index]

        img = self.loader(img_path)
        heat_map = np.load(heat_path)
        vec_map = np.load(vec_path)
        weights_map = np.load(weights_path)

        img, heat_map, vec_map, weights_map = self.crop(img, heat_map, vec_map, weights_map)
        heat_map = np.transpose(heat_map, (2, 0, 1))
        vec_map = np.transpose(vec_map, (2, 0, 1))
        weights_map = np.transpose(weights_map, (2, 0, 1))

        return img, heat_map, vec_map, weights

    def __len__(self):

        return len(self.img_list) 
