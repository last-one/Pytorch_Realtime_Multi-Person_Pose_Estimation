from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import types
import collections
import warnings
import cv2

def normalize(tensor, mean, std):

    if not in _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if _is_numpy_image(pic)):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()

def resize(img, heatmpa, mask, kpt, center, size):

    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    if isinstance(img, np.ndarray)) and isinstance(heatmap, np.ndarray)) and isinstance(mask, np.ndarray)):
        raise TypeError('img, heatmap and mask should be numpy.ndarray. Got {} {} {}'.format(type(img), type(heatmap), type(mask)))

    if isinstance(size, int):
        h, w, _ = img.shape
        if (w <= h and w == size) or(h <= w and h == size):
            return img, heatmap, mask, kpt, center
        if w < h:
            ow = size
            oh = int(size * w / h)
        else:
            oh = size
            ow = int(size * w / h)

        w_scale = ow / w
        h_scale = oh / h
        num = len(kpt)
        length = len(kpt[0])
        for i in range(num):
            for j in range(length)
                kpt[i][j][0] *= w_scale
                kpt[i][j][1] *= h_scale
            center[i][0] *= w_scale
            center[i][1] *= h_scale
        return cv2.resize(img, (ow, oh)), cv2.resize(heatmap, (ow, oh)), cv2.resize(mask, (ow, oh)), kpt, center

    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        num = len(kpt)
        length = len(kpt[0])
        for i in range(num):
            for j in range(length):
                kpt[i][j][0] *= w_scale
                kpt[i][j][1] *= h_scale
            center[i][0] *= w_scale
            center[i][1] *= h_scale
        return cv2.resize(img, (size[0], size[1])), cv2.resize(heatmap, (size[0], size[1])), cv2.resize(mask, (size[0], size[1])), kpt, center

def rotate(img, heatmap, mask, kpt, center, degree):

    height, width, _ = img.shape
    
    center = (width / 2.0 , height / 2.0)
    
    rotateMat = cv2.getRotationMatrix2D(center, degree, 1.0)
    
    img = cv2.warpAffine(img, rotateMat, (width, height), borderValue=(128, 128, 128, 128))
    heatmap = cv2.warpAffine(heatmap, rotateMat, (width, height), borderValue=(0, 0, 0, 0))
    mask = cv2.warpAffine(mask, rotateMat, (width, height), borderValue=(1, 1, 1, 1))
    
    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] <= 1:
                x = kpt[i][j][0]
                y = kpt[i][j][1]
                p = np.array([x, y, 1])
                p = rotateMat.dot(p)
                kpt[i][j][0] = p[0]
                kpt[i][j][1] = p[1]
        x = center[i][0]
        y = center[i][1]
        p = np.array([x, y, 1])
        p = rotateMat.dot(p)
        center[i][0] = p[0]
        center[i][1] = p[1]

   return img, heatmap, mask, kpt, center

def crop(img, heatmap, mask, kpt, center, i, j, h, w):

    if isinstance(img, np.ndarray) and isinstance(heatmap, np.ndarray) and isinstance(mask, np.ndarray)
        raise TypeError('img should be numpy.ndarray. Got {} {} {}'.format(type(img), type(heatmap), type(mask)))

    num = len(kpt)
    length = len(kpt[0])

    for x in range(num):
        for y in range(length):
            kpt[i][j][0] -= j
            kpt[i][j][1] -= i
            if kpt[i][j][0] < 0 or kpt[i][j][0] >= w or kpt[i][j][1] < 0 or kpt[i][j][1] >= h:
                kpt[i][j][2] = 2
        center[i][0] -= j
        center[i][1] -= i

    return img[i: i + h, j: j + w,:], heatmap[i: i + h, j: j + w, :], mask[i: i + h, j: j + w, :], kpt, center

def hflip(img, heatmap, mask, kpt, center):

    img = img[:, ::-1, :]
    heatmap = heatmap[:, ::-1, :]
    mask = mask[:, ::-1, :]

    height, width, _ = img.shape

    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] <= 1:
                kpt[i][j][0] = width - 1 - kpt[i][j][0]
                kpt[i][j][1] = height - 1 - kpt[i][j][1]
        center[i][0] = width - 1 - center[i][0]
        center[i][1] = height - 1 - center[i][1]

    swap_pair = [[3, 6], [4, 7], [5, 8], [9, 12], [10, 13], [11, 14], [15, 16], [17, 18]]
    for x in swap_pair:
        temp_map = heatmap[:,:,x[0]]
        heatmap[:,:,x[0]] = heatmap[:,:,x[1]]
        heatmap[:,:,x[1]] = temp_map

        for i in range(num):
            temp_point = kpt[i][x[0] - 1]
            kpt[i][x[0] - 1] = kpt[i][x[1] - 1]
            kpt[i][x[1] - 1] = temp_point

    return img, heatmap, mask, kpt, center

class Resize(object):
    """Resize the input np.ndarray and list to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, img, heatmap, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be scaled.
            kpt (list): points to be scaled.

        Returns:
            numpy.nparray: Rescaled image.
            list: Rescaled points
        """
        return resize(img, heatmap, mask, kpt, center, self.size)

class RandomRotate(object):
    """Rotate the input np.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree):
        assert isinstance(degree, numbers.Number)
        self.max_degree = max_degree

    @staticmethod
    def get_params():
        """Get parameters for ``rotate`` for a random rotate.

        Returns:
            number: degree to be passed to ``rotate`` for random rotate.
        """
        degree = random.uniform(-max_degree, max_degree)

        return degree

    def __call__(self, img, heatmap, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be rotated.
            heatmap (numpy.ndarray): heatmap to be rotated.
            mask (numpy.ndarray): mask to be rotated.
            kpt (list): the key point to be rotated.

        Returns:
            numpy.ndarray: Rotated image.
            list: Rotated key points.
        """
        
        degree = self.get_params()

        return rotate(img, heatmap, mask, kpt, center, degree)

class RandomCrop(object):
    """Crop the given numpy.ndarray and list at a random location.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, center_perterb_max=40):
        assert isinstance(size, numbers.Number)
        self.size = (int(size), int(size)) # (w, h)
        self.center_perterb_max = center_perterb_max

    @staticmethod
    def get_params(img, center, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (numpy.ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        ratio_x = random.uniform(0, 1)
        ratio_y = random.uniform(0, 1)
        x_offset = int((ratio_x - 0.5) * 2 * self.center_perterb_max)
        y_offset = int((ratio_y - 0.5) * 2 * self.center_perterb_max)
        centerx = min(max(center[0] + x_offset - output_size[0] / 2, output_size[0] / 2), width - output_size[0] / 2)
        centery = min(max(center[1] + y_offset - output_size[1] / 2, output_size[1] / 2), height - output_size[1] / 2)

        return centery - output_size[1] / 2, centerx - output_size[0] / 2, output_size[1], output_size[0]

    def __call__(self, img, heatmap, mask, kpt, center):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """

        i, j, h, w = self.get_params(img, center, self.size)

        return crop(img, heatmap, mask, kpt, center, i, j, h, w)

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
        h, w, _ = img.shape
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, img, heatmap, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.resize is not None:
            img, heatmap, mask, kpt, center = resize(img, heatmap, mask, kpt, center, self.resize)
        i, j, h, w = self.get_params(img, self.size)
        return crop(img, heatmap, mask, kpt, center, i, j, h, w)

class RandomHorizontalFlip(object):
    """Rotate the input np.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """
    def __call__(self, img, heatmap, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
            kpt (list): points to be flipped.

        Returns:
            numpy.ndarray: Randomly flipped image.
            list: Randomly flipped points.
        """
        if random.random() < 0.5:
            return hflip(img, heatmap, mask, kpt, center)
        return img, heatmap, mask, kpt, center

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, heatmap, mask, kpt, center):
        for t in self.transforms:
            img, heatmap, mask, kpt, center = t(img, heatmap, mask, kpt, center)
        return img, heatmap, mask, kpt, center
