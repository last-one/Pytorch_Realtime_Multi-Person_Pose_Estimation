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

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()

def resize(img, mask, kpt, center, size):

    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    
    h, w, _ = img.shape
    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
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
            for j in range(length):
                kpt[i][j][0] *= w_scale
                kpt[i][j][1] *= h_scale
            center[i][0] *= w_scale
            center[i][1] *= h_scale

        return np.ascontiguousarray(cv2.resize(img, (ow, oh))), np.ascontiguousarray(cv2.resize(mask, (ow, oh))), kpt, center
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
        return np.ascontiguousarray(cv2.resize(img, (size[0], size[1]))), np.ascontiguousarray(cv2.resize(mask, (size[0], size[1]))), kpt, center

def rotate(img, mask, kpt, center, degree):

    height, width, _ = img.shape
    
    img_center = (width / 2.0 , height / 2.0)
    
    rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)
    img = cv2.warpAffine(img, rotateMat, (width, height), borderValue=(128, 128, 128, 128))
    mask = cv2.warpAffine(mask, rotateMat, (width, height), borderValue=(1, 1, 1, 1))
    mask = mask.reshape((height, width, 1))

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
                if kpt[i][j][0] < 0 or kpt[i][j][0] >= width or kpt[i][j][1] < 0 or kpt[i][j][1] >= height:
                    kpt[i][j][2] = 2
        x = center[i][0]
        y = center[i][1]
        p = np.array([x, y, 1])
        p = rotateMat.dot(p)
        center[i][0] = p[0]
        center[i][1] = p[1]

    return np.ascontiguousarray(img), np.ascontiguousarray(mask), kpt, center

def crop(img, mask, kpt, center, i, j, h, w):

    num = len(kpt)
    length = len(kpt[0])

    for x in range(num):
        for y in range(length):
            kpt[x][y][0] -= j
            kpt[x][y][1] -= i
            if kpt[x][y][0] < 0 or kpt[x][y][0] >= w or kpt[x][y][1] < 0 or kpt[x][y][1] >= h:
                kpt[x][y][2] = 2
        center[x][0] -= j
        center[x][1] -= i

    return np.ascontiguousarray(img[i: i + h, j: j + w,:]), np.ascontiguousarray(mask[i: i + h, j: j + w, :]), kpt, center

def resized_crop(img, mask, kpt, center, i, j, h, w, size):
    """Crop the given np.ndarray and resize it to desired size.

    Notably used in RandomResizedCrop.

    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
    Returns:
        PIL.Image: Cropped image.
    """
    img, mask, kpt, center = crop(img, mask, kpt, center, i, j, h, w)
    img, mask, kpt, center = resize(img, mask, kpt, center, size)
    return img, mask, kpt, center

def hflip(img, mask, kpt, center):

    height, width, _ = img.shape
    mask = mask.reshape((height, width, 1))

    img = img[:, ::-1, :]
    mask = mask[:, ::-1, :]

    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] <= 1:
                kpt[i][j][0] = width - 1 - kpt[i][j][0]
        center[i][0] = width - 1 - center[i][0]

    swap_pair = [[3, 6], [4, 7], [5, 8], [9, 12], [10, 13], [11, 14], [15, 16], [17, 18]]
    for x in swap_pair:
        for i in range(num):
            temp_point = kpt[i][x[0] - 1]
            kpt[i][x[0] - 1] = kpt[i][x[1] - 1]
            kpt[i][x[1] - 1] = temp_point

    return np.ascontiguousarray(img), np.ascontiguousarray(mask), kpt, center

class RandomResizedCrop(object):
    """Crop the given numpy.ndarray to random size and aspect ratio.

    A crop of random size of (0.5 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        center_perturb_max: the max size of perturb
    """

    def __init__(self, size, center_perturb_max=40):
        self.size = (size, size)
        self.center_perturb_max = center_perturb_max

    @staticmethod
    def get_params(img, center, center_perturb_max):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        height, width, _ = img.shape
        center_perturb_max = min(center_perturb_max, min(width, height) * 0.1)
        for attempt in range(10):
            area = width * height
            target_area = random.uniform(0.5, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= width and h <= height:
                ratio_x = random.uniform(0, 1)
                ratio_y = random.uniform(0, 1)
                x_offset = int((ratio_x - 0.5) * 2 * center_perturb_max)
                y_offset = int((ratio_y - 0.5) * 2 * center_perturb_max)
                centerx = min(max(center[0][0] + x_offset - w / 2, w / 2), width - w / 2)
                centery = min(max(center[0][1] + y_offset - h / 2, h / 2), height - h / 2)
                
                return int(round(centery - h / 2)), int(round(centerx - w / 2)), h, w

        # Fallback
        w = min(width, height)
        i = (height - w) // 2
        j = (width - w) // 2
        return i, j, w, w

    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img, center, self.center_perturb_max)
        return resized_crop(img, mask, kpt, center, i, j, h, w, self.size)

class Resize(object):
    """Resize the input np.ndarray and list to the given size.

    Args:
        size (int): Desired output size. The size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size):
        assert isinstance(size, int)
        self.size = size

    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be scaled.
            kpt (list): points to be scaled.

        Returns:
            numpy.nparray: Rescaled image.
            list: Rescaled points
        """
        return resize(img, mask, kpt, center, self.size)

class RandomRotate(object):
    """Rotate the input np.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree):
        assert isinstance(max_degree, numbers.Number)
        self.max_degree = max_degree

    @staticmethod
    def get_params(max_degree):
        """Get parameters for ``rotate`` for a random rotate.

        Returns:
            number: degree to be passed to ``rotate`` for random rotate.
        """
        degree = random.uniform(-max_degree, max_degree)

        return degree

    def __call__(self, img, mask, kpt, center):
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
        degree = self.get_params(self.max_degree)

        return rotate(img, mask, kpt, center, degree)

class RandomCrop(object):
    """Crop the given numpy.ndarray and list at a random location.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, center_perturb_max=40):
        assert isinstance(size, numbers.Number)
        self.size = (int(size), int(size)) # (w, h)
        self.center_perturb_max = center_perturb_max

    @staticmethod
    def get_params(img, center, output_size, center_perturb_max):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (numpy.ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        ratio_x = random.uniform(0, 1)
        ratio_y = random.uniform(0, 1)
        x_offset = int((ratio_x - 0.5) * 2 * center_perturb_max)
        y_offset = int((ratio_y - 0.5) * 2 * center_perturb_max)
        centerx = min(max(center[0] + x_offset - output_size[0] / 2, output_size[0] / 2), width - output_size[0] / 2)
        centery = min(max(center[1] + y_offset - output_size[1] / 2, output_size[1] / 2), height - output_size[1] / 2)

        return centery - output_size[1] / 2, centerx - output_size[0] / 2, output_size[1], output_size[0]

    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """

        i, j, h, w = self.get_params(img, center, self.size, self.center_perturb_max)

        return crop(img, mask, kpt, center, i, j, h, w)

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

    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.resize is not None:
            img, mask, kpt, center = resize(img, mask, kpt, center, self.resize)
        i, j, h, w = self.get_params(img, self.size)
        return crop(img, mask, kpt, center, i, j, h, w)

class RandomHorizontalFlip(object):
    """Random horizontal flip the image.

    Args:
        prob (number): the probability to flip.
    """
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img, mask, kpt, center):
        """
        Args:
            img (numpy.ndarray): Image to be flipped.
            kpt (list): points to be flipped.

        Returns:
            numpy.ndarray: Randomly flipped image.
            list: Randomly flipped points.
        """
        if random.random() < self.prob:
            return hflip(img, mask, kpt, center)
        return img, mask, kpt, center

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

    def __call__(self, img, mask, kpt, center):
        for t in self.transforms:
            img, mask, kpt, center = t(img, mask, kpt, center)
        return img, mask, kpt, center
