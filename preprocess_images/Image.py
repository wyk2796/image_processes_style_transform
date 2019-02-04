# coding:utf-8
from scipy.misc import imread, imresize
import numpy as np


class Image(object):

    def __init__(self, image_path):
        if image_path is not '' and image_path is not None:
            self.image = imread(image_path, mode="RGB")
            self.original_width = self.image.shape[0]
            self.original_height = self.image[1]
            self.width = self.image.shape[0]
            self.height = self.image[1]
            self.aspect_ratio = self.height / self.width

    def image_resize(self, width=256, height=256):
        if width is None or width < 0:
            width = 256
        if height is None or height < 0:
            height = 256

        self.image = imresize(self.image, (width, height), interp='nearest')
        self.width = width
        self.height = height
        self.image = np.expand_dims(self.image.astype(np.float32), axis=0)
        return self

    def set_image(self, image):
        self.image = image
        self.original_width = self.image.shape[0]
        self.original_height = self.image.shape[1]
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]
        self.aspect_ratio = self.height / self.width
