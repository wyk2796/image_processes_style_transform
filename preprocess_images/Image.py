# coding:utf-8
from scipy.misc import imread, imresize, imsave
import numpy as np


class Image(object):

    def __init__(self, image_path):
        if image_path is not '' and image_path is not None:
            self.image = imread(image_path, mode="RGB")
            self.original_width = self.image.shape[0]
            self.original_height = self.image[1]
            self.width = self.image.shape[0]
            self.height = self.image.shape[1]
            self.aspect_ratio = self.height / self.width

    def image_resize(self, width=256, height=256):
        if width is None or width < 0:
            width = 256
        if height is None or height < 0:
            height = 256

        self.image = imresize(self.image, (width, height), interp='nearest')
        self.width = width
        self.height = height
        return self

    def extend_dim(self):
        self.image = np.expand_dims(self.image.astype(np.float32), axis=0)

    def set_image(self, image):
        self.image = image
        self.original_width = self.image.shape[0]
        self.original_height = self.image.shape[1]
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]
        self.aspect_ratio = self.height / self.width

    def save_img(self, out_path):
        img = np.clip(self.image, 0, 255).astype(np.uint8)
        if self.image.ndim == 4:
            imsave(out_path, img[0])
        else:
            imsave(out_path, img)

    def shape(self):
        return self.image.shape
