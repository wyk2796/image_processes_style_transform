# coding:utf-8
from scipy.misc import imread, imresize
import numpy as np
import tensorflow as tf


def preprocess_image(image_path, img_width=256, img_height=256, load_dims=True, resize=True, size_multiple=4):
    '''
    Preprocess the image so that it can be used by Keras.
    Args:
        image_path: path to the image
        img_width: image width after resizing. Optional: defaults to 256
        img_height: image height after resizing. Optional: defaults to 256
        load_dims: decides if original dimensions of image should be saved,
                   Optional: defaults to False
        vgg_normalize: decides if vgg normalization should be applied to image.
                       Optional: defaults to False
        resize: whether the image should be resided to new size. Optional: defaults to True
        size_multiple: Deconvolution network needs precise input size so as to
                       divide by 4 ("shallow" model) or 8 ("deep" model).
    Returns: an image of shape (3, img_width, img_height) for dim_ordering = "th",
             else an image of shape (img_width, img_height, 3) for dim ordering = "tf"
    '''
    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        global img_WIDTH, img_HEIGHT, aspect_ratio
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    if resize:
        if img_width < 0 or img_height < 0: # We have already loaded image dims
            img_width = (img_WIDTH // size_multiple) * size_multiple # Make sure width is a multiple of 4
            img_height = (img_HEIGHT // size_multiple) * size_multiple # Make sure width is a multiple of 4
        img = imresize(img, (img_width, img_height),interp='nearest')

    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

