# coding:utf-8
from scipy.misc import imresize
import tensorflow as tf
import numpy as np
from skimage import color
from scipy.ndimage.filters import median_filter


def preprocess_reflect_image(image, size_multiple=4):
    org_w = image.shape[0]
    org_h = image.shape[1]

    aspect_ratio = org_h / org_w
    sw = (org_w // size_multiple) * size_multiple  # Make sure width is a multiple of 4
    sh = (org_h // size_multiple) * size_multiple  # Make sure width is a multiple of 4

    size = sw if sw > sh else sh

    pad_w = (size - sw) // 2
    pad_h = (size - sh) // 2

    kvar = tf.keras.backend.variable(value=image)

    paddings = [[pad_w, pad_w], [pad_h, pad_h], [0, 0]]
    squared_img = tf.pad(kvar, paddings, mode='REFLECT', name=None)
    img = tf.keras.backend.eval(squared_img)

    img = imresize(img, (size, size), interp='nearest')
    img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)
    return aspect_ratio, img


def original_colors(original, stylized, original_color):
    ratio = 1. - original_color

    hsv = color.rgb2hsv(original / 255)
    hsv_s = color.rgb2hsv(stylized / 255)

    hsv_s[:, :, 2] = (ratio * hsv_s[:, :, 2]) + (1 - ratio) * hsv[:, :, 2]
    img = color.hsv2rgb(hsv_s)
    return img


def blend(original, stylized, alpha):
    return alpha * original + (1 - alpha) * stylized


def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:, :, d], size=(window_size, window_size))
        ims.append(im_conv_d)
    im_conv = np.stack(ims, axis=2).astype("uint8")
    return im_conv


def crop_image(img, aspect_ratio):
    if aspect_ratio > 1:
        w = img.shape[0]
        h = img.shape[1]
        img = tf.keras.backend.eval(tf.image.crop_to_bounding_box(img, (h-w)//2, 0, w, h))
    else:
        h = img.shape[1]
        w = img.shape[0]
        img = tf.keras.backend.eval(tf.image.crop_to_bounding_box(img, 0, (w-h)//2, w, h))
    return img