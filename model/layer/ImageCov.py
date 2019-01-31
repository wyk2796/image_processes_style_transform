# coding:utf-8
import tensorflow as tf


def un_pooling2D(x, size=(2, 2)):
    shapes = x.get_shape().as_list()
    w = size[0] * shapes[1]
    h = size[1] * shapes[2]
    return tf.image.resize_nearest_neighbor(x, (w, h))


def reflect_padding(x, pad):
    top_pad = pad[0]
    bottom_pad = pad[0]
    left_pad = pad[1]
    right_pad = pad[1]
    return tf.pad(x, [[0, 0], [left_pad, right_pad], [top_pad, bottom_pad], [0, 0]], mode='REFLECT', name=None)



