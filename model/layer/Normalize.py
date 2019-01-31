# coding:utf-8
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda


def vgg_normalize(x):
    x = x[:, :, :, ::-1]
    return tf.add(x, -120)


def denormalize(x):
    return Lambda(tf.multiply(tf.add(x, 1), 127.5))(x)


def normalize(x):
    return Lambda(tf.divide(x, 255))(x)
