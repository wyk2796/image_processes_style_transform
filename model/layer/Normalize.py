# coding:utf-8
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda


def vgg_normalize(x):
    Lambda(lambda t: t[:, :, :, ::-1])(x)
    return Lambda(lambda t: t - 120)(x)



def denormalize(x):
    return Lambda(tf.multiply(tf.add(x, 1), 127.5))(x)


def normalize(x):
    return Lambda(tf.divide(x, 255))(x)
