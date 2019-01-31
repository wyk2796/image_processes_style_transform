# coding:utf-8
import tensorflow as tf
from model.layer.ImageCov import *
from model.layer.Normalize import *
from model.layer.Loss import TVRegularizer
from tensorflow.python.keras.layers import Lambda


class ImageTransform(object):

    def __init__(self, image_width, image_height, tv_weight=1):
        self.width = image_width
        self.height = image_height
        self.tv_weight = tv_weight
        self.model = None

    def create_network(self):
        x = tf.keras.layers.Input(shape=[self.width, self.height, 3])
        # norm = normalize(x)
        a = Lambda(reflect_padding(x, (40, 40)))(x)
        conv1 = self.conv_bn_relu(32, 9, 9, stride=(1, 1))(a)
        conv2 = self.conv_bn_relu(64, 9, 9, stride=(2, 2))(conv1)
        c = self.conv_bn_relu(128, 3, 3, stride=(2, 2))(conv2)
        for i in range(5):
            c = self.res_conv(128, 3, 3)(c)
        dconv1 = self.dconv_bn_nolinear(64, 3, 3)(c)
        dconv2 = self.dconv_bn_nolinear(32, 3, 3)(dconv1)
        dconv3 = self.dconv_bn_nolinear(3, 9, 9, stride=(1, 1), activation="tanh")(dconv2)
        # y = denormalize(dconv3)

        self.model = tf.keras.Model(inputs=x, outputs=dconv3)

        if self.tv_weight > 0:
            self.add_total_variation_loss(self.model.layers[-1])

    def get_model(self):
        return self.model

    def conv_bn_relu(self, nb_filter, nb_row, nb_col, stride):
        def conv_func(x):
            x = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            return x
        return Lambda(conv_func)

    def res_conv(self, nb_filter, nb_row, nb_col, stride=(1, 1)):
        def _res_func(x):
            identity = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)
            a = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(x)
            a = tf.keras.layers.BatchNormalization()(a)
            a = tf.keras.layers.Activation("relu")(a)
            a = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='valid')(a)
            y = tf.keras.layers.BatchNormalization()(a)
            return tf.keras.layers.add([identity, y])
        return Lambda(_res_func)

    def dconv_bn_nolinear(self, nb_filter, nb_row, nb_col, stride=(2, 2), activation="relu"):
        def _dconv_bn(x):
            #TODO: Deconvolution2D
            #x = Deconvolution2D(nb_filter,nb_row, nb_col, output_shape=output_shape, subsample=stride, border_mode='same')(x)
            #x = UpSampling2D(size=stride)(x)
            x = Lambda(un_pooling2D(x, size=stride))(x)
            x = Lambda(reflect_padding)(x, stride)
            x = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col), padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation)(x)
            return x
        return Lambda(_dconv_bn)

    def add_total_variation_loss(self, transform_output_layer):
        # Total Variation Regularization
        layer = transform_output_layer  # Output layer
        tv_regularizer = TVRegularizer(self.tv_weight)(layer)
        layer.add_loss(tv_regularizer)
