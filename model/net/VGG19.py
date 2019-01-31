# coding:utf-8
import tensorflow as tf


class VGG19:
    def __init__(self, input_size):
        self.input_size = input_size
        self.model = None

    def create_network(self):
        img_input = tf.keras.layers.Input(shape=self.input_size)

        # Block 1
        x = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
        x = tf.keras.layers.Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
        x = tf.keras.layers.Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
        x = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
        x = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
        x = tf.keras.layers.Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
        x = tf.keras.layers.Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        self.model = tf.keras.Model(img_input, x, name='vgg19')

    def load_existed_weight(self, weight_path):
        if weight_path is not None and self.model is not None:
            self.model.load_weights(weight_path, by_name=True)

    def save_weight(self, weight_path):
        if self.model is not None:
            self.model.self.model.save_weights(weight_path)

