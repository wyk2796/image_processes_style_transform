# coding:utf-8
from model.net.VGG16 import VGG16
from model.layer.Normalize import *
from model.layer.Loss import *
from model.net.image_transform_net import ImageTransform
from util import *


class StyleTransform(object):

    def __init__(self, session, learning_rate, content_weight, style_weight):
        self.session = session
        tf.keras.backend.set_session(session)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.style_layers_name = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        self.content_layer = 'block3_conv3'
        self.transform_model = None
        self.vgg_model = None
        self.vgg_output_and_layer = dict()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def create_model(self, image_width, image_high, style_image, tv_weight):
        transfer_net = ImageTransform(image_width, image_high, tv_weight)
        transfer_net.create_network()
        self.transform_model = transfer_net.get_model()
        self.loss_net(self.transform_model.output, self.transform_model.input, style_image)
        self.vgg_model.summary()
        self.vgg_model.compile(self.optimizer, self.dummy_loss)

    def loss_net(self, img_in, t_img_in, style_image):

        x = tf.keras.layers.Concatenate(axis=0)([img_in, t_img_in])
        x = vgg_normalize(x)
        vgg = VGG16()
        vgg.create_network(x)
        vgg.load_existed_weight('pretrained/vgg16_tf_kernels_notop.h5')
        self.vgg_model = vgg.model

        for layer in self.vgg_model.layers[-18:]:
            self.vgg_output_and_layer[layer.name] = (layer.output, layer)

        if self.style_weight > 0:
            self.add_style_loss(style_image)

        if self.content_weight > 0:
            self.add_content_loss()

        # Freeze all VGG layers
        for layer in self.vgg_model.layers[-19:]:
            layer.trainable = False

    def add_style_loss(self, style_image):
        print('Getting style features from VGG network.')

        style_layer_outputs = []

        for layer_name in self.style_layers_name:
            style_layer_outputs.append(self.vgg_output_and_layer[layer_name][0])

        vgg_style_func = tf.keras.backend.function([self.vgg_model.layers[-19].input], style_layer_outputs)

        style_features = vgg_style_func([style_image.image])

        # Style Reconstruction Loss
        for i, layer_name in enumerate(self.style_layers_name):
            layer = self.vgg_output_and_layer[layer_name][1]

            feature_var = tf.keras.backend.variable(value=style_features[i][0])
            style_loss = StyleReconstructionRegularizer(
                                style_feature_target=feature_var,
                                weight=self.style_weight)(layer)

            layer.add_loss(style_loss)

    def add_content_loss(self):
        # Feature Reconstruction Loss
        _, layer = self.vgg_output_and_layer[self.content_layer]
        content_regularizer = FeatureReconstructionRegularizer(self.content_weight)(layer)
        layer.add_loss(content_regularizer)

    def dummy_loss(self, y_true, y_pred ):
        return tf.keras.backend.variable(0.0)
