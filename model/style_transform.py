from __future__ import print_function
import functools
from model.net import VGG
import tensorflow as tf
import numpy as np
import util
from model.net.image_transform_net import transform_net
import params as P

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


class StyleTransform(object):

    def __init__(self, style_features, batch_size, content_weight=1, style_weight=1, tv_weight=1, height=256, width=256):
        self.style_features = style_features
        self.batch_size = batch_size
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.loss = None
        self.train_fetch = dict()
        self.output_fetch = dict()
        self.batch_shape = (self.batch_size, width, height, 3)
        self.X_content = tf.placeholder(tf.float32, shape=self.batch_shape, name="X_content")
        self.output = None

    def evaluate_net(self):
        preds = transform_net(self.X_content / 255.0)
        self.output = preds
        self.output_fetch = {"output": self.output}

    def create_network(self):
        X_pre = util.preprocess(self.X_content)
        content_features = {}
        content_net = VGG.net(P.st_vgg_path, X_pre, name='content_feature')
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        preds = transform_net(self.X_content/255.0)
        self.output = preds
        preds_pre = util.preprocess(preds)
        net = VGG.net(P.st_vgg_path, preds_pre, 'after_transform_feature')
        content_size = _tensor_size(content_features[CONTENT_LAYER]) * self.batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])

        with tf.name_scope("content_loss"):
            content_loss = self.content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
            )

        with tf.name_scope("style_loss"):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                tf.summary.image(style_layer, util.unprocess(tf.transpose(layer, (3, 1, 2, 0))[0:10, :, :, :]), max_outputs=10)
                bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
                size = height * width * filters
                feats = tf.reshape(layer, (bs, height * width, filters))
                feats_T = tf.transpose(feats, perm=[0, 2, 1])
                grams = tf.matmul(feats_T, feats) / size
                style_gram = self.style_features[style_layer]
                style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

            style_loss = self.style_weight * functools.reduce(tf.add, style_losses) / self.batch_size

        # total variation denoising
        with tf.name_scope("total_variation_loss"):
            tv_y_size = _tensor_size(preds[:, 1:, :, :])
            tv_x_size = _tensor_size(preds[:, :, 1:, :])
            y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :self.batch_shape[1]-1, :, :])
            x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :self.batch_shape[2]-1, :])
            tv_loss = self.tv_weight * 2 * (x_tv/tv_x_size + y_tv/tv_y_size)/self.batch_size
        self.loss = content_loss + style_loss + tv_loss
        train_step = tf.train.AdamOptimizer(P.learning_rate).minimize(self.loss)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("content_loss", content_loss)
        tf.summary.scalar("style_loss", style_loss)
        tf.summary.scalar("tv_loss", tv_loss)
        merged = tf.summary.merge_all()
        self.train_fetch = {"summary": merged, "train_op": train_step, "loss": self.loss}
        self.output_fetch = {"output": self.output}

    def train(self, session, input_image):
        feed = {self.X_content: input_image}
        result = session.run(self.train_fetch, feed)
        return result

    def predict(self, session, input_image):
        feed = {self.X_content: input_image}
        return session.run(self.output_fetch, feed)


def get_style_feature(input_style_image, device='/cpu:0', writer=None):
    with tf.Graph().as_default(), tf.device(device), tf.Session() as sess:
        style_image = tf.placeholder(dtype=tf.float32, shape=input_style_image.shape(), name='style_image')
        style_image_pre = util.preprocess(style_image)
        net = VGG.net(P.st_vgg_path, style_image_pre, name='style_feature')
        style_features = dict()
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image: input_style_image.image})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram
    if writer is not None:
        writer.add_graph(sess.graph)
    return style_features


def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
