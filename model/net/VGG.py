import tensorflow as tf
import numpy as np
import scipy.io

layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


def net(data_path, input_image, name='default'):

    data = scipy.io.loadmat(data_path)
    weights = data['layers'][0]

    net_layer = [("input", input_image)]

    with tf.name_scope('vgg_{}'.format(name)):
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                net_layer.append((name, _conv_layer(net_layer[i][1], kernels, bias, name=name)))
            elif kind == 'relu':
                net_layer.append((name, _relu_layer(net_layer[i][1], name=name)))
            elif kind == 'pool':
                net_layer.append((name, _pool_layer(net_layer[i][1], name=name)))
            # net[name] = current
    net_dict = dict(net_layer[1:])
    assert len(net_dict) == len(layers)
    return net_dict


def _conv_layer(input, weights, bias, name):
    with tf.name_scope(name):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                            padding='SAME')
        bisa = tf.nn.bias_add(conv, bias)
    return bisa


def _relu_layer(input, name):
    with tf.name_scope(name):
        return tf.nn.relu(input)


def _pool_layer(input, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                              padding='SAME')

