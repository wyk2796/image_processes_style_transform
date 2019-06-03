import tensorflow as tf
import util
WEIGHTS_INIT_STDEV = .1


def transform_net(image):
    with tf.name_scope(name='transform_net'):
        tf.summary.image('input', image)
        conv1 = _conv_layer(image, 32, 9, 1, name='conv_layer_1')
        conv2 = _conv_layer(conv1, 64, 3, 2, name='conv_layer_2')
        conv3 = _conv_layer(conv2, 128, 3, 2, name='conv_layer_3')
        tf.summary.image('conv3', conv3[:, :, :, 0:3] * 150 + 255./2)
        resid1 = _residual_block('residual_block_1', conv3, 3)
        resid2 = _residual_block('residual_block_2', resid1, 3)
        resid3 = _residual_block('residual_block_3', resid2, 3)
        resid4 = _residual_block('residual_block_4', resid3, 3)
        resid5 = _residual_block('residual_block_5', resid4, 3)
        tf.summary.image('resid5', resid5[:, :, :, 0:3] * 150 + 255./2)
        conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, name='conv_layer_t1')
        conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, name='conv_layer_t2')
        conv_t3 = _conv_layer(conv_t2, 3, 9, 1, name='conv_layer_t3', relu=False)
        preds = tf.nn.tanh(conv_t3) * 150 + 255./2
        tf.summary.image('output_image', util.unprocess(preds))
    return preds


def _conv_layer(net, num_filters, filter_size, strides, name, relu=True):
    with tf.name_scope(name):
        weights_init = _conv_init_vars(net, num_filters, filter_size)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        tf.summary.histogram('conv_weight', weights_init)
        net = _instance_norm(net)
        if relu:
            net = tf.nn.relu(net)
    return net


def _conv_tranpose_layer(net, num_filters, filter_size, strides, name):
    with tf.name_scope(name):
        weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, strides, strides, 1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net)
        return tf.nn.relu(net)


def _residual_block(name, net, filter_size=3):
    with tf.name_scope(name):
        tmp = _conv_layer(net, 128, filter_size, 1, name='{}_conv_layer_1'.format(name))
        return net + _conv_layer(tmp, 128, filter_size, 1, relu=False, name='{}_conv_layer_2'.format(name))


def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    tf.summary.histogram('shift', shift)
    tf.summary.histogram('scale', scale)
    normalized = (net - mu) / (sigma_sq + epsilon)**0.5
    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init


def constant_filter_generate(filter_size, in_channels, out_channels):
    return tf.transpose(tf.eye(filter_size, batch_shape=[out_channels, in_channels]), perm=[2, 3, 0, 1])


def up_sampling(net):
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if in_channels > 16:
        strides = 2
        filter_size = 3
        out_channels = in_channels / 2

    else:
        strides = 1
        filter_size = 3
        out_channels = 3
    new_shape = [batch_size, int(rows * strides), int(cols * strides), out_channels]
    return tf.nn.conv2d_transpose(net, constant_filter_generate(filter_size, in_channels, out_channels),
                                  new_shape, [1, strides, strides, 1],
                                  padding='SAME')

def up_sampling_loop(net):
    if net.get_shape()[3] == 3:
        return net
    else:
        net = up_sampling(net)