import time
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications import vgg19
import params as P


# helper functions for reading/processing images
def preprocess_image(image_path, img_nrows, img_ncols):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x, img_nrows, img_ncols):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def kmeans(xs, k):
    assert xs.ndim == 2
    try:
        from sklearn.cluster import k_means
        _, labels, _ = k_means(xs.astype('float64'), k)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(xs, k, missing='raise')
    return labels


def load_mask_labels(img_nrows, img_ncols):
    '''Load both target and style masks.
    A mask image (nr x nc) with m labels/colors will be loaded
    as a 4D boolean tensor:
        (1, m, nr, nc) for 'channels_first' or (1, nr, nc, m) for 'channels_last'
    '''
    target_mask_img = load_img(P.doodle_target_mask_path,
                               target_size=(img_nrows, img_ncols))
    target_mask_img = img_to_array(target_mask_img)
    style_mask_img = load_img(P.doodle_style_mask_path,
                              target_size=(img_nrows, img_ncols))
    style_mask_img = img_to_array(style_mask_img)
    if K.image_data_format() == 'channels_first':
        mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T,
                               target_mask_img.reshape((3, -1)).T])
    else:
        mask_vecs = np.vstack([style_mask_img.reshape((-1, 3)),
                               target_mask_img.reshape((-1, 3))])

    labels = kmeans(mask_vecs, P.num_labels)
    style_mask_label = labels[:img_nrows *
                              img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows *
                               img_ncols:].reshape((img_nrows, img_ncols))

    stack_axis = 0 if K.image_data_format() == 'channels_first' else -1
    style_mask = np.stack([style_mask_label == r for r in range(P.num_labels)],
                          axis=stack_axis)
    target_mask = np.stack([target_mask_label == r for r in range(P.num_labels)],
                           axis=stack_axis)

    return (np.expand_dims(style_mask, axis=0),
            np.expand_dims(target_mask, axis=0))

# Define loss functions
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram


def region_style_loss(style_image, target_image, style_mask, target_mask):
    '''Calculate style loss between style_image and target_image,
    for one common region specified by their (boolean) masks
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 2 == K.ndim(style_mask) == K.ndim(target_mask)
    if K.image_data_format() == 'channels_first':
        masked_style = style_image * style_mask
        masked_target = target_image * target_mask
        num_channels = K.shape(style_image)[0]
    else:
        masked_style = K.permute_dimensions(
            style_image, (2, 0, 1)) * style_mask
        masked_target = K.permute_dimensions(
            target_image, (2, 0, 1)) * target_mask
        num_channels = K.shape(style_image)[-1]
    num_channels = K.cast(num_channels, dtype='float32')
    s = gram_matrix(masked_style) / K.mean(style_mask) / num_channels
    c = gram_matrix(masked_target) / K.mean(target_mask) / num_channels
    return K.mean(K.square(s - c))


def style_loss(style_image, target_image, style_masks, target_masks):
    '''Calculate style loss between style_image and target_image,
    in all regions.
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 3 == K.ndim(style_masks) == K.ndim(target_masks)
    loss = None
    for i in range(P.num_labels):
        if K.image_data_format() == 'channels_first':
            style_mask = style_masks[i, :, :]
            target_mask = target_masks[i, :, :]
        else:
            style_mask = style_masks[:, :, i]
            target_mask = target_masks[:, :, i]
        style_l = region_style_loss(style_image, target_image, style_mask, target_mask)
        if loss is None:
            loss = style_l
        else:
            loss = tf.add(loss, style_l)
    return loss


def content_loss(content_image, target_image):
    return K.sum(K.square(target_image - content_image))


def total_variation_loss(x, img_nrows, img_ncols):
    assert 4 == K.ndim(x)
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


class Evaluator(object):

    def __init__(self, img_nrows, img_cols, session_fetch, session_input, session):
        self.loss_value = None
        self.grads_values = None
        self.grad_values = None
        self.img_nrows = img_nrows
        self.img_ncols = img_cols
        self.session_fetch = session_fetch
        self.session_input = session_input
        self.session = session
        self.write_summary = None
        self.count = 0

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def eval_loss_and_grads(self, x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, self.img_nrows, self.img_ncols))
        else:
            x = x.reshape((1, self.img_nrows, self.img_ncols, 3))
        result = self.session.run(self.session_fetch, {self.session_input: x})
        outs = result['loss_grad']
        self.write_summary.add_summary(result['summary'], self.count)
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


class DoodleModel(object):
    def __init__(self, session):
        self.ref_img = img_to_array(load_img(P.doodle_target_mask_path))
        self.img_nrows, self.img_ncols = self.ref_img.shape[:2]
        if K.image_data_format() == 'channels_first':
            shape = (1, P.num_colors, self.img_nrows, self.img_ncols)
        else:
            shape = (1, self.img_nrows, self.img_ncols, P.num_colors)
        self.style_image = K.variable(preprocess_image(P.doodle_style_img_path, self.img_nrows, self.img_ncols))
        self.target_image = tf.placeholder(shape=shape, dtype=tf.float32)
        if P.use_content_img:
            self.content_image = K.variable(preprocess_image(P.doodle_content_img_path, self.img_nrows, self.img_ncols))
        else:
            self.content_image = K.zeros(shape=shape)
        self.content_feature_layers = ['block5_conv2']
        # To get better generation qualities, use more conv layers for style features
        self.style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                                     'block4_conv1', 'block5_conv1']
        self.session = session
        K.set_session(session)
        self.image_grad = None
        self.image_model = None
        self.mask_model = None
        self.optimizer = None
        self.merged = None
        self.write_summary = None

    def create_network(self):
        # ref_img = img_to_array(load_img(P.target_mask_path))
        # img_nrows, img_ncols = ref_img.shape[:2]
        # Create tensor variables for images
        images = K.concatenate([self.style_image, self.target_image, self.content_image], axis=0)

        # Create tensor variables for masks
        raw_style_mask, raw_target_mask = load_mask_labels(self.img_nrows, self.img_ncols)
        style_mask = K.variable(raw_style_mask.astype('float32'))
        target_mask = K.variable(raw_target_mask.astype('float32'))
        masks = K.concatenate([style_mask, target_mask], axis=0)


        # image model as VGG19
        self.image_model = vgg19.VGG19(include_top=False, input_tensor=images)

        # mask model as a series of pooling
        mask_input = tf.keras.layers.Input(tensor=masks, shape=(None, None, None), name='mask_input')
        x = mask_input
        for layer in self.image_model.layers[1:]:
            name = 'mask_%s' % layer.name
            if 'conv' in layer.name:
                x = tf.keras.layers.AveragePooling2D((3, 3), padding='same', strides=(
                    1, 1), name=name)(x)
            elif 'pool' in layer.name:
                x = tf.keras.layers.AveragePooling2D((2, 2), name=name)(x)
        self.mask_model = tf.keras.Model(mask_input, x)

    def loss_function(self):
        image_features = {}
        mask_features = {}
        for img_layer, mask_layer in zip(self.image_model.layers, self.mask_model.layers):
            if 'conv' in img_layer.name:
                assert 'mask_' + img_layer.name == mask_layer.name
                layer_name = img_layer.name
                img_feat, mask_feat = img_layer.output, mask_layer.output
                tf.summary.image(mask_layer.name,
                                 tf.reshape(mask_feat[1, :, :, 1] * 255, shape=(1, mask_feat.shape[1], mask_feat.shape[2], 1)))
                tf.summary.image(img_layer.name,
                                 tf.reshape(img_feat[0, :, :, 2] * 255, shape=(1, img_feat.shape[1], img_feat.shape[2], 1)))
                image_features[layer_name] = img_feat
                mask_features[layer_name] = mask_feat

        # Overall loss is the weighted sum of content_loss, style_loss and tv_loss
        # Each individual loss uses features from image/mask models.
        content_loss_value = None
        for layer in self.content_feature_layers:
            content_feat = image_features[layer][P.CONTENT, :, :, :]
            target_feat = image_features[layer][P.TARGET, :, :, :]
            content_l = P.doodle_content_weight * content_loss(content_feat, target_feat)
            if content_loss_value is None:
                content_loss_value = content_l
            else:
                content_loss_value = content_loss_value + content_l

        tf.summary.scalar("content_loss", content_loss_value)

        style_loss_value = None
        for layer in self.style_feature_layers:
            style_feat = image_features[layer][P.STYLE, :, :, :]
            target_feat = image_features[layer][P.TARGET, :, :, :]
            style_masks = mask_features[layer][P.STYLE, :, :, :]
            target_masks = mask_features[layer][P.TARGET, :, :, :]
            sl = style_loss(style_feat, target_feat, style_masks, target_masks)
            style_l = (P.doodle_style_weight / len(self.style_feature_layers)) * sl
            if style_loss_value is None:
                style_loss_value = style_l
            else:
                style_loss_value = style_loss_value + style_l

        tf.summary.scalar("style_loss", style_loss_value)

        total_loss = P.doodle_total_variation_weight * total_variation_loss(self.target_image, self.img_nrows, self.img_ncols)
        tf.summary.scalar("total_variation_loss", total_loss)
        loss = content_loss_value + style_loss_value + total_loss
        loss_grads = tf.gradients(loss, self.target_image)
        self.outputs = [loss]

        if self.image_grad is None:
            self.image_grad = loss_grads[0]
        else:
            self.image_grad = self.image_grad + loss_grads[0]

        if isinstance(loss_grads, (list, tuple)):
            self.outputs += loss_grads
        else:
            self.outputs.append(loss_grads)

        tf.summary.scalar('loss', loss)
        tf.summary.image('grad_image', self.image_grad + 127.5)
        self.merged = tf.summary.merge_all()

    def train(self, iterate_num):
        # Generate images by iterative optimization
        if K.image_data_format() == 'channels_first':
            x = np.random.uniform(0, 255, (1, 3, self.img_nrows, self.img_ncols)) - 128.
        else:
            x = np.random.uniform(0, 255, (1, self.img_nrows, self.img_ncols, 3)) - 128.
        fetch = {'loss_grad': self.outputs, 'summary': self.merged}
        evaluator = Evaluator(self.img_nrows, self.img_ncols, fetch, self.target_image, self.session)
        evaluator.write_summary = self.write_summary
        for i in range(iterate_num):
            print('Start of iteration', i)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                             fprime=evaluator.grads, maxfun=20)

            print('Current loss value:', min_val)
            # save current generated image
            img = deprocess_image(x.copy(), self.img_nrows, self.img_ncols)
            fname = P.doodle_target_img_prefix + '_at_iteration_%d.png' % i
            save_img(fname, img)
            evaluator.count += 1
            end_time = time.time()
            print('Image saved as', fname)
            print('Iteration %d completed in %ds' % (i, end_time - start_time))
