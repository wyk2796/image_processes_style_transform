# coding:utf-8
from model.style_transform import StyleTransform
from preprocess_images.Image import Image
import tensorflow as tf
from scipy.misc import imsave
import numpy as np
import params as P
import time


def display_img(i, x, style, is_val=False):
    # save current generated image
    img = x
    if is_val:
        fname = P.output_dir + '/%s_%d_val.png' % (style, i)
    else:
        fname = P.output_dir + '/%s_%d.png' % (style, i)
    imsave(fname, img)
    print('Image saved as %s \n' % fname)


def train():
    style_name = 'la_muse.jpg'
    style_image = Image(P.style_dir + style_name)
    style_image.image_resize(P.width, P.high)
    print('train')
    with tf.Session() as session:
        tf.keras.backend.set_session(session)
        model = StyleTransform(session, P.learning_rate, P.content_weight, P.style_weight)
        model.create_model(P.width, P.high, style_image, P.tv_weight)
        dummy_y = np.zeros((P.batch_size, P.width, P.high, 3))
        skip_to = 0
        t1 = time.time()
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
        i = 0
        for x in data_gen.flow_from_directory(P.train_image_dir, class_mode=None, batch_size=P.batch_size,
                                              target_size=(P.width, P.high), shuffle=False):
            if i > P.n_epoch:
                break
            if i < skip_to:
                i += P.batch_size
                if i % 1000 == 0:
                    print("skip to: %d" % i)
                continue
            hist = model.vgg_model.train_on_batch(x, dummy_y)
            print('\repoc: %d' % i, end='')
            if i % 50 == 0:
                print(hist, (time.time() - t1))
                t1 = time.time()

            if i % 500 == 0:
                print("\repoc: %d\n" % i, end='')
                val_x = model.transform_model.predict(x)

                display_img(i, x[0], style_name)
                display_img(i, val_x[0], style_name, True)
                model.vgg_model.save(style_name + '_weights.h5')
            i += P.batch_size


if __name__ == '__main__':
    train()
