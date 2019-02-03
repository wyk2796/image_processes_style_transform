# coding:utf-8
from model.style_transform import StyleTransform
from preprocess_images.Image import Image
import tensorflow as tf
from scipy.misc import imsave
import numpy as np
import params as P
import time
import os
import util


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
    save_path = P.saved_model + style_name.split(sep='.')[0] + '\\'
    style_image = Image(P.style_dir + style_name)
    style_image.image_resize(P.width, P.high)
    with tf.Session() as session:
        tf.keras.backend.set_session(session)
        model = StyleTransform(session, P.learning_rate, P.content_weight, P.style_weight)
        model.create_model(P.width, P.high, style_image, P.tv_weight)
        saver = tf.train.Saver()
        if os.path.exists(save_path):
            saver.restore(session, save_path)
            # model.vgg_model.load_weights(save_path)
            print('load saved weight from %s' % save_path)
        dummy_y = np.zeros((P.batch_size, P.width, P.high, 3))
        t1 = time.time()
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
        i = 0
        for x in data_gen.flow_from_directory(P.train_image_dir, class_mode=None, batch_size=P.batch_size,
                                              target_size=(P.width, P.high), shuffle=False):
            if i > P.n_epoch:
                break
            hist = model.vgg_model.train_on_batch(x, dummy_y)
            print('\repoc: %d \n' % i, end='')
            if i % 50 == 0:
                print(hist, (time.time() - t1))
                t1 = time.time()
            if i % 500 == 0:
                print("\repoc: %d\n" % i, end='')
                val_x = model.transform_model.predict(x)

                display_img(i, x[0], style_name)
                display_img(i, val_x[0], style_name, True)
                saver.save(session, save_path)
                graph = tf.get_default_graph()
                input = graph.get_tensor_by_name('input_1:0')
                # Pick output for SavedModel
                output = graph.get_tensor_by_name('lambda_8/mul:0')
                tf.saved_model.simple_save(
                    session, P.frozen_model_dir,
                    {"input": input},
                    {"output": output}
                )
                util.save_as_tfjs_model(P.frozen_model_dir, util.load_tensor_name_from_file(P.st_model_tensor_name_path),
                                        P.tfjs_saved_dir + style_name.split(sep='.')[0] + '\\')
                # model.vgg_model.save(save_path)
                #util.generate_encapsulate_model_and_save(util.out_put_layer, model.vgg_model, save_path + '_js')
            i += P.batch_size


if __name__ == '__main__':
    train()
