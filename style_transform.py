# coding:utf-8
import numpy as np
from model.style_transform import StyleTransform
from preprocess_images.Image import Image
from preprocess_images.util import preprocess_reflect_image, crop_image, median_filter_all_colours, blend, original_colors
import tensorflow as tf
import params as P
import time
from scipy.misc import imsave


def predict():
    input_file_path = 'E:\\temp\style_transform\content\\tubingen.jpg'
    saved_model_path = 'E:\\temp\style_transform\pretrained\\la_muse_weights.h5'
    output_file_path = 'E:\\temp\style_transform\output\\tubingen.jpg'
    input_image = Image(input_file_path)
    with tf.Session() as session:
        tf.keras.backend.set_session(session)
        aspect_ratio, in_image = preprocess_reflect_image(input_image.image, size_multiple=4)
        image_width = image_high = in_image.shape[1]
        model = StyleTransform(session, P.learning_rate, image_width, image_high)
        input_image.set_image(in_image)
        model.create_model(image_width, image_high, input_image, P.tv_weight)
        t1 = time.time()

        model.vgg_model.load_weights(saved_model_path)

        y = model.transform_model.predict(in_image)[0]
        y = crop_image(y, aspect_ratio)

        print("process: %s" % (time.time() - t1))

        ox = crop_image(in_image[0], aspect_ratio)

        y = median_filter_all_colours(y, P.media_filter)

        if P.blend_alpha > 0:
            y = blend(ox, y, P.blend_alpha)

        if P.original_color > 0:
            y = original_colors(ox, y, P.original_color)

        imsave('%s_output.png' % output_file_path, y)

if __name__ == '__main__':
    predict()
