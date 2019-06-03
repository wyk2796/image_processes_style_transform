# coding:utf-8
from model.style_transform import StyleTransform
from preprocess_images.Image import Image
import tensorflow as tf
import params as P
import os


style_name = 'udnie'
model_saved_path = P.st_model_saved_dir + style_name + '/'
content_image_path = "resource/image_transform/content/tubingen.jpg"


with tf.device('/cpu:0'), tf.Session() as session:
    image = Image(content_image_path)
    image.extend_dim()
    model = StyleTransform(None,
                           P.st_batch_size,
                           content_weight=P.st_content_weight,
                           style_weight=P.st_style_weight,
                           tv_weight=P.st_tv_weight,
                           height=image.height,
                           width=image.width)
    model.evaluate_net()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if os.path.exists(model_saved_path):
        saver.restore(session, model_saved_path)
        print('load saved weight from %s' % model_saved_path)
    else:
        raise Exception("weight not exist {}".format(model_saved_path))

    file_name = os.path.basename(content_image_path).split('.')[0]
    pre = model.predict(session, image.image)
    output_image = pre['output']
    # style_content = util.unprocess(output_image)
    image.save_img(P.st_output_saved_dir + file_name + '_raw.jpg')
    image.set_image(output_image[0])
    image.save_img(P.st_output_saved_dir + file_name + '_' + style_name +'.jpg')
