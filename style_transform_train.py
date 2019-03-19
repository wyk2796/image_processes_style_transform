import tensorflow as tf
from preprocess_images.Image import Image
from model.style_transform import StyleTransform, get_style_feature
import params as P
import time
import util
import os

style_name = 'starry_night'
style_p = P.st_style_image_path_dict.get(style_name, 'resource/style/mirror.jpg')
style_image = Image(style_p)
style_image.image_resize(P.width, P.high)
style_image.extend_dim()
trains_files = util.get_files(P.st_train_path)
model_saved_path = P.st_model_saved_dir + style_name + '/'
writer = tf.summary.FileWriter(P.st_logs + style_name + "_{}".format(time.time()))

style_features = get_style_feature(style_image, '/gpu:0')

with tf.Session() as session:
    model = StyleTransform(style_features,
                           P.st_batch_size,
                           content_weight=P.st_content_weight,
                           style_weight=P.st_style_weight,
                           tv_weight=P.st_tv_weight)
    model.create_network()
    writer.add_graph(session.graph)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if os.path.exists(model_saved_path):
        saver.restore(session, model_saved_path)
        print('load saved weight from %s' % model_saved_path)
    else:
        print('not exist the pretrained weights!')
    loss = 0
    for i in range(P.st_epoch):

        image = Image(trains_files[i])
        image.image_resize(P.width, P.high)
        image.extend_dim()
        result = model.train(session, image.image)
        loss += result['loss']

        if i % 50 == 0:
            print("\repoch {} loss: {}".format(i, loss / 50), end='')
            summary = result['summary']
            writer.add_summary(summary, i)
            writer.flush()
            loss = 0

        if i % 500 == 0:
            pre = model.predict(session, image.image)
            output_image = pre['output']
            style_content = util.unprocess(output_image)
            image.save_img(P.st_output_saved_dir + str(i) + '_raw.jpg')
            image.set_image(style_content[0])
            image.save_img(P.st_output_saved_dir + str(i) + '.jpg')
            saver.save(session, model_saved_path)



