# coding:utf-8
import tensorflow as tf
import tensorflowjs as tfjs


out_put_layer = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5',
                 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9', 'conv2d_10',
                 'conv2d_11', 'conv2d_12', 'conv2d_13', 'conv2d_14', 'conv2d_15',
                 'cropping2d', 'cropping2d_1', 'cropping2d_2', 'cropping2d_3', 'cropping2d_4',
                 'add', 'add_1', 'add_2', 'add_3', 'add_4',
                 'block1_conv1', 'block1_conv2', 'block1_pool',
                 'block2_conv1', 'block2_conv2', 'block2_pool',
                 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
                 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']


def generate_encapsulate_model_with_output_layer_names(model, output_layer_names):
    enc_model = tf.keras.Model(
        inputs=model.input,
        outputs=list(map(lambda oln: model.get_layer(oln).output, output_layer_names))
    )
    return enc_model


def generate_encapsulate_model_and_save(out_put_layer, model, path):
    enc_model = generate_encapsulate_model_with_output_layer_names(model, out_put_layer)
    tfjs.converters.save_keras_model(enc_model, path)


def print_tensor_name():
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)


def load_tensor_name_from_file(path):
    file = open(path, mode='r', encoding='utf-8')
    tensor_names = []
    for line in file.readlines():
        tensor_names.append(line.strip())
    return tensor_names


def save_as_tfjs_model(model_path, output_node_name, output_dir):
    tfjs.converters.convert_tf_saved_model(model_path, output_node_name, output_dir)

