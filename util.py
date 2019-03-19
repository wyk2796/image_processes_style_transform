# coding:utf-8
import tensorflow as tf
import numpy as np
import os

MEAN_PIXEL = np.array([123.68,  116.779,  103.939])


def get_files(img_dir):
    files = []
    for (_, _, file_names) in os.walk(img_dir):
        files.extend(file_names)
    return [os.path.join(img_dir, x) for x in files]


def preprocess(image):
    return image - MEAN_PIXEL


def unprocess(image):
    return image + MEAN_PIXEL


def generate_encapsulate_model_with_output_layer_names(model, output_layer_names):
    enc_model = tf.keras.Model(
        inputs=model.input,
        outputs=list(map(lambda oln: model.get_layer(oln).output, output_layer_names))
    )
    return enc_model


def print_tensor_name():
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)


def load_tensor_name_from_file(path):
    file = open(path, mode='r', encoding='utf-8')
    tensor_names = []
    for line in file.readlines():
        tensor_names.append(line.strip())
    return tensor_names


