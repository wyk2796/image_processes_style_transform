# coding:utf-8

project_file_dir = "E:\\temp\\style_transform\\"

style_dir = project_file_dir + 'style\\'
content_dir = project_file_dir + 'content\\'
train_image_dir = 'E:\\temp\\train2017\\'
output_dir = project_file_dir + 'output\\'

batch_size = 1
width = high = 256
learning_rate = 1e-3
tv_weight = 1e-6
n_epoch = 82785 * 2

content_weight = 1.0
style_weight = 4.0
