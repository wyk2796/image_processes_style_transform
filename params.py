# coding:utf-8

project_file_dir = "E:\\temp\\style_transform\\"

style_dir = project_file_dir + 'style\\'
content_dir = project_file_dir + 'content\\'
train_image_dir = 'E:\\temp\\train2017\\'
output_dir = project_file_dir + 'output\\'
saved_model = project_file_dir + 'pretrained\\'

tfjs_saved_dir = project_file_dir + 'tfjs_model\\'

frozen_model_dir = project_file_dir + 'frozen_model\\'

st_model_tensor_name_path = 'model_tensor_name/style_transform_model.txt'

batch_size = 1
width = high = 256
learning_rate = 1e-3
tv_weight = 1e-6
n_epoch = 82785 * 2

content_weight = 1.0
style_weight = 4.0

original_color = 0
blend_alpha = 0
media_filter = 3
