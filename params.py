# coding:utf-8

project_file_dir = "E:\\data\\image\\style_transform\\"

style_dir = project_file_dir + 'style\\'
content_dir = project_file_dir + 'content\\'
train_image_dir = 'E:\\data\\image\\picture\\'
output_dir = project_file_dir + 'output\\'
<<<<<<< HEAD
saved_model_dir = project_file_dir + 'pretrained\\'
=======
saved_model = project_file_dir + 'pretrained\\'

tfjs_saved_dir = project_file_dir + 'tfjs_model\\'

frozen_model_dir = project_file_dir + 'frozen_model\\'

st_model_tensor_name_path = 'model_tensor_name/style_transform_model.txt'
>>>>>>> 3a02a5ce6b1e770421924c198cdd8522d6ffef6b

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
