# coding:utf-8

# -----------------------------------------------------------------------
# style transformation
st_vgg_path = 'pretrained/imagenet-vgg-verydeep-19.mat'
st_style_dir = 'resource/image_transform/style/'
st_style_image_path_dict = {'des_glaneuses': st_style_dir + 'des_glaneuses.jpg',
                            'la_muse': st_style_dir + 'la_muse.jpg',
                            'mirror': st_style_dir + 'mirror.jpg',
                            'starry_night': st_style_dir + 'starry_night.jpg',
                            'udnie': st_style_dir + 'udnie.jpg',
                            'wave_crop': st_style_dir + 'wave_crop.jpg',
                            }
st_train_path = 'E:/data/image/picture/train2017'
st_model_saved_dir = 'pretrained/'
st_output_saved_dir = 'resource/image_transform/output/'
st_logs = 'resource/logs/'
st_batch_size = 1
st_epoch = 118278 * 2
st_content_weight = 10
st_style_weight = 200
st_tv_weight = 200
width = 512
high = 512
learning_rate = 1e-4
# ----------------------------------------------------------------------------
# neural doodle
doodle_dir = 'resource/doodle/'
doodle_style_img_path = doodle_dir + 'Monet/style.png'
doodle_style_mask_path = doodle_dir + 'Monet/style_mask.png'
doodle_target_mask_path = doodle_dir + 'Monet/target_mask.png'
doodle_saved_model_dir = doodle_dir + 'pretrained/Monet/'
doodle_logs = 'resource/logs/'
doodle_content_img_path = None
doodle_target_img_prefix = doodle_dir + 'out/'
use_content_img = doodle_content_img_path is not None
num_labels = 4

num_colors = 3
STYLE, TARGET, CONTENT = 0, 1, 2
doodle_total_variation_weight = 50.
doodle_style_weight = 1.
doodle_content_weight = 0.1 if use_content_img else 0
