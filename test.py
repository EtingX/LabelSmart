import labelSmart

#
# label_method = labelSmart.label_change(img_dir='G:\My_Data/train_val_test/train_val/boost\images',
#                                 label_dir='G:\My_Data/train_val_test/train_val/boost\labels',
#                                        save_dir='G:\My_Data/train_val_test/train_val/boost\yolo')
#
# class_file = ['bud','heading']
#
# label_method.voc2yolo(class_file)

# label_method = labelSmart.Label(img_dir='G:/My_Data/train_val_test/train_val/update_final_data/processed/images',
#                                 label_dir='G:/My_Data/train_val_test/train_val/update_final_data/processed/labels',
#                                 out_dir='G:/My_Data/train_val_test/train_val/update_final_data/processed/outputs')
# #
# # replace_label = {'1':0,
# #                  1:0
# #                  }
# #
# # label_method.label_replace(replace_label)
#
# label_method.crop_img()
#
# # import torch
# # print(torch.__version__)
# #
# # print(torch.version.cuda)
# # print(torch.backends.cudnn.version())

import os

image_dir = 'I:/birdcage_2022\wheat_synthesis_project\synthesis/heading'

out_dir = 'G:/My_Data/train_val_test/train_val/update_final_data/processed/outputs/crop_images/heading'

image_list = os.listdir(image_dir)

from random import sample
from shutil import copy
select_list = sample(image_list,1517)

for img in select_list:
    start_dir = os.path.join(image_dir,img)
    end_dir = os.path.join(out_dir,img)
    copy(start_dir,end_dir)