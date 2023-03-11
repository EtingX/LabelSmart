import labelSmart

# class_file = ['bud']
# label_method = labelSmart.label_change(img_dir='crop/final_images',
#                                        label_dir='crop/final_labels',
#                                        class_file=class_file)
# #
# label_method.voc2yolo()

import os
import shutil

# label_dir = 'label_change/yolo2voc'
# img_dir='I:\canola_2022_project\GH_detection\MyData\Image'
# label_list = os.listdir(label_dir)
#
# for label in label_list:
#     name = label[:-4]
#     img_name = name + '.jpg'
#
#     start = os.path.join(img_dir,img_name)
#     end = os.path.join('crop/early_bud_img',img_name)
#     shutil.copy(start,end)

label_method = labelSmart.Label(img_dir='I:\canola_2022_project\GH_detection\My_TrainData\Image',
                                label_dir='I:\canola_2022_project\GH_detection\My_TrainData\Label')
label_method.no_label_search()
label_method.no_img_search()


# import os
#
# label_dir = 'F:\yolov7_YX\My_data\labels/val'
# for label in os.listdir(label_dir):
#     new_label = []
#     label_path = os.path.join(label_dir, label)
#
#     for line in open(label_path):
#         line_list = line.split(' ')
#         if line_list[0] == 5 or line_list[0] == '5':
#             print(label)

# import glob
# import os
#
# label_list = glob.glob(os.path.join("split_dataset/val/Label", "*.txt"))
# print(label_list)
#
# label_dir = 'split_dataset/val/Label'
#
# if len(os.listdir(label_dir)) != len(glob.glob(os.path.join(label_dir, "*.txt"))):
#     print('Some files are not label file, this function only support YOLO label format !')
# else:
#     print('Replace label process start !')