# import os
#
# import wandb
#
#
# # import shutil
# # from random import sample
# #
# label_dir = 'out_dir/replaced_label'
#
# label_list = os.listdir(label_dir)
# #
# # empty_label_list = []
# # for label in label_list:
# #     if not os.path.getsize(os.path.join(label_dir,label)):
# #         empty_label_list.append(label)
# #
# # select_list = sample(empty_label_list, 800)
# #
# # for select_label in select_list:
# #     shutil.copy(os.path.join('out_dir/removed_label',select_label),os.path.join('out_dir/final_label',select_label))
#
# for label in label_list:
#     count = len(open(os.path.join(label_dir,label),'rU').readlines())
#     if count >= 2:
#         print(label)
#
# from google.cloud import storage

# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """Downloads a blob from the bucket."""
#     # The ID of your GCS bucket
#     # bucket_name = "your-bucket-name"
#
#     # The ID of your GCS object
#     # source_blob_name = "storage-object-name"
#
#     # The path to which the file should be downloaded
#     # destination_file_name = "local/path/to/file"
#
#     storage_client = storage.Client()
#
#     bucket = storage_client.bucket(bucket_name)
#
#     # Construct a client side representation of a blob.
#     # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
#     # any content from Google Cloud Storage. As we don't need additional data,
#     # using `Bucket.blob` is preferred here.
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#
#     print(
#         "Downloaded storage object {} from bucket {} to local file {}.".format(
#             source_blob_name, bucket_name, destination_file_name
#         )
#     )
#
# download_blob('canola-2022-dataset-additional', 'My_Data_Maxi_reduce', 'My_Data_Maxi_reduce')

# from google.cloud import storage
# import os
#
# def findOccurrences(s, ch):  # to find position of '/' in blob path ,used to create folders in local storage
#     return [i for i, letter in enumerate(s) if letter == ch]
#
#
# def download_from_bucket(bucket_name, blob_path, local_path):
#     # Create this folder locally
#     if not os.path.exists(local_path):
#         os.makedirs(local_path)
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blobs = list(bucket.list_blobs(prefix=blob_path))
#
#     startloc = 0
#     for blob in blobs:
#         startloc = 0
#         folderloc = findOccurrences(blob.name.replace(blob_path, ''), '/')
#         if (not blob.name.endswith("/")):
#             if (blob.name.replace(blob_path, '').find("/") == -1):
#                 downloadpath = local_path + '/' + blob.name.replace(blob_path, '')
#                 logging.info(downloadpath)
#                 blob.download_to_filename(downloadpath)
#             else:
#                 for folder in folderloc:
#
#                     if not os.path.exists(local_path + '/' + blob.name.replace(blob_path, '')[startloc:folder]):
#                         create_folder = local_path + '/' + blob.name.replace(blob_path, '')[
#                                                            0:startloc] + '/' + blob.name.replace(blob_path, '')[
#                                                                                startloc:folder]
#                         startloc = folder + 1
#                         os.makedirs(create_folder)
#
#                 downloadpath = local_path + '/' + blob.name.replace(blob_path, '')
#
#                 blob.download_to_filename(downloadpath)
#                 logging.info(blob.name.replace(blob_path, '')[0:blob.name.replace(blob_path, '').find("/")])
#
#     logging.info('Blob {} downloaded to {}.'.format(blob_path, local_path))
#
#
# bucket_name = 'canola-2022-dataset-additional'  # do not use gs://
# blob_path = 'My_Data_Maxi_reduce'  # blob path in bucket where data is stored
# local_dir = 'My_Data_Maxi_reduce'  # trainingData folder in local
# download_from_bucket(bucket_name, blob_path, local_dir)

import os
import shutil
from random import sample
#
# train_image = 'I:\canola_2022_project\GH_detection\google\dataset\My_data_large\images/val'
# train_label = 'I:\canola_2022_project\GH_detection\google\dataset\My_data_large\labels/val'
#
# image_list = os.listdir(train_image)
#
# selected_image = sample(image_list, 2250)
#
# for img in selected_image:
#     start = os.path.join(train_image,img)
#     end = os.path.join('I:\canola_2022_project\GH_detection\google\dataset\My_data_large_redude\images/val', img)
#     shutil.copy(start,end)
#
#     label_name = img[:-4] + '.txt'
#     start_l = os.path.join(train_label,label_name)
#     end_l = os.path.join('I:\canola_2022_project\GH_detection\google\dataset\My_data_large_redude\labels/val',label_name)
#     shutil.copy(start_l,end_l)

import os
import labelSmart
import shutil

image_dir = 'I:\canola_2022_project/fpis_test/test/15'
label_dir = 'I:\canola_2022_project/fpis_test/all_label'

save_img_dir = 'I:\canola_2022_project/fpis_test/test/test/15\images'
save_label_dir = 'I:\canola_2022_project/fpis_test/test/test/15\labels'

image_list = os.listdir(image_dir)
label_list = os.listdir(label_dir)

new_img_list = []
new_label_list = []

for img in image_list:
    new_img_list.append(img[:-4])

for label in label_list:
    new_label_list.append(label[:-4])

for new_img in new_img_list:
    if new_img in new_label_list:
        new_img_name = new_img+'.JPG'
        new_label_name = new_img + '.txt'

        img_start = os.path.join(image_dir,new_img_name)
        img_end = os.path.join(save_img_dir,new_img_name)

        shutil.copy(img_start,img_end)

        label_start = os.path.join(label_dir,new_label_name)
        label_end = os.path.join(save_label_dir,new_label_name)

        shutil.copy(label_start,label_end)

    else:
        new_img_name = new_img + '.jpg'

        img_start = os.path.join(image_dir, new_img_name)
        img_end = os.path.join('I:\canola_2022_project/fpis_test/test/no_label', new_img_name)

        shutil.copy(img_start, img_end)