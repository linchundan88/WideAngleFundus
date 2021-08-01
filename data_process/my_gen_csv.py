import os
import sys
sys.path.append(os.path.abspath('..'))
from libs.data_preprocess.my_compute_digest import calcSha1
import csv
from libs.data_preprocess.my_data import write_images_labels_csv
from libs.data_preprocess.my_data_patiend_id import split_dataset_by_pat_id


dir_preprocess = '/disk1/share_8tb/广角眼底2021.04.08/preprocess/384'
list_labels = ['格子样变性', '孔源性视网膜脱离', '视网膜破裂孔', '囊性视网膜突起']  #'正常眼底', '囊性视网膜突起

TRAIN_TYPE = 'wide_angle'
DATA_VERSION = 'v4'
filename_csv = os.path.join(os.path.abspath('../'),
               'datafiles', DATA_VERSION, 'all.csv')
filename_csv_train = os.path.join(os.path.abspath('../'),
                'datafiles', DATA_VERSION, 'train.csv')
filename_csv_valid = os.path.join(os.path.abspath('../'),
                'datafiles', DATA_VERSION, 'valid.csv')
filename_csv_test = os.path.join(os.path.abspath('../'),
                'datafiles', DATA_VERSION, 'test.csv')

dict1 = {}
for dir_path, _, files in os.walk(dir_preprocess):
    for f in files:
        image_file = os.path.join(dir_path, f)
        file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue

        # print(image_file_source)
        sha1 = calcSha1(image_file)
        dict1[sha1] = []

for index, sub_dir in enumerate(list_labels):
    for dir_path, _, files in os.walk(os.path.join(dir_preprocess, sub_dir)):
        for f in files:
            image_file = os.path.join(dir_path, f)
            file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            # print(image_file)
            sha1 = calcSha1(image_file)
            #duplicate file in one subdir
            if len(dict1[sha1]) < index + 1:
                dict1[sha1].append(1)
            else:
                print(f'duplicate file:{image_file}')
    for key in dict1:
        if len(dict1[key]) < index+1:
            dict1[key].append(0)

for key in dict1:
    assert len(dict1[key]) == len(list_labels)


'''
import pickle
with open('dict1.pkl', 'wb') as f:
     pickle.dump(dict1, f)
'''

list_sha1 = []
os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
with open(filename_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'labels'])

    for dir_path, _, files in os.walk(dir_preprocess):
        for f in files:
            image_file = os.path.join(dir_path, f)
            file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            sha1 = calcSha1(image_file)
            if sha1 in list_sha1:
                continue
            list_sha1.append(sha1)

            # print(f'{image_file}')
            list2 = [str(numeric) for numeric in dict1[sha1]]
            csv_writer.writerow([image_file, '_'.join(list2)])

'''
from libs.data_preprocess.my_data import split_dataset
train_files, train_labels, valid_files, valid_labels, test_files, test_labels = \
    split_dataset(filename_csv,  valid_ratio=0.1, test_ratio=0.15, random_state=888)
'''

train_files, train_labels, valid_files, valid_labels, test_files, test_labels =\
    split_dataset_by_pat_id(filename_csv, valid_ratio=0.1, test_ratio=0.15, random_state=86503418)
write_images_labels_csv(train_files, train_labels, filename_csv_train)
write_images_labels_csv(valid_files, valid_labels, filename_csv_valid)
write_images_labels_csv(test_files, test_labels, filename_csv_test)


print('OK')



