'''

write_csv_based_on_dir
  generate labels based on dir, and write images and labels to csv file


split_dataset   (split dataset into training and validation datasets)
split_dataset_cross_validation

split_dataset_by_pat_id
  (split dataset into training and validation datasets, and suppose pat_id is the prefix of image filename)
split_dataset_by_pat_id_cross_validation


split_images_masks(image segmentation)


write_csv_files() write list images and labels to csv file
  image_files and labels after split(based on pat_id or not), write to csv files.



get_images_labels 分类 获取文件名 和 标注类别， 用于计算confusion_matrix等验证

'''


import os
import csv
import pandas as pd
import sklearn
import math


def write_csv_based_on_dir(filename_csv, base_dir, dict_mapping, match_type='header',
       list_file_ext=['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']):

    assert match_type in ['header', 'partial', 'end'], 'match type is error'

    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for dir_path, _, files in os.walk(base_dir, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)

                (filedir, tempfilename) = os.path.split(img_file_source)
                (filename, extension) = os.path.splitext(tempfilename)

                if extension.upper() not in list_file_ext:
                    print('file ext name:', f)
                    continue

                if not filedir.endswith('/'):
                    filedir += '/'

                if '未累及中央的黄斑水肿' in filedir:
                    print('aaa')
                for (k, v) in dict_mapping.items():
                    if match_type == 'header':
                        dir1 = os.path.join(base_dir, k)
                        if not dir1.endswith('/'):
                            dir1 += '/'

                        if dir1 in filedir:
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'partial':
                        if '/' + k + '/' in filedir:
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'end':
                        if filedir.endswith('/' + k + '/'):
                            csv_writer.writerow([img_file_source, v])
                            break


#读取csv文件，分割成训练集和验证集，返回训练集图像文件和标注，以及验证集图像文件和标注
def split_dataset(filename_csv_or_df, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None, field_columns=['images', 'labels']):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(df)*(1-valid_ratio))
        data_train = df[:split_num]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        data_valid = df[split_num:]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        data_train = df[:split_num_train]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        split_num_valid = int(len(df) * (1 - test_ratio))
        data_valid = df[split_num_train:split_num_valid]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        data_test = df[split_num_valid:]
        test_files = data_test[field_columns[0]].tolist()
        test_labels = data_test[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels



def write_images_labels_csv(files, labels, filename_csv, field_columns=['images', 'labels']):
    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow([field_columns[0], field_columns[1]])

        for i, file in enumerate(files):
            csv_writer.writerow([file, labels[i]])

    print('write csv ok!')



# get_list_from_dir and write to csv,  label all 0, used for validation
def write_csv_dir_nolabel(filename_csv, base_dir, replace_dir=False):
    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for dir_path, _, files in os.walk(base_dir, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)

                (filepath, tempfilename) = os.path.split(img_file_source)
                (filename, extension) = os.path.splitext(tempfilename)

                if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                    continue

                if replace_dir:  # remove base dir
                    img_file_source = img_file_source.replace(base_dir, '')

                csv_writer.writerow([img_file_source, 0])


#批量计算(例如confusion_matrix)时候用，全部数据，不拆分  没必要shuffle，保留也无所谓
def get_images_labels(filename_csv_or_pd, shuffle=False):
    if isinstance(filename_csv_or_pd, str):
        df = pd.read_csv(filename_csv_or_pd)
    else:
        df = filename_csv_or_pd

    if shuffle:
        df = sklearn.utils.shuffle(df)

    data_all_image_file = df['images']
    data_all_labels = df['labels']

    all_files = data_all_image_file.tolist()
    all_labels = data_all_labels.tolist()

    return all_files, all_labels



