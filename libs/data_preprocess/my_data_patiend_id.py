import math
import os
import random

import pandas as pd
import sklearn
import csv


# 00007775-20181214@093010-L6  parse 00007775
def get_pat_id(filename):
    _, filename = os.path.split(filename)
    pat_id = filename.split('-')[0]
    return pat_id


def split_dataset_by_pat_id(filename_csv_or_df,
                            valid_ratio=0.1, test_ratio=None, shuffle=True, random_state=None):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    list_patient_id = []
    for _, row in df.iterrows():
        filename = row['images']
        pat_id = get_pat_id(filename)

        print(pat_id, filename)
        if pat_id not in list_patient_id:
            list_patient_id.append(pat_id)

    list_patient_id = sklearn.utils.shuffle(list_patient_id, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(list_patient_id) * (1 - valid_ratio))
        list_patient_id_train = list_patient_id[:split_num]
        list_patient_id_valid = list_patient_id[split_num:]

        train_files = []
        train_labels = []
        valid_files = []
        valid_labels = []

        for _, row in df.iterrows():
            image_file = row['images']
            image_labels = row['labels']

            pat_id = get_pat_id(filename)

            if pat_id in list_patient_id_train:
                train_files.append(image_file)
                train_labels.append(image_labels)

            if pat_id in list_patient_id_valid:
                valid_files.append(image_file)
                valid_labels.append(image_labels)

        return train_files, train_labels, valid_files, valid_labels

    else:
        split_num_train = int(len(list_patient_id) * (1 - valid_ratio - test_ratio))
        list_patient_id_train = list_patient_id[:split_num_train]
        split_num_valid = int(len(list_patient_id) * (1 - test_ratio))
        list_patient_id_valid = list_patient_id[split_num_train:split_num_valid]
        list_patient_id_test = list_patient_id[split_num_valid:]

        train_files = []
        train_labels = []
        valid_files = []
        valid_labels = []
        test_files = []
        test_labels = []

        for _, row in df.iterrows():
            image_file = row['images']
            image_labels = row['labels']
            _, filename = os.path.split(image_file)

            pat_id = get_pat_id(filename)
            if pat_id in list_patient_id_train:
                train_files.append(image_file)
                train_labels.append(image_labels)

            if pat_id in list_patient_id_valid:
                valid_files.append(image_file)
                valid_labels.append(image_labels)

            if pat_id in list_patient_id_test:
                test_files.append(image_file)
                test_labels.append(image_labels)

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels

