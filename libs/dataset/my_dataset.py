'''https://github.com/aleju/imgaug/issues/406
https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=8q8a2Ha9pnaz
'''

import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np



class Dataset_CSV(Dataset):
    def __init__(self, csv_or_df, single_label=None, image_shape=None, imgaug_iaa=None, test_mode=False):
        self.single_label = single_label
        if isinstance(csv_or_df, str):
            assert os.path.exists(csv_or_df), 'csv file does not exists'
            self.df = pd.read_csv(csv_or_df)
        elif isinstance(csv_or_df, pd.DataFrame):
            self.df = csv_or_df
        else:
            raise ValueError("csv_or_df type error")

        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape

        if imgaug_iaa is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                imgaug_iaa.augment_image,
                transforms.ToTensor()
                ])

        self.test_mode = test_mode

    def __getitem__(self, index):
        img_filename = self.df.iloc[index][0]
        assert os.path.exists(img_filename), f'image file: {img_filename} does not exists'
        image = cv2.imread(img_filename)

        #shape: height, width,  resize: width, height
        if (self.image_shape is not None) and (image.shape[:2] != self.image_shape[:2]):
            image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))

        # image = image.transpose(2, 0, 1)  use transform.toTensor instead
        image = self.transform(image)

        if not self.test_mode:
            if self.single_label is not None:
                list_labels = []
                list_labels.append(self.df.iloc[index][1].split('_')[self.single_label])
                labels = np.asarray(list_labels, dtype=np.float32)
                return image, labels
            else:
                list_labels = []
                labels_str = self.df.iloc[index][1]
                for label1 in labels_str.split('_'):
                    list_labels.append(int(label1))
                labels = np.asarray(list_labels, dtype=np.float32)
                return image, labels
        else:
            return image

    def __len__(self):
        return len(self.df)


def get_tensor(img_file, image_shape=None):
    image = cv2.imread(img_file)
    # shape: height, width,  resize: width, height
    if (image_shape is not None) and (image.shape[:2] != image_shape[:2]):
        image = cv2.resize(image, (image_shape[1], image_shape[0]))

    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor_x = transform1(image)
    tensor_x = tensor_x.unsqueeze(0)  #(C,H,W) (N,C,H,W)

    return tensor_x