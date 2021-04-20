import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from libs.Dataset.my_dataset import Dataset_CSV
from libs.NeuralNetworks.Train_Predict.my_train_multi_labels import train
from imgaug import augmenters as iaa
from libs.NeuralNetworks.Helper.my_load_model import load_model

# region setting
save_model_dir = '/tmp2/wide_angel_test'
train_type = 'wide_angle'
data_version = 'v4'
csv_train = os.path.join(os.path.abspath('..'),
                         'datafiles', data_version, 'train.csv')
csv_valid = os.path.join(os.path.abspath('..'),
                         'datafiles', data_version, f'valid.csv')
csv_test = os.path.join(os.path.abspath('..'),
                        'datafiles', data_version, f'test.csv')

iaa = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.25),

    iaa.GaussianBlur(sigma=(0.0, 0.3)),
    iaa.MultiplyBrightness(mul=(0.7, 1.3)),
    iaa.contrast.LinearContrast((0.7, 1.3)),
    iaa.Sometimes(0.9, iaa.Add((-8, 8))),
    iaa.Sometimes(0.9, iaa.Affine(
        scale=(0.98, 1.02),
        translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
        rotate=(-15, 15),
    )),
])

batch_size_train, batch_size_valid = 32, 64
#'单纯性的格子样变性', '单纯性的孔源性视网膜脱离', '单纯性的视网膜破裂孔', '囊性视网膜突起', '正常眼底'
# 872, 1012, 953, 4065, 64

num_class = 1
positive_weights = [100]

num_workers = 4  # when debugging it should be set to 0.

''' do not do resampling in this project
import pandas as pd
import numpy as np
df = pd.read_csv(csv_train)
from torch.utils.data.sampler import WeightedRandomSampler
list_class_samples = []
for label in range(num_class):
    list_class_samples.append(len(df[df['labels'] == label]))
sample_class_weights = 1 / np.power(list_class_samples, 0.5)
sample_class_weights = [1, 1, 1.5, 1]
sample_weights = []
for label in df['labels']:
    sample_weights.append(sample_class_weights[label])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df))
'''

# endregion

# region training
# 'tf_efficientnet_b0', 'res2net50_26w_4s', 'resnet50d' , 'resnest50d'
# 'xception', 'inception_resnet_v2', 'inception_v3'
# for model_name in ['resnest50d', 'resnest101e', 'res2net50_26w_6s', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 'xception', 'inception_resnet_v2', 'inception_v3']:
# for model_name in ['tf_efficientnet_b2', 'tf_efficientnet_b3', 'xception', 'inception_v3', 'inception_resnet_v2', 'resnest50d_4s2x40d', 'resnest101e', 'res2net50_26w_6s']:
for model_name in ['inception_resnet_v2', 'xception', 'inception_v3']:
    model = load_model(model_name, num_class=num_class)

    loss_pos_weights = torch.FloatTensor(positive_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
        loss_pos_weights = loss_pos_weights.cuda()

    # pos_weight:positive negative balance, weight:rescaling weight given to the loss
    # criterion = nn.BCELoss(weight=loss_class_weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_pos_weights)
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=0.001)
    # from libs.NeuralNetworks.my_optimizer import Lookahead
    # optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

    # region determine the input shape by the model type
    if model_name in ['xception', 'inception_resnet_v2', 'inception_v3']:
        image_shape = (299, 299)
    elif 'efficientnet_b0' in model_name:
        image_shape = (224, 224)
    elif 'efficientnet_b1' in model_name:
        image_shape = (240, 240)
    elif 'efficientnet_b2' in model_name:
        image_shape = (260, 260)
    elif 'efficientnet_b3' in model_name:
        image_shape = (300, 300)
    elif 'efficientnet_b4' in model_name:
        image_shape = (380, 380)
    elif 'efficientnet_b5' in model_name:
        image_shape = (456, 456)
    elif 'efficientnet_b6' in model_name:
        image_shape = (528, 528)
    elif 'efficientnet_b7' in model_name:
        image_shape = (600, 600)
    else:
        image_shape = (224, 224)
    # endregion

    ds_train = Dataset_CSV(csv_or_df=csv_train, single_label=3, imgaug_iaa=iaa, image_shape=image_shape)
    loader_train = DataLoader(ds_train, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers)
    ds_valid = Dataset_CSV(csv_or_df=csv_valid, single_label=3, image_shape=image_shape)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid,
                              num_workers=num_workers)
    ds_test = Dataset_CSV(csv_or_df=csv_test, single_label=3, image_shape=image_shape)
    loader_test = DataLoader(ds_test, batch_size=batch_size_valid,
                             num_workers=num_workers)

    scheduler = StepLR(optimizer, step_size=2, gamma=0.3)
    epochs_num = 10

    train(model,
          loader_train=loader_train,
          criterion=criterion,
          optimizer=optimizer, scheduler=scheduler,
          epochs_num=epochs_num, log_interval_train=10,
          loader_valid=loader_valid, loader_test=loader_test,
          save_model_dir=os.path.join(save_model_dir, data_version, model_name)
          )

    del model
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()

# endregion

print('OK')
