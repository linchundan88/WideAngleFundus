import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from libs.dataset.my_dataset import Dataset_CSV
from libs.neural_networks.helper.my_train_multi_labels import train
from imgaug import augmenters as iaa
from libs.neural_networks.models.my_load_model import load_model
from pathlib import Path

# region setting
save_model_dir = '/tmp2/wide_angel/indepedent_classifier1'
train_type = 'wide_angle'
data_version = 'v5'

num_class = 1   #single label, sigmoid

# single_label_no = 0  # the targeted class label
# positive_weights = [4]
# single_label_no = 1
# positive_weights = [3]
single_label_no = 2
positive_weights = [3]
# single_label_no = 3
# positive_weights = [100]

'''
单纯性的格子样变性', '单纯性的孔源性视网膜脱离', '单纯性的视网膜破裂孔', '囊性视网膜突起'
class no:0
label 0:5131, label 0:827
class no:1
label 0:5005, label 0:953
class no:2
label 0:4947, label 0:1011
class no:3
label 0:5893, label 0:65
'''

csv_train = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'train.csv')
csv_valid = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'valid.csv')
csv_test = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'test.csv')

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
    model, image_shape = load_model(model_name, num_class=num_class, get_image_shape=True)

    loss_pos_weights = torch.FloatTensor(positive_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
        loss_pos_weights = loss_pos_weights.cuda()

    ds_train = Dataset_CSV(data_source=csv_train, single_label=single_label_no, imgaug_iaa=iaa, image_shape=image_shape)
    loader_train = DataLoader(ds_train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    ds_valid = Dataset_CSV(data_source=csv_valid, single_label=single_label_no, image_shape=image_shape)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid, num_workers=num_workers)
    ds_test = Dataset_CSV(data_source=csv_test, single_label=single_label_no, image_shape=image_shape)
    loader_test = DataLoader(ds_test, batch_size=batch_size_valid, num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_pos_weights)
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=0.001)
    # from libs.neural_networks.my_optimizer import Lookahead
    # optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.3)
    epochs_num = 10

    train(model,
          loader_train=loader_train,
          criterion=criterion, optimizer=optimizer, scheduler=scheduler,
          epochs_num=epochs_num, amp=True, log_interval_train=10,
          loader_valid=loader_valid, loader_test=loader_test,
          save_model_dir=os.path.join(save_model_dir, data_version, str(single_label_no), model_name)
          )

    del model
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()

# endregion

print('OK')
