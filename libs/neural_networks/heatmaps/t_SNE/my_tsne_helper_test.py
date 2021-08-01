import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from libs.NeuralNetworks.models.my_load_model import load_model
from libs.dataset.my_dataset import get_tensor
from libs.NeuralNetworks.heatmaps.t_SNE.my_tsne_helper import compute_features

num_classes = 4

model_name = 'inception_v3'

if model_name == 'xception':
    model_file = '/tmp2/wide_angel/v3/xception/epoch7.pth'
    model = load_model(model_name, num_class=num_classes, model_file=model_file)
    image_shape = (299, 299)
    layer_features = model.global_pool
    activation = 'sigmoid'

if model_name == 'inception_v3':
    model_file = '/tmp2/wide_angel/v3/inception_v3/epoch8.pth'
    model = load_model(model_name, num_class=num_classes, model_file=model_file)
    image_shape = (299, 299)
    layer_features = model.global_pool
    activation = 'sigmoid'

if model_name == 'inception_resnet_v2':
    model_file = '/tmp2/wide_angel/v3/inception_resnet_v2/epoch8.pth'
    model = load_model(model_name, num_class=num_classes, model_file=model_file)
    image_shape = (299, 299)
    layer_features = model.global_pool
    activation = 'sigmoid'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval()
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

file_image = '/disk1/share_8tb/广角眼底2021.04.12/preprocess/384/视网膜破裂孔/01263635-20181227@170013-R2.jpg'
tensor_x = get_tensor(file_image, image_shape=image_shape)
tensor_x = tensor_x.to(device)
features = compute_features(model, tensor_x, layer_features)

print(features.shape)

