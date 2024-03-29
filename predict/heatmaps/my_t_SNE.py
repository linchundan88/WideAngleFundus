import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from libs.neural_networks.models.my_load_model import load_model
from torch.utils.data import DataLoader
from libs.dataset.my_dataset import Dataset_CSV
from libs.neural_networks.heatmaps.t_SNE.my_tsne_helper import compute_features_files, gen_tse_features, draw_tsne

num_classes = 4
csv_file = os.path.join(os.path.abspath('../..'), 'datafiles', 'v5', 'test.csv')
save_features = False

model_name = 'inception_v3'

#region load model
if model_name == 'xception':
    model_file = '/tmp2/wide_angel/v3/xception/epoch7.pth'
    model, image_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
    layer_features = model.global_pool
    batch_size = 32

if model_name == 'inception_v3':
    model_file = '/tmp2/wide_angel/v3/inception_v3/epoch8.pth'
    model, image_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
    layer_features = model.global_pool
    batch_size = 32

if model_name == 'inception_resnet_v2':
    model_file = '/tmp2/wide_angel/v3/inception_resnet_v2/epoch8.pth'
    model, image_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
    layer_features = model.global_pool
    batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval()
if torch.cuda.device_count() > 0:
    model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

#endregion


ds_test = Dataset_CSV(data_source=csv_file, image_shape=image_shape, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=batch_size,
                             num_workers=4)

features = compute_features_files(model, layer_features, loader_test)

X_tsne = gen_tse_features(features)

if save_features:
    npy_file_features = "/disk1/share_8tb/广角眼底2021.04.12/results/T-SNE/test"
    os.makedirs(os.path.dirname(npy_file_features), exist_ok=True)
    import numpy as np
    np.save(npy_file_features, X_tsne)  # X_tsne = np.load(save_npy_file)


for i in range(num_classes):
    tsne_image_file = f'/disk1/share_8tb/广角眼底2021.04.22/results/T-SNE/test/{i}_{model_name}.png'
    os.makedirs(os.path.dirname(tsne_image_file), exist_ok=True)
    from libs.data_preprocess.my_multi_labels import get_labels
    draw_tsne(X_tsne, get_labels(csv_file, class_index=i), nb_classes=2, save_tsne_image=tsne_image_file,
             labels_text=['0', '1'], colors=['g', 'r'])



# '格子样变性', '孔源性视网膜脱离', '视网膜破裂孔', '囊性视网膜突起'

print('OK')

