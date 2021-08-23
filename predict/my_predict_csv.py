import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath('..'))
import pandas as pd
from libs.data_preprocess.my_data import get_images_labels
from libs.neural_networks.models.my_load_model import load_model
from sklearn.metrics import confusion_matrix
import shutil


dir_original = '/disk1/share_8tb/广角眼底2021.04.12/original'
dir_preprocess = '/disk1/share_8tb/广角眼底2021.04.12/preprocess/384'
dir_dest = '/disk1/share_8tb/广角眼底2021.05.03/results/test'

save_probs = True
pkl_prob = os.path.join(dir_dest, 'probs.pkl')

export_confusion_files = False

train_type = 'wide_angle'
data_version = 'v5'
csv_file = os.path.join(os.path.abspath('..'),
                'datafiles', data_version, 'test.csv')

num_classes = 4
thresholds = [0.5 for x in range(num_classes)]

dicts_models = []

model_name = 'xception'
model_file = '/tmp2/wide_angel/v3/xception/epoch7.pth'
model, input_shape = load_model(model_name, num_class=num_classes, model_file=model_file, get_image_shape=True)
dict_model1 = {'model': model,
                'input_shape': input_shape, 'model_weight': 1, 'batch_size': 64}
dicts_models.append(dict_model1)

model_name = 'inception_v3'
model_file = '/tmp2/wide_angel/v3/inception_v3/epoch8.pth'
model, input_shape = load_model(model_name, num_class=num_classes, model_file=model_file, get_image_shape=True)
dict_model1 = {'model': model,
                'input_shape': input_shape, 'model_weight': 0.8, 'batch_size': 64}
dicts_models.append(dict_model1)

model_name = 'inception_resnet_v2'
model_file = '/tmp2/wide_angel/v3/inception_resnet_v2/epoch8.pth'
model, input_shape = load_model(model_name, num_class=num_classes, model_file=model_file, get_image_shape=True)
dict_model1 = {'model': model,
                'input_shape': input_shape, 'model_weight': 1, 'batch_size': 64}
dicts_models.append(dict_model1)

df = pd.read_csv(csv_file)
files, labels = get_images_labels(filename_csv_or_pd=df)

from libs.neural_networks.helper.my_predict import predict_csv_multiple_model
prob_total, prob_lists = predict_csv_multiple_model(dicts_models, csv_file, activation='sigmoid')

if save_probs:
    import pickle
    os.makedirs(os.path.dirname(pkl_prob), exist_ok=True)
    with open(pkl_prob, 'wb') as f:
        pickle.dump((prob_total, prob_lists), f)


preds = prob_total > thresholds

list_labels = []
for j in range(num_classes):
    list1 = []
    for label_str in labels:
        label = label_str.split('_')[j]
        list1.append(int(label))
    list_labels.append(list1)

for j in range(num_classes):
    cf1 = confusion_matrix(list_labels[j], preds[:, j])
    print(cf1)


if export_confusion_files:
    for index, (filename, labels) in enumerate(zip(files, labels)):
        for i in range(num_classes):
            label_gt = int(labels.split('_')[i])
            if preds[index][i]:
                label_pred = 1
            else:
                label_pred = 0

            if label_gt != label_pred:
                file_source = filename.replace(dir_preprocess, dir_original)
                file_dest = os.path.join(dir_dest, str(i), f'{str(label_gt)}_{str(label_pred)}',
                                  f'prob{str(int(prob_total[index][i]*100))}_' + os.path.split(filename)[1])
                os.makedirs(os.path.dirname(file_dest), exist_ok=True)
                print(f'copy file to {file_dest}')
                shutil.copy(file_source, file_dest)

print('OK')

