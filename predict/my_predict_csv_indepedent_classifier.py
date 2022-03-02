import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath('..'))
import os
from pathlib import Path
import pandas as pd
from libs.data_preprocess.my_data import get_images_labels
from libs.neural_networks.models.my_load_model import load_model
from sklearn.metrics import confusion_matrix
import shutil


num_classes = 1  # single label
single_label_no = 2   # the targeted class label

dir_original = '/disk1/share_8tb/广角眼底2021.04.12/original'
dir_preprocess = '/disk1/share_8tb/广角眼底2021.04.12/preprocess/384'
dir_dest = os.path.join('/disk1/share_8tb/广角眼底2022.0.7/results/test', str(single_label_no))

save_probs = True
pkl_prob = Path(dir_dest).joinpath('probs.pkl')
export_confusion_files = False

train_type = 'wide_angle'
data_version = 'v5'
csv_file = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'test.csv')

thresholds = [0.5 for x in range(num_classes)]

dicts_models = []

#single_label_no:3
# model_name = 'inception_resnet_v2'
# model_file = '/tmp2/wide_angel/indepedent_classifier/v4/inception_resnet_v2/epoch1.pth'
# model, input_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
# dict_model1 = {'model': model, 'input_shape': input_shape, 'model_weight': 1, 'batch_size': 64}
# dicts_models.append(dict_model1)

model_name = 'inception_resnet_v2'
model_file = f'/tmp2/wide_angel/indepedent_classifier1/v5/{single_label_no}/inception_resnet_v2/epoch8.pth'
model, input_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
dict_model1 = {'model': model, 'input_shape': input_shape, 'model_weight': 1, 'batch_size': 64}
dicts_models.append(dict_model1)

model_name = 'xception'
model_file = f'/tmp2/wide_angel/indepedent_classifier1/v5/{single_label_no}/xception/epoch8.pth'
model, input_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
dict_model2 = {'model': model, 'input_shape': input_shape, 'model_weight': 1, 'batch_size': 64}
dicts_models.append(dict_model2)

model_name = 'inception_v3'
model_file = f'/tmp2/wide_angel/indepedent_classifier1/v5/{single_label_no}/inception_v3/epoch8.pth'
model, input_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
dict_model3 = {'model': model, 'input_shape': input_shape, 'model_weight': 1, 'batch_size': 64}
dicts_models.append(dict_model3)

df = pd.read_csv(csv_file)
files, labels = get_images_labels(filename_csv_or_pd=df)

from libs.neural_networks.helper.my_predict import predict_csv_multiple_model
prob_total, prob_lists = predict_csv_multiple_model(dicts_models, csv_file, activation='sigmoid')
preds = prob_total > thresholds

if save_probs:
    import pickle
    pkl_prob.parent.mkdir(parents=True, exist_ok=True)
    with pkl_prob.open(mode='wb') as f:
        pickle.dump((prob_total, prob_lists), f)


list_labels = []
for label_str in labels:
    label = label_str.split('_')[single_label_no]
    list_labels.append(int(label))


cf1 = confusion_matrix(list_labels, preds[:, 0])
print(cf1)


if export_confusion_files:
    for index, (filename, labels) in enumerate(zip(files, labels)):
        for i in range(num_classes):
            label_gt = int(labels.split('_')[i])
            label_pred = 1 if preds[index][i] else 0

            if label_gt != label_pred:
                file_source = filename.replace(dir_preprocess, dir_original)
                file_dest = Path(dir_dest).joinpath(str(i), f'{str(label_gt)}_{str(label_pred)}',
                            f'prob{str(int(prob_total[index][i]*100))}_' + os.path.split(filename)[1])
                file_dest.parent.mkdir(parents=True, exist_ok=True)
                print(f'copy file to {file_dest}')
                shutil.copy(file_source, file_dest)

print('OK')

