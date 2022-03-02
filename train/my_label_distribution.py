import pandas as pd
from pathlib import Path
import numpy as np

class_num = 4
list_labels = [[] for i in range(class_num)]

train_type = 'wide_angle'
data_version = 'v5'

csv_all = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'all.csv')
csv_train = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'train.csv')
csv_valid = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'valid.csv')
csv_test = Path(__file__).parent.parent.absolute().joinpath('datafiles', data_version, 'test.csv')


df = pd.read_csv(csv_all)

for index, row in df.iterrows():
    list_tmp = row['labels'].split('_')
    for index, label in enumerate(list_tmp):
        list_labels[index].append(int(label))

for i in range(class_num):
    list_np = np.array(list_labels[i])
    print(f'class no:{i}')
    print(f'label 0:{np.sum(list_np == 0)}, label 0:{np.sum(list_np == 1)}')


print('OK')

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
