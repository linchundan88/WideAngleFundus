import os
from libs.dataset.my_dataset import Dataset_CSV
import numpy as np
import torch
import pandas as pd
import sklearn

def gen_baselines(csv_file, label, image_shape, sample_size):

    df = pd.read_csv(csv_file)
    df = df[df['labels'] == label]
    assert len(df) > sample_size, 'the length of df should great then the sample size.'
    df = sklearn.utils.shuffle(df, random_state=22222)

    # add_black_interval = 16

    tensor_shape = list(image_shape)
    tensor_shape.insert(0, 3)
    tensor_shape.insert(0, 1)
    # tensor_shape = tuple(tensor_shape)
    # tensor_shape = (1, 3, 299, 299)

    # img_black = np.zeros(tensor_shape)

    ds = Dataset_CSV(csv_or_df=df, multi_labels=True, image_shape=image_shape)

    for i in range(sample_size):
        data1 = ds[i][0].cpu().numpy()
        data1 = np.expand_dims(data1, axis=0)

        # if (i % add_black_interval == 0):
        #     data1 = np.concatenate((data1, img_black), axis=0)

        if 'data_x' not in locals().keys():
            data_x = data1
        else:
            data_x = np.concatenate((data_x, data1), axis=0)

    data_x = torch.from_numpy(data_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_x = data_x.to(device)

    return data_x


if __name__ == "__main__":
    csv_file = os.path.join(os.path.abspath('../../..'),
                                 'datafiles', 'train_v1.csv')
    baselines = gen_baselines(csv_file, label='0_0_0', image_shape=(299, 299), sample_size=64)

    print(baselines.shape)

