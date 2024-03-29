
import os
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from libs.dataset.my_dataset import Dataset_CSV
# from libs.neural_networks.helper.obsoleted_my_is_inception import is_inception_model


@torch.no_grad()
def predict_csv_single_model(model, filename_csv, image_shape,
                      activation, batch_size=64):

    # assert os.path.exists(filename_csv), "csv file not exist"
    assert filename_csv.exists(), "csv file not exist"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    dataset = Dataset_CSV(data_source=filename_csv, image_shape=image_shape, test_mode=True)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=4)
    list_probs = []
    for batch_idx, inputs in enumerate(data_loader):
        print('batch:', batch_idx)
        inputs = inputs.to(device)
        outputs = model(inputs)
        if activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        # if activation == 'softmax':
        #     outputs = torch.softmax(outputs, dim=1)
        # probabilities = F.softmax(outputs, dim=1).data
        list_probs.append(outputs.cpu().numpy())

    probs = np.vstack(list_probs)

    '''release memory It may not be necessary.
    del model
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()
    '''

    return probs


def predict_csv_multiple_model(dicts_models, filename_csv,
                             activation=None):
    prob_lists = []
    for dict_model in dicts_models:
        probs = predict_csv_single_model(dict_model['model'], filename_csv,
            dict_model['input_shape'], activation=activation,
            batch_size=dict_model['batch_size'])
        prob_lists.append(probs)

    sum_models_weights = 0
    for i, prob1 in enumerate(prob_lists):
        if 'model_weight' not in dicts_models[i]:
            model_weight = 1
        else:
            model_weight = dicts_models[i]['model_weight']

        if i == 0:
            prob_total = prob1 * model_weight
        else:
            prob_total += prob1 * model_weight

        sum_models_weights += model_weight

    prob_total /= sum_models_weights

    return prob_total, prob_lists


