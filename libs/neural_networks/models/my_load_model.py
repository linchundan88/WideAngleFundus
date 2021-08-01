

import torch
import torch.nn as nn
import timm

def load_model(model_name, num_class=2, model_file=None):

    model = timm.create_model(model_name, pretrained=True)
    if model_name == 'inception_resnet_v2':
        num_filters = model.classif.in_features
        model.classif = nn.Linear(num_filters, num_class)
    elif 'efficientnet' in model_name:
        num_filters = model.classifier.in_features
        model.classifier = nn.Linear(num_filters, num_class)
    else:
        num_filters = model.fc.in_features
        model.fc = nn.Linear(num_filters, num_class)

    if model_file is not None:
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)

    # if torch.cuda.device_count() > 0:
    #     model.cuda()
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    return model