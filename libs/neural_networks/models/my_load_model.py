

import torch
import torch.nn as nn
import timm

def load_model(model_name, num_class=2, model_file=None, get_image_shape=True):

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

    if get_image_shape:
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

        return model, image_shape
    else:
        return model