import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from libs.NeuralNetworks.Helper.my_load_model import load_model


num_classes = 4

model_name = 'xception'
model_file = '/tmp2/wide_angel/v3/xception/epoch7.pth'
model = load_model(model_name, num_class=num_classes, model_file=model_file)

'''
model_name = 'inception_v3'
model_file = '/tmp2/wide_angel/v3/inception_v3/epoch8.pth'
model = load_model(model_name, num_class=num_classes, model_file=model_file)

model_name = 'inception_resnet_v2'
model_file = '/tmp2/wide_angel/v3/inception_resnet_v2/epoch8.pth'
model = load_model(model_name, num_class=num_classes, model_file=model_file)
'''

scripted_module = torch.jit.script(model)

# model.eval()
#import numpy as np
# inputs_test = torch.from_numpy(np.zeros((1, 3, 299, 299), dtype=float))
# store = torch.jit.trace(model, inputs_test)

model_file_saved = model_file.replace('.pth', '.pth_jit')
model_file_saved = '/tmp/aaa.pth'
torch.jit.save(scripted_module, model_file_saved)

print('model save completed.')

model = torch.jit.load(model_file_saved)

#torch serve