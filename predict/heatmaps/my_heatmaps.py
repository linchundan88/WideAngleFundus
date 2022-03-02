'''a good article
https://gilberttanner.com/blog/interpreting-pytorch-models-with-captum
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from libs.neural_networks.models.my_load_model import load_model
from captum.attr import *
from captum.attr import visualization as viz
import pandas as pd
from libs.dataset.my_dataset import get_tensor
from libs.neural_networks.heatmaps.my_gen_baselines import gen_baselines
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4
thresholds = [0.5 for x in range(num_classes)]

dataset_split = 'train'
heatmap_type = 'LayerGradCam'  # GuidedBackprop, IntegratedGradients, LayerGradCam, DeepLiftShap, GradientShap

dir_dest_base = f'/disk1/share_8tb/广角眼底2021.04.12/results/heatmaps/'

#region load model, setting image_shape and activation function.
model_name = 'inception_v3'
if model_name == 'xception':
    activation = 'sigmoid'
    model_file = '/tmp2/wide_angel/v3/xception/epoch7.pth'
    model, image_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
if model_name == 'inception_v3':
    activation = 'sigmoid'
    model_file = '/tmp2/wide_angel/v3/inception_v3/epoch8.pth'
    model, image_shape = load_model(model_name, num_class=num_classes, model_file=model_file)
if model_name == 'inception_resnet_v2':
    activation = 'sigmoid'
    model_file = '/tmp2/wide_angel/v3/inception_resnet_v2/epoch8.pth'
    model,image_shape = load_model(model_name, num_class=num_classes, model_file=model_file)

if torch.cuda.device_count() > 0:
    model.to(device)

model = model.eval()
#endregion

dir_dest = os.path.join(dir_dest_base, heatmap_type, model_name, dataset_split)

train_type = 'wide_angle'
data_version = 'v4'
csv_file = os.path.join(os.path.abspath('../..'),
                'datafiles', data_version, f'{dataset_split}.csv')


if heatmap_type == 'GuidedBackprop':
    guidedBackprop = GuidedBackprop(model)
if heatmap_type == 'IntegratedGradients':
    integratedGradients = IntegratedGradients(model)
if heatmap_type == 'DeepLift':
    deepLift = DeepLift(model)
    baselines = gen_baselines(csv_file, label='0_0_0', image_shape=image_shape, sample_size=32)

use_baselines = False
if heatmap_type == 'DeepLiftShap':
    deepLiftShap = DeepLiftShap(model)
    if use_baselines:
        baselines = gen_baselines(csv_file, label='0_0_0', image_shape=image_shape, sample_size=32)

    # with torch.no_grad():
    #     outputs = model(baselines).squeeze()  #remove batch num
    #     if activation == 'sigmoid':
    #                 outputs = torch.sigmoid(outputs)
    # outputs = outputs.cpu().numpy()
    # preds = outputs > thresholds

if heatmap_type in ['GuidedBackprop', 'IntegratedGradients', 'DeepLift', 'DeepLiftShap']:
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

if heatmap_type == 'LayerGradCam':
    if model_name == 'xception':
        #https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/xception.py
        layer_gc = LayerGradCam(model, model.act4)
    if model_name == 'inception_v3':
        layer_gc = LayerGradCam(model, model.Mixed_7c)
    if model_name == 'inception_resnet_v2':
        layer_gc = LayerGradCam(model, model.conv2d_7b)


df = pd.read_csv(csv_file)
for index, row in df.iterrows():
    file_image = row['images']
    file_label = row['labels']
    tensor_x = get_tensor(file_image, image_shape=image_shape)
    tensor_x = tensor_x.to(device)
    with torch.no_grad():
        outputs = model(tensor_x).squeeze()  #remove batch num
        if activation == 'sigmoid':
                    outputs = torch.sigmoid(outputs)
    outputs = outputs.cpu().numpy()
    preds = outputs > thresholds

    # print(file_image, preds)

    for class_index in range(num_classes):
        if preds[class_index]:  # multi-label positive
            label = file_label.split('_')[class_index]
            file_dest_heatmap = os.path.join(dir_dest, str(class_index),
                                             label, os.path.split(file_image)[1])
            os.makedirs(os.path.dirname(file_dest_heatmap), exist_ok=True)
            print(file_dest_heatmap)

            if heatmap_type == 'GuidedBackprop':
                attribution = guidedBackprop.attribute(tensor_x, target=class_index)

            if heatmap_type == 'IntegratedGradients':
                attribution = integratedGradients.attribute(tensor_x, n_steps=50, target=class_index)

            if heatmap_type == 'DeepLift':
                if not use_baselines:
                    attribution = deepLift.attribute(tensor_x, target=class_index)
                else:
                    attribution = deepLift.attribute(tensor_x, baselines=baselines, target=class_index)

            if heatmap_type == 'DeepLiftShap':
                attribution = deepLiftShap.attribute(tensor_x, baselines=baselines, target=class_index)

            if heatmap_type in ['GuidedBackprop', 'IntegratedGradients', 'DeepLift', 'DeepLiftShap']:
                figure, subplot = viz.visualize_image_attr_multiple(attr=np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                                      original_image=None,
                                                      methods=["heat_map"],
                                                      signs=["positive"],
                                                      fig_size=(6, 6), use_pyplot=False,
                                                      cmap=default_cmap,
                                                      show_colorbar=False)

                figure.savefig(file_dest_heatmap)

                file_dest_heatmap = file_dest_heatmap.replace('.jpg', '_process.jpg')
                file_dest_heatmap = file_dest_heatmap.replace('.png', '_process.png')
                shutil.copy(file_image, file_dest_heatmap)

            if heatmap_type == 'LayerGradCam':
                attr = layer_gc.attribute(tensor_x, class_index, relu_attributions=True)
                upsampled_attr = LayerAttribution.interpolate(attr, image_shape, interpolate_mode='bilinear')
                cam = upsampled_attr.detach().cpu().numpy().squeeze()

                # cam = np.maximum(cam, 0)  # using relu_attributions=True
                cam = cam / np.max(cam)  # heatmap:0-1

                from matplotlib import pyplot as plt
                plt.axis("off")  # turns off axes
                # plt.axis("tight")  # gets rid of white border
                plt.imshow(cam, alpha=0.5, cmap='jet')
                # plt.show()

                #cam = cv2.applyColorMap(np.uint8(255 * grads), cv2.COLORMAP_JET)

                plt.savefig(file_dest_heatmap, bbox_inches='tight', pad_inches=0)
                plt.close()

                file_dest_heatmap = file_dest_heatmap.replace('.jpg', '_process.jpg')
                file_dest_heatmap = file_dest_heatmap.replace('.png', '_process.png')
                shutil.copy(file_image, file_dest_heatmap)

print('OK')



'''            
data = attribution.cpu().numpy()
data = np.squeeze(data, axis=0)
data = np.transpose(data, axes=(1, 2, 0))  #(C,H,W)  (H,W,C)
data = np.mean(data, -1)
abs_max = np.percentile(np.abs(data), 100)
abs_min = abs_max

cmap = 'seismic'
plt.axis('off')

plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.margins(0, 0)

plt.axis("off")  # turns off axes
plt.axis("tight")  # gets rid of white border
# plt.axis("image")  # square up the image instead of filling the "figure" space

plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
save_filename1 = 'aaa.jpg'
plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
plt.close()

'''

'''
if blend_original_image:
    plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    save_filename1 = list_images[i]
    plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
    plt.close()

    img_heatmap = cv2.imread(list_images[i])
    (tmp_height, tmp_width) = img_original.shape[:-1]
    img_heatmap = cv2.resize(img_heatmap, (tmp_width, tmp_height))
    img_heatmap_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.jpg'.format(i))
    cv2.imwrite(img_heatmap_file, img_heatmap)

    dst = cv2.addWeighted(img_original, 0.65, img_heatmap, 0.35, 0)
    img_blend_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_blend_{0}.jpg'.format(i))
    cv2.imwrite(img_blend_file, dst)

    # region create gif
    import imageio

    mg_paths = [img_original_file, img_heatmap_file, img_blend_file]
    gif_images = []
    for path in mg_paths:
        gif_images.append(imageio.imread(path))
    img_file_gif = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.gif'.format(i))
    imageio.mimsave(img_file_gif, gif_images, fps=gif_fps)
    list_images[i] = img_file_gif
    # endregion
else:
    plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    save_filename1 = 'aaa.jpg'
    plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
    plt.close()
'''
