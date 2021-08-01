
import os
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
from libs.dataset.my_dataset import Dataset_CSV
from imgaug import augmenters as iaa

print(os.path.abspath('.'))

csv_file_test = os.path.join(os.path.abspath('.'),
                'test.csv')

iaa = iaa.Sequential([
                    iaa.flip.Fliplr(p=0.5),
                    iaa.flip.Flipud(p=0.5),
                    iaa.GaussianBlur(sigma=(0.0, 0.1)),
                    iaa.MultiplyBrightness(mul=(0.65, 1.35)),
                ])
custom_ds = Dataset_CSV(csv_or_df=csv_file_test, imgaug_iaa=iaa, image_shape=[299, 299])
# custom_dl = DataLoader(custom_ds, batch_size=1,
#                        num_workers=1, pin_memory=True)

image_tensor, label = custom_ds[2]
image_np = image_tensor.numpy() * 255
image_np = image_np.astype('uint8')
image_np = image_np.transpose(1, 2, 0)

import cv2
cv2.imwrite('/tmp6/abcdefg.jpg', image_np)
print('OK')

exit(0)

import torch
train_loader = torch.utils.data.DataLoader(custom_ds, batch_size=4,
                                          shuffle=False, num_workers=2)
for x, y in train_loader:
    print('a')

