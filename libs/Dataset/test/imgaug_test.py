import os
import cv2
from imgaug import augmenters as iaa

imgaug_iaa = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    # iaa.Fliplr(0.5),  # horizontally flip 50% of the images

    iaa.GaussianBlur(sigma=(0.0, 1)),
    # iaa.MultiplyBrightness(mul=(0.7, 1.3)),
    # iaa.contrast.LinearContrast((0.7, 1.3)),
    # iaa.Sometimes(0.9, iaa.Add((-8, 8))),
    # iaa.Add((-8, 8))
    # iaa.Sometimes(0.9, iaa.Affine(
    #     scale=(0.98, 1.02),
    #     translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
    #     rotate=(-10, 10),  # rotate by -10 to +10 degrees
    # )),
])


img_file = '/media/ubuntu/data2/3D_OCT_DME_ALL/preprocess/128_128_128/TOPOCON/累及中央的黄斑水肿/02-000035_20160607_091454_OPT_R_001/i_6.png'
dest_dir = '/tmp7/a'
os.makedirs(dest_dir, exist_ok=True)

assert os.path.exists(img_file), 'file not found!'
img1 = cv2.imread(img_file)
cv2.imwrite(os.path.join(dest_dir, 'original.jpg'), img1)
for i in range(50):
    img2 = imgaug_iaa(image=img1)
    cv2.imwrite(os.path.join(dest_dir, f'{i}.jpg'), img2)

print('OK')