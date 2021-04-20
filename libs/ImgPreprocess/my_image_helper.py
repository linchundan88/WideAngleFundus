
import numpy as np
import math
import cv2
import os


def resize_images_dir(source_dir, dest_dir, convert_image_to_square=False, image_shape=(299, 299)):
    if not source_dir.endswith('/'):
        source_dir += '/'
    if not dest_dir.endswith('/'):
        dest_dir += '/'

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)
            file_base, file_ext = os.path.splitext(image_file_source)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue
            img1 = cv2.imread(image_file_source)
            if img1 is None:
                print('error file:', image_file_source)
                continue

            if convert_image_to_square:
                img1 = image_to_square(img1)
            img1 = cv2.resize(img1, image_shape)

            image_file_dest = image_file_source.replace(source_dir, dest_dir)
            os.makedirs(os.path.dirname(image_file_dest), exist_ok=True)
            cv2.imwrite(image_file_dest, img1)
            print(image_file_source)

# 1.square, 2.resize
def image_to_square(image1, image_shape=None, grayscale=False):
    if isinstance(image1, str):
        image1 = cv2.imread(image1)

    height, width = image1.shape[0:2]

    if width > height:
        #original size can be odd or even number,
        padding_top = math.floor((width - height) / 2)
        padding_bottom = math.ceil((width - height) / 2)

        if image1.ndim == 3:
            image_channel = image1.shape[2]
            image_padding_top = np.zeros((padding_top, width, image_channel), dtype=np.uint8)
            image_padding_bottom = np.zeros((padding_bottom, width, image_channel), dtype=np.uint8)

            image1 = np.concatenate((image_padding_top, image1,image_padding_bottom), axis=0)

        if image1.ndim == 2:
            image_padding_top = np.zeros((padding_top, width), dtype=np.uint8)
            image_padding_bottom = np.zeros((padding_bottom, width), dtype=np.uint8)

            image1 = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)

    elif width < height:
        padding_left = math.floor((height - width) / 2)
        padding_right = math.ceil((height - width) / 2)

        if image1.ndim == 3:
            image_channel = image1.shape[2]
            image_padding_left = np.zeros((height, padding_left, image_channel), dtype=np.uint8)
            image_padding_right = np.zeros((height, padding_right, image_channel), dtype=np.uint8)

            image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)

        if image1.ndim == 2:
            image_padding_left = np.zeros((height, padding_left), dtype=np.uint8)
            image_padding_right = np.zeros((height, padding_right), dtype=np.uint8)

            image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)

    if image_shape is not None:
        # height, width = image1.shape[0:2]  #image1 is square now

        if image1.shape[0:2] != image_shape:
            image1 = cv2.resize(image1, (image_shape.shape[1], image_shape.shape[0]))

    if grayscale:
        #cv2.cvtColor only support unsigned int (8U, 16U) or 32 bit float (32F).
        # image_output = np.uint8(image_output)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    return image1

def image_border_padding(image1,
                         padding_top, padding_bottom, padding_left, padding_right):

    if image1.ndim == 2:
        image1 = np.expand_dims(image1, axis=-1)
    (height, width) = image1.shape[:-1]

    image_padding_top = np.zeros((padding_top, width, 3), dtype=np.uint8)
    image_padding_bottom = np.zeros((padding_bottom, width, 3), dtype=np.uint8)

    image1 = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)

    (height, width) = image1.shape[:-1]

    image_padding_left = np.zeros((height, padding_left, 3), dtype=np.uint8)
    image_padding_right = np.zeros((height, padding_right, 3), dtype=np.uint8)

    image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)

    return image1


def load_resize_images(image_files, image_shape=None, grayscale=False):
    list_image = []

    if isinstance(image_files, list):   # list of image files
        for image_file in image_files:
            image_file = image_file.strip()

            if grayscale:
                image1 = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            else:
                image1 = cv2.imread(image_file)

            try:
                if (image_shape is not None) and (image1.shape[:2] != image_shape[:2]):
                        image1 = cv2.resize(image1, image_shape[:2])
            except:
                raise Exception("Image shape error:" + image_file)

            if image1 is None:
                raise Exception("Invalid image:" + image_file)

            if image1.ndim == 2:
                image1 = np.expand_dims(image1, axis=-1)

            list_image.append(image1)
    else:
        if isinstance(image_files, str):
            if grayscale:
                image1 = cv2.imread(image_files, cv2.IMREAD_GRAYSCALE)
            else:
                image1 = cv2.imread(image_files)
        else:
            if grayscale and image_files.ndim == 3:
                image1 = cv2.cvtColor(image_files, cv2.COLOR_BGR2GRAY)
            else:
                image1 = image_files

        try:
            if (image_shape is not None) and (image1.shape[:2] != image_shape[:2]):
                image1 = cv2.resize(image1, image_shape[:2])
        except:
            raise Exception("Invalid image:" + image_files)

        if image1 is None:
            raise Exception("Invalid image:" + image_files)

        if image1.ndim == 2:
            image1 = np.expand_dims(image1, axis=-1)

        list_image.append(image1)

    return list_image