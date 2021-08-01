import os
import sys
sys.path.append(os.path.abspath('..'))

dir_original = '/disk1/share_8tb/广角眼底2021.04.12/original/'
dir_preprocess = '/disk1/share_8tb/广角眼底2021.04.12/preprocess/384/'

#prob97#10188519 01240793-20200831@153002-R2
for dir_path, _, files in os.walk(dir_original):
    for f in files:
        file_source = os.path.join(dir_path, f)
        file_base, file_ext = os.path.splitext(file_source)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue
        if f.startswith('prob'):
            list_s = f.split('#')
            assert len(list_s) == 2, 'error'
            list_s = list_s[1:]
            tuple_s = tuple(list_s)
            file_dest = os.path.join(dir_path, ''.join(tuple_s))

            print(f'rename from {file_source} to {file_dest}')
            os.rename(file_source, file_dest)


from libs.img_preprocess.my_image_helper import resize_images_dir
resize_images_dir(dir_original, dir_preprocess, image_shape=(384, 384),
                  convert_image_to_square=True)


print('OK')

