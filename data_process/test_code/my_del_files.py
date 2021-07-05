import os
from libs.DataPreprocess.my_compute_digest import calcSha1

list_sha1 = []
dir1 = '/disk1/share_8tb/2021.4.7第一个模型 错图/我分错的 非目标病变'
print('start computing sha1.')
for dir_path, _, files in os.walk(dir1):
    for f in files:
        image_file = os.path.join(dir_path, f)
        file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue

        # print(image_file)
        sha1 = calcSha1(image_file)
        list_sha1.append(sha1)

print('computing sha1 completed.')


dir2 = '/disk1/share_8tb/广角眼底2021.04.08/original'
for dir_path, _, files in os.walk(dir2):
    for f in files:
        image_file = os.path.join(dir_path, f)
        file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
        if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            continue
        sha1 = calcSha1(image_file)
        if(sha1 in list_sha1):
            print(f'remove file: {image_file}')
            os.remove(image_file)


print('OK')

