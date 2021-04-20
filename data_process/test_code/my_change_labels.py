import os
import shutil

dir1 = '/disk1/share_8tb/2021.4.9 新一模 错图/'
dir2 = '/disk1/share_8tb/广角眼底2021.04.08/original'

for label in ['格子样变性', '孔源性视网膜脱离', '视网膜破裂孔']:
    for dir_path, subpaths, files in os.walk(os.path.join(dir1, label)):
        for f in files:
            image_file = os.path.join(dir_path, f)
            _, filename = os.path.split(image_file)
            file_base, file_ext = os.path.splitext(filename)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            #remove prefix:prob22_    prob22_00566485 - 20190813 @ 145928 - R1.jpg
            if filename.startswith('prob'):
                filename = ''.join(filename.split('_')[1:])

            if '/0-1' in dir_path:
                filename_dest = os.path.join(dir2, label, filename)
                if not os.path.exists(filename_dest):
                    print(f'copy file:{filename_dest}')
                    shutil.copy(image_file, filename_dest)

            if '/1-0' in dir_path:
                filename_dest = os.path.join(dir2, label, filename)
                if os.path.exists(filename_dest):
                    print(f'remove :{filename_dest}')
                    os.remove(filename_dest)


            #os.path.join(dir2, label)