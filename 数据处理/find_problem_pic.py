import os

from pyxllib.xl import TicToc, XlPath, round_int, browser
from shutil import copy

# img_dir = '/home/wengkangming/map_file/entrate_hesuan/核酸结果、健康码、行程码_原数据_需要标注的数据_重新确认——22.6h_7'
img_dir = '/home/wengkangming/map_file/entrate_hesuan/complete_packge_pic'

data_root = XlPath(img_dir)
files = list(data_root.glob_images('**/*'))

count = 0

with open('/home/wengkangming/map_file/entrate_hesuan/data/有问题.txt', 'w') as f:
    for pic in files:
        if '有问题' in pic.name:
            f.writeline( pic.name + '\n')

print(f"total handle {count} picture")