import os

from pyxllib.xl import TicToc, XlPath, round_int, browser
from shutil import copy

img_dir = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/Summary/'

data_root = XlPath(img_dir)
files = list(data_root.glob_files('**/*'))

def eli_dup_pic_name(file):
    for pic in files:
        #上一级目录和当前文件名生成新的文件名
        pic_name = str(pic).split('/')[-1]
        pic_parentdir_path = os.path.dirname(pic)
        pic_par_dir = pic_parentdir_path.split('/')[-1] #文件的目录

        #新的文件名
        new_pic_name = pic_par_dir + pic_name

        new_pic = str(pic).replace(pic_name, new_pic_name)
        #旧的文件名
        os.rename(pic, new_pic)

eli_dup_pic_name(files)

