import os

from pyxllib.xl import TicToc, XlPath, round_int, browser
from shutil import copy

# img_dir = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/training/'
img_dir = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/Summary/'
save_dir = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/Summary/Summary_pic'

# RecogDir = "/home/wengkangming/map_file/entrate_hesuan/data/reg_data/"
# SaveRecogDir = '/home/wengkangming/map_file/entrate_hesuan/data/reg_data/recog_complete_packge_pic'

def packImgDir(img_dir, save_dir):
    data_root = XlPath(img_dir)
    files = list(data_root.glob_images('**/*'))
    img_shuffix = ['.png', '.jpg', '.jpeg']

    count = 0
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for pic in files:
        if pic.suffix.lower() in img_shuffix:
            copy(pic, save_dir)
            count += 1
            print(f"{save_dir}")
        else:
            print(f"can't copy {pic.name}")

    print(f"total handle {count} picture")

# packImgDir(RecogDir, SaveRecogDir)
packImgDir(img_dir, save_dir)