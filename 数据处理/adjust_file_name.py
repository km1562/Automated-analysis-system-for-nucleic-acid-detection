import os
from pyxllib.xl import TicToc, XlPath, round_int, browser
from shutil import copy
import json

def get_json_data(json_path):
    # 获取json里面数据
    with open(json_path, 'r') as f:
        params = json.load(f)
        dict = params

    return dict

def write_json_data(dict, json_path1):
    with open(json_path1, 'w') as r:
        # 定义为写模式，名称定义为r

        json.dump(dict, r, ensure_ascii=False)
        # 将dict写入名称为r的文件中

img_dir = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/Summary/'

data_root = XlPath(img_dir)
files = list(data_root.glob_files('**/*'))

for json_file in files:
    shuffix = os.path.splitext(json_file)[-1]
    if shuffix == ".json":
        try:
            pic_gt = get_json_data(json_file)

            # 处理新的文件名字，json里面读取到的图片路径，换成 目录 + json + 修改后缀 .json变成.jpg
            imagePath = pic_gt["imagePath"]  # '08EDA01411823550F5D0610ACA9523B1.png'
            img_shuffix = os.path.splitext(imagePath)[-1] #'.png'
            json_file_name = str(json_file).split('/')[-1]
            new_pic_file_name = json_file_name.replace(".json", img_shuffix)
            print(f"new pic file name's {new_pic_file_name}")
            pic_gt["imagePath"] = new_pic_file_name
            write_json_data(pic_gt, json_file)
        except:
            print(f"waring open file failes, {json_file}", json_file)

# test_labelme_json = "/home/wengkangming/map_file/entrate_hesuan/核酸结果、健康码、行程码_标注好的数据/19级-hanjj完成-6h/9月1日返校（16上报+1人）/1922031124-陈伟哲-que/1922031124-陈伟哲-que6cd25f5dcae5e3995f40b3d62989a47.json"








# pic_gt = get_json_data(test_labelme_json)
#
#
# imagePath = pic_gt["imagePath"]
# (filepath, tempfilename) = os.path.split(imagePath)
# _, extension = os.path.splitext(tempfilename)
#
# json_file_name = str(test_labelme_json).split('/')[-1]
# pic_file_name = json_file_name.replace(".json", extension)
#
# pic_gt["imagePath"] = pic_file_name
#
# write_json_data(pic_gt, test_labelme_json, )