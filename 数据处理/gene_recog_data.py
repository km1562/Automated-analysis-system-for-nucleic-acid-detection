# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/19 11:08
# @Author  : wkm
# @QQmail  : 690772123@qq.com


#切出照片，然后保存text
#按照类别切分，类别+计数（全局） ann
import json
import os

from pyxllib.xlcv import xlcv
from pycocotools.coco import COCO
from collections import defaultdict
from PIL import Image
import numpy
from pyxllib.algo.geo import xywh2ltrb

# def ConvertCocoBBox(pts):
#     LeftTopX, LeftTopy = pts[0], pts[1]
#     RightBott

def write_ann_txt(txt_file, img_file, text):
    with open(txt_file, 'a+') as f:
        f.writelines(img_file + '\t' + text + '\n')
        # f.writelines(img_file + ' ' + text + '\n')

def get_count_dict(coco):
    """
    返回类别，以及他们的计数，方便之后存取

    :param coco:
    :return:
    """
    category_dicts = coco.cats
    count_dict=defaultdict(int)
    for key, cat_dict in category_dicts.items():
        # if cat_dict['name'] == '其他类':
        #     continue
        # else:
        #     count_dict[cat_dict['name']] = 0
        count_dict[cat_dict['name']] = 0
    return count_dict

def save_img(coco):
    """
    传进来一个coco，
    把所有的ann_bbox框都裁切出来

    :param coco:
    :return:
    """
    img_to_anns = coco.imgToAnns
    for key, AnnsInImg in img_to_anns.items():  #单证图片包含的所有ann
        img_id = AnnsInImg[0]['image_id']
        os.chdir(img_dir)
        img_info = coco.loadImgs(img_id)
        ImgFileName =img_info[0]['file_name']

        try :
            im = Image.open(ImgFileName)
        except:
            print(f"{im.filename} has problem")

        for box_ann_dict in AnnsInImg:
            t = box_ann_dict['category']
            # img_id = box_ann_dict['image_id']
            if t in count_dict.keys():
                # pass
                # 拿到坐标
                pts = box_ann_dict['bbox']
                pts = xywh2ltrb(pts)
                pts = numpy.array(pts).reshape(2,2)

                try:
                    subim = xlcv.get_sub(im, pts)
                except:
                    print(f"{im} have problem")
                    continue

                #图片名字
                SaveCatDir = save_dir + t
                if not os.path.exists(SaveCatDir):
                    os.makedirs(SaveCatDir)

                subim = Image.fromarray(subim)
                BaseName = t + "_" + str(count_dict[t]) + '.jpg'
                SaveFileName = SaveCatDir +'/' + BaseName

                subim.save(SaveFileName)
                count_dict[t] += 1
                write_ann_txt(ann_txt_file, BaseName, box_ann_dict['text'])

gt_file = "/home/wengkangming/map_file/entrate_hesuan/data/entrate_data_368.json"
img_dir = "/home/wengkangming/map_file/entrate_hesuan/data/complete_packge_pic/"

save_dir = "/home/wengkangming/map_file/entrate_hesuan/data/reg_data/"
ann_txt_file = "/home/wengkangming/map_file/entrate_hesuan/data/reg_data/rec_gt_train.txt"

coco = COCO(gt_file)
count_dict = get_count_dict(coco)
save_img(coco)