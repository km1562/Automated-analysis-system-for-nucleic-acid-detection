# from pyxllib.data.coco import Coco2Labelme, CocoGtData
# from pyxllib.xl import TicToc, XlPath, round_int, browser

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyxlpr.data.labelme import LabelmeDataset
from pyxlpr.data.coco import CocoGtData
import json

#转为coco

#读取每一张图片

# img_path = "/home/wengkangming/map_file/entrate_hesuan/data/paper_data/training/"
img_path = "/home/wengkangming/map_file/entrate_hesuan/data/paper_data/Summary/"

# {'id': 0, 'name': '姓名', 'supercategory': ''} ,
#         {'id': 1, 'name': '身份证', 'supercategory': ''},
#         {'id': 2, 'name': '联系方式', 'supercategory': ''},
#         {'id': 3, 'name': '采样时间', 'supercategory': ''},
#         {'id': 4, 'name': '检测时间', 'supercategory': ''},
#         {'id': 5, 'name': '核酸结果', 'supercategory': ''},
#         {'id': 6, 'name': '14天经过或途经', 'supercategory': ''},
#         {'id': 7, 'name': '健康码颜色', 'supercategory': ''},


#生成字典的categories_list
categories_list = ['姓名', '身份证', '联系方式', '采样时间', '检测时间', '核酸结果','14天经过或途经', '健康码颜色', '其他类']
categories_dict = CocoGtData.gen_categories(categories_list) #{{'id': 1, 'name': '姓名', 'supercategory': ''}, ...}

#直接生成coco格式的gt, 但是category都是默认0
labelme_to_coco = LabelmeDataset(img_path)
coco_gt_dict = labelme_to_coco.to_coco_gt_dict(
    categories_dict
)

#更新一下coco的category_id
category2id = {
    x['name']: x['id'] for x in categories_dict
}

annotations = coco_gt_dict["annotations"]

for ann_dict in annotations:
    ann_dict_cat = ann_dict["category"]
    cat_id = category2id[ann_dict_cat]
    ann_dict['category_id'] =cat_id
    # ann_dict["category"] =

coco_gt_dict["annotations"] = annotations

# save_path = '/home/wengkangming/map_file/entrate_hesuan/entrate_data_368.json'
# save_path = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/traing.json'
save_path = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/Summary/summary.json'
with open(save_path, "w") as f:
    json.dump(coco_gt_dict, f, ensure_ascii=False)
