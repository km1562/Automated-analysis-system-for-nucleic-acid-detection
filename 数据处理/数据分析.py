# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 11:00
# @Author  : wkm
# @QQmail  : 690772123@qq.com

from pycocotools.coco import COCO

# dataDir='/path/to/your/cocoDataset'
# dataType='val2017'
annFile = '/home/wengkangming/map_file/entrate_hesuan/data/paper_data/Summary/summary.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cat_nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))

# 统计各类的图片数量和标注框数量
for cat_name in cat_nms:
    catId = coco.getCatIds(catNms=cat_name)
    imgId = coco.getImgIds(catIds=catId)
    annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)

    print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))
