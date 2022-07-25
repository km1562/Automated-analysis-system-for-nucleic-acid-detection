#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 厦门理工计算机学院 王大寒/(PRIU)福建省模式识别与图像理解重点实验室
# @Document : https://www.yuque.com/code4101/python/hesuan
# @Email  : (陈)877362867@qq.com
# @Address: 厦门理工综合楼1905
# @Date   : 2022/04/16

import os
import re
import datetime
import sys
import time

import pandas as pd
# import fire
from tqdm import tqdm
import numpy as np

# from PyQt5.QtCore import QSize, Qt, QUrl
# from PyQt5.QtGui import QFont, QIcon, QDesktopServices
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QGridLayout, QWidget,
#                              QLineEdit, QToolButton, QFileDialog, QProgressBar)

# pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
import layoutparser as lp

from pyxllib.xl import TicToc, XlPath, round_int, browser, SingletonForEveryClass, matchpairs
from pyxllib.xlcv import xlcv
from pyxllib.gui.qt import get_input_widget
from pyxllib.file.xlsxlib import openpyxl

from pyxlpr.paddleocr import PaddleOCR
from pyxlpr.data.imtextline import TextlineShape

# 使用显卡的时候，windows运行可能有问题：OMP: Error #15: Initializing libiomp5md.dll，需改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_GPU = True

HESUAN_PATH = '/home/chenkunze/data/hesuan'


class PPOCR:
    def __init__(self, *, ppocr=True, pplp=True):
        if ppocr:
            recdir = f'{HESUAN_PATH}/resources/hesuan_ocr_v6_final'
            self.ppocr = PaddleOCR.build_ppocr(use_gpu=USE_GPU, rec_model_dir=recdir,
                                               rec_char_dict_path=f'{recdir}/char_dict.txt')
        if pplp:
            # 使用版面分析功能，直接提取出关键信息，这样后处理会更简单
            self.pplp = lp.PaddleDetectionLayoutModel(model_path=f"{HESUAN_PATH}/resources/hesuan_det_v4",
                                                      threshold=0.5,
                                                      label_map={0: "其他类", 1: "姓名", 2: "身份证号", 3: "联系电话",
                                                                 4: "采样时间", 5: "检测时间", 6: "核酸结果"},
                                                      enforce_cpu=not USE_GPU,  # 有gpu显卡会自动调用，能大大提速
                                                      enable_mkldnn=True)

    def supplement_attrs(self, attrs, texts):
        """ 可能会有没见过的模板数据，如果一些字段不存在，则扩展一些手段自动找值
        """

        # 1 通过关键词
        def find_by_key(attr, keys):
            for i in range(len(texts) - 1):
                t = texts[i]
                for k in keys:
                    if k in t:
                        attrs[attr] = texts[i + 1]
                        return

        for t, keys in {'姓名': ['姓名'],
                        '身份证号': ['证件号码', '身份证号码'],
                        '联系电话': ['联系电话'],
                        '采样时间': ['采样时间'],
                        '检测时间': ['检测时间', '报告时间'],
                        '核酸结果': ['检测结果']}.items():
            if t not in attrs:
                find_by_key(t, keys)

        # 2 通过正则分析文本
        def find_by_patter(attr, patter):
            if attr in attrs:
                return
            for t in texts:
                m = re.search(patter, t)
                if m:
                    attrs[attr] = m.group()
                    return

        find_by_patter('身份证号', r'\d{2}[\d\*]{14}\d{2}')
        find_by_patter('联系电话', r'\d{2}[\d\*]{7}\d{2}')
        find_by_patter('采样时间', r'20\d{2}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?')
        find_by_patter('检测时间', r'20\d{2}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?')

        if '核酸结果' not in attrs:
            for t in texts:
                if '阴性' in t:
                    attrs['核酸结果'] = '阴性'
        if '核酸结果' not in attrs:
            for t in texts:
                if '阳性' in t:
                    attrs['核酸结果'] = '阳性'

        return attrs

    def refine_attrs(self, attrs):
        """ 对识别出来的文本内容进行优化 """

        def refine_date(x):
            x = re.sub(r'^(\d{3}).(\d{2})\.(\d{2})', r'2\1-\2-\3', x)  # 有的开头漏识别一个2
            x = re.sub(r'(\d{4}).(\d{2})\.(\d{2})', r'\1-\2-\3', x)  # 改成-连接符
            x = re.sub(r'(\d+-\d+-\d{2})(\S)', r'\1 \2', x)
            return x

        if '采样时间' in attrs:
            attrs['采样时间'] = refine_date(attrs['采样时间'])
        if '检测时间' in attrs:
            attrs['检测时间'] = refine_date(attrs['检测时间'])
        if '核酸结果' in attrs:
            if '阴' in attrs['核酸结果']:
                attrs['核酸结果'] = '阴性'
            if '阳' in attrs['核酸结果']:
                attrs['核酸结果'] = '阳性'
        if '联系电话' in attrs:
            attrs['联系电话'] = re.sub(r'[^\d\*]+', '', attrs['联系电话'])  # 删除非数字内容
        if '身份证号' in attrs:
            attrs['身份证号'] = re.sub(r'[^\d\*X]+', '', attrs['身份证号'])  # 删除非数字内容
            attrs['身份证号'] = re.sub(r'X+(?!X)', '', attrs['身份证号'])  # 不是末尾的X删除
        return attrs

    def parse_det(self, file):
        """ 基于普通检测识别模型做的框架 """
        texts = self.ppocr.ocr2texts(file, True)
        attrs = {}
        self.supplement_attrs(attrs, texts)
        self.refine_attrs(attrs)
        return attrs

    def _parse_layout(self, file):
        # 调用版面分析模型获得类别
        image = xlcv.read(file, 1)[..., ::-1]
        lay = self.pplp.detect(image)
        blocks = lay.to_dict()['blocks']
        for block in blocks:
            block['ltrb'] = [round_int(block[x]) for x in ['x_1', 'y_1', 'x_2', 'y_2']]
        blocks.sort(key=lambda x: TextlineShape(x['ltrb']))
        return image, blocks

    def parse_layout(self, file):
        """ 基于版面分析做的信息提取框架
        """
        # 1 识别
        image, blocks = self._parse_layout(file)
        ls = []
        columns = ['type', 'text', 'score', 'points']
        for block in blocks:
            t = block['type']
            im = xlcv.get_sub(image, block['ltrb'])  # 识别每个框里的文本内容
            text, score = self.ppocr.rec_singleline(im)
            ls.append([t, text, round((block['score'] + score) / 2, 4), block['ltrb']])
        # 按照几何关系，从上到下，从左到右排列识别出来的文本框
        df = pd.DataFrame.from_records(ls, columns=columns)
        # browser(df)

        # 2 优化结果
        attrs = {}
        for idx, row in df.iterrows():
            k, v = row['type'], row['text']
            if k == '其他类':
                continue
            if k not in attrs:
                attrs[k] = v

        self.supplement_attrs(attrs, df['text'])
        self.refine_attrs(attrs)
        return attrs

    def show_layout(self, file):
        """ 可视化查看效果 """
        image = xlcv.read(file, 1)
        layout = self.pplp.detect(image[..., ::-1])
        im = lp.draw_box(image, layout, box_width=3, show_element_type=True)
        im.show()

    def parse_light(self, file):
        """ 因为rec比较慢，可以配合layout做过滤，这是快速版本的解析实现 """
        image, blocks = self._parse_layout(file)  # 版面分析基本识别
        attrs = {}
        for block in blocks:
            t = block['type']
            if t == '其他类' or t in attrs:
                continue
            im = xlcv.get_sub(image, block['ltrb'])  # 识别每个框里的文本内容
            text, score = self.ppocr.rec_singleline(im)
            attrs[t] = text

        self.refine_attrs(attrs)  # 后处理
        return attrs

    def parse_online(self):
        """ 使用联网api进行识别
        :return:
        """
        raise NotImplementedError


# 得到图片

USE_GPU = True
HESUAN_PATH = "../../核酸结果自动汇总统计工具v1.0/resources/hesuan_det_v4"

img_path = "/home/wengkangming/map_file/entrate_hesuan/核酸结果、健康码、行程码_原数据"

data_root = XlPath(img_path)
dir_names = data_root.glob_dirs('*')

model = PPOCR()

for dir in dir_names:
    for pic in dir.glob_images('*'):
        pic_result = model.parse_layout(pic)
        pass

# 送入ppocr

# 得到结构