import os
import re
import datetime
import sys
import time

import pandas as pd
import fire
from tqdm import tqdm
import numpy as np

from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QFont, QIcon, QDesktopServices
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QGridLayout, QWidget,
                             QLineEdit, QToolButton, QFileDialog, QProgressBar, QMessageBox)

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

HESUAN_PATH = r"D:\BaiduSyncdisk\map_file\entrate_hesuan\sanma"


# HESUAN_PATH = "/home/wengkangming/map_file/entrate_hesuan/三码合一入学检测/"
# HESUAN_PATH = r"D:\BaiduSyncdisk\map_file\entrate_hesuan\核酸结果自动汇总统计工具v1.0"

class PPOCR:
    def __init__(self, label_map, model_dir='task1', *, ppocr=True, pplp=True):
        # TODO, 加载不同的权重
        if ppocr:
            # recdir = rf"{HESUAN_PATH}/resources/hesuan_ocr_v6_final"
            # recdir = rf"{HESUAN_PATH}\resources\{model_dir}\ocr_include_other"
            recdir = rf"{HESUAN_PATH}\resources\{model_dir}\hesuan_ocr_v6_final"
            self.ppocr = PaddleOCR.build_ppocr(use_gpu=USE_GPU, rec_model_dir=recdir)
            # self.ppocr = PaddleOCR.build_ppocr(use_gpu=USE_GPU)

        if pplp:
            # 使用版面分析功能，直接提取出关键信息，这样后处理会更简单
            # self.pplp = lp.PaddleDetectionLayoutModel(model_path=rf"{HESUAN_PATH}/resources/layout_model",
            self.pplp = lp.PaddleDetectionLayoutModel(model_path=rf"{HESUAN_PATH}\resources\{model_dir}\layout_model",
            # self.pplp = lp.PaddleDetectionLayoutModel(model_path=rf"{HESUAN_PATH}\resources\{model_dir}\cascade_rcnn",
                                                      # self.pplp = lp.PaddleDetectionLayoutModel(model_path=rf"{HESUAN_PATH}\resources\hesuan_det_v4",
                                                      threshold=0.5,
                                                      label_map=label_map,
                                                      enforce_cpu=not USE_GPU,  # 有gpu显卡会自动调用，能大大提速
                                                      enable_mkldnn=True,
                                                      )

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

        # TODO,好像有两码的,这里也可以全文搜索
        find_by_patter('14天经过或途经', r'.*省')
        find_by_patter('14天经过或途经', r'.*市')

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

            x = re.sub(r'(\(|:)?(\d{4}).?(\d{2})(\.|/|-)?(\d{2})(\s\))?', r'\2-\3-\5', x)  # 改成-连接符,开头和结尾可能有括号
            x = re.sub(r'(\(|:)?(\d{4}).?(\d{1})(\.|/|-)(\d{1})(\s\))?', r'\2-0\3-0\5', x)  #月跟日可能存在一位，需要补0
            x = re.sub(r'(\(|:)?(\d{4}).?(\d{1})(\.|/|-)(\d{2})(\s\))?', r'\2-0\3-\5', x)  #月可能存在一位, 日是两位，日不补0

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
        if '14天经过或途经' in attrs:
            province_list = ["河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东",
                             "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾"]
            non_province_list = ["内蒙古自治区", "广西壮族自治区", "西藏自治区", "宁夏回族自治区", "新疆维吾尔自治区", "北京市", "天津市", "上海市", "重庆市",
                                 "香港特别行政区", "澳门特别行政区"]
            names = set()
            for t in province_list:  # 可以扩展
                names |= set(re.findall(t + '省.+?市', attrs["14天经过或途经"]))

            for t in non_province_list:
                names |= set(re.findall(t, attrs["14天经过或途经"]))
            attrs["14天经过或途经"] = ",".join(names)

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

        for block in blocks:
            t = block['type']
            im = xlcv.get_sub(image, block['ltrb'])  # 识别每个框里的文本内容
            text, score = self.ppocr.rec_singleline(im)
            ls.append([t, text, round((block['score'] + score) / 2, 4), block['ltrb']])
        # 按照几何关系，从上到下，从左到右排列识别出来的文本框

        return ls

    def get_attrs(self, ls):
        """
        后处理结果
        """
        columns = ['type', 'text', 'score', 'points']
        df = pd.DataFrame.from_records(ls, columns=columns)
        # 2 优化结果
        attrs = {}
        for idx, row in df.iterrows():
            k, v = row['type'], row['text']
            if k == '其他类':
                continue
            if k not in attrs:
                attrs[k] = v
            elif k == '14天经过或途经':  # '14天途径在里面了，全部加起来
                attrs[k] += v

        self.supplement_attrs(attrs, df['text'])
        self.refine_attrs(attrs)
        return attrs

    def parse_multi_layout(self, file_list):
        """
        多张图片的版面分析
        """
        res_ls = []
        for file in file_list:
            res_ls += self.parse_layout(file)

        attrs = self.get_attrs(res_ls)
        return attrs

    def show_layout(self, file):
        """ 可视化查看效果 """
        image = xlcv.read(file, 1)
        layout = self.pplp.detect(image[..., ::-1])
        im = lp.draw_box(image, layout, box_width=3, show_element_type=True)
        im.show()

    def parse_multi_layout_light(self, file_list):
        attrs = {}
        for file in file_list:
            attrs = self.parse_light(file, attrs)
        self.refine_attrs(attrs)  # 后处理
        return attrs

    def parse_light(self, file, attrs):
        """ 因为rec比较慢，可以配合layout做过滤，这是快速版本的解析实现 """
        image, blocks = self._parse_layout(file)  # 版面分析基本识别
        for block in blocks:
            t = block['type']
            if t != '14天经过或途经' and (t == '其他类' or t in attrs):  # 必须不等于14天，然后重复了或者为其他类，才可以跳过
                continue
            im = xlcv.get_sub(image, block['ltrb'])  # 识别每个框里的文本内容
            text, score = self.ppocr.rec_singleline(im)

            if t == '14天经过或途经':
                if t in attrs.keys():
                    attrs[t] += text
                else:
                    attrs[t] = text
            else:
                attrs[t] = text
        return attrs

    def parse_online(self):
        """ 使用联网api进行识别
        :return:
        """
        raise NotImplementedError


class Hesuan(metaclass=SingletonForEveryClass):
    """ 核酸检测的各种功能接口 """

    def __init__(self, task):
        self.task = task
        # self.columns = columns
        # self.stu_colunms = stu_solunms
        # self.file_num = file_num
        if self.task == '日常核酸检测报告分析':
            self.columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '身份证号']
            self.stu_columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '身份证号', '联系电话']
            self.file_num = 1
            self.model_dir = 'task1'
            self.label_map = {0: "其他类", 1: "姓名", 2: "身份证号", 3: "联系电话", 4: "采样时间", 5: "检测时间", 6: "核酸结果"}
            self.model_dir = 'task1'
        elif self.task == '双码（健康码、行程码）分析':
            self.columns = ['文件', '姓名', '联系电话', '14天经过或途经', '健康码颜色']
            self.stu_columns = ['班级', '文件', '学号', '姓名', '联系电话', '14天经过或途经', '健康码颜色']
            self.file_num = 2
            self.model_dir = 'task2'
            self.label_map = {0: "姓名", 1: "身份证号", 2: "联系方式", 3: "采样时间",
                              4: "检测时间", 5: "核酸结果", 6: "14天经过或途经", 7: "健康码颜色",
                              8: "其他类"}
            self.model_dir = 'task2'
        elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
            self.columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '14天经过或途经', '健康码颜色', '身份证号']
            self.stu_columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '14天经过或途经', '健康码颜色', '身份证号',
                                '联系电话']
            self.file_num = 3
            self.model_dir = 'task3'
            self.label_map = {0: "姓名", 1: "身份证号", 2: "联系方式", 3: "采样时间", 4: "检测时间", 5: "核酸结果", 6: "14天经过或途经", 7: "健康码颜色", 8: "其他类"}

        self.ppocr = PPOCR(label_map=self.label_map, model_dir=self.model_dir)

    def ensure_images(self, imdir):
        """ 一些非图片个数数据自动转图片

        其中压缩包只支持zip格式
        """
        # 1 先解压所有的压缩包
        for f in imdir.rglob_files('*.zip'):
            from pyxllib.file.packlib import unpack_archive
            # 没有对应同名目录则解压
            if not f.with_name(f.stem).is_dir():
                unpack_archive(f, wrap=1)

        # 2 再检索所有的pdf
        for f in imdir.rglob_files('*'):
            # 可能存在pdf格式的文件，将其转成jpg图片
            suffix = f.suffix.lower()
            if suffix == '.pdf':
                f2 = f.with_suffix('.jpg')
                if not f2.exists():
                    from pyxllib.file.pdflib import FitzDoc
                    # im = FitzDoc(f).load_page(0).get_pil_image()
                    f = FitzDoc(f)
                    im = f.load_page(0).get_pil_image()
                    im.save(f2)

    def similar(self, x, y):
        """ 计算两条数据间的相似度

        这里不适合用编辑距离，因为levenshtein涉及到c++编译，会让配置变麻烦。
        这里就用正则等一些手段计算相似度。

        相关字段：姓名(+文件)、联系电话、身份证号
        """
        t = 0
        if y['姓名'] in x['文件']:
            t += 100
        if y['姓名'] == x['姓名']:
            t += 200
        elif re.match(re.sub(r'\*+', r'.*', x['姓名']) + '$', y['姓名']):
            t += 50

        def check_key_by_asterisk(k):
            # 带星号*的相似度匹配
            nonlocal t
            if isinstance(y[k], str):
                if y[k] == x[k]:
                    t += 100
                else:
                    if y[k][-2:] == x[k][-2:]:
                        t += 20
                    if y[k][:2] == x[k][:2]:
                        t += 10

        check_key_by_asterisk('联系电话')
        check_key_by_asterisk('身份证号')
        return t

    def link_table(self, df1, df2):
        # 1 找出df1中每一张图片，匹配的是df2中的谁
        idxs = []
        for idx1, row1 in df1.iterrows():
            max_idx2, max_sim = -1, 0
            for idx2, row2 in df2.iterrows():
                sim = self.similar(row1, row2)
                if sim > max_sim:
                    max_idx2 = idx2
                    max_sim = sim
            if max_idx2 == -1:
                # 如果没有匹配到，到时候要直接展示原来的数据
                idxs.append(-idx1)
            else:
                idxs.append(max_idx2)

        for k in ['姓名', '联系电话', '身份证号']:
            df1[k] = [(df2[k][i] if i > 0 else df1[k][-i]) for i in idxs]
        for k in ['班级', '学号']:
            df1[k] = [(df2[k][i] if i > 0 else '') for i in idxs]

        # 2 列出涉及到的班级的成员名单
        classes = set(df1['班级'])  # 0.7版是只显示图片涉及到的班级
        # classes = set(df2['班级'])  # 0.9版是直接以清单中的情况显示
        idxs_ = set(idxs)
        for idx2, row2 in df2.iterrows():
            if idx2 not in idxs_ and row2['班级'] in classes:
                line = {}
                for k in ['文件', '采样时间', '检测时间', '核酸结果']:
                    line[k] = ''
                for k in ['姓名', '联系电话', '身份证号', '班级', '学号']:
                    line[k] = row2[k].strip()
                df1 = df1.append(line, ignore_index=True)

        # 3 重新调整列
        # 保护隐私，不显示电话、身份证
        df = df1[['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果']]
        df.sort_values(['班级', '学号'], inplace=True)
        df.index = np.arange(1, len(df) + 1)

        return df

    def link_table2(self, df1, df2):
        # 1 匹配情况
        xs = [row for idx, row in df1.iterrows()]
        ys = [row for idx, row in df2.iterrows()]
        ms = matchpairs(xs, ys, self.similar, least_score=40, index=True)
        idxs = {m[1]: m[0] for m in ms}

        # + 辅助函数
        def quote(s):
            # 带上括号注释
            return f'({s})' if s else ''

        # df2如果有扩展列，也显示出来
        custom_cols = []
        for c in df2.columns:  # 按照原表格的列顺序展示
            c = str(c)
            if c not in {'班级', '学号', '姓名', '身份证号', '联系电话'}:
                custom_cols.append(c)

        def extend_cols(y=None):
            if y is None:
                return [''] * len(custom_cols)
            else:
                return [('' if y[c] != y[c] else y[c]) for c in custom_cols]

        # 2 模板表清单
        ls = []
        # columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '身份证号', '联系电话'] + custom_cols
        # colunms = ['班级', '文件', '学号', '姓名', '联系电话', '14天经过或途经', '健康码颜色'] + custom_cols
        # columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '14天经过或途经', '健康码颜色', '身份证号', '联系电话'] + custom_cols
        columns = self.stu_columns + custom_cols
        for idx, y in df2.iterrows():
            i = idxs.get(int(idx), -1)
            if self.task == '日常核酸检测报告分析':
                record = [y['班级'], '', y['学号'], y['姓名'], '', '', '', y['身份证号'], y['联系电话']] + extend_cols(y)
            elif self.task == '双码（健康码、行程码）分析':
                record = [y['班级'], '', y['学号'], y['姓名'], y['联系电话'], '', '', ] + extend_cols(y)
            elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
                record = [y['班级'], '', y['学号'], y['姓名'], '', '', '', '', '', y['身份证号'], y['联系电话']] + extend_cols(y)
            else:
                raise NotImplementedError

            if i != -1:  # 找得到匹配项
                x = xs[i]
                if self.task == '日常核酸检测报告分析':
                    record[1] = x['文件']
                    record[3] += quote(x['姓名'])
                    record[4] = x['采样时间']
                    record[5] = x['检测时间']
                    record[6] = x['核酸结果']
                    record[7] += quote(x['身份证号'])
                    record[8] += quote(x['联系电话'])
                elif self.task == '双码（健康码、行程码）分析':
                    record[1] = x['文件']
                    record[3] += quote(x['姓名'])
                    record[4] += quote(x['联系电话'])
                    record[5] = x['14天经过或途经']
                    record[6] = x['健康码颜色']
                elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
                    record[1] = x['文件']
                    record[3] += quote(x['姓名'])
                    record[4] = x['采样时间']
                    record[5] = x['检测时间']
                    record[6] = x['核酸结果']
                    record[7] = x['14天经过或途经']
                    record[8] = x['健康码颜色']
                    record[9] += quote(x['身份证号'])
                    record[10] += quote(x['联系电话'])

            ls.append(record)

        # 3 df1中剩余未匹配图片加在最后面
        idxs = {m[0] for m in ms}
        for i, x in enumerate(xs):
            if i not in idxs:
                if self.task == '日常核酸检测报告分析':
                    record = ['', x['文件'], '', quote(x['姓名']),
                              x['采样时间'], x['检测时间'], x['核酸结果'],
                              quote(x['身份证号']), quote(x['联系电话'])]
                elif self.task == '双码（健康码、行程码）分析':
                    record = ['', x['文件'], '', quote(x['姓名']),
                              x['14天经过或途经'], x['健康码颜色']]
                elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
                    record = ['', x['文件'], '', quote(x['姓名']),
                              x['采样时间'], x['检测时间'], x['核酸结果'], x['14天经过或途经'], x['健康码颜色'],
                              quote(x['身份证号']), quote(x['联系电话']), ]
                else:
                    raise NotImplementedError

                ls.append(record + extend_cols())

        df = pd.DataFrame.from_records(ls, columns=columns)
        df.index = np.arange(1, len(df) + 1)
        return df

    # def parse(self, imdir, students=None, *, parse_mode='light', pb=None):
    def parse(self, imdir, students=None, *, parse_mode='multi_layout_light', pb=None):
        """
        :param imdir: 图片所在目录
        :param students: 【可选】参考学生清单
        :param pb: qt的进度条控件
        """
        # 1 基本的信息解析
        parser = getattr(self.ppocr, 'parse_' + parse_mode)  # 核心入口
        ls = []
        # columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '14天经过或途经', '健康码颜色', '身份证号']
        # columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '身份证号']

        imdir = XlPath(imdir)
        self.ensure_images(imdir)

        # files = list(imdir.glob_images('**/*'))
        img_lists = self.get_img_fileslist(imdir, self.file_num)

        total_number = len(img_lists)
        if pb:
            pb.setMaximum(total_number)

        tt = time.time()
        for i, img_list in tqdm(enumerate(img_lists), '识别中'):
            if pb:
                pb.setFormat(f'%v/%m，已用时{int(time.time() - tt)}秒')
                pb.setValue(i)
                QApplication.processEvents()  # 好像用这个刷新就能避免使用QThread来解决刷新问题

            if self.file_num == 1 and img_list[0].name == 'xmut.jpg':
                # 有的人没设目录，直接在根目录全量查找了，此时需要过滤掉我的一个资源图 片
                continue
            attrs = parser(img_list)

            for f in img_list:
                rf, ff = f.relpath(imdir).as_posix(), f.resolve()
                # if attrs['文件'] != '':
                if '文件' in attrs.keys():
                    attrs['文件'] = attrs['文件'] + ' ' + f'<a href="{ff}" target="_blank">{rf}</a><br/>'
                else:
                    attrs['文件'] = f'<a href="{ff}" target="_blank">{rf}</a><br/>'
            # 如果要展示图片，可以加：<img src="{ff}" width=100/>，但实测效果不好
            row = []
            for col in self.columns:
                row.append(attrs.get(col, ''))
            ls.append(row)
        if pb:
            pb.setFormat(f'识别图片%v/%m，已用时{int(time.time() - tt)}秒')
            pb.setValue(total_number)
            QApplication.processEvents()

        df = pd.DataFrame.from_records(ls, columns=self.columns)

        # 2 如果有表格清单，进一步优化数据展示形式
        if students is not None:
            df = self.link_table2(df, students)

        # 3 不同的日期用不同的颜色标记
        # https://www.color-hex.com/
        # 绿，蓝，黄，青，紫，红; 然后深色版本再一套
        colors = ['70ea2a', '187dd8', 'f3cf83', '99ffcc', 'ccaacc', 'ff749e',
                  '138808', '9999ff', 'd19a3f', '99ccaa', 'aa33aa', 'cc0000']

        def set_color(m):
            s1, s2 = m.groups()
            # 每个日期颜色是固定的，而不是按照相距今天的天数来选的
            i = (datetime.date.fromisoformat(s1) - datetime.date(2022, 1, 1)).days
            c = colors[i % len(colors)]
            return f'<td bgcolor="#{c}">{s1}{s2}</td>'

        res = df.to_html(escape=False)
        res = re.sub(r'<td>(\d{4}-\d{2}-\d{2})(.*?)</td>', set_color, res)
        res = re.sub(r'<td>阳性</td>', r'<td bgcolor="#cc0000">阳性</td>', res)

        return res

    def browser(self):
        """ 打开浏览器查看 """
        pass

    def to_excel(self):
        """ 导出xlsx文件

        我前面功能格式是基于html设计的，这里要导出excel高亮格式的话，还得额外开发工作量。
        而且又会导致一大堆依赖包安装，暂时不考虑扩展。
        """
        raise NotImplementedError

    def get_img_fileslist(self, dir_path, filenum):
        """
        返回目录下大于filenum的文件列表

        :param dirpath:
        :param filenum:
        :return:
        """
        root = XlPath(dir_path)
        files_lists = []
        for _dir in root.rglob_dirs('*'):
            files_list = list(_dir.glob_images('*'))
            if len(files_list) >= filenum:
                files_lists.append(files_list)
        return files_lists


class MainWindow(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()

        # 设置中文尝试
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # 1 窗口核心属性
        self.setMinimumSize(QSize(900, 300))
        self.setWindowIcon(QIcon('resources/xmut.jpg'))
        self.setWindowTitle("核酸检测分析 @(厦门理工学院)福建省模式识别与图像理解重点实验室")

        # 2 使用网格布局
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        gridLayout = QGridLayout(self)
        centralWidget.setLayout(gridLayout)

        # 3 每列标签
        for i, text in enumerate(["核酸截图目录：", "人员名单【可选】：", "sheet：", "导出报表：", "检测任务", "进度条："]):
            q = QLabel(text, self)
            q.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            gridLayout.addWidget(q, i, 0)

        # 4 每行具体功能
        def add_line1():
            """ 第一行的控件，选择截图目录 """
            # 1 输入框
            le = self.srcdirEdit = QLineEdit(self)
            le.setText(os.path.abspath(os.curdir))  # 默认当前目录
            gridLayout.addWidget(le, 0, 1)

            # 2 浏览选择
            def select_srcdir():
                directory = str(QFileDialog.getExistingDirectory()).replace('/', '\\')
                if directory:
                    self.srcdirEdit.setText(directory)

            btn = QToolButton(self)
            btn.setText(u"浏览选择...")
            btn.clicked.connect(select_srcdir)
            gridLayout.addWidget(btn, 0, 2)

        def add_line2():
            """ 第二行的控件，选择excel文件位置 """

            # 1 输入框
            def auto_search_xlsx(root=None):
                """
                :param root: 参考目录，可以不输入，默认以当前工作环境为准
                """
                if root is None:
                    root = XlPath('.')
                root = XlPath(root)

                res = ''
                for f in root.glob('*.xlsx'):
                    res = f  # 取最后一个文件
                return str(res)

            le = self.refxlsxEdit = QLineEdit(self)
            # 自动找当前目录下是否有excel文件，没有则置空
            le.setText(auto_search_xlsx(self.srcdir))
            gridLayout.addWidget(le, 1, 1)

            # 2 浏览选择
            def select_refxlsx():
                file = str(QFileDialog.getOpenFileName(filter='*.xlsx')[0]).replace('/', '\\')
                if file:
                    self.refxlsxEdit.setText(file)

            btn = QToolButton(self)
            btn.setText(u"浏览选择...")
            btn.clicked.connect(select_refxlsx)
            gridLayout.addWidget(btn, 1, 2)

        def add_line3():
            def update_combo_box():
                xlsxfile = self.refxlsx
                if xlsxfile and xlsxfile.is_file():
                    wb = openpyxl.open(str(xlsxfile), read_only=True, data_only=True)
                    self.sheets_cb.reset_items(tuple([ws.title for ws in wb.worksheets]))
                else:
                    self.sheets_cb.reset_items(tuple())

            self.sheets_cb = get_input_widget(tuple(), parent=self)
            gridLayout.addWidget(self.sheets_cb, 2, 1)
            update_combo_box()
            self.refxlsxEdit.textChanged.connect(update_combo_box)

        def add_line4():
            self.dstfileEdit = QLineEdit(self)
            self.dstfileEdit.setText(str(XlPath('./报表.html').absolute()))
            gridLayout.addWidget(self.dstfileEdit, 3, 1)

        def add_line5():
            # 任务选择、单码、双码、三码、
            def get_value():
                self.task = self.task_cb.currentText()
                # if self.task == '日常核酸检测报告分析':
                #     self.columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '身份证号']
                #     self.stu_columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '身份证号', '联系电话']
                #     self.file_num = 1
                # elif self.task == '双码（健康码、行程码）分析':
                #     self.columns = ['文件', '姓名', '联系电话', '14天经过或途经', '健康码颜色']
                #     self.stu_columns = ['班级', '文件', '学号', '姓名', '联系电话', '14天经过或途经', '健康码颜色']
                #     self.file_num = 2
                # elif self.task == '双码（健康码、行程码）+24小时核酸检测报告分析':
                #     self.columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '14天经过或途经', '健康码颜色', '身份证号']
                #     self.stu_columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '14天经过或途经', '健康码颜色', '身份证号',
                #                         '联系电话']
                #     self.file_num = 3
                # if self.task not in ['日常核酸检测报告分析', '双码（健康码、行程码）分析', '双码（健康码、行程码）+24小时核酸检测报告分析']:
                #     QMessageBox.about(self,
                #                       '请选择任务'
                #                       )
                # return self.task

            self.task_cb = get_input_widget(['日常核酸检测报告分析', '双码（健康码、行程码）分析', '双码（健康码、行程码）+ 24小时核酸检测报告分析'],
                                            cur_value='请选择任务', parent=self)
            gridLayout.addWidget(self.task_cb, 4, 1)
            self.task_cb.activated.connect(get_value)

        def add_line6():
            # 进度条
            pb = self.pb = QProgressBar(self)
            pb.setAlignment(Qt.AlignHCenter)
            gridLayout.addWidget(pb, 5, 1)

        def add_line7():
            # 1 生成报表
            btn = QToolButton(self)
            btn.setText("生成报表")
            btn.setFont(QFont('Times', 20))
            btn.clicked.connect(self.stat)
            gridLayout.addWidget(btn, 6, 1)

            # 2 帮助文档
            def help():
                url = QUrl("https://www.yuque.com/docs/share/1b90b806-3f1d-43f4-b004-64a98944a414?#")
                QDesktopServices.openUrl(url)

            btn = QToolButton(self)
            btn.setText("使用手册")
            btn.setFont(QFont('Times', 20))
            btn.clicked.connect(help)
            gridLayout.addWidget(btn, 6, 1, alignment=Qt.AlignHCenter)

        add_line1()
        add_line2()
        add_line3()
        add_line4()
        add_line5()
        add_line6()
        add_line7()

    @property
    def srcdir(self):
        return XlPath(self.srcdirEdit.text())

    @property
    def refxlsx(self):
        p = XlPath(self.refxlsxEdit.text())
        if p.is_file():
            return p

    def get_sheet_name(self):
        return self.sheets_cb.currentText()

    def get_students(self):
        if self.refxlsx:
            df = pd.read_excel(self.refxlsx, self.get_sheet_name(),
                               dtype={'学号': str, '联系电话': str, '身份证号': str, '姓名': str})
            df = df[(~df['姓名'].isna()) & (~df['班级'].isna()) & (~df['学号'].isna())]
            return df
        return None

    @property
    def dstfile(self):
        return XlPath(self.dstfileEdit.text())

    def stat(self):
        if hasattr(self, 'task') == False:
            QMessageBox.warning(self, '警告', '请选择任务')
        else:
            hs = Hesuan(task=self.task)
            res = hs.parse(self.srcdir, self.get_students(), pb=self.pb)
            dstfile = self.dstfile
            dstfile.write_text(res, encoding='utf8')
            os.startfile(dstfile)


def gui():
    """ 打开可视化窗口使用 """
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


def show_layout(file):
    ppocr = PPOCR()
    ppocr.show_layout(file)


if __name__ == '__main__':
    os.chdir(HESUAN_PATH)
    if len(sys.argv) == 1:  # 默认执行main函数
        sys.argv += ['gui']
    with TicToc(__name__):
        fire.Fire()
