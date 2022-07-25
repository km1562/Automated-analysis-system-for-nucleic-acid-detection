import json
import os
from pyxllib.xl import TicToc, XlPath, round_int, browser
from pyxllib.algo.geo import rect_bounds
import pprint
import numpy as np
from pyxllib.xlcv import xlcv

from pyxlpr.paddleocr import PaddleOCR
from pyxlpr.data.imtextline import TextlineShape
import pandas as pd

import layoutparser as lp
from pyxlpr.data.coco import CocoGtData
from statics_3_pic import get_img_filelist
import re

USE_GPU = True
HESUAN_PATH = "/home/wengkangming/map_file/entrate_hesuan/PaddleDetection/inference_model/ppyolov2_r50vd_dcn_all_entrate_dataset_batch_20_lrdiv8_pretrained_epoch_3900_3gpu_2580epoch_rerun_340rerun_1000epoch/"  #配置不同，类别就不同
img_path = "/home/wengkangming/map_file/entrate_hesuan/test_v2_entrate_hesuan/"

class PPOCR:
    def __init__(self, *, ppocr=True, pplp=True):

        # categories_list = ['姓名', '身份证', '联系方式', '采样时间', '检测时间', '核酸结果', '14天经过或途经', '健康码颜色', '其他类']
        # categories_dict = CocoGtData.gen_categories(categories_list)

        self.ppocr = PaddleOCR.build_ppocr(use_gpu=USE_GPU)
        self.pplp = lp.PaddleDetectionLayoutModel(model_path=HESUAN_PATH,
                                                  threshold=0.5,
                                                  label_map={0: "姓名", 1: "身份证", 2: "联系方式", 3: "采样时间",
                                                             4: "检测时间", 5: "核酸结果", 6: "14天经过或途经", 7: "健康码颜色", 8: "其他类"},
                                                  enforce_cpu=not USE_GPU,  # 有gpu显卡会自动调用，能大大提速
                                                  enable_mkldnn=True,
                                                  )
        self.count = 0

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

        for t, keys in {'姓名': ['姓名'],  #t为attr，attrs为结果集
                        '身份证号': ['证件号码', '身份证号码'],
                        '联系电话': ['联系电话'],
                        '采样时间': ['采样时间'],
                        '检测时间': ['检测时间', '报告时间'],
                        '核酸结果': ['检测结果']}.items():
            if t not in attrs:
                find_by_key(t, keys)

        # 2 通过正则分析文本
        def find_by_patter(attr, patter):
            #TODO 这里可能需要罗列所有的途径
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

        #TODO,好像有两码的
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
        if '14天经过或途经' in attrs:
            province_list = ["河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "海南", "四川", \
                            "贵州", "云南", "陕西", "甘肃", "青海", "台湾"]
            non_province_list = ["内蒙古自治区", "广西壮族自治区", "西藏自治区", "宁夏回族自治区", "新疆维吾尔自治区", "北京市", "天津市", "上海市", "重庆市", "香港特别行政区", "澳门特别行政区"]
            names = set()
            for t in province_list:  # 可以扩展
                names |= set(re.findall(t + '省.+?市', attrs["14天经过或途经"]))

            for t in non_province_list:
                names |= set(re.findall(t, attrs["14天经过或途经"]))
            attrs["14天经过或途经"] = ",".join(names)

        return attrs


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
        # browser(df)

    def multi_parse_layout(self, file_list):
        res_ls = []
        for file in file_list:
           res_ls += self.parse_layout(file)

        attrs = self.get_attrs(res_ls)
        return attrs

    def get_attrs(self, ls):

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
            elif k == '14天经过或途经': #'14天途径在里面了，全部加起来
                attrs[k] += v

        self.supplement_attrs(attrs, df['text'])
        self.refine_attrs(attrs)
        return attrs

    def show_layout(self, file):
        """ 可视化查看效果 """
        image = xlcv.read(file, 1)
        layout = self.pplp.detect(image[..., ::-1])
        im = lp.draw_box(image, layout, box_width=3, show_element_type=True)
        im.show()

if __name__ == "__main__":
    EntratePPocr = PPOCR()

# dirspath = "/home/wengkangming/map_file/entrate_hesuan/核酸结果、健康码、行程码_原数据_需要标注的数据_重新确认——22.6h_7/20级计2-陈杰贞/20级计2/"
    block = []
    file_lists = get_img_filelist(img_path, 3)
    for file_list in file_lists:
        EntratePPocr.multi_parse_layout(file_list)



