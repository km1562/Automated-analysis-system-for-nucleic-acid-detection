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

USE_GPU = True
# HESUAN_PATH = "/home/wengkangming/map_file/entrate_hesuan/PaddleDetection/inference_model/ppyolov2_r50vd_dcn_365e_entrate_dataset/"  #配置不同，类别就不同
HESUAN_PATH = "/home/wengkangming/map_file/entrate_hesuan/PaddleDetection/inference_model/ppyolov2_r50vd_dcn_all_entrate_dataset_batch_20_lrdiv8_pretrained_epoch_3900_3gpu_2580epoch_rerun_340rerun_1000epoch/"  #配置不同，类别就不同
img_path = "/home/wengkangming/map_file/entrate_hesuan/data_backups/19魏_补充数据_12_my_pretrain"
RECOG_PATH = "/home/wengkangming/map_file/entrate_hesuan/核酸结果自动汇总统计工具v1.0/resources/hesuan_ocr_v6_tmp_v1/"

class PPOCR:
    def __init__(self, *, ppocr=True, pplp=True):

        # categories_list = ['姓名', '身份证', '联系方式', '采样时间', '检测时间', '核酸结果', '14天经过或途经', '健康码颜色', '其他类']
        # categories_dict = CocoGtData.gen_categories(categories_list)

        # recdir = f'{HESUAN_PATH}/resources/hesuan_ocr_v7_tmp_v2'
        recdir = "/home/wengkangming/map_file/entrate_hesuan/PaddleOCR/inference/rec_crnn/"
        # self.ppocr = PaddleOCR.build_ppocr(use_gpu=USE_GPU, rec_model_dir=recdir,
        #                                        rec_char_dict_path=f'{recdir}/char_dict.txt')
        recdir = "/home/wengkangming/map_file/entrate_hesuan/PaddleOCR/inference/rec_crnn/"
        self.ppocr = PaddleOCR.build_ppocr(use_gpu=USE_GPU, rec_model_dir=recdir)
        # self.ppocr = PaddleOCR.build_ppocr(use_gpu=USE_GPU)
        self.pplp = lp.PaddleDetectionLayoutModel(model_path=HESUAN_PATH,
                                                  threshold=0.5,
                                                  # label_map={0: "其他类", 1: "姓名", 2: "身份证号", 3: "联系电话",
                                                  #            4: "采样时间", 5: "检测时间", 6: "核酸结果"},
                                                  label_map={0: "姓名", 1: "身份证", 2: "联系方式", 3: "采样时间",
                                                             4: "检测时间", 5: "核酸结果", 6: "14天经过或途经", 7: "健康码颜色", 8: "其他类"},
                                                  enforce_cpu=not USE_GPU,  # 有gpu显卡会自动调用，能大大提速
                                                  enable_mkldnn=True,
                                                  )
        self.count = 0

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
        columns = ['type', 'text', 'score', 'points']

        from pyxlpr.data.labelme import LabelmeDict
        json_dict = LabelmeDict.gen_data(file)

        for block in blocks:
            t = block['type']
            im = xlcv.get_sub(image, block['ltrb'])  # 识别每个框里的文本内容
            text, score = self.ppocr.rec_singleline(im)
            score = round((block['score'] + score) / 2, 4)
            points = block['ltrb']  # [2,2]

            shape_dict_label_dict = {"text": text, "category": t, "text_kv": "value", "score": score}

            # 生成json包括flag,gourp_id，label，points
            shape_dict = LabelmeDict.gen_shape(shape_dict_label_dict, points, )

            json_dict["shapes"].append(shape_dict)

        return json_dict

    def save_json_label(self, file, json_dict):
        file_shuffix = file.suffix
        file_shuffix_str = str(file_shuffix)
        save_file = str(file).replace(file_shuffix_str, '.json')
        with open(save_file, "w") as f:
            print("save success", save_file)
            json.dump(json_dict, f)
            self.count += 1

    def gen_Labelme_json(self, img_path):
        data_root = XlPath(img_path)
        print("data_root:", data_root)
        self.ensure_images(data_root)

        files = list(data_root.glob_images('**/*'))
        for pic in files:
            json_dict = self.parse_layout(pic)
            self.save_json_label(pic, json_dict)

        print(f"total handle {self.count} picture")

if __name__ == "__main__":
    EntratePPocr = PPOCR()
    EntratePPocr.gen_Labelme_json(img_path=img_path)

