# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 8:59
# @Author  : wkm
# @QQmail  : 690772123@qq.com
import os

from pyxllib.xl import TicToc, XlPath, round_int, browser

img_dir = "/home/wengkangming/map_file/entrate_hesuan/三码合一入学检测/test_v2_entrate_hesuan/"
# test_img_dir = "/home/wengkangming/map_file/entrate_hesuan/核酸结果、健康码、行程码_原数据_需要标注的数据_重新确认——22.6h_7/20级计2-陈杰贞/20级计2/9月3日（1人）/"

# def get_img_filelist(dirpath, filenum):
#     """
#     返回目录下大于filenum的文件列表
#
#     :param dirpath:
#     :param filenum:
#     :return:
#     """
#     assert isinstance(filenum, int)
#     dir_root = XlPath(dirpath)
#     dirs = list(dir_root.rglob_dirs())  #直接for即可，不需要list，迭代器，同时还是xlPATH
#     file_list = []
#
#     for dir in dirs:
#         os.chdir(dir)
#         ImgDir = XlPath(dir)
#         ImgUnderDirList = list(ImgDir.glob_images())
#         if len(ImgUnderDirList) >= filenum:
#             file_list.append(ImgUnderDirList)
#
#     return file_list

def get_img_filelist(dir, filenum):
    """
    返回目录下大于filenum的文件列表

    :param dirpath:
    :param filenum:
    :return:
    """
    root = XlPath(dir)
    file_lists = []
    for _dir in root.rglob_dirs():
        file_list = list(_dir.glob_images())
        if len(file_list) >= filenum:
            file_lists.append(file_list)
    return file_lists

img_filelist = get_img_filelist( img_dir, 3)
pass