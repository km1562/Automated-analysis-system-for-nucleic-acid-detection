# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 11:10
# @Author  : wkm
# @QQmail  : 690772123@qq.com
from pyxllib.xl import XlPath

imdir = r'C:\Users\weng\Desktop\杜博士\作业批改\WEB前端框架实训_共有92份作业'
imdir = XlPath(imdir)

for f in imdir.rglob_files('*.zip'):
    from pyxllib.file.packlib import unpack_archive

    # 没有对应同名目录则解压d
    if not f.with_name(f.stem).is_dir():
        unpack_archive(f, wrap=1)