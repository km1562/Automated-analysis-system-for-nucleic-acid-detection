# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 10:22
# @Author  : wkm
# @QQmail  : 690772123@qq.com

import cpca
# location_str = ["您于前14天内到达或途经：福建省厦门市，福建省泉州市，福建省漳州市"]
#
# df = cpca.transform(location_str)
# print(df)
import re
str_test = "您于前14天内到达或途经：福建省厦门市，福建省泉州市，福建省漳州市您近14天主要到访过：福建省厦门市、福建省泉州市、福建省漳州市"

ans = re.findall(r'省(.+?市)', str_test)
ans = set(ans)
pass