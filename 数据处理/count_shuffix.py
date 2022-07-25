from pyxllib.xl import TicToc, XlPath, round_int, browser
import pathlib

img_path = "/home/wengkangming/map_file/entrate_hesuan/核酸结果、健康码、行程码_原数据_test"
data_root = XlPath(img_path)
files = list(data_root.glob_images('**/*'))

count_suffix_map = {}

for pic in files:
    suffix = pic.suffix
    str_suffix = str(suffix)
    if str_suffix == '.jpeg':
        print(pic)

    if str_suffix not in count_suffix_map:
        count_suffix_map[str_suffix] = 1
    else:
        count_suffix_map[str_suffix] += 1

for key, value in count_suffix_map.items():
    print(f"the suffix's {key}, number's {value}", key, value)