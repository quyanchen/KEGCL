# 修改
import torch
import numpy as np
from torch_geometric.data import Data

# import re
#
# clean_lines = []
# with open("ppi/Attribute_biogridnew.txt", "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         # 删除不可见字符，替换所有类型的空格为标准空格
#         clean_line = re.sub(r"[^\S\r\n]+", " ", line).strip()  # 包括全角空格
#         if clean_line:
#             try:
#                 # 尝试转换为浮点数，确保格式正确
#                 row = list(map(float, clean_line.split()))
#                 clean_lines.append(clean_line)
#             except ValueError as e:
#                 print(f"Skipping invalid line {i}: {repr(line)} - {e}")
#
# # 保存清理后的文件
# with open("Abiogridnew_cleaned.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(clean_lines))

import re
import os

# 输入文件夹和输出文件夹路径
# input_folder = "ppi"
# output_folder = "cleaned_ppi"

input_folder = "./"
output_folder = "./"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有的文件
for i in range(1):  # 从 0 到 11
    # input_file = os.path.join(input_folder, f"Attribute_dip{i}.txt")
    # output_file = os.path.join(output_folder, f"Attribute_dip_cleaned{i}.txt")
    input_file = os.path.join(input_folder, "Attribute_dipnew.txt")
    output_file = os.path.join(output_folder, "Adipnew_cleaned.txt")

    clean_lines = []  # 保存清理后的行

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f):
                # 删除不可见字符，替换所有类型的空格为标准空格
                clean_line = re.sub(r"[^\S\r\n]+", " ", line).strip()
                if clean_line:
                    try:
                        # 确保行内容可以正确解析为浮点数
                        row = list(map(float, clean_line.split()))
                        clean_lines.append(clean_line)
                    except ValueError as e:
                        print(f"Skipping invalid line {line_number} in {input_file}: {repr(line)} - {e}")

        # 保存清理后的文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_lines))

        print(f"Cleaned {input_file} -> {output_file}")

    except FileNotFoundError:
        print(f"File {input_file} not found. Skipping.")

print("All files processed.")
