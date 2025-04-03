import os
from PIL import Image
import numpy as np

# 指定包含JPEG文件的目录
directory = '/home/zyx/zbb/improved-diffusion/datasets/anime-faces-val'

# 创建一个空列表来存储图像数据
images = []

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 构建完整的文件路径
    file_path = os.path.join(directory, filename)
    # 打开并读取图像
    with Image.open(file_path) as img:
        # 将图像转换为RGB格式（如果需要）
        img = img.convert('RGB')
        # 将图像转换为NumPy数组并添加到列表
        images.append(np.array(img))

# 将所有图像数组转换为一个NumPy数组
image_array = np.array(images)

# 保存为NPZ文件，压缩格式
np.savez_compressed("/home/zyx/zbb/improved-diffusion/datasets/anime-faces-val.npz", arr_0=image_array)

print("Images are successfully saved as NPZ.")
