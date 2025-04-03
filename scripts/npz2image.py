import numpy as np
from PIL import Image
import os

# 2k /media/zyx/aniface/ckpt/log/samples_12-06-13.npz
# 4k /media/zyx/aniface/ckpt/log/samples_12-06-17.npz
# 6k /media/zyx/aniface/ckpt/log/samples_12-07-06.npz
# 8k /media/zyx/aniface/ckpt/log/samples_12-07-13.npz
# 10k /media/zyx/aniface/ckpt/log/samples_12-07-19.npz
# 20k /media/zyx/aniface/ckpt/log/samples_12-08-05.npz
# 30k /media/zyx/aniface/ckpt/log/samples_12-09-00.npz


# 加载 .npz 文件
data = np.load('/media/zyx/aniface/ckpt/log/samples_12-09-00.npz')
print(data.files[0])
# 假设 .npz 文件中包含一个名为 'image_array' 的数组
image_data = data[data.files[0]]

# 确保图像数据是正确的形状和类型
image_data = image_data.astype('uint8')

# 遍历图像数据并保存为图片
for i, img_array in enumerate(image_data):
    if i>32:
        break
    img = Image.fromarray(img_array)
    img.save(os.path.join("/home/zyx/zbb/improved-diffusion/datasets/ea_aniface",f'image30k_{i}.png'))

# 关闭 .npz 文件
print("complete!")
data.close()
