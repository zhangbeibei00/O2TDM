import matplotlib.pyplot as plt
import numpy as np

f = open('data1.txt')
line = f.readline()
data_list = []
while line:
    num = list(map(float, line.split()))
    data_list.append(num)
    line = f.readline()
f.close()
data_array = np.array(data_list)
# print(data_array)
x = np.arange(0, 1, 0.01)  # start,stop,step
y1 = data_array
y2 = 500 - data_array

# 设置全局字体
# Times New Roman: 一种常用的衬线字体，通常用于学术文档和印刷品。
# Arial: 一种无衬线字体，通常用于现代、清晰的设计。
# Helvetica: Helvetica是Arial的Mac OS X 和Linux系统上的等价字体，也是一种无衬线字体。
# Courier New: 一种等宽字体，适用于显示代码或文本表格。
# Verdana: 一种无衬线字体，具有较大的x高度，易于阅读。
# Georgia: 一种衬线字体，具有一些装饰性特点，适用于标题和印刷文档。
# Palatino: 另一种衬线字体，适用于印刷和书籍排版。
# Cursive: 一种手写风格的字体，用于装饰性的设计。
# Comic Sans MS: 一种装饰性和非正式的字体，通常不适用于正式文档。
# Symbol: 一种包含各种符号和特殊字符的字体。
plt.rcParams['font.family'] = 'Arial'

fig, ax1 = plt.subplots()

ax1.plot(x, y1, 'b-', label='Y1',linewidth=1)
ax1.set_xlabel('X data', fontsize=16, fontweight='bold')
ax1.set_ylabel('Loss1', color='k', fontsize=16, fontweight='bold')  # 设置Y1轴标题
ax1.tick_params(axis='y', labelcolor='k', labelsize=14, width=2)

ax2 = ax1.twinx()
ax2.plot(x, y2, 'r--', label='Y2',linewidth=1)
ax2.set_ylabel('Loss2', color='k', fontsize=16, fontweight='bold')  # 设置Y2轴标题
plt.tick_params(axis='y', labelcolor='k', labelsize=14, width=2)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2

# 设置X轴标签为粗体
ax1.legend(lines, labels, loc='upper left')
plt.show()
