# '''S1>0,S2<0 加180°，  S1<0,S2>0, 加90°, S1>0 , S2>0 取值0°-45°, S1<0,S2<0,0°到45°'''
import cv2
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# 设置字体为中文
plt.rcParams['font.family'] = 'SimHei'

# 读取图像文件
I = cv2.imread( r"C:\Users\17925\Desktop\SFP\ship\0.bmp")
# 将读取到的图像转换为灰度图
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

N = 1.0  # 归一化因子，此处为1表示不对图像进行缩放
# 通过子采样提取不同角度的图像数据：
I0 = I[0::2, 0::2] / N      # I0：取偶数行、偶数列，对应0°方向
I45 = I[1::2, 0::2] / N      # I45：取奇数行、偶数列，对应45°方向
I90 = I[1::2, 1::2] / N      # I90：取奇数行、奇数列，对应90°方向
I135 = I[0::2, 1::2] / N     # I135：取偶数行、奇数列，对应135°方向

# 计算总强度信号 S0 和两个差分信号 S1 与 S2
S0 = (I0 + I45 + I90 + I135) / 2  # S0为总信号
S1 = I0 - I90                   # S1为差分信号，反映某一方向上的亮度差
S2 = I45 - I135                 # S2为另一差分信号，反映另一个方向上的亮度差

# 获取I0的尺寸，作为后续计算phi角度矩阵的尺寸
i, j = I0.shape
# 初始化phi矩阵，用于存放计算得到的偏振角（单位：弧度）
phi = np.zeros((i, j))
# 双重循环遍历每个像素点，计算对应的偏振角phi
for m in range(i):
     for k in range(j):
         # 确保索引m, k在S1的有效范围内
         if m < S1.shape[0] and k < S1.shape[1]:
             # 当S1>0且S2>=0时，直接计算偏振角（单位：弧度）
             if S1[m, k] > 0 and S2[m, k] >= 0:
                 phi[m, k] = 1 / 2 * np.arctan(S2[m, k] / S1[m, k])
             # 当S1>0且S2<0时，计算偏振角后加180°（即加math.pi）
             if S1[m, k] > 0 and S2[m, k] < 0:
                 phi[m, k] = (1 / 2 * np.arctan(S2[m, k] / S1[m, k])) + math.pi
             # 当S1<0时，无论S2正负，计算偏振角后加90°（即加math.pi/2）
             if (S1[m, k] < 0 and S2[m, k] <= 0) or (S1[m, k] < 0 and S2[m, k] > 0):
                 phi[m, k] = (1 / 2 * np.arctan(S2[m, k] / S1[m, k])) + math.pi/2
             # 当S1==0且S2>0时，偏振角固定为45°（即math.pi/4）
             if S1[m, k] == 0 and S2[m, k] > 0:
                 phi[m, k] = math.pi/4
             # 当S1==0且S2<0时，偏振角固定为135°（即3*math.pi/4）
             if S1[m, k] == 0 and S2[m, k] < 0:
                 phi[m, k] = math.pi/4 * 3

# 获取phi矩阵的高度和宽度
height, width = phi.shape

# 将每个像素点的phi值转换为角度（单位：度）并保存到一个二维列表中
pixel_values = []
for y in range(height):
     row = []
     for x in range(width):
         # 角度转换公式：弧度 * 180/π
         pixel = phi[y, x] * 180 / math.pi
         row.append(pixel)
     pixel_values.append(row)

# 创建一个Pandas DataFrame，将二维列表转换为表格，每列命名为Pixel_0, Pixel_1, ...
df = pd.DataFrame(pixel_values, columns=[f'Pixel_{i}' for i in range(width)])
# 将DataFrame保存为Excel文件，不包含行索引
df.to_excel(r"C:\Users\17925\OneDrive\Desktop\φ.xlsx", index=False)
phi_deg = phi * 180 / math.pi
# 归一化方位角 φ
phi_normalized = ((phi - np.min(phi)) / (np.max(phi) - np.min(phi))) * 255
phi_normalized = phi_normalized.astype(np.uint8)
cv2.imwrite(r"C:\Users\17925\OneDrive\Desktop\phi.png", phi_normalized)  # 保存方位角图像
# 生成坐标网格
height, width = phi_normalized.shape
x = np.arange(width)  # 列坐标 (Column index)
y = np.arange(height)  # 行坐标 (Row index)

# 创建二维热力图
fig = go.Figure(data=go.Heatmap(
    z=phi_deg,                         # 偏振度数据
    x=x,                         # 横坐标
    y=y,                         # 纵坐标
    colorscale='Plasma',            # 红绿蓝颜色映射
    colorbar=dict(
        title='Angle of Polarization',      # 颜色条标题
        titleside='right'
    )
))

# 设置图形布局
fig.update_layout(
    title={
        'text': "Phi",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Column Index',  # x轴标签
    yaxis_title='Row Index',     # y轴标签
    width=612,                   # 图像宽度
    height=512,                  # 图像高度
    autosize=False,              # 关闭自动缩放
)

# 反转y轴方向以匹配图像坐标系
fig.update_yaxes(autorange="reversed")

# 显示交互式图形
fig.show()