import cv2
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# 设置字体为中文
plt.rcParams['font.family'] = 'SimHei'
# 读取彩色图像
image = cv2.imread(r"C:\Users\17925\Desktop\SFP\ship\0.bmp")
# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 使用Otsu阈值处理，自动选择最佳阈值
_, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 形态学操作来去除噪点并保留前景区域
kernel = np.ones((1, 1), np.uint8)

# 腐蚀和膨胀操作
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
# 生成坐标网格
height, width = mask.shape
x = np.arange(width)  # 列坐标 (Column index)
y = np.arange(height)  # 行坐标 (Row index)

# 创建二维热力图（二值掩码专用）
fig = go.Figure(data=go.Heatmap(
    z=mask,
    x=x,
    y=y,
    colorscale=[[0, 'black'], [1, 'white']],  # 二值色系
    zmin=0,                                   # 固定数据范围
    zmax=255,
    colorbar=dict(
        title='Binary Mask',
        titleside='right',

    )
))

# 设置图形布局
fig.update_layout(
    title={
        'text': "Mask",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Column Index',
    yaxis_title='Row Index',
    width=612,
    height=512,
    autosize=False,
)

fig.update_yaxes(autorange="reversed")
fig.show()

# 保存生成的掩膜图像
cv2.imwrite(r"C:\Users\17925\Desktop\mask.png", mask)
