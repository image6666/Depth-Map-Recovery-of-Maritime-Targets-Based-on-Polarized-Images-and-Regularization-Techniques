import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, zoom
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# 设置字体为中文
plt.rcParams['font.family'] = 'SimHei'

# 读取深度图
original_depth_map = cv2.imread(r"C:\Users\17925\Desktop\depthFC.png", cv2.IMREAD_UNCHANGED)

# 如果是彩色图像，转换为灰度图
if len(original_depth_map.shape) == 3:
    original_depth_map = cv2.cvtColor(original_depth_map, cv2.COLOR_BGR2GRAY)

# 将深度图类型转换为 float32
depth_map = original_depth_map.astype(np.float64)

# 归一化深度图到 [0, 1]
depth_map -= np.min(depth_map)
depth_map /= np.max(depth_map)
# 将深度图重新归一化到 [0, 255] 范围
depth_map = depth_map * 255  # 将 [0, 1] 归一化到 [0, 255]
depth_map = np.clip(depth_map, 0, 255)  # 确保数值不超过 [0, 255] 范围
# 缩放深度图，使用 scipy.ndimage.zoom 保留更多细节
scale_factor = 0.8# 缩放因子
depth_map = zoom(depth_map, scale_factor, order=1)  # 使用线性插值缩放

# 应用高斯滤波，sigma=0.5 保留更多细节
depth_map = gaussian_filter(depth_map, sigma=0.7)

# 打印深度图的最小值和最大值
print(f"Depth map min value: {np.min(depth_map)}, max value: {np.max(depth_map)}")

# 创建 x 和 y 坐标网格
x = np.arange(0, depth_map.shape[1])
y = np.arange(0, depth_map.shape[0])
x, y = np.meshgrid(x, y)

# 打印网格信息
print(f"x shape: {x.shape}, y shape: {y.shape}, depth_map shape: {depth_map.shape}")

# 使用 Plotly 创建三维表面图
fig = go.Figure(data=[go.Surface(z=depth_map, x=x[0, :], y=y[:, 0], colorscale='Viridis')])

# 设置三维图的布局和参数
fig.update_layout(
    title='三维表面重建',  # 标题
    autosize=False,  # 禁用自动调整大小
    width=1000,  # 设定图的宽度
    height=1000,  # 设定图的高度
    margin=dict(l=65, r=50, b=65, t=90),  # 设置边距
    scene=dict(  # 三维场景设置
        xaxis=dict(nticks=10, range=[0, depth_map.shape[1]]),  # x轴刻度
        yaxis=dict(nticks=10, range=[0, depth_map.shape[0]]),  # y轴刻度
        zaxis=dict(nticks=10, range=[np.min(depth_map), np.max(depth_map)]),  # z轴刻度和范围
        camera=dict(  # 摄像机角度设置
            eye=dict(x=1.87, y=1.87, z=1.87)  # 摄像机的观察位置
        )
    ),
    dragmode='orbit'  # 设置拖拽模式为轨道视角
)
# 显示生成的三维图
fig.show()
