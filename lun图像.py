import cv2
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# 设置字体为中文
plt.rcParams['font.family'] = 'SimHei'
# 1. 读取原始图像并转换为灰度图
image_path = r"C:\Users\17925\Desktop\SFP\ship\0.bmp"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"无法找到图像：{image_path}")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 子采样提取四向偏振分量
N = 1.0  # 归一化因子
I0   = gray_img[0::2, 0::2] / N   # 0°方向
I45  = gray_img[1::2, 0::2] / N   # 45°方向
I90  = gray_img[1::2, 1::2] / N   # 90°方向
I135 = gray_img[0::2, 1::2] / N   # 135°方向

# 3. 计算斯托克斯参数
S0 = (I0 + I45 + I90 + I135) / 2  # 总光强
S1 = I0 - I90                     # 水平偏振分量
S2 = I45 - I135                   # 对角偏振分量

# 4. 计算非光强偏振图像 Iun = S0 - 偏振分量
eps = 1e-8  # 防止零除
polarization = np.sqrt(S1**2 + S2**2)  # 偏振强度
Iun = S0 - polarization  # 非偏振成分

# 5. 归一化并保存图像
Iun_normalized = cv2.normalize(Iun, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite(r"C:\Users\17925\Desktop\Iun.png", Iun_normalized)

# 6. 二维可视化
height, width = Iun.shape
x = np.arange(width)
y = np.arange(height)

fig = go.Figure(data=go.Heatmap(
    z=Iun,
    x=x,
    y=y,
    colorscale=[[0, 'black'], [1, 'white']],  # 二值色系
    zmin=0,  # 固定数据范围
    zmax=255,
    colorbar=dict(
        title='Non-polarized Intensity',
        titleside='right'
    )
))

fig.update_layout(
    title={
        'text': "Iun",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Column Index',
    yaxis_title='Row Index',
    width=612,  # 图像宽度
    height=512,  # 图像高度
    autosize=False,  # 关闭自动缩放
)

fig.update_yaxes(autorange="reversed")
fig.show()