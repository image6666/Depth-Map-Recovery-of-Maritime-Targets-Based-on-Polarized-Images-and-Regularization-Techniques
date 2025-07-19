import cv2
import numpy as np
import pandas as pd

# 1. 读取图像并转换为灰度图
image_path = r"C:\Users\17925\Desktop\SFP\ship\0.bmp" # 图像路径
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"无法找到图像：{image_path}")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 2. 子采样处理，将图像分为四个方向的子图
N = 1.0  # 归一化因子（此处为1，可根据需要调整）
I0 = gray_img[0::2, 0::2] / N      # 0°方向的子图：偶数行、偶数列
I45 = gray_img[1::2, 0::2] / N      # 45°方向的子图：奇数行、偶数列
I90 = gray_img[1::2, 1::2] / N      # 90°方向的子图：奇数行、奇数列
I135 = gray_img[0::2, 1::2] / N     # 135°方向的子图：偶数行、奇数列

# 3. 计算 S0、S1、S2 信号
S0 = (I0 + I45 + I90 + I135) / 2    # 总强度信号（综合亮度）
S1 = I0 - I90                     # 水平与垂直分量的差异
S2 = I45 - I135                   # 45°与135°分量的差异

# 4. 打印原图像的最小和最大灰度值
min_value = gray_img.min()
max_value = gray_img.max()
print("最小灰度值：", min_value)
print("最大灰度值：", max_value)

# 5. 计算极化度 p
p = np.sqrt(S1 ** 2 + S2 ** 2) / S0

# 6. 将 p 的像素值保存到二维列表中
height, width = p.shape
pixel_values = [list(p[y, :]) for y in range(height)]

# 7. 利用 Pandas 将 p 数据保存为 Excel 文件
df = pd.DataFrame(pixel_values, columns=[f'Pixel_{i}' for i in range(width)])
df.to_excel(r"C:\Users\17925\Desktop\P.xlsx", index=False)

# 8. 将 p 映射到 0-255 范围以便保存为图像
min_p = np.min(p)
max_p = np.max(p)
p_normalized = ((p - min_p) / (max_p - min_p)) * 255
p_normalized = p_normalized.astype(np.uint8)

# 9. 保存归一化后的 p 图像
result_path = r"C:\Users\17925\Desktop\P.png"
cv2.imwrite(result_path, p_normalized)

# 10. 定义归一化并保存图像的函数
def normalize_and_save(image, filename):
    """
    将输入图像归一化到0-255范围，并保存为PNG图像
    """
    img_min = np.min(image)
    img_max = np.max(image)
    # 防止除0错误
    if img_max - img_min == 0:
        norm_img = np.zeros_like(image, dtype=np.uint8)
    else:
        norm_img = ((image - img_min) / (img_max - img_min)) * 255
        norm_img = norm_img.astype(np.uint8)
    cv2.imwrite(filename, norm_img)

# 11. 对 I0, I45, I90, I135 四通道图像归一化并保存到桌面
normalize_and_save(I0,   r"C:\Users\17925\Desktop\I0.png")
normalize_and_save(I45,  r"C:\Users\17925\Desktop\I45.png")
normalize_and_save(I90,  r"C:\Users\17925\Desktop\I90.png")
normalize_and_save(I135, r"C:\Users\17925\Desktop\I135.png")
