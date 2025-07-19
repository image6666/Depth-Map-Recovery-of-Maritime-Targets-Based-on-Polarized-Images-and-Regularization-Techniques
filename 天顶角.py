import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# 设置字体为中文
plt.rcParams['font.family'] = 'SimHei'
def convert_to_numeric(df):
    """
    将 DataFrame 中的所有列转换为数值类型，如果转换失败则置为 NaN。
    """
    for col in df.columns:
        # 对每一列调用 pd.to_numeric，将无法转换的值设置为 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calculate_theta(p0_value, n=1.5):
    """
    根据给定的 p0_value 和折射率 n 计算对应的入射角 theta（单位：度）。
    该函数通过求解方程：
        P0(theta) - p0_target = 0
    来确定 theta，其中 P0 由一系列光学公式计算得出。
    """

    def equation(theta, n, p0_target):
        # 计算 sinθ 和 cosθ 及其平方
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_sq = sin_theta ** 2
        cos_theta_sq = cos_theta ** 2
        n_sq = n ** 2

        # 计算 A 参数，反映反射系数中某一分量
        A = ((cos_theta - n * np.cos(np.arcsin(sin_theta / n))) ** 2) / \
            ((cos_theta + n * np.cos(np.arcsin(sin_theta / n))) ** 2)

        # 计算 B 参数，另一反射系数分量
        B = ((n * cos_theta - np.cos(np.arcsin(sin_theta / n))) ** 2) / \
            ((n * cos_theta + np.cos(np.arcsin(sin_theta / n))) ** 2)

        # 计算透射光功率分量 Pt
        Pt = ((n - 1 / n) ** 2 * sin_theta_sq) / \
             (2 + 2 * n ** 2 - (n + 1 / n) ** 2 * sin_theta_sq + 4 * cos_theta * np.sqrt(n_sq - sin_theta_sq))

        # 计算反射光功率分量 Pr
        Pr = 2 * np.sqrt((sin_theta ** 4) * (cos_theta ** 2) * (n_sq - sin_theta_sq)) / \
             (sin_theta ** 4 + cos_theta_sq * (n_sq - sin_theta_sq))

        # 计算总反射比 P0
        P0 = ((A + B) * Pt + (2 - A - B) * Pr) / 2

        # 返回计算得到的 P0 与目标 p0_target 之间的差值
        return P0 - p0_target

    # 利用 minimize_scalar 在 [0, π/2] 范围内搜索 theta，使得方程绝对值最小
    result = minimize_scalar(lambda theta: abs(equation(theta, n, p0_value)),
                             bounds=(0, np.pi / 2), method='bounded')
    # 将结果从弧度转换为角度后返回
    return np.rad2deg(result.x)


def process_excel(file_path):
    """
    读取 Excel 文件中的 p0 值，转换为数值型，
    计算每个 p0 值对应的 theta（入射角）值，
    并将结果保存到新的 Excel 文件中。
    """
    # 读取 Excel 文件，假设文件中无表头
    df_p0 = pd.read_excel(file_path, header=None)
    # 将所有数据转换为数值类型，确保后续计算不会出现数据类型问题
    df_p0 = convert_to_numeric(df_p0)

    # 检查是否存在 NaN 值（转换失败的值）
    if df_p0.isnull().any().any():
        print("输入数据中存在无法转换为数值的内容，请检查 Excel 文件！")
        return

    # 初始化与 df_p0 形状相同的数组，用于存储计算得到的 theta 值
    theta_values = np.zeros_like(df_p0, dtype=float)

    # 遍历 Excel 中的每个 p0 值，并计算对应的 theta 值
    for index, p0_value in np.ndenumerate(df_p0.values):
        theta_values[index] = calculate_theta(p0_value)
    # 归一化天顶角 θ
    theta_normalized = (theta_values / np.max(theta_values)) * 255
    theta_normalized = theta_normalized.astype(np.uint8)
    cv2.imwrite(r"C:\Users\17925\OneDrive\Desktop\theta.png", theta_normalized)  # 保存天顶角图像
    # 生成坐标网格
    height, width = theta_normalized.shape
    x = np.arange(width)  # 列坐标 (Column index)
    y = np.arange(height)  # 行坐标 (Row index)

    # 创建二维热力图
    fig = go.Figure(data=go.Heatmap(
        z=theta_normalized,  # 偏振度数据
        x=x,  # 横坐标
        y=y,  # 纵坐标
        colorscale='Viridis',  # 红绿蓝颜色映射
        colorbar=dict(
            title='Zenith Angle',  # 颜色条标题
            titleside='right'
        )
    ))

    # 设置图形布局
    fig.update_layout(
        title={
            'text': "Theta",
            'y': 0.85,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Column Index',  # x轴标签
        yaxis_title='Row Index',  # y轴标签
        width=612,  # 图像宽度
        height=512,  # 图像高度
        autosize=False,  # 关闭自动缩放
    )

    # 反转y轴方向以匹配图像坐标系
    fig.update_yaxes(autorange="reversed")

    # 显示交互式图形
    fig.show()
    # 将计算结果转换为 DataFrame 格式
    df_theta = pd.DataFrame(theta_values)
    # 定义输出文件路径
    output_path = r"C:\Users\17925\OneDrive\Desktop\θ.xlsx"
    # 将结果保存到 Excel 文件（不保存索引和表头）
    df_theta.to_excel(output_path, index=False, header=None)
    print(f"计算完成，结果已保存至 {output_path}")


if __name__ == "__main__":
    # 设置输入 Excel 文件的路径
    file_path = r"C:\Users\17925\Desktop\P.xlsx"
    process_excel(file_path)
