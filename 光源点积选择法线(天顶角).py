import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
plt.rcParams['font.family'] = 'SimHei'  # 设置字体
def lambertian_sfp(theta, phi, mask_resized, n, s, albedo, Iun_resized):
    """
    基于拉姆伯特模型的偏振形状恢复 (Shape-from-Polarization, SFP) 算法。
    """
    # Step 1: 反演DOP表达式以计算天顶角 theta
    min_angle = np.min(theta)
    max_angle = np.max(theta)
    theta_normalized = ((theta - min_angle) / (max_angle - min_angle)) * 255
    theta_normalized = theta_normalized.astype(np.uint8)

    # Step 2: 调整非极化光强度
    Iun = np.clip(Iun_resized, 0, albedo)  # 确保 Iun 在 [0, albedo] 范围内
    Iun = Iun / albedo  # 归一化光强度#
    # (将实际测量的非极化光强转化为一个反映“实际反射率”与理论反射率之间差异的因子，从而为后续基于拉姆伯特反射模型的法线计算提供了亮度约束)
    # Step 3: 计算两种可能的极化法线（偏振信息获取法线）
    N1 = np.zeros((theta.shape[0], theta.shape[1], 3))  # 第一种法线
    N2 = np.zeros((theta.shape[0], theta.shape[1], 3))  # 第二种法线

    # 为避免零值问题，对法线计算增加一个小的epsilon
    epsilon = 1e-6
    cos_theta = np.clip(np.cos(theta), epsilon, 1)
    sin_theta = np.clip(np.sin(theta), epsilon, 1)
    N1[:, :, 0] = np.cos(phi) * sin_theta
    N1[:, :, 1] = np.sin(phi) * sin_theta
    N1[:, :, 2] = np.cos(theta)

    N2[:, :, 0] = np.cos(phi + np.pi) * sin_theta
    N2[:, :, 1] = np.sin(phi + np.pi) * sin_theta
    N2[:, :, 2] = np.cos(theta)

    # Step 4: 遍历每个像素点，选择最匹配的法线（Iun 提供非极化强度信息，即使用光强信息求取法线）
    N = np.full_like(N1, np.nan)  # 用于存储最终的法线
    for row in range(mask_resized.shape[0]):
        for col in range(mask_resized.shape[1]):
            if mask_resized[row, col]:  # 只处理前景区域
                a, b, c = s[0], s[1], s[2] * np.cos(theta[row, col]) - Iun[row, col]
                #s 表示光源方向，s[2]表示沿 z 轴的反射分量，Iun 提供非极化强度信息(即使用光强信息对确定法线方向进行约束）。
                d = -np.sin(theta[row, col]) ** 2
                sqrt_term = np.sqrt(np.maximum(0, -d * a ** 2 - d * b ** 2 - c ** 2))
                denominator = np.maximum(a ** 2 + b ** 2, epsilon)
                ny1 = -(b * c + a * sqrt_term) / denominator
                ny2 = -(b * c - a * sqrt_term) / denominator

                nx1 = np.sqrt(np.sin(theta[row, col]) ** 2 - ny1 ** 2)
                nx2 = -np.sqrt(np.sin(theta[row, col]) ** 2 - ny1 ** 2)

                # 两种法线 n1 和 n2
                n1 = np.array([nx1, ny1, np.cos(theta[row, col])])
                n2 = np.array([nx2, ny2, np.cos(theta[row, col])])

                # 避免零向量的比较
                if np.linalg.norm(n1) > epsilon and np.linalg.norm(n2) > epsilon:
                    n1best = max(np.dot(n1, N1[row, col, :]), np.dot(n1, N2[row, col, :]))
                    n2best = max(np.dot(n2, N1[row, col, :]), np.dot(n2, N2[row, col, :]))

                    na = n1 if n1best > n2best else n2
                    nb = N1[row, col, :] if np.dot(na, N1[row, col, :]) > np.dot(na, N2[row, col, :]) else N2[row, col, :]

                    n = na + nb
                    n[:2] = n[:2] / np.linalg.norm(n[:2]) * np.sin(theta[row, col])
                    n[2] = np.cos(theta[row, col])

                    N[row, col, :] = np.real(n)

    # Step 6: 显示和保存最匹配法线图像
    display_and_save_normals(N)

    return N  # 返回法线矩阵

def display_and_save_normals(N):
    """
    显示和保存法线图像。
    Args:
        N (numpy array): 法线矩阵 (rows x cols x 3)
    """
    # 将法线分量归一化到 [0, 255] 范围
    N_normalized = (N + 1) / 2.0 * 255  # 法线值可能在 [-1, 1]，因此归一化到 [0, 255]
    N_normalized = np.clip(N_normalized, 0, 255).astype(np.uint8)

    # 创建RGB图像
    normal_map = cv2.cvtColor(N_normalized, cv2.COLOR_BGR2RGB)

    # 显示法线图像
    plt.imshow(normal_map)
    plt.title("法线方向图")
    plt.axis('off')
    plt.show()

    # 保存法线图像
    output_path =r"C:\Users\17925\Desktop\normal.png"
    cv2.imwrite(output_path, normal_map)
    print(f"法线图像已保存至 {output_path}")
def compute_depth_map(N, mask_resized):
    """
    根据法线计算深度图。
    Args:
        N (numpy array): 法线矩阵 (rows x cols x 3)
        mask (numpy array): 掩膜 (rows x cols)
    Returns:
        depth_map (numpy array): 深度图
    """
    # 初始化深度图
    depth_map = np.zeros(N.shape[:2], dtype=np.float32)

    # 计算深度图
    for row in range(N.shape[0]):
        for col in range(N.shape[1]):
            if mask_resized[row, col]:  # 只处理前景区域
                depth_map[row, col] = -N[row, col, 2]  # 使用法线的z分量

    # 归一化深度图到 [0, 255]
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_map_normalized.astype(np.uint8)

def display_and_save_depth_map(depth_map):
    """
    显示和保存深度图像。
    Args:
        depth_map (numpy array): 深度图 (rows x cols)
    """
    # 显示深度图像
    plt.imshow(depth_map, cmap='gray')
    plt.title("深度图")
    plt.axis('off')
    plt.show()
    # 保存深度图图像
    output_path =r"C:\Users\17925\OneDrive\Desktop\depth.png"
    cv2.imwrite(output_path, depth_map)
    print(f"深度图像已保存至 {output_path}")
def load_polarization_images(dop_path, phi_path, intensity_path, mask_path=None):
    """
    加载偏振图像 (DOP, 相位角, 非偏振强度) 和掩膜图.
    """
    rho = cv2.imread(dop_path, cv2.IMREAD_GRAYSCALE) / 255.0  # 归一化
    phi = cv2.imread(phi_path, cv2.IMREAD_GRAYSCALE) / 255.0 * np.pi  # 归一化为相位角
    Iun = cv2.imread(intensity_path, cv2.IMREAD_GRAYSCALE) / 255.0  # 归一化非偏振强度

    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0  # 二值掩膜

    # 检查并调整所有图像的尺寸一致
    target_shape = Iun.shape
    rho_resized = cv2.resize(rho, (target_shape[1], target_shape[0]))
    phi_resized = cv2.resize(phi, (target_shape[1], target_shape[0]))
    if mask is not None:
        mask_resized = cv2.resize(mask, (target_shape[1], target_shape[0]))
    else:
        mask_resized = np.ones_like(rho_resized)  # 如果没有提供掩膜，使用全白掩膜

    return rho_resized, phi_resized, Iun, mask_resized

# 主程序
if __name__ == "__main__":
    # 假设光源方向和折射率等参数
    n = 1.5
    s = np.array([0,0,1])
    albedo = 0.6

    # 加载图像
    theta, phi, Iun_resized, mask_resized = load_polarization_images(
        r"C:\Users\17925\Desktop\SFP\ship\theta.png",
        r"C:\Users\17925\Desktop\SFP\ship\phi.png",
        r"C:\Users\17925\Desktop\SFP\ship\mask_resized.png",
        r"C:\Users\17925\Desktop\SFP\ship\mask_resized.png")
    # 计算法线
    N = lambertian_sfp(theta, phi, mask_resized, n, s, albedo, Iun_resized)
    # 计算并保存深度图
    depth_map = compute_depth_map(N, mask_resized)
    display_and_save_depth_map(depth_map)