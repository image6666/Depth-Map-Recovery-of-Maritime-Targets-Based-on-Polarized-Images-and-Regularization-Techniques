# Depth-Map-Recovery-of-Maritime-Targets-Based-on-Polarized-Images-and-Regularization-Techniques
Depth Map Recovery
#首先根据斯托克斯原理通过原始偏振图像求解偏振度图像P
#再根据自适应阈值算法完成MASK图像以及IUN图像的求解
#再根据镜\漫反射联和光照估计模型完成对天顶角图像的求解
#再根据斯托克斯原理求解方位角图像
#再根据MASK图像、IUN图像、天顶角图像以及方位角图像，通过光源点积确定唯一法线原理求解法向量图像
#再根据五点差分法将法向量图像离散化大型稀疏矩阵的形式，并首次使用双共轭梯度法求解泊松方程获得更准确的深度信息
#最后提出自适应正则化权重选择公式对泊松方程稳定求解，以得到更准确的目标深度信息。
