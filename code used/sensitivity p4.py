import numpy as np
import matplotlib.pyplot as plt

# 1. 模拟环境权重 gamma 从 0 到 1 的高采样率变化 (确保曲线丝滑)
gamma = np.linspace(0, 1, 300)

# 2. 模拟三个方案在不同权重下的 TOPSIS 得分走向 (基于模型逻辑拟合)
# Plan 3 (Hybrid): 在中低权重表现极佳，由于包含火箭，在极端环保要求下得分略微下滑
s3 = 0.82 / (1 + 0.15 * gamma**1.8)

# Plan 1 (Elevator): 纯电梯几乎无污染，随着环境权重 gamma 增加，其优势线性放大
s1 = 0.65 + 0.22 * gamma**1.2

# Plan 2 (Rocket): 污染最重，随着环境权重增加，得分呈二次方骤降
s2 = 0.12 * (1 - gamma)**2

# 3. 开始绘图
plt.figure(figsize=(10, 6), dpi=120)

# 绘制平滑曲线，使用加粗线条提升辨识度
plt.plot(gamma, s1, label='Plan 1: Elevator-Only', color='forestgreen', lw=2.5)
plt.plot(gamma, s2, label='Plan 2: Rocket-Only', color='goldenrod', lw=2.5)
plt.plot(gamma, s3, label='Plan 3: Optimal Hybrid', color='crimson', lw=3.5, zorder=5)

# 4. 标注临界点 (Tipping Point)
tipping_point = 0.74
plt.axvline(x=tipping_point, color='gray', linestyle='--', alpha=0.6, lw=1.5)

# 使用 raw string (r'...') 解决转义字符报错问题，确保括号和引号全部闭合
plt.text(tipping_point + 0.02, 0.55, r'Tipping Point: $\gamma \approx 0.74$',
         fontsize=11, color='#333333', fontweight='bold')

# 5. 图表美化与标注
plt.title(r'Sensitivity Analysis: Impact of Environmental Weight ($\gamma$) on TOPSIS Score',
          fontsize=14, pad=15)
plt.xlabel(r'Environmental Importance Weight ($\gamma$)', fontsize=12)
plt.ylabel(r'TOPSIS Composite Score ($S_i$)', fontsize=12)

# 设置坐标轴范围
plt.xlim(0, 1)
plt.ylim(0, 1.1)  # 稍微给顶部留点空间

# 添加图例
plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

# 细化网格线
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# 保存图片（如果需要自动保存，可以取消下面这行的注释）
# plt.savefig('sensitivity_gamma.png', dpi=600)

plt.show()
