import numpy as np
import matplotlib.pyplot as plt

# 1. 模拟数据（基于物流动力学模型）
a_vals = np.linspace(0.08, 0.95, 300)
time = 120 / a_vals + 30
cost = 350 * a_vals + 80

# 2. 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 7))

# 3. 绘制帕累托前沿背景点 (s=8)
sc = ax.scatter(cost, time, c=a_vals, cmap='coolwarm', s=8, alpha=0.4, edgecolors='none', label='Possible Strategies')

# --- 4. 关键点绘制（调整顺序以控制图例排列） ---

# Plan 1: 纯电梯方案
p1_idx = 280
plan1 = ax.scatter(cost[p1_idx], time[p1_idx], color='forestgreen', s=150, edgecolors='white', lw=2, label='Plan 1: Elevator Only', zorder=5)

# Plan 2: 纯火箭方案
p2_idx = 5
plan2 = ax.scatter(cost[p2_idx], time[p2_idx], color='goldenrod', s=150, edgecolors='white', lw=2, label='Plan 2: Rocket Only', zorder=5)

# Plan 3: 混合最优方案
p3_idx = 110
plan3 = ax.scatter(cost[p3_idx], time[p3_idx], color='crimson', s=200, edgecolors='white', lw=2.5, label='Plan 3: Optimal Hybrid', zorder=6)

# 5. 美化细节
ax.set_title('Pareto Frontier: Global Comparison of Three Plans', fontsize=15, pad=15, fontweight='bold')
ax.set_xlabel('Total Mission Cost (Trillion USD)', fontsize=12)
ax.set_ylabel('Mission Duration (Years)', fontsize=12)

# 6. 图例与色带
# 手动控制顺序：Plan 1 -> Plan 2 -> Plan 3
handles, labels = ax.get_legend_handles_labels()
# 索引说明：0 是背景点，1 是 Plan 1，2 是 Plan 2，3 是 Plan 3
order = [1, 2, 3, 0]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          frameon=True, fontsize=10, loc='upper right', facecolor='white', framealpha=0.9)

# 调整色带
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Allocation Ratio $a$ (Elevator Usage)', fontsize=10)

# 设置坐标轴显示范围
ax.set_ylim(min(time)-50, max(time)+150)
ax.set_xlim(min(cost)-50, max(cost)+50)

plt.tight_layout()
# plt.savefig('pareto_final_123.png', dpi=600)
plt.show()