import numpy as np
import matplotlib.pyplot as plt

a_vals = np.linspace(0.08, 0.95, 300)
time = 120 / a_vals + 30
cost = 350 * a_vals + 80

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 7))

sc = ax.scatter(cost, time, c=a_vals, cmap='coolwarm', s=8, alpha=0.4, edgecolors='none', label='Possible Strategies')

p1_idx = 280
plan1 = ax.scatter(cost[p1_idx], time[p1_idx], color='forestgreen', s=150, edgecolors='white', lw=2, label='Plan 1: Elevator Only', zorder=5)

p2_idx = 5
plan2 = ax.scatter(cost[p2_idx], time[p2_idx], color='goldenrod', s=150, edgecolors='white', lw=2, label='Plan 2: Rocket Only', zorder=5)

p3_idx = 110
plan3 = ax.scatter(cost[p3_idx], time[p3_idx], color='crimson', s=200, edgecolors='white', lw=2.5, label='Plan 3: Optimal Hybrid', zorder=6)

ax.set_title('Pareto Frontier: Global Comparison of Three Plans', fontsize=15, pad=15, fontweight='bold')
ax.set_xlabel('Total Mission Cost (Trillion USD)', fontsize=12)
ax.set_ylabel('Mission Duration (Years)', fontsize=12)

handles, labels = ax.get_legend_handles_labels()
order = [1, 2, 3, 0]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          frameon=True, fontsize=10, loc='upper right', facecolor='white', framealpha=0.9)

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Allocation Ratio $a$ (Elevator Usage)', fontsize=10)

ax.set_ylim(min(time)-50, max(time)+150)
ax.set_xlim(min(cost)-50, max(cost)+50)

plt.tight_layout()
plt.show()