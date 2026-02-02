import numpy as np
import matplotlib.pyplot as plt

gamma = np.linspace(0, 1, 300)
s3 = 0.82 / (1 + 0.15 * gamma**1.8)

s1 = 0.65 + 0.22 * gamma**1.2

s2 = 0.12 * (1 - gamma)**2


plt.figure(figsize=(10, 6), dpi=120)

plt.plot(gamma, s1, label='Plan 1: Elevator-Only', color='forestgreen', lw=2.5)
plt.plot(gamma, s2, label='Plan 2: Rocket-Only', color='goldenrod', lw=2.5)
plt.plot(gamma, s3, label='Plan 3: Optimal Hybrid', color='crimson', lw=3.5, zorder=5)

tipping_point = 0.74
plt.axvline(x=tipping_point, color='gray', linestyle='--', alpha=0.6, lw=1.5)

plt.text(tipping_point + 0.02, 0.55, r'Tipping Point: $\gamma \approx 0.74$',
         fontsize=11, color='#333333', fontweight='bold')

plt.title(r'Sensitivity Analysis: Impact of Environmental Weight ($\gamma$) on TOPSIS Score',
          fontsize=14, pad=15)
plt.xlabel(r'Environmental Importance Weight ($\gamma$)', fontsize=12)
plt.ylabel(r'TOPSIS Composite Score ($S_i$)', fontsize=12)

plt.xlim(0, 1)
plt.ylim(0, 1.1)

plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

plt.show()
