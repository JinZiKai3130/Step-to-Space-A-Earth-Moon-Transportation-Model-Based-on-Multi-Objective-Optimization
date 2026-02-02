import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sites = ['Alaska', 'California', 'Texas', 'Florida', 'Virginia',
         'Baikonur', 'Kourou', 'Satish Dhawan', 'Taiyuan', 'Majia']
data_2018_2023 = np.array([
    [1, 0, 2, 3, 1, 3],
    [10, 8, 6, 10, 19, 24],
    [10, 11, 18, 22, 23, 21],
    [20, 16, 30, 31, 57, 70],
    [1, 2, 2, 5, 1, 3],
    [13, 16, 13, 13, 13, 11],
    [11, 8, 5, 7, 5, 3],
    [7, 6, 2, 2, 5, 7],
    [10, 13, 14, 16, 16, 17],
    [3, 6, 7, 6, 9, 10]
])

def independent_growth_mc(data, n_sim=3000, n_years=27):
    n_sites = data.shape[0]
    results = np.zeros((n_sites, n_sim, n_years + 1))
    results[:, :, 0] = data[:, -1].reshape(-1, 1) @ np.ones((1, n_sim))
    np.random.seed(42)
    params = {
        'init_growth': [0.08, 0.12, 0.15, 0.18, 0.07, 0.02, 0.05, 0.10, 0.14, 0.16],
        'tech_bottleneck': [0.3, 0.6, 0.7, 0.8, 0.4, 0.2, 0.5, 0.6, 0.7, 0.65],
        'capacity': [30, 250, 300, 350, 40, 20, 60, 80, 200, 120],
        'growth_std': [0.15, 0.12, 0.10, 0.08, 0.20, 0.25, 0.18, 0.15, 0.10, 0.12],
        'breakthrough_prob': [0.01, 0.03, 0.05, 0.08, 0.02, 0.01, 0.02, 0.04, 0.06, 0.04],
        'breakthrough_strength': [0.02, 0.03, 0.04, 0.05, 0.02, 0.01, 0.02, 0.03, 0.04, 0.03]
    }
    rand_growth = np.random.randn(n_sites, n_sim, n_years)
    rand_breakthrough = np.random.rand(n_sites, n_sim, n_years)
    for t in range(1, n_years + 1):
        for i in range(n_sites):
            years_from_start = t - 1
            beta = 0.05 * (1 - params['tech_bottleneck'][i]) + 0.01
            r_min = 0.02 * params['tech_bottleneck'][i]
            base_growth_rate = (params['init_growth'][i] - r_min) * np.exp(-beta * years_from_start) + r_min
            growth_rate = base_growth_rate + params['growth_std'][i] * rand_growth[i, :, t - 1]
            growth_rate = np.maximum(growth_rate, 0)
            breakthrough_mask = rand_breakthrough[i, :, t - 1] < params['breakthrough_prob'][i]
            if np.any(breakthrough_mask):
                growth_rate[breakthrough_mask] += params['breakthrough_strength'][i]
            new_vals = results[i, :, t - 1] * (1 + growth_rate)
            capacity = params['capacity'][i]
            util_ratio = new_vals / capacity
            limit_factor = 1.0 / (1.0 + np.exp(8.0 * (util_ratio - 0.85)))
            new_vals = np.where(new_vals > capacity, capacity, new_vals)
            results[i, :, t] = np.maximum(new_vals * limit_factor, 0)
    return results, params

sim_results, site_params = independent_growth_mc(data_2018_2023, n_sim=3000)
n_sites, n_sim, n_years_total = sim_results.shape
target_year = 27
summary_2050 = {}

print(f"{'sites':<12} {'2023':<6} {'2050 median':<10} {'90% interval':<18} {'growth':<8}")
for i, site in enumerate(sites):
    values_2050 = sim_results[i, :, target_year]
    median = np.median(values_2050)
    p5, p95 = np.percentile(values_2050, [5, 95])
    initial = data_2018_2023[i, -1]
    cagr = (median / initial) ** (1 / 27) - 1 if initial > 0 and median > 0 else 0
    summary_2050[site] = {'median': median, 'p5': p5, 'p95': p95, 'cagr': cagr, 'capacity': site_params['capacity'][i], 'tech_bottleneck': site_params['tech_bottleneck'][i]}
    print(f"{site:<12} {initial:<6.0f} {median:<10.1f} [{p5:.1f}-{p95:.1f}] {cagr:<8.2%}")

total_2050 = np.sum(sim_results[:, :, target_year], axis=0)
total_median = np.median(total_2050)
total_initial = np.sum(data_2018_2023[:, -1])
total_cagr = (total_median / total_initial) ** (1 / 27) - 1
print(f"{'Total':<12} {total_initial:<6.0f} {total_median:<10.1f} [-] {total_cagr:<8.2%}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('launch prediction results', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
box = ax1.boxplot([sim_results[i, :, target_year] for i in range(n_sites)], labels=sites, vert=True, patch_artist=True)
colors = plt.cm.RdYlGn(1 - np.array(site_params['tech_bottleneck']))
for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
ax1.set_ylabel('launches (per year)')
ax1.set_title('distribution（color stands for technology bottleneck）')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

ax2 = axes[0, 1]
years = np.arange(2024, 2051)
for i in range(min(4, n_sites)):
    beta = 0.05 * (1 - site_params['tech_bottleneck'][i]) + 0.01
    r_min = 0.02 * site_params['tech_bottleneck'][i]
    growth_curve = [(site_params['init_growth'][i] - r_min) * np.exp(-beta * t) + r_min for t in range(27)]
    ax2.plot(years, growth_curve, label=sites[i], linewidth=2)
ax2.set_xlabel('year'); ax2.set_ylabel('growth rate'); ax2.set_title('growth rate decay'); ax2.legend(); ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
utilization_matrix = np.zeros((n_sites, 6))
for i in range(n_sites):
    for j, year_idx in enumerate([0, 5, 10, 15, 20, 27]):
        utilization_matrix[i, j] = (np.median(sim_results[i, :, year_idx]) / site_params['capacity'][i]) * 100
im = ax3.imshow(utilization_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
ax3.set_yticks(range(n_sites)); ax3.set_yticklabels(sites); ax3.set_xticks(range(6)); ax3.set_xticklabels(['2023', '2028', '2033', '2038', '2043', '2050']); ax3.set_title('capacity utilization (%)')
plt.colorbar(im, ax=ax3)

ax4 = axes[1, 0]
top_sites_idx = np.argsort([summary_2050[s]['median'] for s in sites])[-4:][::-1]
for idx in top_sites_idx:
    ax4.plot(np.arange(2023, 2051), np.median(sim_results[idx, :, :], axis=0), label=sites[idx], linewidth=2.5)
    ax4.axhline(y=site_params['capacity'][idx], color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('year'); ax4.set_ylabel('launches'); ax4.set_title('trends vs capacity'); ax4.legend(); ax4.grid(True, alpha=0.3)

ax5 = axes[1, 1]
final_growths = [(site_params['init_growth'][i] - (0.02 * site_params['tech_bottleneck'][i])) * np.exp(-(0.05 * (1 - site_params['tech_bottleneck'][i]) + 0.01) * 27) + (0.02 * site_params['tech_bottleneck'][i]) for i in range(n_sites)]
ax5.scatter(site_params['tech_bottleneck'], final_growths, s=100, c=range(n_sites), cmap='tab10')
ax5.set_xlabel('bottleneck coefficient'); ax5.set_ylabel('2050 growth rate'); ax5.set_title('bottleneck vs growth'); ax5.grid(True, alpha=0.3)

ax6 = axes[1, 2]
ax6.barh(sites, [27 * p * s for p, s in zip(site_params['breakthrough_prob'], site_params['breakthrough_strength'])], color='steelblue')
ax6.set_xlabel('breakthrough impact'); ax6.set_title('breakthrough contribution'); ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

summary_df = pd.DataFrame({'site': sites, 'launches_2023': data_2018_2023[:, -1], 'median_2050': [summary_2050[s]['median'] for s in sites], 'cagr': [summary_2050[s]['cagr'] for s in sites], 'capacity': [summary_2050[s]['capacity'] for s in sites]})
with pd.ExcelWriter('launch_prediction_2050.xlsx') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    pd.DataFrame(data_2018_2023.T, columns=sites).to_excel(writer, sheet_name='History')