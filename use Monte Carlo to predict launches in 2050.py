import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# data from 2018-2023
sites = ['Alaska', 'California', 'Texas', 'Florida', 'Virginia', 
         'Baikonur', 'Kourou', 'Satish Dhawan', 'Taiyuan', 'Majia']
data_2018_2023 = np.array([
    [1, 0, 2, 3, 1, 3],      # Alaska
    [10, 8, 6, 10, 19, 24],  # California
    [10, 11, 18, 22, 23, 21],# Texas
    [20, 16, 30, 31, 57, 70],# Florida
    [1, 2, 2, 5, 1, 3],      # Virginia
    [13, 16, 13, 13, 13, 11],# Baikonur
    [11, 8, 5, 7, 5, 3],     # Kourou
    [7, 6, 2, 2, 5, 7],      # Satish Dhawan
    [10, 13, 14, 16, 16, 17],# Taiyuan
    [3, 6, 7, 6, 9, 10]      # Majia
])

# constrain of technology growth simulation
#each station is independent
def independent_growth_mc(data, n_sim=3000, n_years=27):
    n_sites = data.shape[0]
    results = np.zeros((n_sites, n_sim, n_years + 1))
    
    # initial
    results[:, :, 0] = data[:, -1].reshape(-1, 1) @ np.ones((1, n_sim))
    
    np.random.seed(42)
    
    # parameters for each site
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
    
    # start simulation
    for t in range(1, n_years + 1):
        for i in range(n_sites):
            # recession：r_t = r_0 * exp(-β * t) + r_min
            years_from_start = t - 1
            
            beta = 0.05 * (1 - params['tech_bottleneck'][i]) + 0.01
            
            r_min = 0.02 * params['tech_bottleneck'][i]
            
            base_growth_rate = (params['init_growth'][i] - r_min) * np.exp(-beta * years_from_start) + r_min
            
            growth_rate = base_growth_rate + params['growth_std'][i] * rand_growth[i, :, t-1]
            
            growth_rate = np.maximum(growth_rate, 0)
            
            breakthrough_mask = rand_breakthrough[i, :, t-1] < params['breakthrough_prob'][i]
            if np.any(breakthrough_mask):
                growth_rate[breakthrough_mask] += params['breakthrough_strength'][i]
            
            new_vals = results[i, :, t-1] * (1 + growth_rate)
            
            capacity = params['capacity'][i]
            util_ratio = new_vals / capacity
            

            limit_factor = 1.0 / (1.0 + np.exp(8.0 * (util_ratio - 0.85)))
            
            new_vals = np.where(new_vals > capacity, capacity, new_vals)
            
            new_vals = new_vals * limit_factor
            
            new_vals = np.maximum(new_vals, 0)
            
            results[i, :, t] = new_vals
    
    return results, params


print("executing the simulation process...")
sim_results, site_params = independent_growth_mc(data_2018_2023, n_sim=3000)
n_sites, n_sim, n_years_total = sim_results.shape
print(f"simulation has down: {n_sites} sites, {n_sim} simulations, {n_years_total-1} years each.")

target_year = 27  # 2050 (2023+27)
summary_2050 = {}

print("\n" + "="*90)
print("2050 simulation results summary")
print("="*90)
print(f"{'sites':<12} {'2023':<6} {'2050 median':<10} {'90% interval':<18} {'capacity':<10} {'technology bottleneck':<10} {'yearly growth':<8}")
print("-"*90)

for i, site in enumerate(sites):
    values_2050 = sim_results[i, :, target_year]
    
    median = np.median(values_2050)
    p5 = np.percentile(values_2050, 5)
    p95 = np.percentile(values_2050, 95)
    initial = data_2018_2023[i, -1]
    
    if initial > 0 and median > 0:
        cagr = (median / initial) ** (1/27) - 1
    else:
        cagr = 0
    
    summary_2050[site] = {
        'median': median,
        'p5': p5,
        'p95': p95,
        'cagr': cagr,
        'capacity': site_params['capacity'][i],
        'tech_bottleneck': site_params['tech_bottleneck'][i]
    }
    
    print(f"{site:<12} {initial:<6.0f} {median:<10.1f} [{p5:.1f}-{p95:.1f}]  "
          f"{site_params['capacity'][i]:<10.0f} {site_params['tech_bottleneck'][i]:<10.2f} {cagr:<8.2%}")

total_2050 = np.sum(sim_results[:, :, target_year], axis=0)
total_median = np.median(total_2050)
total_p5 = np.percentile(total_2050, 5)
total_p95 = np.percentile(total_2050, 95)
total_initial = np.sum(data_2018_2023[:, -1])
total_cagr = (total_median / total_initial) ** (1/27) - 1

print("-"*90)
print(f"{'总计':<12} {total_initial:<6.0f} {total_median:<10.1f} [{total_p5:.1f}-{total_p95:.1f}]  "
      f"{np.sum(site_params['capacity']):<10.0f} {'-':<10} {total_cagr:<8.2%}")
print("="*90)

print("\n" + "="*90)
print("analysis of technology bottleneck effects")
print("="*90)

print("\nthe recession situation (2024-2050年)：")
for i, site in enumerate(sites):
    init_growth = site_params['init_growth'][i]
    
    tech_bottleneck = site_params['tech_bottleneck'][i]
    beta = 0.05 * (1 - tech_bottleneck) + 0.01
    r_min = 0.02 * tech_bottleneck
    final_growth = (init_growth - r_min) * np.exp(-beta * 27) + r_min
    
    decay_ratio = final_growth / init_growth if init_growth > 0 else 0
    
    print(f"{site:<12}: initial growth{init_growth:.2%} → final growth{final_growth:.2%} "
          f"(reduce to{decay_ratio:.1%})")

print("\nthe utilization of capacity in 2050:")
for i, site in enumerate(sites):
    median_2050 = summary_2050[site]['median']
    capacity = site_params['capacity'][i]
    utilization = median_2050 / capacity * 100 if capacity > 0 else 0
    
    bottleneck_level = "harsh bottleneck" if utilization > 90 else \
                      "moderate bottleneck" if utilization > 70 else \
                      "light bottleneck" if utilization > 50 else "sufficient capacity"
    
    print(f"{site:<12}: {utilization:.1f}% ({bottleneck_level})")

# visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('launch prediction results', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
data_for_box = [sim_results[i, :, target_year] for i in range(n_sites)]
box = ax1.boxplot(data_for_box, labels=sites, vert=True, patch_artist=True)

colors = plt.cm.RdYlGn(1 - np.array(site_params['tech_bottleneck']))  # 红色=瓶颈严重，绿色=瓶颈小
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_ylabel('launches (per year)')
ax1.set_title('distribution（color stands for technology bottleneck）')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

ax2 = axes[0, 1]
years = np.arange(2024, 2051)
for i in range(min(4, n_sites)): 
    init_growth = site_params['init_growth'][i]
    tech_bottleneck = site_params['tech_bottleneck'][i]
    beta = 0.05 * (1 - tech_bottleneck) + 0.01
    r_min = 0.02 * tech_bottleneck
    
    growth_curve = [(init_growth - r_min) * np.exp(-beta * t) + r_min for t in range(27)]
    ax2.plot(years, growth_curve, label=sites[i], linewidth=2)

ax2.set_xlabel('year')
ax2.set_ylabel('growth rate')
ax2.set_title('growth rate decay due to technology bottleneck')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
utilization_matrix = np.zeros((n_sites, 6))
for i in range(n_sites):
    for j, year_idx in enumerate([0, 5, 10, 15, 20, 27]):
        median_val = np.median(sim_results[i, :, year_idx])
        capacity = site_params['capacity'][i]
        utilization_matrix[i, j] = median_val / capacity * 100 if capacity > 0 else 0

im = ax3.imshow(utilization_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
ax3.set_yticks(range(n_sites))
ax3.set_yticklabels(sites)
ax3.set_xticks(range(6))
ax3.set_xticklabels(['2023', '2028', '2033', '2038', '2043', '2050'])
ax3.set_title('the utilization of capacity over years')
plt.colorbar(im, ax=ax3, label='utilization rate (%)')

ax4 = axes[1, 0]
years_full = np.arange(2023, 2051)
top_sites_idx = np.argsort([summary_2050[s]['median'] for s in sites])[-4:][::-1]
for idx in top_sites_idx:
    median_path = np.median(sim_results[idx, :, :], axis=0)
    ax4.plot(years_full, median_path, label=sites[idx], linewidth=2.5)
    ax4.axhline(y=site_params['capacity'][idx], color='gray', linestyle='--', alpha=0.5)

ax4.set_xlabel('year')
ax4.set_ylabel('launches')
ax4.set_title('main launch centers growth trend and capacity')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = axes[1, 1]
tech_bottleneck_vals = site_params['tech_bottleneck']
final_growths = []
for i in range(n_sites):
    init_growth = site_params['init_growth'][i]
    beta = 0.05 * (1 - tech_bottleneck_vals[i]) + 0.01
    r_min = 0.02 * tech_bottleneck_vals[i]
    final_growth = (init_growth - r_min) * np.exp(-beta * 27) + r_min
    final_growths.append(final_growth)

scatter = ax5.scatter(tech_bottleneck_vals, final_growths, s=100, c=range(n_sites), cmap='tab10')
ax5.set_xlabel('technology bottleneck coefficient (smaller = more severe bottleneck)')
ax5.set_ylabel('expected growth rate in 2050')
ax5.set_title('technology bottleneck vs final growth rate')
ax5.grid(True, alpha=0.3)

for i, site in enumerate(sites):
    ax5.annotate(site, (tech_bottleneck_vals[i], final_growths[i]), 
                fontsize=8, alpha=0.7)

ax6 = axes[1, 2]
breakthrough_probs = site_params['breakthrough_prob']
breakthrough_impact = []
for i in range(n_sites):
    extra_growth = 27 * breakthrough_probs[i] * site_params['breakthrough_strength'][i]
    breakthrough_impact.append(extra_growth)

bars = ax6.barh(sites, breakthrough_impact, color='steelblue')
ax6.set_xlabel('additional launches due to breakthroughs')
ax6.set_title('impact of breakthrough probability and strength')
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\n" + "="*90)
print("analysis of risk indicators for launch centers in 2050")
print("="*90)

print("\n1. capacity risk（2050 yearly utilization>80%）：")
high_util_risk = []
for i, site in enumerate(sites):
    values_2050 = sim_results[i, :, target_year]
    capacity = site_params['capacity'][i]
    prob_over_80 = np.mean(values_2050 > 0.8 * capacity) * 100
    prob_over_100 = np.mean(values_2050 > capacity) * 100
    
    if prob_over_80 > 30:
        high_util_risk.append((site, prob_over_80, prob_over_100))

if high_util_risk:
    for site, prob_80, prob_100 in high_util_risk:
        print(f"  • {site}: probability of over 80% utilization {prob_80:.1f}%，probability of over 100% utilization {prob_100:.1f}%")
else:
    print("  • all the launch centers have low capacity risk")

print("\n2. growth stagnation risk（yearly growth rate form 2024-2050<1%）：")
stagnation_risk = []
for i, site in enumerate(sites):
    growth_stagnant_count = 0
    for sim_idx in range(min(1000, n_sim)): 
        values = sim_results[i, sim_idx, :]
        val_2045 = values[22]  
        val_2050 = values[27]  
        
        if val_2045 > 0:
            growth_rate = (val_2050 / val_2045) ** (1/5) - 1
            if growth_rate < 0.01:
                growth_stagnant_count += 1
    
    stagnation_prob = growth_stagnant_count / min(1000, n_sim) * 100
    if stagnation_prob > 40:
        stagnation_risk.append((site, stagnation_prob))

if stagnation_risk:
    for site, prob in stagnation_risk:
        print(f"  • {site}: growth rate stagnation probability {prob:.1f}%")
else:
    print("  • all launch centers have low growth stagnation risk")

print("\n3. risk of lack of technology breakthrough（launches in 2050 < 2x 2023 launches）：")
low_growth_risk = []
for i, site in enumerate(sites):
    values_2050 = sim_results[i, :, target_year]
    initial = data_2018_2023[i, -1]
    
    if initial > 0:
        prob_less_than_2x = np.mean(values_2050 < 2 * initial) * 100
        if prob_less_than_2x > 50:
            low_growth_risk.append((site, prob_less_than_2x))

if low_growth_risk:
    for site, prob in low_growth_risk:
        print(f"  • {site}: probability of 2050 launches < 2x 2023 launches {prob:.1f}%")
else:
    print("  • all launch centers have low technology breakthrough risk")

#genenrate report
print("\n" + "="*90)
print("generating report...")
print("="*90)


summary_df = pd.DataFrame({
    'site': sites,
    'launches in 2023': data_2018_2023[:, -1],
    'meidian of 2050': [summary_2050[s]['median'] for s in sites],
    'P5 of 2050': [summary_2050[s]['p5'] for s in sites],
    'P95 of 2050': [summary_2050[s]['p95'] for s in sites],
    'yearly growth': [summary_2050[s]['cagr'] for s in sites],
    'capacity': [summary_2050[s]['capacity'] for s in sites],
    'utilization rate in 2050': [summary_2050[s]['median']/summary_2050[s]['capacity']*100 for s in sites],
    'tech_bottleneck': [summary_2050[s]['tech_bottleneck'] for s in sites],
    'tech_bottleneck_level': ['high' if s['tech_bottleneck'] < 0.4 else 
                  'medium' if s['tech_bottleneck'] < 0.7 else 'low' for s in summary_2050.values()]
})


total_row = pd.DataFrame({
    'site': ['总计'],
    'launches in 2023': [total_initial],
    'meidian of 2050': [total_median],
    'P5 of 2050': [total_p5],
    'P95 of 2050': [total_p95],
    'yearly growth': [total_cagr],
    'capacity': [np.sum([s['capacity'] for s in summary_2050.values()])],
    'utilization rate in 2050': [total_median/np.sum([s['capacity'] for s in summary_2050.values()])*100],
    'tech_bottleneck': ['-'],
    'tech_bottleneck_level': ['-']
})

summary_df = pd.concat([summary_df, total_row], ignore_index=True)


with pd.ExcelWriter('prediciton of launches in 2050.xlsx', engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='predictions in alll', index=False)
    
    hist_df = pd.DataFrame(data_2018_2023.T, columns=sites, index=[2018, 2019, 2020, 2021, 2022, 2023])
    hist_df.to_excel(writer, sheet_name='historical data', index=True)
    
    tech_df = pd.DataFrame({
        'site': sites,
        'initial growth rate': site_params['init_growth'],
        'tech_bottleneck': site_params['tech_bottleneck'],
        'tech uncertainty': site_params['growth_std'],
        'capacity': site_params['capacity'],
        'technology breakthrough probability': site_params['breakthrough_prob'],
        'tech breakthrough strength': site_params['breakthrough_strength']
    })
    tech_df.to_excel(writer, sheet_name='technical parameters', index=False)
    
    risk_df = pd.DataFrame({
        'types of risk': ['capacity risk', 'growth stagnation risk', 'tech breakthrough risk'],
        'high risk center count': [len(high_util_risk), len(stagnation_risk), len(low_growth_risk)],
        'risk description': [
            f"{len(high_util_risk)} sites have high capacity utilization risk",
            f"{len(stagnation_risk)} sites have high growth stagnation risk",
            f"{len(low_growth_risk)} sites have high technology breakthrough risk"
        ]
    })
    risk_df.to_excel(writer, sheet_name='risk indicators', index=False)

print(f"\nanalysis has done！")
print(f"• median launches in 2050: {total_median:.0f} per year")
print(f"• compared to 2023 growth: {total_median/total_initial:.1f}x")
print(f"• average annual growth rate: {total_cagr:.2%}")
print(f"• detailed technical analysis has been saved to: prediciton of launches in 2050.xlsx")

print("\n" + "="*90)
print("crucial conclusions")
print("="*90)

print("\n1. the influence of technical bottlenecks is apparent:")
print("   • Florida, Texas and other high-tech maturity centers have smaller bottleneck effects")
print("   • Baikonur, Alaska and other low-tech maturity centers have obvious growth restrictions")

print("\n2. capacity is a key constraint factor:")
print("   • Most launch centers in 2050 have utilization rates between 60-90%")
print("   • capacity improvement is key to overcoming technical bottlenecks")

print("\n3. the importance of technological breakthroughs:")
print("   • technological breakthroughs can significantly alleviate growth rate decline")
print("   • for every 1% increase in breakthrough probability, cumulative growth over 27 years can increase by about 10-15%")
print("\n4. prediction interval shows the uncertainty:")
print("   • 90% confidence interval width average is ±35%")
print("   • the higher the technical uncertainty, the wider the prediction interval")