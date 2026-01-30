import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ====================== 1. Data Preparation ======================
# Historical data (2018-2023)
sites = ['Alaska', 'California', 'Texas', 'Florida', 'Virginia', 
         'Baikonur', 'Kourou', 'Satish Dhawan', 'Taiyuan', 'Mahia']
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
    [3, 6, 7, 6, 9, 10]      # Mahia
])

# ====================== 2. Monte Carlo Simulation Function ======================
def monte_carlo_forecast(data, n_sim=5000, n_years=27, market_capacity=1800):
    """
    Simplified Monte Carlo simulation forecast
    data: 6 years historical data (6 columns)
    n_sim: number of simulations
    n_years: forecast years (2024-2050)
    market_capacity: global total market capacity
    """
    n_sites = data.shape[0]
    results = np.zeros((n_sites, n_sim, n_years + 1))
    
    # Initialization: 2023 as starting point
    results[:, :, 0] = data[:, -1].reshape(-1, 1) @ np.ones((1, n_sim))
    
    np.random.seed(42)
    
    # Parameters for each launch site
    params = {
        'growth_mean': [0.08, 0.12, 0.15, 0.18, 0.07, 0.02, 0.05, 0.10, 0.14, 0.16],  # Mean growth rate
        'growth_std': [0.15, 0.12, 0.10, 0.08, 0.20, 0.25, 0.18, 0.15, 0.10, 0.12],   # Growth rate std
        'capacity': [30, 250, 300, 350, 40, 20, 60, 80, 200, 120],                    # Capacity
        'policy': [0.6, 0.8, 0.9, 0.95, 0.7, 0.4, 0.75, 0.8, 0.9, 0.85],              # Policy support
        'tech': [0.5, 0.7, 0.8, 0.9, 0.6, 0.4, 0.7, 0.65, 0.8, 0.75]                  # Technology maturity
    }
    
    # Pre-generate random numbers
    rand_growth = np.random.randn(n_sites, n_sim, n_years)
    rand_global = np.random.randn(n_sim, n_years) * 0.1
    rand_shock = np.random.rand(n_sites, n_sim, n_years)
    
    # Year-by-year simulation
    for t in range(1, n_years + 1):
        # Independent growth for each launch site
        for i in range(n_sites):
            # Growth rate = base growth rate × policy support × tech maturity × decay factor
            decay = np.exp(-0.02 * (t-1))  # 2% annual decay
            mu = params['growth_mean'][i] * decay * params['policy'][i]
            sigma = params['growth_std'][i] * (1.5 - params['tech'][i])  # More volatility with less mature tech
            
            # Lognormal growth
            growth = np.exp(mu + sigma * rand_growth[i, :, t-1]) - 1
            new_vals = results[i, :, t-1] * (1 + growth)
            
            # Add global market factor
            new_vals *= (1 + rand_global[:, t-1])
            
            # Capacity constraint (sigmoid smoothing)
            util = new_vals / params['capacity'][i]
            limit_factor = 1 / (1 + np.exp(5 * (util - 0.8)))  # Significant constraint above 80% utilization
            new_vals *= limit_factor
            
            # Random shocks (3% probability, reduce 20-50%)
            shock_mask = rand_shock[i, :, t-1] < 0.03
            if np.any(shock_mask):
                shock_strength = 0.2 + 0.3 * np.random.rand(np.sum(shock_mask))
                new_vals[shock_mask] *= (1 - shock_strength)
            
            results[i, :, t] = np.maximum(new_vals, 0)
        
        # Market competition adjustment: ensure total doesn't exceed market capacity
        if t > 5:  # Market competition effect starts after 5 years
            total = np.sum(results[:, :, t], axis=0)
            exceed_mask = total > market_capacity
            if np.any(exceed_mask):
                scale_factor = market_capacity / total[exceed_mask]
                results[:, exceed_mask, t] *= scale_factor.reshape(1, -1)
    
    return results, params

# ====================== 3. Run Simulation ======================
print("Running Monte Carlo simulation...")
sim_results, site_params = monte_carlo_forecast(data_2018_2023, n_sim=3000)
n_sites, n_sim, n_years_total = sim_results.shape
print(f"Simulation complete: {n_sites} launch sites, {n_sim} simulations, {n_years_total-1} years forecast")

# ====================== 4. Analyze 2050 Results ======================
target_year = 27  # 2050 (2023+27)
summary_2050 = {}

print("\n" + "="*80)
print("2050 Launch Forecast for Each Launch Site (Monte Carlo Simulation)")
print("="*80)
print(f"{'Launch Site':<15} {'2023':<8} {'2050 Median':<12} {'90% Interval':<20} {'Capacity':<10} {'Growth Rate':<8}")
print("-"*80)

for i, site in enumerate(sites):
    values_2050 = sim_results[i, :, target_year]
    
    # Statistics
    median = np.median(values_2050)
    p5 = np.percentile(values_2050, 5)
    p95 = np.percentile(values_2050, 95)
    initial = data_2018_2023[i, -1]
    growth_rate = (median / initial) ** (1/27) - 1 if initial > 0 else 0
    
    summary_2050[site] = {
        'median': median,
        'p5': p5,
        'p95': p95,
        'growth_rate': growth_rate
    }
    
    print(f"{site:<15} {initial:<8.0f} {median:<12.1f} [{p5:.1f}-{p95:.1f}]  "
          f"{site_params['capacity'][i]:<10.0f} {growth_rate:<8.2%}")

# Calculate total
total_2050 = np.sum(sim_results[:, :, target_year], axis=0)
total_median = np.median(total_2050)
total_p5 = np.percentile(total_2050, 5)
total_p95 = np.percentile(total_2050, 95)

print("-"*80)
print(f"{'Total':<15} {np.sum(data_2018_2023[:,-1]):<8.0f} {total_median:<12.1f} [{total_p5:.1f}-{total_p95:.1f}]")
print("="*80)

# ====================== 5. Visualization ======================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Top 10 Launch Sites 2050 Launch Forecast Analysis', fontsize=14, fontweight='bold')

# 5.1 2050 Forecast Box Plot
ax1 = axes[0, 0]
data_for_box = [sim_results[i, :, target_year] for i in range(n_sites)]
ax1.boxplot(data_for_box, tick_labels=sites, vert=False)
ax1.set_xlabel('Launch Count (per year)')
ax1.set_title('2050 Launch Forecast Distribution by Site')
plt.setp(ax1.get_yticklabels(), fontsize=9)

# 5.2 Growth Trends (Top 5 sites)
ax2 = axes[0, 1]
years = np.arange(2023, 2051)
top_sites_idx = np.argsort([summary_2050[s]['median'] for s in sites])[-5:][::-1]
for idx in top_sites_idx:
    median_path = np.median(sim_results[idx, :, :], axis=0)
    ax2.plot(years, median_path, label=sites[idx], linewidth=2)
ax2.set_xlabel('Year')
ax2.set_ylabel('Launch Count')
ax2.set_title('Major Launch Sites Growth Trends (Median)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 5.3 2050 Total Launch Distribution
ax3 = axes[0, 2]
ax3.hist(total_2050, bins=40, edgecolor='black', alpha=0.7, density=True)
ax3.axvline(total_median, color='red', linestyle='--', label=f'Median: {total_median:.0f}')
ax3.axvline(np.mean(total_2050), color='green', linestyle='-', label=f'Mean: {np.mean(total_2050):.0f}')
ax3.set_xlabel('Global Total Launch Count')
ax3.set_ylabel('Probability Density')
ax3.set_title('2050 Global Total Launch Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 5.4 Market Share Changes
ax4 = axes[1, 0]
market_share_2023 = data_2018_2023[:, -1] / np.sum(data_2018_2023[:, -1])
market_share_2050 = np.array([summary_2050[s]['median'] for s in sites]) / total_median

x = np.arange(n_sites)
width = 0.35
ax4.barh(x - width/2, market_share_2023*100, width, label='2023', color='skyblue')
ax4.barh(x + width/2, market_share_2050*100, width, label='2050', color='lightcoral')
ax4.set_yticks(x)
ax4.set_yticklabels(sites, fontsize=9)
ax4.set_xlabel('Market Share (%)')
ax4.set_title('Market Share Changes (2023 vs 2050)')
ax4.legend()

# 5.5 Capacity Utilization
ax5 = axes[1, 1]
utilization = []
for i, site in enumerate(sites):
    median_2050 = summary_2050[site]['median']
    capacity = site_params['capacity'][i]
    utilization.append(min(median_2050 / capacity * 100, 120))  # Cap at 120%
    
colors = ['green' if u < 70 else 'orange' if u < 90 else 'red' for u in utilization]
bars = ax5.barh(sites, utilization, color=colors)
ax5.set_xlabel('Capacity Utilization (%)')
ax5.set_title('2050 Capacity Utilization')
ax5.axvline(80, color='orange', linestyle=':', alpha=0.7, label='80% Warning Line')
ax5.axvline(100, color='red', linestyle=':', alpha=0.7, label='100% Capacity')
ax5.legend(fontsize=9)

# 5.6 Growth Multiplier Ranking
ax6 = axes[1, 2]
growth_factors = []
for i, site in enumerate(sites):
    initial = data_2018_2023[i, -1]
    median_2050 = summary_2050[site]['median']
    if initial > 0:
        growth_factors.append(median_2050 / initial)
    else:
        growth_factors.append(0)

sorted_idx = np.argsort(growth_factors)[::-1]  # Descending
sorted_sites = [sites[i] for i in sorted_idx]
sorted_growth = [growth_factors[i] for i in sorted_idx]

ax6.barh(sorted_sites, sorted_growth, color='steelblue')
ax6.set_xlabel('Growth Multiplier (2050/2023)')
ax6.set_title('Growth Multiplier Ranking')
for i, (site, growth) in enumerate(zip(sorted_sites, sorted_growth)):
    ax6.text(growth + 0.1, i, f'{growth:.1f}x', va='center')

plt.tight_layout()
# plt.show()

# ====================== 6. Probabilistic Risk Assessment ======================
print("\n" + "="*80)
print("Probabilistic Risk Assessment")
print("="*80)

# 6.1 Total Launch Risk
print(f"\nGlobal Total Launch Risk:")
print(f"• 95% confidence worst-case: {np.percentile(total_2050, 5):.0f} launches/year")
print(f"• 99% confidence worst-case: {np.percentile(total_2050, 1):.0f} launches/year")
print(f"• Probability of below 1000/year: {np.mean(total_2050 < 1000)*100:.1f}%")
print(f"• Probability of exceeding 2000/year: {np.mean(total_2050 > 2000)*100:.1f}%")

# 6.2 Overcapacity Risk by Site
print(f"\nOvercapacity Risk (exceeding 80% capacity):")
high_risk_sites = []
for i, site in enumerate(sites):
    values = sim_results[i, :, target_year]
    capacity = site_params['capacity'][i]
    prob_over_80 = np.mean(values > 0.8 * capacity) * 100
    if prob_over_80 > 30:
        high_risk_sites.append((site, prob_over_80))

if high_risk_sites:
    for site, prob in high_risk_sites:
        print(f"• {site}: {prob:.1f}%")
else:
    print("• All launch sites have low risk")

# 6.3 Market Concentration Risk
print(f"\nMarket Concentration Risk:")
shares = np.array([summary_2050[s]['median'] for s in sites]) / total_median
hhi = np.sum(shares**2) * 10000  # Herfindahl-Hirschman Index
print(f"• HHI Index: {hhi:.0f}")
if hhi < 1500:
    print("• Risk Level: Low (competitive)")
elif hhi < 2500:
    print("• Risk Level: Medium (moderately concentrated)")
else:
    print("• Risk Level: High (highly monopolistic)")

# ====================== 7. Generate Excel Output ======================
print("\n" + "="*80)
print("Generating detailed report...")
print("="*80)

# Create summary table
summary_df = pd.DataFrame({
    'Launch Site': sites,
    '2023 Launches': data_2018_2023[:, -1],
    '2050 Median': [summary_2050[s]['median'] for s in sites],
    '2050 P5': [summary_2050[s]['p5'] for s in sites],
    '2050 P95': [summary_2050[s]['p95'] for s in sites],
    'Annual Growth Rate': [summary_2050[s]['growth_rate'] for s in sites],
    'Capacity': site_params['capacity'],
    'Policy Support': site_params['policy'],
    'Tech Maturity': site_params['tech']
})

# Add total row
total_row = pd.DataFrame({
    'Launch Site': ['Total'],
    '2023 Launches': [np.sum(data_2018_2023[:, -1])],
    '2050 Median': [total_median],
    '2050 P5': [total_p5],
    '2050 P95': [total_p95],
    'Annual Growth Rate': [(total_median / np.sum(data_2018_2023[:, -1])) ** (1/27) - 1],
    'Capacity': [np.sum(site_params['capacity'])],
    'Policy Support': [np.mean(site_params['policy'])],
    'Tech Maturity': [np.mean(site_params['tech'])]
})

summary_df = pd.concat([summary_df, total_row], ignore_index=True)

# Save to Excel
with pd.ExcelWriter('Launch_Sites_Forecast_2050.xlsx', engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Forecast Summary', index=False)
    
    # Historical data sheet
    hist_df = pd.DataFrame(data_2018_2023.T, columns=sites, index=[2018, 2019, 2020, 2021, 2022, 2023])
    hist_df.to_excel(writer, sheet_name='Historical Data')
    
    # Risk indicators sheet
    risk_df = pd.DataFrame({
        'Risk Indicator': ['Global Total Launch Median', '90% Confidence Lower Bound', '90% Confidence Upper Bound', 
                   'Probability Below 1000/year', 'Probability Exceeding 2000/year', 'HHI Index', 'Risk Level'],
        'Value': [f"{total_median:.0f} launches", f"{total_p5:.0f} launches", f"{total_p95:.0f} launches",
                f"{np.mean(total_2050 < 1000)*100:.1f}%", f"{np.mean(total_2050 > 2000)*100:.1f}%",
                f"{hhi:.0f}", "Low" if hhi < 1500 else "Medium" if hhi < 2500 else "High"]
    })
    risk_df.to_excel(writer, sheet_name='Risk Indicators', index=False)

print(f"\nForecast complete!")
print(f"• 2050 Global Total Launch Median: {total_median:.0f} launches/year")
print(f"• Growth multiplier vs 2023: {total_median/np.sum(data_2018_2023[:,-1]):.1f}x")
print(f"• Detailed results saved to: Launch_Sites_Forecast_2050.xlsx")