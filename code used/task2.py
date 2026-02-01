import numpy as np
import matplotlib.pyplot as plt

# ====================== Parameter Setup ======================
M = 1e8  # Total transport mass, metric tons
# Annual transport capacity under ideal conditions (tons/year)
E_elevator_perfect = 537000  # Space elevator (three harbors)
E_rocket_perfect = 88450     # Rockets (all launch sites combined)

# Cost parameters (USD)
cost_elec_per_ton = 1900          # Elevator electricity cost, USD/ton
cost_rocket_per_ton = 6.8e6       # Rocket launch cost, USD/ton (based on 850M per 125 tons)
maintain_elevator_per_year = 1e8  # Elevator annual maintenance, USD/year
maintain_rocket_per_year = 1.69e10 # Rocket annual maintenance, USD/year (16.9 billion)

# Efficiency factors (assumed values, range 0~1)
w_e = 0.9   # Space elevator average efficiency (accounting for failures, delays)
w_r = 0.8   # Rocket average efficiency (accounting for launch failures, weather delays)

# Actual annual transport capacity (accounting for efficiency)
E_elevator = w_e * E_elevator_perfect
E_rocket = w_r * E_rocket_perfect

# ====================== Helper Functions: Pure Strategies ======================
def pure_elevator():
    """Pure space elevator strategy"""
    T = M / E_elevator  # Time, years
    C = cost_elec_per_ton * M + maintain_elevator_per_year * T
    return T, C

def pure_rocket():
    """Pure rocket strategy"""
    T = M / E_rocket
    C = cost_rocket_per_ton * M + maintain_rocket_per_year * T
    return T, C

# Calculate pure strategies as baselines
T_pure_e, C_pure_e = pure_elevator()
T_pure_r, C_pure_r = pure_rocket()
print(f"Pure Elevator Strategy: Time = {T_pure_e:.2f} years, Cost = {C_pure_e:.2e} USD")
print(f"Pure Rocket Strategy: Time = {T_pure_r:.2f} years, Cost = {C_pure_r:.2e} USD")

# ====================== Hybrid Strategy Calculation ======================
def hybrid_performance(a):
    """Hybrid strategy: a is the proportion assigned to elevator, returns time and cost"""
    M_e = a * M          # Mass transported by elevator
    M_r = (1 - a) * M    # Mass transported by rocket
    
    T_e = M_e / E_elevator if M_e > 0 else 0
    T_r = M_r / E_rocket if M_r > 0 else 0
    T = max(T_e, T_r)   # Parallel operation, take the longer time
    
    # Cost = elevator electricity + rocket launch + elevator maintenance + rocket maintenance
    C = (cost_elec_per_ton * M_e +
         cost_rocket_per_ton * M_r +
         maintain_elevator_per_year * T_e +
         maintain_rocket_per_year * T_r)
    
    return T, C

# ====================== Pareto Frontier Analysis ======================
# Generate a range of a values
a_values = np.linspace(0, 1, 101)  # 0 to 1, 101 points
T_values = []
C_values = []

for a in a_values:
    T, C = hybrid_performance(a)
    T_values.append(T)
    C_values.append(C)

# Normalize (divide by pure rocket values, as rocket strategy typically worst-case)
T_norm = np.array(T_values) / T_pure_r
C_norm = np.array(C_values) / C_pure_r

# Objective function: weighted sum, weights b and c (b + c = 1)
b = 0.5
c = 1 - b
F_values = b * T_norm + c * C_norm

# Find optimal a (minimizing F)
optimal_idx = np.argmin(F_values)
optimal_a = a_values[optimal_idx]
optimal_T, optimal_C = hybrid_performance(optimal_a)

print(f"\nOptimal hybrid proportion a = {optimal_a:.2%}")
print(f"Corresponding time T = {optimal_T:.2f} years")
print(f"Corresponding cost C = {optimal_C:.2e} USD")

# ====================== Visualization ======================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Pareto Frontier: Time vs Cost
ax1 = axes[0, 0]
scatter = ax1.scatter(T_norm, C_norm, c=a_values, cmap='viridis', alpha=0.7)
ax1.scatter(T_norm[optimal_idx], C_norm[optimal_idx], color='red', s=100, label='Optimal')
ax1.set_xlabel('Normalized Time (T/T_pure_rocket)')
ax1.set_ylabel('Normalized Cost (C/C_pure_rocket)')
ax1.set_title('Pareto Frontier: Time vs Cost')
ax1.legend()
ax1.grid(True)
plt.colorbar(scatter, ax=ax1, label='Elevator Proportion (a)')

# 2. Objective Function F vs a
ax2 = axes[0, 1]
ax2.plot(a_values, F_values, 'b-')
ax2.axvline(optimal_a, color='red', linestyle='--', label=f'Optimal a = {optimal_a:.2%}')
ax2.set_xlabel('a (Elevator Proportion)')
ax2.set_ylabel('Objective F')
ax2.set_title('Objective Function vs a')
ax2.legend()
ax2.grid(True)

# 3. Time and Cost vs a
ax3 = axes[1, 0]
ax3.plot(a_values, T_values, 'g-', label='Time (years)')
ax3.set_xlabel('a (Elevator Proportion)')
ax3.set_ylabel('Time (years)', color='g')
ax3.tick_params(axis='y', labelcolor='g')
ax3_2 = ax3.twinx()
ax3_2.plot(a_values, C_values, 'orange', label='Cost (USD)')
ax3_2.set_ylabel('Cost (USD)', color='orange')
ax3_2.tick_params(axis='y', labelcolor='orange')
ax3.set_title('Time and Cost vs a')
lines3, labels3 = ax3.get_legend_handles_labels()
lines3_2, labels3_2 = ax3_2.get_legend_handles_labels()
ax3.legend(lines3 + lines3_2, labels3 + labels3_2, loc='upper left')
ax3.grid(True)

# 4. Comparison of Strategies
ax4 = axes[1, 1]
categories = ['Pure Elevator', 'Pure Rocket', f'Hybrid (a={optimal_a:.2%})']
times = [T_pure_e, T_pure_r, optimal_T]
costs = [C_pure_e, C_pure_r, optimal_C]
x = np.arange(len(categories))
width = 0.35
bars1 = ax4.bar(x - width/2, times, width, label='Time (years)', color='lightgreen')
bars2 = ax4.bar(x + width/2, costs, width, label='Cost (USD)', color='lightcoral')
ax4.set_xlabel('Strategy')
ax4.set_ylabel('Value')
ax4.set_title('Comparison of Strategies')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(True)

plt.tight_layout()
# plt.show()

# ====================== Sensitivity Analysis: Optimal a under Different Efficiencies ======================
print("\n=== Sensitivity Analysis: Optimal a under Different Efficiency Combinations ===")
efficiency_combinations = [
    (0.9, 0.8),   # Baseline
    (0.8, 0.9),   # Lower elevator efficiency, higher rocket efficiency
    (0.95, 0.7),  # Higher elevator efficiency, lower rocket efficiency
    (0.7, 0.95),  # Lower elevator efficiency, higher rocket efficiency
]

results = []
for w_e_test, w_r_test in efficiency_combinations:
    E_e_test = w_e_test * E_elevator_perfect
    E_r_test = w_r_test * E_rocket_perfect
    
    # Recalculate pure rocket baseline for normalization
    T_pure_r_test = M / E_r_test
    C_pure_r_test = cost_rocket_per_ton * M + maintain_rocket_per_year * T_pure_r_test
    
    F_min = float('inf')
    a_opt = 0
    T_opt = 0
    C_opt = 0
    
    # Iterate over a to find optimal
    for a in np.linspace(0, 1, 101):
        M_e = a * M
        M_r = (1 - a) * M
        T_e = M_e / E_e_test if M_e > 0 else 0
        T_r = M_r / E_r_test if M_r > 0 else 0
        T = max(T_e, T_r)
        C = (cost_elec_per_ton * M_e +
             cost_rocket_per_ton * M_r +
             maintain_elevator_per_year * T_e +
             maintain_rocket_per_year * T_r)
        
        # Normalize
        T_n = T / T_pure_r_test
        C_n = C / C_pure_r_test
        F = b * T_n + c * C_n
        
        if F < F_min:
            F_min = F
            a_opt = a
            T_opt = T
            C_opt = C
    
    results.append((w_e_test, w_r_test, a_opt, T_opt, C_opt))
    print(f"w_e = {w_e_test:.2f}, w_r = {w_r_test:.2f} -> a = {a_opt:.2%}, T = {T_opt:.2f} years, C = {C_opt:.2e} USD")

# Print sensitivity analysis table
print("\nEfficiency Combinations and Corresponding Optimal a:")
print("w_e\tw_r\tOptimal a\tTime (years)\tCost (USD)")
for r in results:
    print(f"{r[0]:.2f}\t{r[1]:.2f}\t{r[2]:.2%}\t{r[3]:.2f}\t{r[4]:.2e}")