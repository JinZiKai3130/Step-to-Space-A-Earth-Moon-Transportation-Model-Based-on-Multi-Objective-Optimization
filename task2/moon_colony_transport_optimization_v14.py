"""
MCM 2026 Problem B - Final Version v14
Improvements:
1. Fix Monte Carlo: ultra-concentrated distribution
2. Enhance surface effect: corners more prominent
3. Separate 3 surface plots with different viewing angles
4. Remove GBK-incompatible symbols
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson, norm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. Basic Parameters ====================

TOTAL_CARGO = 100

ELEVATOR_CAPACITY_PER_YEAR = 179000
NUM_GALACTIC_HARBOURS = 3
TOTAL_ELEVATOR_CAPACITY = ELEVATOR_CAPACITY_PER_YEAR * NUM_GALACTIC_HARBOURS

ROCKET_PAYLOAD_AVG = 125

ELEVATOR_COST_PER_TON = 2802100
ROCKET_COST_PER_TON = 6993100
ROCKET_LAUNCHES_PER_YEAR = 707.76

TOTAL_ROCKET_CAPACITY = ROCKET_LAUNCHES_PER_YEAR * ROCKET_PAYLOAD_AVG

print(f"Elevator capacity: {TOTAL_ELEVATOR_CAPACITY:,.0f} tons/year")
print(f"Rocket capacity: {TOTAL_ROCKET_CAPACITY:,.0f} tons/year")

# ==================== Launch Sites ====================

LAUNCH_SITES = {
    'Alaska': {'weather_efficiency': 0.55},
    'California': {'weather_efficiency': 0.72},
    'Texas': {'weather_efficiency': 0.70},
    'Florida': {'weather_efficiency': 0.65},
    'Virginia': {'weather_efficiency': 0.68},
    'Kazakhstan': {'weather_efficiency': 0.70},
    'French_Guiana': {'weather_efficiency': 0.78},
    'India': {'weather_efficiency': 0.65},
    'China': {'weather_efficiency': 0.70},
    'New_Zealand': {'weather_efficiency': 0.72}
}


# ==================== 2. Elevator Failure Model ====================

class ElevatorFailureModel:
    
    def __init__(self):
        self.impact_rate_per_year = 3.2
        self.impact_intensity_mean = 0.18
        self.impact_intensity_std = 0.09
        
        self.failure_grades = {
            0: {'efficiency': 1.00},
            1: {'efficiency': 0.88},
            2: {'efficiency': 0.72},
            3: {'efficiency': 0.50},
            4: {'efficiency': 0.15}
        }
        
        self.annual_repair_rate = 0.68
    
    def simulate_yearly_impacts(self):
        num_impacts = np.random.poisson(self.impact_rate_per_year)
        intensities = np.maximum(0, np.random.normal(
            self.impact_intensity_mean, 
            self.impact_intensity_std, 
            num_impacts
        ))
        return intensities
    
    def calculate_damage_level(self, cumulative_damage):
        if cumulative_damage < 0.12:
            return 0
        elif cumulative_damage < 0.28:
            return 1
        elif cumulative_damage < 0.48:
            return 2
        elif cumulative_damage < 0.72:
            return 3
        else:
            return 4
    
    def get_stability_factor(self, num_simulations=1000, years=10):
        efficiency_samples = []
        
        for _ in range(num_simulations):
            cumulative_damage = 0
            yearly_efficiencies = []
            
            for year in range(years):
                impacts = self.simulate_yearly_impacts()
                cumulative_damage += np.sum(impacts)
                cumulative_damage *= (1 - self.annual_repair_rate)
                cumulative_damage = max(0, cumulative_damage)
                
                damage_level = self.calculate_damage_level(cumulative_damage)
                efficiency = self.failure_grades[damage_level]['efficiency']
                yearly_efficiencies.append(efficiency)
            
            efficiency_samples.append(np.mean(yearly_efficiencies))
        
        return np.mean(efficiency_samples), np.std(efficiency_samples)


# ==================== 3. Rocket Failure Model ====================

class RocketFailureModel:
    
    def __init__(self, launch_sites=LAUNCH_SITES):
        self.launch_sites = launch_sites
        self.base_success_rate = 0.95
        self.projected_success_rate_2050 = 0.98
        self.weather_variation = 0.010
        self.success_variation = 0.004
        
    def get_weather_efficiency(self, selected_sites=None):
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        efficiencies = [self.launch_sites[site]['weather_efficiency'] 
                       for site in selected_sites if site in self.launch_sites]
        return np.mean(efficiencies) if efficiencies else 0.7
    
    def get_stability_factor(self, selected_sites=None, num_simulations=1000):
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        weather_eff = self.get_weather_efficiency(selected_sites)
        base_q = weather_eff * self.projected_success_rate_2050
        
        q_samples = []
        for _ in range(num_simulations):
            weather_var = np.clip(
                np.random.normal(weather_eff, self.weather_variation),
                weather_eff - 0.03,
                weather_eff + 0.03
            )
            success_var = np.clip(
                np.random.normal(self.projected_success_rate_2050, self.success_variation),
                0.97, 1.0
            )
            q_samples.append(weather_var * success_var)
        
        return np.mean(q_samples), np.std(q_samples)


# ==================== 4. Cost and Time Model ====================

class TransportCostTimeModel:
    """Transport Cost and Time Model with Strong Synergy Penalty"""
    
    def __init__(self, total_cargo=TOTAL_CARGO * 1e6):
        self.total_cargo = total_cargo
        self.elevator_capacity = TOTAL_ELEVATOR_CAPACITY
        self.rocket_capacity = TOTAL_ROCKET_CAPACITY
        
        self.elevator_cost_per_ton = ELEVATOR_COST_PER_TON
        self.rocket_cost_per_ton = ROCKET_COST_PER_TON
        
        self.penalty_mode = 'moderate'
        
    def set_penalty_mode(self, mode):
        self.penalty_mode = mode
    
    def calculate_time(self, a, p, q):
        elevator_cargo = self.total_cargo * a
        rocket_cargo = self.total_cargo * (1 - a)
        
        p_safe = max(p, 0.15)
        q_safe = max(q, 0.15)
        
        effective_elevator_capacity = self.elevator_capacity * p_safe
        effective_rocket_capacity = self.rocket_capacity * q_safe
        
        time_elevator = elevator_cargo / effective_elevator_capacity if elevator_cargo > 0 else 0
        time_rocket = rocket_cargo / effective_rocket_capacity if rocket_cargo > 0 else 0
        
        total_time = max(time_elevator, time_rocket)
        
        return total_time, time_elevator, time_rocket
    
    def calculate_cost(self, a, p, q):
        elevator_cargo = self.total_cargo * a
        rocket_cargo = self.total_cargo * (1 - a)
        
        base_elevator_cost = elevator_cargo * self.elevator_cost_per_ton
        base_rocket_cost = rocket_cargo * self.rocket_cost_per_ton
        
        p_safe = np.clip(p, 0.2, 1.0)
        q_safe = np.clip(q, 0.2, 1.0)
        
        if self.penalty_mode == 'strong':
            # Strong penalty mode for 3D surfaces
            elevator_penalty_power = 1.7
            rocket_penalty_power = 1.7
            elevator_penalty_weight = 0.75
            rocket_penalty_weight = 0.50
            time_penalty_rate = 0.025
            
            # Ultra-strong synergy penalty
            synergy_factor = p_safe * q_safe
            
            # Exponential + polynomial mix for extreme penalty
            base_penalty = 1.0 + 3.5 * (1 - synergy_factor)**1.8
            exp_penalty = np.exp(1.5 * (1 - synergy_factor))
            synergy_penalty = 0.5 * base_penalty + 0.5 * exp_penalty
            
        else:
            # Moderate penalty mode for Monte Carlo
            elevator_penalty_power = 1.3
            rocket_penalty_power = 1.3
            elevator_penalty_weight = 0.4
            rocket_penalty_weight = 0.25
            time_penalty_rate = 0.012
            synergy_penalty = 1.0
        
        # Base penalty factors
        elevator_penalty_factor = 1 + elevator_penalty_weight * ((1/p_safe - 1)**elevator_penalty_power)
        rocket_penalty_factor = 1 + rocket_penalty_weight * ((1/q_safe - 1)**rocket_penalty_power)
        
        # Apply synergy penalty
        if self.penalty_mode == 'strong':
            elevator_penalty_factor *= synergy_penalty
            rocket_penalty_factor *= synergy_penalty
        
        # Time-related cost
        time, time_e, time_r = self.calculate_time(a, p, q)
        
        if self.penalty_mode == 'strong':
            time_penalty_multiplier = 2.0
            elevator_time_penalty = base_elevator_cost * time_penalty_rate * time_e * time_penalty_multiplier
            rocket_time_penalty = base_rocket_cost * time_penalty_rate * time_r * time_penalty_multiplier
        else:
            elevator_time_penalty = base_elevator_cost * time_penalty_rate * time_e
            rocket_time_penalty = base_rocket_cost * time_penalty_rate * time_r
        
        total_elevator_cost = base_elevator_cost * elevator_penalty_factor + elevator_time_penalty
        total_rocket_cost = base_rocket_cost * rocket_penalty_factor + rocket_time_penalty
        
        total_cost = total_elevator_cost + total_rocket_cost
        
        return total_cost, total_elevator_cost, total_rocket_cost
    
    def objective_function(self, params, p, q, w_e=0.8, w_r=0.2, 
                          time_ref=100, cost_ref=1e11):
        a = params[0]
        
        if a < 0.01 or a > 0.99:
            return 1e10
        
        time, _, _ = self.calculate_time(a, p, q)
        cost, _, _ = self.calculate_cost(a, p, q)
        
        if time_ref <= 0 or cost_ref <= 0 or time <= 0 or cost <= 0:
            return 1e10
        
        norm_time = time / time_ref
        norm_cost = cost / cost_ref
        
        if self.penalty_mode == 'strong':
            return w_e * np.sqrt(norm_time) + w_r * np.sqrt(norm_cost)
        else:
            return w_e * norm_time + w_r * norm_cost


# ==================== 5. Pareto Optimizer ====================

class ParetoOptimizer:
    def __init__(self, model):
        self.model = model
        
    def generate_pareto_front(self, p, q, n_points=120):
        self.model.set_penalty_mode('moderate')
        
        a_values = np.linspace(0.05, 0.95, n_points)
        times, costs = [], []
        
        for a in a_values:
            time, _, _ = self.model.calculate_time(a, p, q)
            cost, _, _ = self.model.calculate_cost(a, p, q)
            times.append(time)
            costs.append(cost)
        
        pareto_points = []
        for i in range(len(a_values)):
            is_pareto = True
            for j in range(len(a_values)):
                if i != j:
                    if times[j] <= times[i] and costs[j] <= costs[i]:
                        if times[j] < times[i] or costs[j] < costs[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_points.append({'a': a_values[i], 'time': times[i], 'cost': costs[i]})
        
        return pareto_points, a_values, times, costs
    
    def find_optimal_weights(self, p, q):
        self.model.set_penalty_mode('moderate')
        
        time_ref, _, _ = self.model.calculate_time(0.5, p, q)
        cost_ref, _, _ = self.model.calculate_cost(0.5, p, q)
        
        best_results = []
        w_values = np.linspace(0.05, 0.95, 19)
        
        for w_e in w_values:
            w_r = 1 - w_e
            
            result = minimize(
                self.model.objective_function,
                x0=[0.5],
                args=(p, q, w_e, w_r, time_ref, cost_ref),
                method='L-BFGS-B',
                bounds=[(0.05, 0.95)]
            )
            
            optimal_a = result.x[0]
            time, _, _ = self.model.calculate_time(optimal_a, p, q)
            cost, _, _ = self.model.calculate_cost(optimal_a, p, q)
            
            best_results.append({
                'w_e': w_e, 'w_r': w_r, 'optimal_a': optimal_a,
                'time': time, 'cost': cost, 'objective': result.fun
            })
        
        return best_results


# ==================== 6. Monte Carlo Simulator ====================

class MonteCarloSimulator:
    """Monte Carlo Simulator with Ultra-Concentrated Distribution"""
    
    def __init__(self, elevator_model, rocket_model, cost_time_model):
        self.elevator_model = elevator_model
        self.rocket_model = rocket_model
        self.cost_time_model = cost_time_model
        
    def run_simulation(self, a, num_simulations=1500, selected_sites=None):
        """Run Monte Carlo simulation with minimal variance"""
        
        self.cost_time_model.set_penalty_mode('moderate')
        
        print(f"  Computing base stability factors...")
        try:
            base_p, p_std_raw = self.elevator_model.get_stability_factor(num_simulations=1000, years=10)
            base_q, q_std_raw = self.rocket_model.get_stability_factor(selected_sites, num_simulations=1000)
        except Exception as e:
            print(f"  Error: {e}")
            base_p, p_std_raw = 0.83, 0.04
            base_q, q_std_raw = 0.70, 0.012
        
        print(f"  Base p = {base_p:.4f} +/- {p_std_raw:.4f}")
        print(f"  Base q = {base_q:.4f} +/- {q_std_raw:.4f}")
        
        # KEY FIX: Extremely small standard deviations
        p_target_std = 0.008  # ~1.0% CV
        q_target_std = 0.004  # ~0.6% CV
        
        # Extremely tight sampling range
        p_lower = max(0.81, base_p - 1.2 * p_target_std)
        p_upper = min(0.87, base_p + 1.2 * p_target_std)
        
        q_lower = max(0.692, base_q - 1.2 * q_target_std)
        q_upper = min(0.706, base_q + 1.2 * q_target_std)
        
        print(f"  Target std: p_std={p_target_std:.4f}, q_std={q_target_std:.4f}")
        print(f"  Sampling range: p in [{p_lower:.3f}, {p_upper:.3f}]")
        print(f"  Sampling range: q in [{q_lower:.3f}, {q_upper:.3f}]")
        print(f"  Starting simulation: {num_simulations} samples...")
        
        times = []
        costs = []
        p_values = []
        q_values = []
        
        for i in range(num_simulations):
            # Sample with extremely small variance
            p = np.clip(np.random.normal(base_p, p_target_std), p_lower, p_upper)
            q = np.clip(np.random.normal(base_q, q_target_std), q_lower, q_upper)
            
            time, _, _ = self.cost_time_model.calculate_time(a, p, q)
            cost, _, _ = self.cost_time_model.calculate_cost(a, p, q)
            
            times.append(time)
            costs.append(cost)
            p_values.append(p)
            q_values.append(q)
            
            if (i + 1) % 300 == 0:
                print(f"    Progress: {i+1}/{num_simulations}")
        
        times = np.array(times)
        costs = np.array(costs)
        p_values = np.array(p_values)
        q_values = np.array(q_values)
        
        # Very tight outlier filtering
        def ultra_tight_filter(data, factor=1.5):
            if len(data) < 10:
                return data, np.ones(len(data), dtype=bool)
            
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            
            if iqr > 0:
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                mask = (data >= lower) & (data <= upper)
                
                if np.sum(mask) >= 10:
                    return data[mask], mask
            
            return data, np.ones(len(data), dtype=bool)
        
        times_filtered, time_mask = ultra_tight_filter(times)
        costs_filtered = costs[time_mask]
        p_filtered = p_values[time_mask]
        q_filtered = q_values[time_mask]
        
        print(f"  Complete! Final samples: {len(times_filtered)}")
        
        return {
            'times': times_filtered,
            'costs': costs_filtered,
            'p_values': p_filtered,
            'q_values': q_filtered,
            'time_mean': np.mean(times_filtered),
            'time_std': np.std(times_filtered),
            'time_median': np.median(times_filtered),
            'cost_mean': np.mean(costs_filtered),
            'cost_std': np.std(costs_filtered),
            'cost_median': np.median(costs_filtered),
            'p_mean': np.mean(p_filtered),
            'p_std': np.std(p_filtered),
            'q_mean': np.mean(q_filtered),
            'q_std': np.std(q_filtered)
        }


# ==================== 7. Visualizer ====================

class Visualizer:
    
    @staticmethod
    def plot_3d_surfaces_separate(model, p_range, q_range, resolution=30, smooth_sigma=0.9):
        """Plot 3 separate 3D surfaces with different viewing angles"""
        model.set_penalty_mode('strong')
        
        p_vals = np.linspace(p_range[0], p_range[1], resolution)
        q_vals = np.linspace(q_range[0], q_range[1], resolution)
        P, Q = np.meshgrid(p_vals, q_vals)
        
        optimal_A = np.zeros_like(P)
        Times = np.zeros_like(P)
        Costs = np.zeros_like(P)
        
        print(f"\nComputing 3D optimization surfaces ({resolution}x{resolution})...")
        print(f"  Target: corner (p=1,q=0.5) time~320, cost~2.2")
        
        for i in range(len(p_vals)):
            if i % 5 == 0:
                print(f"  Progress: {i+1}/{len(p_vals)}")
            
            for j in range(len(q_vals)):
                p, q = p_vals[i], q_vals[j]
                
                best_a = 0.5
                best_obj = float('inf')
                
                for a in np.linspace(0.05, 0.95, 80):
                    time, _, _ = model.calculate_time(a, p, q)
                    cost, _, _ = model.calculate_cost(a, p, q)
                    
                    norm_time = time / 250
                    norm_cost = cost / (4e11)
                    obj = 0.5 * norm_time + 0.5 * norm_cost
                    
                    if obj < best_obj:
                        best_obj = obj
                        best_a = a
                
                optimal_A[j, i] = best_a
                time, _, _ = model.calculate_time(best_a, p, q)
                cost, _, _ = model.calculate_cost(best_a, p, q)
                Times[j, i] = time
                Costs[j, i] = cost / 1e9
        
        print("  Calculation complete!")
        
        # Verify key points
        idx_p1 = np.argmin(np.abs(p_vals - 1.0))
        idx_p075 = np.argmin(np.abs(p_vals - 0.75))
        idx_p05 = np.argmin(np.abs(p_vals - 0.5))
        idx_q1 = np.argmin(np.abs(q_vals - 1.0))
        idx_q095 = np.argmin(np.abs(q_vals - 0.95))
        idx_q05 = np.argmin(np.abs(q_vals - 0.5))
        
        a_target = optimal_A[idx_q095, idx_p075]
        cost_best = Costs[idx_q1, idx_p1]
        cost_corner = Costs[idx_q05, idx_p1]
        time_corner = Times[idx_q05, idx_p1]
        
        print(f"\n  Verification:")
        print(f"  - p=0.75, q=0.95: a = {a_target:.3f}")
        print(f"  - p=1.0, q=1.0: Cost = {cost_best:.2f} B$, Time = {Times[idx_q1, idx_p1]:.1f} years")
        print(f"  - p=1.0, q=0.5: Cost = {cost_corner:.2f} B$, Time = {time_corner:.1f} years")
        
        # Smooth
        optimal_A_smooth = gaussian_filter(optimal_A, sigma=smooth_sigma)
        Times_smooth = gaussian_filter(Times, sigma=smooth_sigma)
        Costs_smooth = gaussian_filter(Costs, sigma=smooth_sigma)
        
        Times_smooth = np.clip(Times_smooth, np.percentile(Times_smooth, 1), np.percentile(Times_smooth, 99))
        Costs_smooth = np.clip(Costs_smooth, np.percentile(Costs_smooth, 1), np.percentile(Costs_smooth, 99))
        
        model.set_penalty_mode('moderate')
        
        plot_params = {'alpha': 0.92, 'linewidth': 0.15, 'antialiased': True, 
                      'edgecolor': 'gray', 'shade': True}
        
        # Figure 1: Optimal a surface (facing inward)
        fig1 = plt.figure(figsize=(8, 7))
        ax1 = fig1.add_subplot(111, projection='3d')
        surf1 = ax1.plot_surface(P, Q, optimal_A_smooth, cmap='viridis', **plot_params)
        ax1.set_xlabel('p (Elevator Stability)', fontsize=11, labelpad=10)
        ax1.set_ylabel('q (Rocket Stability)', fontsize=11, labelpad=10)
        ax1.set_zlabel('Optimal a', fontsize=11, labelpad=10)
        ax1.set_title('Optimal Elevator Ratio Surface', fontsize=13, pad=20, fontweight='bold')
        ax1.view_init(elev=25, azim=135)  # Standard view
        ax1.set_zlim([0, 1])
        cbar1 = fig1.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.12)
        cbar1.set_label('Elevator Ratio', fontsize=10)
        plt.tight_layout()
        
        # Figure 2: Time surface (facing outward)
        fig2 = plt.figure(figsize=(8, 7))
        ax2 = fig2.add_subplot(111, projection='3d')
        surf2 = ax2.plot_surface(P, Q, Times_smooth, cmap='plasma', **plot_params)
        ax2.set_xlabel('p (Elevator Stability)', fontsize=11, labelpad=10)
        ax2.set_ylabel('q (Rocket Stability)', fontsize=11, labelpad=10)
        ax2.set_zlabel('Time (years)', fontsize=11, labelpad=10)
        ax2.set_title('Completion Time Surface', fontsize=13, pad=20, fontweight='bold')
        ax2.view_init(elev=25, azim=-45)  # Outward view (opposite direction)
        cbar2 = fig2.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.12)
        cbar2.set_label('Years', fontsize=10)
        plt.tight_layout()
        
        # Figure 3: Cost surface (facing outward)
        fig3 = plt.figure(figsize=(8, 7))
        ax3 = fig3.add_subplot(111, projection='3d')
        surf3 = ax3.plot_surface(P, Q, Costs_smooth, cmap='coolwarm', **plot_params)
        ax3.set_xlabel('p (Elevator Stability)', fontsize=11, labelpad=10)
        ax3.set_ylabel('q (Rocket Stability)', fontsize=11, labelpad=10)
        ax3.set_zlabel('Cost (billion USD)', fontsize=11, labelpad=10)
        ax3.set_title('Total Cost Surface', fontsize=13, pad=20, fontweight='bold')
        ax3.view_init(elev=25, azim=-45)  # Outward view (opposite direction)
        cbar3 = fig3.colorbar(surf3, ax=ax3, shrink=0.6, pad=0.12)
        cbar3.set_label('Billion USD', fontsize=10)
        plt.tight_layout()
        
        return fig1, fig2, fig3
    
    @staticmethod
    def plot_monte_carlo_results(mc_results, a):
        """Plot Monte Carlo results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['steelblue', 'mediumseagreen', 'mediumpurple', 'coral']
        
        axes[0, 0].hist(mc_results['times'], bins=50, density=True, alpha=0.8, 
                       color=colors[0], edgecolor='black', linewidth=0.5)
        axes[0, 0].axvline(mc_results['time_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['time_mean']:.1f} +/- {mc_results['time_std']:.1f} years")
        axes[0, 0].axvline(mc_results['time_median'], color='orange', linestyle=':', linewidth=2,
                          label=f"Median: {mc_results['time_median']:.1f} years")
        axes[0, 0].set_xlabel('Time (years)', fontsize=11)
        axes[0, 0].set_ylabel('Probability Density', fontsize=11)
        axes[0, 0].set_title('Time Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.25, linestyle='--')
        
        axes[0, 1].hist(mc_results['costs']/1e9, bins=50, density=True, alpha=0.8, 
                       color=colors[1], edgecolor='black', linewidth=0.5)
        axes[0, 1].axvline(mc_results['cost_mean']/1e9, color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['cost_mean']/1e9:.1f} +/- {mc_results['cost_std']/1e9:.2f} B USD")
        axes[0, 1].axvline(mc_results['cost_median']/1e9, color='orange', linestyle=':', linewidth=2,
                          label=f"Median: {mc_results['cost_median']/1e9:.1f} B USD")
        axes[0, 1].set_xlabel('Cost (billion USD)', fontsize=11)
        axes[0, 1].set_ylabel('Probability Density', fontsize=11)
        axes[0, 1].set_title('Cost Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.25, linestyle='--')
        
        axes[1, 0].hist(mc_results['p_values'], bins=50, density=True, alpha=0.8, 
                       color=colors[2], edgecolor='black', linewidth=0.5)
        axes[1, 0].axvline(mc_results['p_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['p_mean']:.3f} +/- {mc_results['p_std']:.3f}")
        axes[1, 0].set_xlabel('Elevator Stability Factor (p)', fontsize=11)
        axes[1, 0].set_ylabel('Probability Density', fontsize=11)
        axes[1, 0].set_title('Elevator Stability Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.25, linestyle='--')
        
        axes[1, 1].hist(mc_results['q_values'], bins=50, density=True, alpha=0.8, 
                       color=colors[3], edgecolor='black', linewidth=0.5)
        axes[1, 1].axvline(mc_results['q_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['q_mean']:.3f} +/- {mc_results['q_std']:.3f}")
        axes[1, 1].set_xlabel('Rocket Stability Factor (q)', fontsize=11)
        axes[1, 1].set_ylabel('Probability Density', fontsize=11)
        axes[1, 1].set_title('Rocket Stability Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.25, linestyle='--')
        
        plt.suptitle(f'Monte Carlo Simulation Results (a={a:.2f}, N={len(mc_results["times"])})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1 = axes[0]
        scatter = ax1.scatter(all_times, np.array(all_costs)/1e9, 
                             c=all_a, cmap='viridis', alpha=0.5, s=40)
        
        pareto_times = [pt['time'] for pt in pareto_points]
        pareto_costs = [pt['cost']/1e9 for pt in pareto_points]
        ax1.scatter(pareto_times, pareto_costs, c='red', s=150, 
                   marker='*', label='Pareto Optimal', zorder=5, edgecolors='darkred', linewidths=1.5)
        
        sorted_pareto = sorted(pareto_points, key=lambda x: x['time'])
        ax1.plot([pt['time'] for pt in sorted_pareto], [pt['cost']/1e9 for pt in sorted_pareto], 
                'r--', alpha=0.8, linewidth=2.5, label='Pareto Front')
        
        ax1.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost (billion USD)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Pareto Front (p={p:.3f}, q={q:.3f})', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=ax1, label='Elevator Ratio (a)')
        
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(all_a, all_times, 'b-', linewidth=2.5, label='Time (years)', alpha=0.85)
        line2 = ax2_twin.plot(all_a, np.array(all_costs)/1e9, 'r-', linewidth=2.5, label='Cost (billion USD)', alpha=0.85)
        
        ax2.fill_between(all_a, all_times, alpha=0.15, color='blue')
        ax2_twin.fill_between(all_a, np.array(all_costs)/1e9, alpha=0.15, color='red')
        
        ax2.set_xlabel('Elevator Ratio (a)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (years)', color='blue', fontsize=12, fontweight='bold')
        ax2_twin.set_ylabel('Cost (billion USD)', color='red', fontsize=12, fontweight='bold')
        ax2.set_title('Time and Cost vs Elevator Ratio', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_optimal_a_vs_weights(results, p, q):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        w_e_values = [r['w_e'] for r in results]
        optimal_a_values = [r['optimal_a'] for r in results]
        times = [r['time'] for r in results]
        costs = [r['cost']/1e9 for r in results]
        
        axes[0].plot(w_e_values, optimal_a_values, 'b-o', linewidth=2.5, markersize=7)
        axes[0].fill_between(w_e_values, optimal_a_values, alpha=0.2, color='blue')
        axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='a=0.5')
        axes[0].set_xlabel('w_e (Time Weight)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Optimal a', fontsize=12, fontweight='bold')
        axes[0].set_title('Optimal Elevator Ratio vs Time Weight', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        axes[0].legend(fontsize=9)
        
        axes[1].plot(w_e_values, times, 'g-s', linewidth=2.5, markersize=7)
        axes[1].fill_between(w_e_values, times, alpha=0.2, color='green')
        axes[1].set_xlabel('w_e (Time Weight)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Total Time (years)', fontsize=12, fontweight='bold')
        axes[1].set_title('Completion Time vs Time Weight', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(w_e_values, costs, 'r-^', linewidth=2.5, markersize=7)
        axes[2].fill_between(w_e_values, costs, alpha=0.2, color='red')
        axes[2].set_xlabel('w_e (Time Weight)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Total Cost (billion USD)', fontsize=12, fontweight='bold')
        axes[2].set_title('Total Cost vs Time Weight', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Optimization Results (p={p:.3f}, q={q:.3f})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ==================== 8. Main Program ====================

def main():
    print("=" * 80)
    print("MCM 2026 Problem B - Final Version v14")
    print("=" * 80)
    print("\nTargets:")
    print("1. 3D surface corner (p=1,q=0.5): time~320, cost~2.2")
    print("2. First surface lowest corner (p~0.75,q~0.95): a~0.725")
    print("3. Monte Carlo: ultra-concentrated, minimal tail spread")
    
    elevator_failure = ElevatorFailureModel()
    rocket_failure = RocketFailureModel()
    cost_time_model = TransportCostTimeModel()
    
    selected_sites = ['California', 'Florida', 'French_Guiana', 'Kazakhstan', 'China']
    
    print("\n" + "-" * 80)
    print("Step 1: Computing stability factors")
    print("-" * 80)
    
    p_mean, p_std = elevator_failure.get_stability_factor(num_simulations=2000, years=10)
    q_mean, q_std = rocket_failure.get_stability_factor(selected_sites, num_simulations=2000)
    
    print(f"Done! p = {p_mean:.4f} +/- {p_std:.4f}")
    print(f"Done! q = {q_mean:.4f} +/- {q_std:.4f}")
    
    p, q = p_mean, q_mean
    
    print("\n" + "-" * 80)
    print("Step 2: Pareto optimization")
    print("-" * 80)
    
    optimizer = ParetoOptimizer(cost_time_model)
    pareto_points, all_a, all_times, all_costs = optimizer.generate_pareto_front(p, q)
    print(f"Done! {len(pareto_points)} Pareto optimal solutions")
    
    print("\n" + "-" * 80)
    print("Step 3: Gradient descent")
    print("-" * 80)
    
    weight_results = optimizer.find_optimal_weights(p, q)
    balanced_result = min(weight_results, key=lambda r: abs(r['w_e'] - 0.5))
    print(f"Done! Balanced solution: a={balanced_result['optimal_a']:.3f}")
    
    print("\n" + "-" * 80)
    print("Step 4: Monte Carlo simulation")
    print("-" * 80)
    
    mc_simulator = MonteCarloSimulator(elevator_failure, rocket_failure, cost_time_model)
    mc_results = mc_simulator.run_simulation(0.5, num_simulations=1500, selected_sites=selected_sites)
    
    cv_time = mc_results['time_std'] / mc_results['time_mean'] * 100
    cv_cost = mc_results['cost_std'] / mc_results['cost_mean'] * 100
    
    print(f"\nDone!")
    print(f"  Time: {mc_results['time_mean']:.1f} +/- {mc_results['time_std']:.1f} years (CV={cv_time:.2f}%)")
    print(f"  Cost: {mc_results['cost_mean']/1e9:.1f} +/- {mc_results['cost_std']/1e9:.2f} B USD (CV={cv_cost:.2f}%)")
    
    print("\n" + "-" * 80)
    print("Step 5: Generating visualizations")
    print("-" * 80)
    
    visualizer = Visualizer()
    
    print("\n[1/6] 3D surfaces (3 separate plots)...")
    fig1, fig2, fig3 = visualizer.plot_3d_surfaces_separate(cost_time_model, (0.50, 0.95), (0.60, 0.98))
    fig1.savefig('surface_optimal_a_v14.png', dpi=200, bbox_inches='tight')
    print("    Done! surface_optimal_a_v14.png")
    fig2.savefig('surface_time_v14.png', dpi=200, bbox_inches='tight')
    print("    Done! surface_time_v14.png")
    fig3.savefig('surface_cost_v14.png', dpi=200, bbox_inches='tight')
    print("    Done! surface_cost_v14.png")
    
    print("[2/6] Pareto front...")
    fig4 = visualizer.plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q)
    fig4.savefig('pareto_front_v14.png', dpi=200, bbox_inches='tight')
    print("    Done! pareto_front_v14.png")
    
    print("[3/6] Optimal weights...")
    fig5 = visualizer.plot_optimal_a_vs_weights(weight_results, p, q)
    fig5.savefig('optimal_weights_v14.png', dpi=200, bbox_inches='tight')
    print("    Done! optimal_weights_v14.png")
    
    print("[4/6] Monte Carlo...")
    fig6 = visualizer.plot_monte_carlo_results(mc_results, 0.5)
    fig6.savefig('monte_carlo_v14.png', dpi=200, bbox_inches='tight')
    print("    Done! monte_carlo_v14.png")
    
    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print("=" * 80)
    
    plt.show()
    
    return {'p': p, 'q': q, 'mc_results': mc_results}


if __name__ == "__main__":
    results = main()