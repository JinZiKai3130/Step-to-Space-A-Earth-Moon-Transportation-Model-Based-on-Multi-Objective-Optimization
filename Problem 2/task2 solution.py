"""
MCM 2026 Problem B - No Titles Version
Silent execution with no figure titles
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
    """Transport Cost and Time Model - Fixed for realistic cost curves"""
    
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
        
        min_elevator_ops = 0.1 * self.total_cargo * self.elevator_cost_per_ton
        min_rocket_ops = 0.05 * self.total_cargo * self.rocket_cost_per_ton
        
        elevator_cost_with_min = base_elevator_cost + min_elevator_ops * (1 if a > 0.01 else 0)
        rocket_cost_with_min = base_rocket_cost + min_rocket_ops * (1 if a < 0.99 else 0)
        
        p_safe = np.clip(p, 0.2, 1.0)
        q_safe = np.clip(q, 0.2, 1.0)
        
        elevator_stability_penalty = 1.0 + 0.3 * ((1/p_safe - 1.0)**1.5)
        rocket_stability_penalty = 1.0 + 0.4 * ((1/q_safe - 1.0)**1.5)
    
        time, time_e, time_r = self.calculate_time(a, p, q)
        
        elevator_time_cost = 1 + 0.005 * time_e
        rocket_time_cost = 1 + 0.003 * time_r
    
        if a > 0.85:
            elevator_scale_penalty = 1 + 0.15 * (a - 0.85)**2
        else:
            elevator_scale_penalty = 1.0
    
        if (1-a) > 0.5:
            rocket_scale_penalty = 1 + 0.2 * ((1-a) - 0.5)**2
        else:
            rocket_scale_penalty = 1.0
    
        total_elevator_cost = (elevator_cost_with_min * elevator_stability_penalty * 
                              elevator_time_cost * elevator_scale_penalty)
    
        total_rocket_cost = (rocket_cost_with_min * rocket_stability_penalty * 
                            rocket_time_cost * rocket_scale_penalty)
    
        total_cost = total_elevator_cost + total_rocket_cost
    
        return total_cost, total_elevator_cost, total_rocket_cost
    
    def objective_function(self, params, p, q, w_e=0.5, w_r=0.5, 
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
            synergy_factor = p * q
            target_a = 0.725
            a_bias = 0.0
            
            if synergy_factor > 0.85:
                a_deviation = abs(a - target_a)
                a_bias = 0.15 * (a_deviation / 0.275)**1.5
            
            return w_e * np.sqrt(norm_time) + w_r * np.sqrt(norm_cost) + a_bias
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
        
        times = np.array(times)
        costs = np.array(costs)
        
        pareto_mask = np.ones(len(a_values), dtype=bool)
        
        for i in range(len(a_values)):
            for j in range(len(a_values)):
                if i != j:
                    if (times[j] <= times[i] and costs[j] < costs[i]) or \
                       (times[j] < times[i] and costs[j] <= costs[i]):
                        pareto_mask[i] = False
                        break
        
        pareto_a = a_values[pareto_mask]
        pareto_times = times[pareto_mask]
        pareto_costs = costs[pareto_mask]
        
        sort_idx = np.argsort(pareto_times)
        pareto_a = pareto_a[sort_idx]
        pareto_times = pareto_times[sort_idx]
        pareto_costs = pareto_costs[sort_idx]
        
        pareto_points = []
        for i in range(len(pareto_a)):
            pareto_points.append({
                'a': pareto_a[i],
                'time': pareto_times[i],
                'cost': pareto_costs[i]
            })
        
        return pareto_points, a_values, times.tolist(), costs.tolist()
    
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
    
    def __init__(self, elevator_model, rocket_model):
        self.elevator_model = elevator_model
        self.rocket_model = rocket_model
        
    def run_simulation(self, num_simulations=1000, selected_sites=None):
        p_values = []
        q_values = []
        
        base_p, p_std = self.elevator_model.get_stability_factor(num_simulations=1000, years=10)
        base_q, q_std = self.rocket_model.get_stability_factor(selected_sites, num_simulations=1000)
        
        for _ in range(num_simulations):
            p = np.clip(np.random.normal(base_p, p_std), 0.4, 1.0)
            q = np.clip(np.random.normal(base_q, q_std), 0.55, 1.0)
            
            p_values.append(p)
            q_values.append(q)
        
        return {
            'p_values': np.array(p_values),
            'q_values': np.array(q_values),
            'p_mean': np.mean(p_values),
            'p_std': np.std(p_values),
            'q_mean': np.mean(q_values),
            'q_std': np.std(q_values)
        }


# ==================== 7. Visualizer ====================

class Visualizer:
    
    @staticmethod
    def plot_3d_surfaces_merged(model, p_range, q_range, resolution=30, smooth_sigma=0.9):
        model.set_penalty_mode('strong')
        
        p_vals = np.linspace(p_range[0], p_range[1], resolution)
        q_vals = np.linspace(q_range[0], q_range[1], resolution)
        P, Q = np.meshgrid(p_vals, q_vals)
        
        optimal_A = np.zeros_like(P)
        Times = np.zeros_like(P)
        Costs = np.zeros_like(P)
        
        for i in range(len(p_vals)):
            for j in range(len(q_vals)):
                p, q = p_vals[i], q_vals[j]
                
                best_a = 0.5
                best_obj = float('inf')
                
                for a in np.linspace(0.05, 0.95, 100):
                    obj = model.objective_function([a], p, q, 0.5, 0.5, 250, 4e11)
                    
                    if obj < best_obj:
                        best_obj = obj
                        best_a = a
                
                optimal_A[j, i] = best_a
                time, _, _ = model.calculate_time(best_a, p, q)
                cost, _, _ = model.calculate_cost(best_a, p, q)
                Times[j, i] = time
                Costs[j, i] = cost / 1e9
        
        optimal_A_smooth = gaussian_filter(optimal_A, sigma=smooth_sigma)
        Times_smooth = gaussian_filter(Times, sigma=smooth_sigma)
        Costs_smooth = gaussian_filter(Costs, sigma=smooth_sigma)
        
        Times_smooth = np.clip(Times_smooth, np.percentile(Times_smooth, 1), np.percentile(Times_smooth, 99))
        Costs_smooth = np.clip(Costs_smooth, np.percentile(Costs_smooth, 1), np.percentile(Costs_smooth, 99))
        
        model.set_penalty_mode('moderate')
        
        plot_params = {'alpha': 0.92, 'linewidth': 0.15, 'antialiased': True, 
                      'edgecolor': 'gray', 'shade': True}
        
        fig1 = plt.figure(figsize=(8, 7))
        ax1 = fig1.add_subplot(111, projection='3d')
        surf1 = ax1.plot_surface(P, Q, optimal_A_smooth, cmap='viridis', **plot_params)
        ax1.set_xlabel('p (Elevator Stability)', fontsize=11, labelpad=10)
        ax1.set_ylabel('q (Rocket Stability)', fontsize=11, labelpad=10)
        ax1.set_zlabel('Optimal a', fontsize=11, labelpad=10)
        ax1.view_init(elev=25, azim=135)
        ax1.set_zlim([0.70, 1.0])
        cbar1 = fig1.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.12)
        cbar1.set_label('Elevator Ratio', fontsize=10)
        plt.tight_layout()
        
        fig2 = plt.figure(figsize=(16, 7))
        
        ax2 = fig2.add_subplot(121, projection='3d')
        surf2 = ax2.plot_surface(P, Q, Times_smooth, cmap='plasma', **plot_params)
        ax2.set_xlabel('p (Elevator Stability)', fontsize=11, labelpad=10)
        ax2.set_ylabel('q (Rocket Stability)', fontsize=11, labelpad=10)
        ax2.set_zlabel('Time (years)', fontsize=11, labelpad=10)
        ax2.view_init(elev=25, azim=-45)
        cbar2 = fig2.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.12)
        cbar2.set_label('Years', fontsize=10)
        
        ax3 = fig2.add_subplot(122, projection='3d')
        surf3 = ax3.plot_surface(P, Q, Costs_smooth, cmap='coolwarm', **plot_params)
        ax3.set_xlabel('p (Elevator Stability)', fontsize=11, labelpad=10)
        ax3.set_ylabel('q (Rocket Stability)', fontsize=11, labelpad=10)
        ax3.set_zlabel('Cost (billion USD)', fontsize=11, labelpad=10)
        ax3.view_init(elev=25, azim=-45)
        cbar3 = fig2.colorbar(surf3, ax=ax3, shrink=0.6, pad=0.12)
        cbar3.set_label('Billion USD', fontsize=10)
        
        plt.tight_layout()
        
        return fig1, fig2
    
    @staticmethod
    def plot_elevator_stability(mc_results):
        fig = plt.figure(figsize=(10, 7))
        
        plt.hist(mc_results['p_values'], bins=50, density=True, alpha=0.85, 
                 color='mediumpurple', edgecolor='black', linewidth=0.8)
        plt.axvline(mc_results['p_mean'], color='red', linestyle='--', linewidth=3,
                   label=f"Mean: {mc_results['p_mean']:.4f} ± {mc_results['p_std']:.4f}")
        
        p_median = np.median(mc_results['p_values'])
        plt.axvline(p_median, color='orange', linestyle=':', linewidth=2.5,
                   label=f"Median: {p_median:.4f}")
        
        plt.xlabel('Elevator Stability Factor (p)', fontsize=14, fontweight='bold')
        plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        textstr = f'N = {len(mc_results["p_values"])} simulations\n'
        textstr += f'Range: [{mc_results["p_values"].min():.4f}, {mc_results["p_values"].max():.4f}]'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.98, 0.97, textstr, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rocket_stability(mc_results):
        fig = plt.figure(figsize=(10, 7))
        
        plt.hist(mc_results['q_values'], bins=50, density=True, alpha=0.85, 
                 color='coral', edgecolor='black', linewidth=0.8)
        plt.axvline(mc_results['q_mean'], color='red', linestyle='--', linewidth=3,
                   label=f"Mean: {mc_results['q_mean']:.4f} ± {mc_results['q_std']:.4f}")
        
        q_median = np.median(mc_results['q_values'])
        plt.axvline(q_median, color='orange', linestyle=':', linewidth=2.5,
                   label=f"Median: {q_median:.4f}")
        
        plt.xlabel('Rocket Stability Factor (q)', fontsize=14, fontweight='bold')
        plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        textstr = f'N = {len(mc_results["q_values"])} simulations\n'
        textstr += f'Range: [{mc_results["q_values"].min():.4f}, {mc_results["q_values"].max():.4f}]'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.98, 0.97, textstr, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        all_costs_array = np.array(all_costs)
        all_times_array = np.array(all_times)
        
        ax1 = axes[0]
        
        scatter = ax1.scatter(all_times, all_costs_array/1e9, 
                             c=all_a, cmap='viridis', alpha=0.5, s=40, 
                             edgecolors='none', label='All points')
        
        pareto_times = [pt['time'] for pt in pareto_points]
        pareto_costs = [pt['cost']/1e9 for pt in pareto_points]
        
        if len(pareto_points) > 1:
            ax1.plot(pareto_times, pareto_costs, 'r--', 
                    alpha=0.8, linewidth=2.5, label='Pareto Front', zorder=4)
        
        ax1.scatter(pareto_times, pareto_costs, 
                   marker='*', s=200, c='red', 
                   edgecolors='darkred', linewidths=1.2,
                   label='Pareto Optimal', zorder=5)
        
        ax1.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost (billion USD)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Elevator Ratio (a)', fontsize=10)
        
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(all_a, all_times, 'b-', linewidth=2.5, 
                        label='Time (years)', alpha=0.85)
        line2 = ax2_twin.plot(all_a, all_costs_array/1e9, 'r-', 
                             linewidth=2.5, label='Cost (billion USD)', alpha=0.85)
        
        ax2.fill_between(all_a, all_times, alpha=0.15, color='blue')
        ax2_twin.fill_between(all_a, all_costs_array/1e9, alpha=0.15, color='red')
        
        ax2.set_xlabel('Elevator Ratio (a)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (years)', color='blue', fontsize=12, fontweight='bold')
        ax2_twin.set_ylabel('Cost (billion USD)', color='red', fontsize=12, fontweight='bold')
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
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        axes[0].legend(fontsize=9)
        
        axes[1].plot(w_e_values, times, 'g-s', linewidth=2.5, markersize=7)
        axes[1].fill_between(w_e_values, times, alpha=0.2, color='green')
        axes[1].set_xlabel('w_e (Time Weight)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Total Time (years)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(w_e_values, costs, 'r-^', linewidth=2.5, markersize=7)
        axes[2].fill_between(w_e_values, costs, alpha=0.2, color='red')
        axes[2].set_xlabel('w_e (Time Weight)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Total Cost (billion USD)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ==================== 8. Main Program ====================

def main():
    elevator_failure = ElevatorFailureModel()
    rocket_failure = RocketFailureModel()
    cost_time_model = TransportCostTimeModel()
    
    selected_sites = ['California', 'Florida', 'French_Guiana', 'Kazakhstan', 'China']
    
    p_mean, p_std = elevator_failure.get_stability_factor(num_simulations=2000, years=10)
    q_mean, q_std = rocket_failure.get_stability_factor(selected_sites, num_simulations=2000)
    
    p, q = p_mean, q_mean
    
    optimizer = ParetoOptimizer(cost_time_model)
    pareto_points, all_a, all_times, all_costs = optimizer.generate_pareto_front(p, q)
    
    weight_results = optimizer.find_optimal_weights(p, q)
    
    mc_simulator = MonteCarloSimulator(elevator_failure, rocket_failure)
    mc_results = mc_simulator.run_simulation(num_simulations=1000, selected_sites=selected_sites)
    
    visualizer = Visualizer()
    
    fig1, fig2 = visualizer.plot_3d_surfaces_merged(cost_time_model, (0.50, 0.95), (0.60, 0.98))
    fig1.savefig('surface_optimal_a_v21.png', dpi=200, bbox_inches='tight')
    fig2.savefig('surface_time_cost_merged_v21.png', dpi=200, bbox_inches='tight')
    
    fig3 = visualizer.plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q)
    fig3.savefig('pareto_front_v21.png', dpi=200, bbox_inches='tight')
    
    fig4 = visualizer.plot_optimal_a_vs_weights(weight_results, p, q)
    fig4.savefig('optimal_weights_v21.png', dpi=200, bbox_inches='tight')
    
    fig5 = visualizer.plot_elevator_stability(mc_results)
    fig5.savefig('elevator_stability_v21.png', dpi=200, bbox_inches='tight')
    
    fig6 = visualizer.plot_rocket_stability(mc_results)
    fig6.savefig('rocket_stability_v21.png', dpi=200, bbox_inches='tight')
    
    plt.show()
    
    return {'p': p, 'q': q, 'mc_results': mc_results}


if __name__ == "__main__":
    results = main()