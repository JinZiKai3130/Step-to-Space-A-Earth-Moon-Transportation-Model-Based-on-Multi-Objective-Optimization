"""
MCM 2026 Problem B - 第二问：平滑优化版本
修复：曲面平滑、图表数量调整
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

# ==================== 1. 基础参数设置 ====================

TOTAL_CARGO = 100

ELEVATOR_CAPACITY_PER_YEAR = 179000
NUM_GALACTIC_HARBOURS = 3
TOTAL_ELEVATOR_CAPACITY = ELEVATOR_CAPACITY_PER_YEAR * NUM_GALACTIC_HARBOURS

ROCKET_PAYLOAD_MIN = 100
ROCKET_PAYLOAD_MAX = 150
ROCKET_PAYLOAD_AVG = (ROCKET_PAYLOAD_MIN + ROCKET_PAYLOAD_MAX) / 2

# 成本参数
ELEVATOR_COST_PER_TON = 2802100
ROCKET_COST_PER_TON = 6993100
ROCKET_LAUNCHES_PER_YEAR = 707.76

# ==================== 发射场数据 ====================

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


# ==================== 2. 电梯故障模型 ====================

class ElevatorFailureModel:
    """太空电梯故障模型"""
    
    def __init__(self):
        self.impact_rate_per_year = 3.5
        self.impact_intensity_mean = 0.22
        self.impact_intensity_std = 0.10
        
        self.failure_grades = {
            0: {'efficiency': 1.0},
            1: {'efficiency': 0.88},
            2: {'efficiency': 0.70},
            3: {'efficiency': 0.45},
            4: {'efficiency': 0.10}
        }
        
        self.annual_repair_rate = 0.65
    
    def simulate_yearly_impacts(self):
        """模拟一年的撞击事件"""
        num_impacts = np.random.poisson(self.impact_rate_per_year)
        intensities = np.maximum(0, np.random.normal(
            self.impact_intensity_mean, 
            self.impact_intensity_std, 
            num_impacts
        ))
        return intensities
    
    def calculate_damage_level(self, cumulative_damage):
        """根据累积损伤计算故障等级"""
        if cumulative_damage < 0.12:
            return 0
        elif cumulative_damage < 0.30:
            return 1
        elif cumulative_damage < 0.50:
            return 2
        elif cumulative_damage < 0.75:
            return 3
        else:
            return 4
    
    def get_stability_factor(self, num_simulations=1000, years=10):
        """计算电梯稳定性因子 p"""
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


# ==================== 3. 火箭故障模型 ====================

class RocketFailureModel:
    """火箭故障模型"""
    
    def __init__(self, launch_sites=LAUNCH_SITES):
        self.launch_sites = launch_sites
        self.base_success_rate = 0.95
        self.projected_success_rate_2050 = 0.98
        self.weather_variation = 0.02
        self.success_variation = 0.008
        
    def get_weather_efficiency(self, selected_sites=None):
        """计算综合天气效率"""
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        efficiencies = [self.launch_sites[site]['weather_efficiency'] 
                       for site in selected_sites if site in self.launch_sites]
        return np.mean(efficiencies) if efficiencies else 0.7
    
    def get_stability_factor(self, selected_sites=None, num_simulations=1000):
        """计算火箭稳定性因子 q"""
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        weather_eff = self.get_weather_efficiency(selected_sites)
        base_q = weather_eff * self.projected_success_rate_2050
        
        q_samples = []
        for _ in range(num_simulations):
            weather_var = np.clip(
                np.random.normal(weather_eff, self.weather_variation),
                weather_eff - 0.08, weather_eff + 0.08
            )
            success_var = np.clip(
                np.random.normal(self.projected_success_rate_2050, self.success_variation),
                0.96, 1.0
            )
            q_samples.append(weather_var * success_var)
        
        return np.mean(q_samples), np.std(q_samples)


# ==================== 4. 成本与时间模型 - 平滑优化 ====================

class TransportCostTimeModel:
    """运输成本和时间模型 - 平滑版"""
    
    def __init__(self, total_cargo=TOTAL_CARGO * 1e6):
        self.total_cargo = total_cargo
        self.elevator_capacity = TOTAL_ELEVATOR_CAPACITY
        self.rocket_payload = ROCKET_PAYLOAD_AVG
        self.rockets_per_year = ROCKET_LAUNCHES_PER_YEAR
        
        self.elevator_cost_per_ton = ELEVATOR_COST_PER_TON
        self.rocket_cost_per_ton = ROCKET_COST_PER_TON
        
    def calculate_time(self, a, p, q):
        """计算完成运输任务所需时间"""
        elevator_cargo = self.total_cargo * a
        rocket_cargo = self.total_cargo * (1 - a)
        
        # ===== 关键调整1: 添加最小效率保护 =====
        p_safe = max(p, 0.1)  # 防止除零
        q_safe = max(q, 0.1)
        
        effective_elevator_capacity = self.elevator_capacity * p_safe
        effective_rocket_capacity = self.rockets_per_year * self.rocket_payload * q_safe
        
        if elevator_cargo > 0:
            time_elevator = elevator_cargo / effective_elevator_capacity
        else:
            time_elevator = 0
            
        if rocket_cargo > 0:
            time_rocket = rocket_cargo / effective_rocket_capacity
        else:
            time_rocket = 0
        
        total_time = max(time_elevator, time_rocket)
        
        return total_time, time_elevator, time_rocket
    
    def calculate_cost(self, a, p, q):
        """计算总成本 - 平滑版本"""
        elevator_cargo = self.total_cargo * a
        rocket_cargo = self.total_cargo * (1 - a)
        
        base_elevator_cost = elevator_cargo * self.elevator_cost_per_ton
        base_rocket_cost = rocket_cargo * self.rocket_cost_per_ton
        
        # ===== 关键调整2: 使用平滑的多项式惩罚函数替代指数 =====
        # 二次或三次多项式比指数函数更平滑
        p_safe = max(p, 0.1)
        q_safe = max(q, 0.1)
        
        # 多项式惩罚：随着效率降低，成本上升，但不会爆炸式增长
        elevator_penalty_factor = 1 + 0.5 * ((1/p_safe - 1)**1.0)
        rocket_penalty_factor = 1 + 0.3 * ((1/q_safe - 1)**1.0)
        
        total_elevator_cost = base_elevator_cost * elevator_penalty_factor
        total_rocket_cost = base_rocket_cost * rocket_penalty_factor
        
        total_cost = total_elevator_cost + total_rocket_cost
        
        return total_cost, total_elevator_cost, total_rocket_cost
    
    def objective_function(self, params, p, q, w_e=0.5, w_r=0.5, 
                          time_ref=100, cost_ref=1e11):
        """目标函数"""
        a = params[0]
        
        if a < 0.01 or a > 0.99:
            return 1e10
        
        time, _, _ = self.calculate_time(a, p, q)
        cost, _, _ = self.calculate_cost(a, p, q)
        
        norm_time = time / time_ref
        norm_cost = cost / cost_ref
        
        return w_e * norm_time + w_r * norm_cost


# ==================== 5. 帕累托优化器 ====================

class ParetoOptimizer:
    def __init__(self, model):
        self.model = model
        
    def generate_pareto_front(self, p, q, n_points=100):
        """生成帕累托前沿"""
        a_values = np.linspace(0.05, 0.95, n_points)
        
        times = []
        costs = []
        
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
                pareto_points.append({
                    'a': a_values[i],
                    'time': times[i],
                    'cost': costs[i]
                })
        
        return pareto_points, a_values, times, costs
    
    def find_optimal_weights(self, p, q):
        """找到最优的权重系数"""
        time_ref, _, _ = self.model.calculate_time(0.5, p, q)
        cost_ref, _, _ = self.model.calculate_cost(0.5, p, q)
        
        best_results = []
        w_values = np.linspace(0.1, 0.9, 17)
        
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
                'w_e': w_e,
                'w_r': w_r,
                'optimal_a': optimal_a,
                'time': time,
                'cost': cost,
                'objective': result.fun
            })
        
        return best_results


# ==================== 6. 蒙特卡洛模拟 ====================

class MonteCarloSimulator:
    """蒙特卡洛模拟器"""
    
    def __init__(self, elevator_model, rocket_model, cost_time_model):
        self.elevator_model = elevator_model
        self.rocket_model = rocket_model
        self.cost_time_model = cost_time_model
        
    def run_simulation(self, a, num_simulations=1000, selected_sites=None):
        """运行蒙特卡洛模拟"""
        times = []
        costs = []
        p_values = []
        q_values = []
        
        base_p, p_std = self.elevator_model.get_stability_factor(num_simulations=500, years=5)
        base_q, q_std = self.rocket_model.get_stability_factor(selected_sites, num_simulations=500)
        
        for _ in range(num_simulations):
            p = np.clip(np.random.normal(base_p, p_std), 0.3, 1.0)
            q = np.clip(np.random.normal(base_q, q_std), 0.5, 1.0)
            
            time, _, _ = self.cost_time_model.calculate_time(a, p, q)
            cost, _, _ = self.cost_time_model.calculate_cost(a, p, q)
            
            times.append(time)
            costs.append(cost)
            p_values.append(p)
            q_values.append(q)
        
        return {
            'times': np.array(times),
            'costs': np.array(costs),
            'p_values': np.array(p_values),
            'q_values': np.array(q_values),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'p_mean': np.mean(p_values),
            'p_std': np.std(p_values),
            'q_mean': np.mean(q_values),
            'q_std': np.std(q_values)
        }


# ==================== 7. 可视化 - 平滑优化 ====================

class Visualizer:
    
    @staticmethod
    def plot_3d_optimization_surface(model, p_range, q_range, resolution=30, smooth=True):
        """绘制3D优化曲面 - 平滑版"""
        
        # ===== 关键调整3: 适度的p,q范围 =====
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
                
                # 搜索最优a
                for a in np.linspace(0.1, 0.9, 40):
                    time, _, _ = model.calculate_time(a, p, q)
                    cost, _, _ = model.calculate_cost(a, p, q)
                    
                    norm_time = time / 200
                    norm_cost = cost / 5e11
                    obj = 0.5 * norm_time + 0.5 * norm_cost
                    
                    if obj < best_obj:
                        best_obj = obj
                        best_a = a
                
                optimal_A[j, i] = best_a
                time, _, _ = model.calculate_time(best_a, p, q)
                cost, _, _ = model.calculate_cost(best_a, p, q)
                Times[j, i] = time
                Costs[j, i] = cost / 1e9
        
        # ===== 关键调整4: 应用高斯平滑滤波 =====
        if smooth:
            optimal_A = gaussian_filter(optimal_A, sigma=1.5)
            Times = gaussian_filter(Times, sigma=1.5)
            Costs = gaussian_filter(Costs, sigma=1.5)
        
        # ===== 关键调整5: 裁剪异常值使曲面更平滑 =====
        Times = np.clip(Times, np.percentile(Times, 1), np.percentile(Times, 99))
        Costs = np.clip(Costs, np.percentile(Costs, 1), np.percentile(Costs, 99))
        
        # 创建3D图
        fig = plt.figure(figsize=(18, 6))
        
        # 图1: 最优a曲面
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(P, Q, optimal_A, cmap='viridis', alpha=0.95,
                                  linewidth=0.2, antialiased=True, 
                                  edgecolor='gray', shade=True)
        ax1.set_xlabel('p (Elevator Stability)', fontsize=10, labelpad=8)
        ax1.set_ylabel('q (Rocket Stability)', fontsize=10, labelpad=8)
        ax1.set_zlabel('Optimal a', fontsize=10, labelpad=8)
        ax1.set_title('Optimal Elevator Ratio Surface', fontsize=12, pad=15)
        ax1.view_init(elev=20, azim=135)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)
        
        # 图2: 时间曲面
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(P, Q, Times, cmap='plasma', alpha=0.95,
                                  linewidth=0.2, antialiased=True, 
                                  edgecolor='gray', shade=True)
        ax2.set_xlabel('p (Elevator Stability)', fontsize=10, labelpad=8)
        ax2.set_ylabel('q (Rocket Stability)', fontsize=10, labelpad=8)
        ax2.set_zlabel('Time (years)', fontsize=10, labelpad=8)
        ax2.set_title('Completion Time Surface', fontsize=12, pad=15)
        ax2.view_init(elev=20, azim=135)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)
        
        # 图3: 成本曲面
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(P, Q, Costs, cmap='coolwarm', alpha=0.95,
                                  linewidth=0.2, antialiased=True, 
                                  edgecolor='gray', shade=True)
        ax3.set_xlabel('p (Elevator Stability)', fontsize=10, labelpad=8)
        ax3.set_ylabel('q (Rocket Stability)', fontsize=10, labelpad=8)
        ax3.set_zlabel('Cost (billion $)', fontsize=10, labelpad=8)
        ax3.set_title('Total Cost Surface', fontsize=12, pad=15)
        ax3.view_init(elev=20, azim=135)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.1)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q):
        """绘制帕累托前沿"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        scatter = ax1.scatter(all_times, np.array(all_costs)/1e9, 
                             c=all_a, cmap='viridis', alpha=0.6, s=60)
        
        pareto_times = [pt['time'] for pt in pareto_points]
        pareto_costs = [pt['cost']/1e9 for pt in pareto_points]
        ax1.scatter(pareto_times, pareto_costs, c='red', s=180, 
                   marker='*', label='Pareto optimal', zorder=5, edgecolors='black', linewidths=1.5)
        
        sorted_pareto = sorted(pareto_points, key=lambda x: x['time'])
        pareto_times_sorted = [pt['time'] for pt in sorted_pareto]
        pareto_costs_sorted = [pt['cost']/1e9 for pt in sorted_pareto]
        ax1.plot(pareto_times_sorted, pareto_costs_sorted, 'r--', alpha=0.8, linewidth=2.5)
        
        ax1.set_xlabel('Time (years)', fontsize=12)
        ax1.set_ylabel('Cost (billion $)', fontsize=12)
        ax1.set_title(f'Pareto Front (p={p:.3f}, q={q:.3f})', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Elevator Ratio (a)')
        
        ax2 = axes[1]
        ax2.plot(all_a, all_times, 'b-', linewidth=2.5, label='Time (years)', alpha=0.8)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(all_a, np.array(all_costs)/1e9, 'r-', linewidth=2.5, 
                     label='Cost (billion $)', alpha=0.8)
        
        ax2.set_xlabel('Elevator Ratio (a)', fontsize=12)
        ax2.set_ylabel('Time (years)', color='blue', fontsize=12)
        ax2_twin.set_ylabel('Cost (billion $)', color='red', fontsize=12)
        ax2.set_title('Time and Cost vs Elevator Ratio', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_monte_carlo_results(mc_results, a):
        """绘制蒙特卡洛模拟结果"""
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        
        # 时间分布
        axes[0, 0].hist(mc_results['times'], bins=50, density=True, alpha=0.75, 
                       color='steelblue', edgecolor='black', linewidth=1.2)
        axes[0, 0].axvline(mc_results['time_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"μ = {mc_results['time_mean']:.1f} ± {mc_results['time_std']:.1f} years")
        axes[0, 0].set_xlabel('Time (years)', fontsize=11)
        axes[0, 0].set_ylabel('Probability Density', fontsize=11)
        axes[0, 0].set_title('Time Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # 成本分布
        axes[0, 1].hist(mc_results['costs']/1e9, bins=50, density=True, alpha=0.75, 
                       color='mediumseagreen', edgecolor='black', linewidth=1.2)
        axes[0, 1].axvline(mc_results['cost_mean']/1e9, color='red', linestyle='--', linewidth=2.5,
                          label=f"μ = {mc_results['cost_mean']/1e9:.1f} ± {mc_results['cost_std']/1e9:.1f} B$")
        axes[0, 1].set_xlabel('Cost (billion $)', fontsize=11)
        axes[0, 1].set_ylabel('Probability Density', fontsize=11)
        axes[0, 1].set_title('Cost Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # p值分布
        axes[1, 0].hist(mc_results['p_values'], bins=50, density=True, alpha=0.75, 
                       color='mediumpurple', edgecolor='black', linewidth=1.2)
        axes[1, 0].axvline(mc_results['p_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"μ = {mc_results['p_mean']:.3f} ± {mc_results['p_std']:.3f}")
        axes[1, 0].set_xlabel('Elevator Stability Factor (p)', fontsize=11)
        axes[1, 0].set_ylabel('Probability Density', fontsize=11)
        axes[1, 0].set_title('Elevator Stability Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # q值分布
        axes[1, 1].hist(mc_results['q_values'], bins=50, density=True, alpha=0.75, 
                       color='coral', edgecolor='black', linewidth=1.2)
        axes[1, 1].axvline(mc_results['q_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"μ = {mc_results['q_mean']:.3f} ± {mc_results['q_std']:.3f}")
        axes[1, 1].set_xlabel('Rocket Stability Factor (q)', fontsize=11)
        axes[1, 1].set_ylabel('Probability Density', fontsize=11)
        axes[1, 1].set_title('Rocket Stability Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'Monte Carlo Simulation Results (a={a:.2f}, N=1000)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_optimal_a_vs_weights(results, p, q):
        """绘制最优a与权重的关系"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        w_e_values = [r['w_e'] for r in results]
        optimal_a_values = [r['optimal_a'] for r in results]
        times = [r['time'] for r in results]
        costs = [r['cost']/1e9 for r in results]
        
        # 最优a
        axes[0].plot(w_e_values, optimal_a_values, 'b-o', linewidth=2.5, markersize=8)
        axes[0].fill_between(w_e_values, optimal_a_values, alpha=0.25, color='blue')
        axes[0].set_xlabel('$w_e$ (Time Weight)', fontsize=12)
        axes[0].set_ylabel('Optimal $a$ (Elevator Ratio)', fontsize=12)
        axes[0].set_title('Optimal Elevator Ratio vs Time Weight', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_ylim([0, 1])
        
        # 时间
        axes[1].plot(w_e_values, times, 'g-s', linewidth=2.5, markersize=8)
        axes[1].fill_between(w_e_values, times, alpha=0.25, color='green')
        axes[1].set_xlabel('$w_e$ (Time Weight)', fontsize=12)
        axes[1].set_ylabel('Total Time (years)', fontsize=12)
        axes[1].set_title('Completion Time vs Time Weight', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # 成本
        axes[2].plot(w_e_values, costs, 'r-^', linewidth=2.5, markersize=8)
        axes[2].fill_between(w_e_values, costs, alpha=0.25, color='red')
        axes[2].set_xlabel('$w_e$ (Time Weight)', fontsize=12)
        axes[2].set_ylabel('Total Cost (billion $)', fontsize=12)
        axes[2].set_title('Total Cost vs Time Weight', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'Optimization Results (p={p:.3f}, q={q:.3f})', 
                    fontsize=14, y=1.02, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sensitivity_analysis(model, base_p, base_q, base_a):
        """敏感性分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 变化范围
        p_range = np.linspace(0.65, 0.92, 50)
        q_range = np.linspace(0.68, 0.92, 50)
        a_range = np.linspace(0.1, 0.9, 50)
        
        # p的敏感性
        times_p = [model.calculate_time(base_a, p, base_q)[0] for p in p_range]
        costs_p = [model.calculate_cost(base_a, p, base_q)[0]/1e9 for p in p_range]
        
        axes[0, 0].plot(p_range, times_p, 'b-', linewidth=2.5)
        axes[0, 0].axvline(base_p, color='red', linestyle='--', alpha=0.7, label=f'Base p={base_p:.2f}')
        axes[0, 0].set_xlabel('p (Elevator Stability)', fontsize=11)
        axes[0, 0].set_ylabel('Time (years)', fontsize=11)
        axes[0, 0].set_title('Time Sensitivity to p', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].legend()
        
        axes[1, 0].plot(p_range, costs_p, 'r-', linewidth=2.5)
        axes[1, 0].axvline(base_p, color='blue', linestyle='--', alpha=0.7, label=f'Base p={base_p:.2f}')
        axes[1, 0].set_xlabel('p (Elevator Stability)', fontsize=11)
        axes[1, 0].set_ylabel('Cost (billion $)', fontsize=11)
        axes[1, 0].set_title('Cost Sensitivity to p', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].legend()
        
        # q的敏感性
        times_q = [model.calculate_time(base_a, base_p, q)[0] for q in q_range]
        costs_q = [model.calculate_cost(base_a, base_p, q)[0]/1e9 for q in q_range]
        
        axes[0, 1].plot(q_range, times_q, 'b-', linewidth=2.5)
        axes[0, 1].axvline(base_q, color='red', linestyle='--', alpha=0.7, label=f'Base q={base_q:.2f}')
        axes[0, 1].set_xlabel('q (Rocket Stability)', fontsize=11)
        axes[0, 1].set_ylabel('Time (years)', fontsize=11)
        axes[0, 1].set_title('Time Sensitivity to q', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].legend()
        
        axes[1, 1].plot(q_range, costs_q, 'r-', linewidth=2.5)
        axes[1, 1].axvline(base_q, color='blue', linestyle='--', alpha=0.7, label=f'Base q={base_q:.2f}')
        axes[1, 1].set_xlabel('q (Rocket Stability)', fontsize=11)
        axes[1, 1].set_ylabel('Cost (billion $)', fontsize=11)
        axes[1, 1].set_title('Cost Sensitivity to q', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        axes[1, 1].legend()
        
        # a的敏感性
        times_a = [model.calculate_time(a, base_p, base_q)[0] for a in a_range]
        costs_a = [model.calculate_cost(a, base_p, base_q)[0]/1e9 for a in a_range]
        
        axes[0, 2].plot(a_range, times_a, 'b-', linewidth=2.5)
        axes[0, 2].axvline(base_a, color='red', linestyle='--', alpha=0.7, label=f'Base a={base_a:.2f}')
        axes[0, 2].set_xlabel('a (Elevator Ratio)', fontsize=11)
        axes[0, 2].set_ylabel('Time (years)', fontsize=11)
        axes[0, 2].set_title('Time Sensitivity to a', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, linestyle='--')
        axes[0, 2].legend()
        
        axes[1, 2].plot(a_range, costs_a, 'r-', linewidth=2.5)
        axes[1, 2].axvline(base_a, color='blue', linestyle='--', alpha=0.7, label=f'Base a={base_a:.2f}')
        axes[1, 2].set_xlabel('a (Elevator Ratio)', fontsize=11)
        axes[1, 2].set_ylabel('Cost (billion $)', fontsize=11)
        axes[1, 2].set_title('Cost Sensitivity to a', fontsize=12, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3, linestyle='--')
        axes[1, 2].legend()
        
        plt.suptitle(f'Sensitivity Analysis (p={base_p:.2f}, q={base_q:.2f}, a={base_a:.2f})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ==================== 8. 主程序 ====================

def main():
    print("=" * 80)
    print("MCM 2026 Problem B - 第二问：平滑优化版本")
    print("=" * 80)
    
    elevator_failure = ElevatorFailureModel()
    rocket_failure = RocketFailureModel()
    cost_time_model = TransportCostTimeModel()
    
    selected_sites = ['California', 'Florida', 'French_Guiana', 'Kazakhstan', 'China']
    
    print("\n" + "-" * 40)
    print("1. 计算稳定性因子 p 和 q")
    print("-" * 40)
    
    p_mean, p_std = elevator_failure.get_stability_factor(num_simulations=2000, years=10)
    print(f"电梯稳定性因子 p: {p_mean:.4f} ± {p_std:.4f}")
    
    q_mean, q_std = rocket_failure.get_stability_factor(selected_sites, num_simulations=2000)
    print(f"火箭稳定性因子 q: {q_mean:.4f} ± {q_std:.4f}")
    print(f"选用发射场: {', '.join(selected_sites)}")
    
    p, q = p_mean, q_mean
    
    print("\n" + "-" * 40)
    print("2. 帕累托最优求解")
    print("-" * 40)
    
    optimizer = ParetoOptimizer(cost_time_model)
    pareto_points, all_a, all_times, all_costs = optimizer.generate_pareto_front(p, q)
    print(f"找到 {len(pareto_points)} 个帕累托最优解")
    
    print("\n" + "-" * 40)
    print("3. 梯度下降求解最优权重")
    print("-" * 40)
    
    weight_results = optimizer.find_optimal_weights(p, q)
    
    print("\n不同权重下的最优解:")
    print(f"{'w_e':^8} {'w_r':^8} {'Optimal a':^12} {'Time (yrs)':^12} {'Cost (B$)':^12}")
    print("-" * 55)
    for r in weight_results[::2]:
        print(f"{r['w_e']:^8.2f} {r['w_r']:^8.2f} {r['optimal_a']:^12.3f} "
              f"{r['time']:^12.1f} {r['cost']/1e9:^12.2f}")
    
    print("\n" + "-" * 40)
    print("4. 蒙特卡洛模拟")
    print("-" * 40)
    
    mc_simulator = MonteCarloSimulator(elevator_failure, rocket_failure, cost_time_model)
    
    test_a_values = [0.3, 0.5, 0.7]
    for test_a in test_a_values:
        mc_results = mc_simulator.run_simulation(test_a, num_simulations=1000, 
                                                  selected_sites=selected_sites)
        print(f"\na = {test_a}:")
        print(f"  时间: {mc_results['time_mean']:.1f} ± {mc_results['time_std']:.1f} 年")
        print(f"  成本: {mc_results['cost_mean']/1e9:.2f} ± {mc_results['cost_std']/1e9:.2f} 十亿美元")
        print(f"  p: {mc_results['p_mean']:.4f} ± {mc_results['p_std']:.4f}")
        print(f"  q: {mc_results['q_mean']:.4f} ± {mc_results['q_std']:.4f}")
    
    print("\n" + "-" * 40)
    print("5. 生成可视化图表（共5张）")
    print("-" * 40)
    
    visualizer = Visualizer()
    
    # 图1: 3D优化曲面（平滑版）
    # ===== 关键调整6: 使用适中的范围和分辨率 =====
    fig1 = visualizer.plot_3d_optimization_surface(
        cost_time_model, 
        p_range=(0.5, 0.95),   # 适中范围，避免极端值
        q_range=(0.55, 0.95),  # 适中范围
        resolution=25,          # 适中分辨率
        smooth=True             # 启用高斯平滑
    )
    fig1.savefig('optimization_surface_3d_smooth.png', dpi=150, bbox_inches='tight')
    print(" 已保存: optimization_surface_3d_smooth.png")
    
    # 图2: 帕累托前沿
    fig2 = visualizer.plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q)
    fig2.savefig('pareto_front.png', dpi=150, bbox_inches='tight')
    print(" 已保存: pareto_front.png")
    
    # 图3: 最优权重关系
    fig3 = visualizer.plot_optimal_a_vs_weights(weight_results, p, q)
    fig3.savefig('optimal_weights.png', dpi=150, bbox_inches='tight')
    print(" 已保存: optimal_weights.png")
    
    # 图4: 蒙特卡洛结果
    mc_results = mc_simulator.run_simulation(0.5, num_simulations=1000, 
                                              selected_sites=selected_sites)
    fig4 = visualizer.plot_monte_carlo_results(mc_results, 0.5)
    fig4.savefig('monte_carlo_results.png', dpi=150, bbox_inches='tight')
    print(" 已保存: monte_carlo_results.png")
    
    # 图5: 敏感性分析
    fig5 = visualizer.plot_sensitivity_analysis(cost_time_model, p, q, 0.5)
    fig5.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print(" 已保存: sensitivity_analysis.png")
    
    print("\n" + "-" * 40)
    print("6. 结果总结")
    print("-" * 40)
    
    balanced_result = [r for r in weight_results if abs(r['w_e'] - 0.5) < 0.05][0]
    
    print(f"""
    稳定性因子:
    - 电梯稳定性 p = {p:.4f} ± {p_std:.4f}
    - 火箭稳定性 q = {q:.4f} ± {q_std:.4f}
    
    平衡优化结果 (w_e = w_r = 0.5):
    - 最优电梯占比 a = {balanced_result['optimal_a']:.3f}
    - 预计完成时间: {balanced_result['time']:.1f} 年
    - 预计总成本: {balanced_result['cost']/1e9:.2f} 十亿美元
    
    蒙特卡洛模拟结果 (a = 0.5):
    - 时间: {mc_results['time_mean']:.1f} ± {mc_results['time_std']:.1f} 年
    - 成本: {mc_results['cost_mean']/1e9:.2f} ± {mc_results['cost_std']/1e9:.2f} 十亿美元
    - p分布: {mc_results['p_mean']:.4f} ± {mc_results['p_std']:.4f}
    - q分布: {mc_results['q_mean']:.4f} ± {mc_results['q_std']:.4f}
    """)
    
    plt.show()
    
    return {
        'p': p, 'p_std': p_std,
        'q': q, 'q_std': q_std,
        'pareto_points': pareto_points,
        'weight_results': weight_results,
        'mc_results': mc_results
    }


if __name__ == "__main__":
    results = main()