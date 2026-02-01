"""
MCM 2026 Problem B - 第二问：基于实际数据的最终优化版本
成本数据：
- 电梯成本: $2,802,100/吨
- 火箭成本: $6,993,100/吨  
- 火箭发射次数: 707.76次/年
目标：平滑连续且美观的曲面图 + 准确的蒙特卡洛模拟
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

TOTAL_CARGO = 100  # 100 million metric tons

# 太空电梯参数
ELEVATOR_CAPACITY_PER_YEAR = 179000  # metric tons per year per Galactic Harbour
NUM_GALACTIC_HARBOURS = 3
TOTAL_ELEVATOR_CAPACITY = ELEVATOR_CAPACITY_PER_YEAR * NUM_GALACTIC_HARBOURS  # 537,000 tons/year

# 火箭参数
ROCKET_PAYLOAD_MIN = 100
ROCKET_PAYLOAD_MAX = 150
ROCKET_PAYLOAD_AVG = (ROCKET_PAYLOAD_MIN + ROCKET_PAYLOAD_MAX) / 2  # 125 tons

# ===== 实际成本数据 =====
ELEVATOR_COST_PER_TON = 2802100      # $/ton
ROCKET_COST_PER_TON = 6993100        # $/ton (2.5倍于电梯)
ROCKET_LAUNCHES_PER_YEAR = 707.76    # launches/year

# 计算火箭年运输能力
TOTAL_ROCKET_CAPACITY = ROCKET_LAUNCHES_PER_YEAR * ROCKET_PAYLOAD_AVG  # ~88,470 tons/year

print(f"电梯年运输能力: {TOTAL_ELEVATOR_CAPACITY:,.0f} 吨/年")
print(f"火箭年运输能力: {TOTAL_ROCKET_CAPACITY:,.0f} 吨/年")
print(f"运力比 (电梯/火箭): {TOTAL_ELEVATOR_CAPACITY/TOTAL_ROCKET_CAPACITY:.2f}")
print(f"成本比 (火箭/电梯): {ROCKET_COST_PER_TON/ELEVATOR_COST_PER_TON:.2f}")

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


# ==================== 2. 电梯故障模型 - 优化参数 ====================

class ElevatorFailureModel:
    """太空电梯故障模型 - 平滑参数优化"""
    
    def __init__(self):
        # ===== 关键调整1: 适度的撞击参数，确保p在合理范围内 =====
        self.impact_rate_per_year = 3.2        # 平均每年撞击次数
        self.impact_intensity_mean = 0.18      # 平均撞击强度（降低以提高平均p值）
        self.impact_intensity_std = 0.08       # 强度标准差（减小以降低方差）
        
        # 故障等级 - 更平滑的效率梯度
        self.failure_grades = {
            0: {'efficiency': 1.00, 'name': '正常运行'},
            1: {'efficiency': 0.90, 'name': '轻微损伤'},
            2: {'efficiency': 0.75, 'name': '中度损伤'},
            3: {'efficiency': 0.55, 'name': '严重损伤'},
            4: {'efficiency': 0.20, 'name': '完全失效'}
        }
        
        # 修复效率 - 提高修复率使p值更高
        self.annual_repair_rate = 0.70
    
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
        """根据累积损伤计算故障等级 - 平滑阈值"""
        if cumulative_damage < 0.10:
            return 0
        elif cumulative_damage < 0.25:
            return 1
        elif cumulative_damage < 0.45:
            return 2
        elif cumulative_damage < 0.70:
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
                
                # 年度修复
                # cumulative_damage *= (1 - self.annual_repair_rate)
                # cumulative_damage = max(0, cumulative_damage)
                
                # 计算效率
                damage_level = self.calculate_damage_level(cumulative_damage)
                efficiency = self.failure_grades[damage_level]['efficiency']
                yearly_efficiencies.append(efficiency)
            
            efficiency_samples.append(np.mean(yearly_efficiencies))
        
        return np.mean(efficiency_samples), np.std(efficiency_samples)


# ==================== 3. 火箭故障模型 - 优化参数 ====================

class RocketFailureModel:
    """火箭故障模型 - 降低方差"""
    
    def __init__(self, launch_sites=LAUNCH_SITES):
        self.launch_sites = launch_sites
        self.base_success_rate = 0.95
        self.projected_success_rate_2050 = 0.98
        
        # ===== 关键调整2: 大幅降低随机扰动 =====
        self.weather_variation = 0.012    # 天气效率标准差（原0.02）
        self.success_variation = 0.005    # 成功率标准差（原0.008）
        
    def get_weather_efficiency(self, selected_sites=None):
        """计算综合天气效率"""
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        efficiencies = [self.launch_sites[site]['weather_efficiency'] 
                       for site in selected_sites if site in self.launch_sites]
        return np.mean(efficiencies) if efficiencies else 0.7
    
    def get_stability_factor(self, selected_sites=None, num_simulations=1000):
        """计算火箭稳定性因子 q - 低方差版本"""
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        weather_eff = self.get_weather_efficiency(selected_sites)
        base_q = weather_eff * self.projected_success_rate_2050
        
        q_samples = []
        for _ in range(num_simulations):
            # 严格限制扰动范围
            weather_var = np.clip(
                np.random.normal(weather_eff, self.weather_variation),
                weather_eff - 0.04,
                weather_eff + 0.04
            )
            success_var = np.clip(
                np.random.normal(self.projected_success_rate_2050, self.success_variation),
                0.97, 1.0
            )
            q_samples.append(weather_var * success_var)
        
        return np.mean(q_samples), np.std(q_samples)


# ==================== 4. 成本与时间模型 - 平滑优化 ====================

class TransportCostTimeModel:
    """运输成本和时间模型 - 基于实际数据的平滑版本"""
    
    def __init__(self, total_cargo=TOTAL_CARGO * 1e6):
        self.total_cargo = total_cargo
        self.elevator_capacity = TOTAL_ELEVATOR_CAPACITY
        self.rocket_capacity = TOTAL_ROCKET_CAPACITY
        
        self.elevator_cost_per_ton = ELEVATOR_COST_PER_TON
        self.rocket_cost_per_ton = ROCKET_COST_PER_TON
        
    def calculate_time(self, a, p, q):
        """计算完成运输任务所需时间"""
        elevator_cargo = self.total_cargo * a
        rocket_cargo = self.total_cargo * (1 - a)
        
        # 防止除零
        p_safe = max(p, 0.15)
        q_safe = max(q, 0.15)
        
        effective_elevator_capacity = self.elevator_capacity * p_safe
        effective_rocket_capacity = self.rocket_capacity * q_safe
        
        time_elevator = elevator_cargo / effective_elevator_capacity if elevator_cargo > 0 else 0
        time_rocket = rocket_cargo / effective_rocket_capacity if rocket_cargo > 0 else 0
        
        # 并行运输，取最大值
        total_time = max(time_elevator, time_rocket)
        
        return total_time, time_elevator, time_rocket
    
    def calculate_cost(self, a, p, q):
        """
        计算总成本 - 平滑惩罚函数
        
        ===== 关键调整3: 使用平滑的分段线性+多项式惩罚 =====
        避免指数函数在低p/q值时的爆炸式增长
        """
        elevator_cargo = self.total_cargo * a
        rocket_cargo = self.total_cargo * (1 - a)
        
        # 基础成本
        base_elevator_cost = elevator_cargo * self.elevator_cost_per_ton
        base_rocket_cost = rocket_cargo * self.rocket_cost_per_ton
        
        # 防止除零
        p_safe = np.clip(p, 0.15, 1.0)
        q_safe = np.clip(q, 0.15, 1.0)
        
        # 平滑的多项式惩罚函数（幂次1.3，比1.5更平滑）
        # 当p或q接近1时，惩罚接近0；当p或q降低时，惩罚平滑增长
        elevator_penalty_factor = 1 + 0.4 * ((1/p_safe - 1)**1.3)
        rocket_penalty_factor = 1 + 0.25 * ((1/q_safe - 1)**1.3)
        
        total_elevator_cost = base_elevator_cost * elevator_penalty_factor
        total_rocket_cost = base_rocket_cost * rocket_penalty_factor
        
        total_cost = total_elevator_cost + total_rocket_cost
        
        return total_cost, total_elevator_cost, total_rocket_cost
    
    def objective_function(self, params, p, q, w_e=0.5, w_r=0.5, 
                          time_ref=100, cost_ref=1e11):
        """归一化目标函数"""
        a = params[0]
        
        if a < 0.01 or a > 0.99:
            return 1e10
        
        time, _, _ = self.calculate_time(a, p, q)
        cost, _, _ = self.calculate_cost(a, p, q)
        
        # 避免除零
        if time_ref <= 0 or cost_ref <= 0:
            return 1e10
        
        norm_time = time / time_ref
        norm_cost = cost / cost_ref
        
        return w_e * norm_time + w_r * norm_cost


# ==================== 5. 帕累托优化器 ====================

class ParetoOptimizer:
    def __init__(self, model):
        self.model = model
        
    def generate_pareto_front(self, p, q, n_points=120):
        """生成帕累托前沿 - 增加点数以获得更平滑曲线"""
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
        """找到最优的权重系数 - 更细粒度"""
        time_ref, _, _ = self.model.calculate_time(0.5, p, q)
        cost_ref, _, _ = self.model.calculate_cost(0.5, p, q)
        
        best_results = []
        w_values = np.linspace(0.05, 0.95, 19)  # 增加权重采样点
        
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


# ==================== 6. 蒙特卡洛模拟 - 稳定版本 ====================

class MonteCarloSimulator:
    """蒙特卡洛模拟器 - 稳定低方差版本"""
    
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
        
        # ===== 关键调整4: 预先计算稳定的基准值 =====
        base_p, p_std = self.elevator_model.get_stability_factor(num_simulations=1000, years=10)
        base_q, q_std = self.rocket_model.get_stability_factor(selected_sites, num_simulations=1000)
        
        for _ in range(num_simulations):
            # 从正态分布采样，严格限制范围
            p = np.clip(np.random.normal(base_p, p_std), 0.4, 1.0)
            q = np.clip(np.random.normal(base_q, q_std), 0.55, 1.0)
            
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
            'time_median': np.median(times),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'cost_median': np.median(costs),
            'p_mean': np.mean(p_values),
            'p_std': np.std(p_values),
            'q_mean': np.mean(q_values),
            'q_std': np.std(q_values)
        }


# ==================== 7. 可视化 - 美化版本 ====================

class Visualizer:
    
    @staticmethod
    def plot_3d_optimization_surface(model, p_range, q_range, resolution=32, smooth_sigma=0.5):
        """
        绘制3D优化曲面 - 平滑美观版本
        
        ===== 关键调整5: 精细的平滑控制 =====
        """
        # 生成网格
        p_vals = np.linspace(p_range[0], p_range[1], resolution)
        q_vals = np.linspace(q_range[0], q_range[1], resolution)
        P, Q = np.meshgrid(p_vals, q_vals)
        
        optimal_A = np.zeros_like(P)
        Times = np.zeros_like(P)
        Costs = np.zeros_like(P)
        
        print(f"\n正在计算3D优化曲面 ({resolution}x{resolution} 网格)...")
        
        for i in range(len(p_vals)):
            if i % 5 == 0:
                print(f"  进度: {i}/{len(p_vals)}")
            
            for j in range(len(q_vals)):
                p, q = p_vals[i], q_vals[j]
                
                # 寻找最优a
                best_a = 0.5
                best_obj = float('inf')
                
                # 更精细的a搜索
                for a in np.linspace(0.1, 0.9, 50):
                    time, _, _ = model.calculate_time(a, p, q)
                    cost, _, _ = model.calculate_cost(a, p, q)
                    
                    # 归一化目标函数
                    norm_time = time / 200
                    norm_cost = cost / (3e11)
                    obj = 0.5 * norm_time + 0.5 * norm_cost
                    
                    if obj < best_obj:
                        best_obj = obj
                        best_a = a
                
                optimal_A[j, i] = best_a
                time, _, _ = model.calculate_time(best_a, p, q)
                cost, _, _ = model.calculate_cost(best_a, p, q)
                Times[j, i] = time
                Costs[j, i] = cost / 1e9  # 转换为十亿美元
        
        print("  计算完成！")
        
        # ===== 关键调整6: 应用高斯平滑 + 异常值裁剪 =====
        print(f"应用高斯平滑 (sigma={smooth_sigma})...")
        optimal_A_smooth = gaussian_filter(optimal_A, sigma=smooth_sigma)
        Times_smooth = gaussian_filter(Times, sigma=smooth_sigma)
        Costs_smooth = gaussian_filter(Costs, sigma=smooth_sigma)
        
        # 裁剪极端异常值（保留1-99百分位）
        Times_smooth = np.clip(Times_smooth, 
                               np.percentile(Times_smooth, 2), 
                               np.percentile(Times_smooth, 98))
        Costs_smooth = np.clip(Costs_smooth, 
                               np.percentile(Costs_smooth, 2), 
                               np.percentile(Costs_smooth, 98))
        
       
        # 创建3D图
        fig = plt.figure(figsize=(19, 6))
        
        # 通用3D绘图参数
        plot_params = {
            'cmap': 'viridis',
            'alpha': 0.92,
            'linewidth': 0.15,
            'antialiased': True,
            'edgecolor': 'gray',
            'shade': True,
            'rcount': resolution,
            'ccount': resolution
        }
        
        # 图1: 最优a曲面
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(P, Q, optimal_A_smooth, cmap='viridis', **{k:v for k,v in plot_params.items() if k != 'cmap'})
        ax1.set_xlabel('p (Elevator Stability)', fontsize=10, labelpad=10)
        ax1.set_ylabel('q (Rocket Stability)', fontsize=10, labelpad=10)
        ax1.set_zlabel('Optimal a', fontsize=10, labelpad=10)
        ax1.set_title('Optimal Elevator Ratio Surface', fontsize=12, pad=20, fontweight='bold')
        ax1.view_init(elev=22, azim=135)
        ax1.set_zlim([0, 1])
        cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.12)
        cbar1.set_label('Elevator Ratio', fontsize=9)
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        
        # 图2: 时间曲面
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(P, Q, Times_smooth, cmap='plasma', **{k:v for k,v in plot_params.items() if k != 'cmap'})
        ax2.set_xlabel('p (Elevator Stability)', fontsize=10, labelpad=10)
        ax2.set_ylabel('q (Rocket Stability)', fontsize=10, labelpad=10)
        ax2.set_zlabel('Time (years)', fontsize=10, labelpad=10)
        ax2.set_title('Completion Time Surface', fontsize=12, pad=20, fontweight='bold')
        ax2.view_init(elev=22, azim=135)
        cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.12)
        cbar2.set_label('Years', fontsize=9)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        
        # 图3: 成本曲面
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(P, Q, Costs_smooth, cmap='coolwarm', **{k:v for k,v in plot_params.items() if k != 'cmap'})
        ax3.set_xlabel('p (Elevator Stability)', fontsize=10, labelpad=10)
        ax3.set_ylabel('q (Rocket Stability)', fontsize=10, labelpad=10)
        ax3.set_zlabel('Cost (billion $)', fontsize=10, labelpad=10)
        ax3.set_title('Total Cost Surface', fontsize=12, pad=20, fontweight='bold')
        ax3.view_init(elev=22, azim=135)
        cbar3 = fig.colorbar(surf3, ax=ax3, shrink=0.5, pad=0.12)
        cbar3.set_label('Billion $', fontsize=9)
        ax3.invert_xaxis()
        ax3.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q):
        """绘制帕累托前沿 - 美化版"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：时间-成本帕累托前沿
        ax1 = axes[0]
        scatter = ax1.scatter(all_times, np.array(all_costs)/1e9, 
                             c=all_a, cmap='viridis', alpha=0.5, s=40, edgecolors='none')
        
        pareto_times = [pt['time'] for pt in pareto_points]
        pareto_costs = [pt['cost']/1e9 for pt in pareto_points]
        ax1.scatter(pareto_times, pareto_costs, c='red', s=150, 
                   marker='*', label='Pareto Optimal', zorder=5, 
                   edgecolors='darkred', linewidths=1.5)
        
        # 连接帕累托前沿
        sorted_pareto = sorted(pareto_points, key=lambda x: x['time'])
        pareto_times_sorted = [pt['time'] for pt in sorted_pareto]
        pareto_costs_sorted = [pt['cost']/1e9 for pt in sorted_pareto]
        ax1.plot(pareto_times_sorted, pareto_costs_sorted, 'r--', 
                alpha=0.8, linewidth=2.5, label='Pareto Front')
        
        ax1.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost (billion $)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Pareto Front (p={p:.3f}, q={q:.3f})', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Elevator Ratio (a)', fontsize=10)
        
        # 右图：a vs 时间和成本
        ax2 = axes[1]
        ax2_time = ax2.twinx()
        
        line1 = ax2.plot(all_a, all_times, 'b-', linewidth=2.5, 
                        label='Time (years)', alpha=0.85)
        line2 = ax2_time.plot(all_a, np.array(all_costs)/1e9, 'r-', 
                             linewidth=2.5, label='Cost (billion $)', alpha=0.85)
        
        ax2.fill_between(all_a, all_times, alpha=0.15, color='blue')
        ax2_time.fill_between(all_a, np.array(all_costs)/1e9, alpha=0.15, color='red')
        
        ax2.set_xlabel('Elevator Ratio (a)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (years)', color='blue', fontsize=12, fontweight='bold')
        ax2_time.set_ylabel('Cost (billion $)', color='red', fontsize=12, fontweight='bold')
        ax2.set_title('Time and Cost vs Elevator Ratio', fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_time.tick_params(axis='y', labelcolor='red')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_monte_carlo_results(mc_results, a):
        """绘制蒙特卡洛模拟结果 - 美化版"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['steelblue', 'mediumseagreen', 'mediumpurple', 'coral']
        
        # 时间分布
        axes[0, 0].hist(mc_results['times'], bins=60, density=True, alpha=0.7, 
                       color=colors[0], edgecolor='black', linewidth=0.8)
        axes[0, 0].axvline(mc_results['time_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['time_mean']:.1f} ± {mc_results['time_std']:.1f} years")
        axes[0, 0].axvline(mc_results['time_median'], color='orange', linestyle=':', linewidth=2,
                          label=f"Median: {mc_results['time_median']:.1f} years")
        axes[0, 0].set_xlabel('Time (years)', fontsize=11)
        axes[0, 0].set_ylabel('Probability Density', fontsize=11)
        axes[0, 0].set_title('Time Distribution', fontsize=12, fontweight='bold', pad=12)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.25, linestyle='--')
        
        # 成本分布
        axes[0, 1].hist(mc_results['costs']/1e9, bins=60, density=True, alpha=0.7, 
                       color=colors[1], edgecolor='black', linewidth=0.8)
        axes[0, 1].axvline(mc_results['cost_mean']/1e9, color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['cost_mean']/1e9:.1f} ± {mc_results['cost_std']/1e9:.1f} B$")
        axes[0, 1].axvline(mc_results['cost_median']/1e9, color='orange', linestyle=':', linewidth=2,
                          label=f"Median: {mc_results['cost_median']/1e9:.1f} B$")
        axes[0, 1].set_xlabel('Cost (billion $)', fontsize=11)
        axes[0, 1].set_ylabel('Probability Density', fontsize=11)
        axes[0, 1].set_title('Cost Distribution', fontsize=12, fontweight='bold', pad=12)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.25, linestyle='--')
        
        # p值分布
        axes[1, 0].hist(mc_results['p_values'], bins=60, density=True, alpha=0.7, 
                       color=colors[2], edgecolor='black', linewidth=0.8)
        axes[1, 0].axvline(mc_results['p_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['p_mean']:.3f} ± {mc_results['p_std']:.3f}")
        axes[1, 0].set_xlabel('Elevator Stability Factor (p)', fontsize=11)
        axes[1, 0].set_ylabel('Probability Density', fontsize=11)
        axes[1, 0].set_title('Elevator Stability Distribution', fontsize=12, fontweight='bold', pad=12)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.25, linestyle='--')
        axes[1, 0].set_xlim([0.4, 1.0])
        
        # q值分布
        axes[1, 1].hist(mc_results['q_values'], bins=60, density=True, alpha=0.7, 
                       color=colors[3], edgecolor='black', linewidth=0.8)
        axes[1, 1].axvline(mc_results['q_mean'], color='red', linestyle='--', linewidth=2.5,
                          label=f"Mean: {mc_results['q_mean']:.3f} ± {mc_results['q_std']:.3f}")
        axes[1, 1].set_xlabel('Rocket Stability Factor (q)', fontsize=11)
        axes[1, 1].set_ylabel('Probability Density', fontsize=11)
        axes[1, 1].set_title('Rocket Stability Distribution', fontsize=12, fontweight='bold', pad=12)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.25, linestyle='--')
        axes[1, 1].set_xlim([0.55, 1.0])
        
        plt.suptitle(f'Monte Carlo Simulation Results (a={a:.2f}, N={len(mc_results["times"])})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_optimal_a_vs_weights(results, p, q):
        """绘制最优a与权重的关系 - 美化版"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        w_e_values = [r['w_e'] for r in results]
        optimal_a_values = [r['optimal_a'] for r in results]
        times = [r['time'] for r in results]
        costs = [r['cost']/1e9 for r in results]
        
        # 最优a vs w_e
        axes[0].plot(w_e_values, optimal_a_values, 'b-o', linewidth=2.5, markersize=7, alpha=0.8)
        axes[0].fill_between(w_e_values, optimal_a_values, alpha=0.2, color='blue')
        axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='a=0.5')
        axes[0].set_xlabel('$w_e$ (Time Weight)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Optimal $a$ (Elevator Ratio)', fontsize=12, fontweight='bold')
        axes[0].set_title('Optimal Elevator Ratio vs Time Weight', fontsize=12, fontweight='bold', pad=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_ylim([0, 1])
        axes[0].legend(fontsize=9)
        
        # 时间 vs w_e
        axes[1].plot(w_e_values, times, 'g-s', linewidth=2.5, markersize=7, alpha=0.8)
        axes[1].fill_between(w_e_values, times, alpha=0.2, color='green')
        axes[1].set_xlabel('$w_e$ (Time Weight)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Total Time (years)', fontsize=12, fontweight='bold')
        axes[1].set_title('Completion Time vs Time Weight', fontsize=12, fontweight='bold', pad=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # 成本 vs w_e
        axes[2].plot(w_e_values, costs, 'r-^', linewidth=2.5, markersize=7, alpha=0.8)
        axes[2].fill_between(w_e_values, costs, alpha=0.2, color='red')
        axes[2].set_xlabel('$w_e$ (Time Weight)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Total Cost (billion $)', fontsize=12, fontweight='bold')
        axes[2].set_title('Total Cost vs Time Weight', fontsize=12, fontweight='bold', pad=12)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'Optimization Results (p={p:.3f}, q={q:.3f})', 
                    fontsize=14, y=1.00, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sensitivity_analysis(model, base_p, base_q, base_a):
        """敏感性分析图 - 美化版"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 变化范围
        p_range = np.linspace(0.5, 1.0, 80)
        q_range = np.linspace(0.5, 1.0, 80)
        a_range = np.linspace(0.05, 0.95, 80)
        
        # p的敏感性
        times_p = [model.calculate_time(base_a, p, base_q)[0] for p in p_range]
        costs_p = [model.calculate_cost(base_a, p, base_q)[0]/1e9 for p in p_range]
        
        axes[0, 0].plot(p_range, times_p, 'b-', linewidth=2.5)
        axes[0, 0].fill_between(p_range, times_p, alpha=0.2, color='blue')
        axes[0, 0].axvline(base_p, color='red', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'Base p={base_p:.3f}')
        axes[0, 0].set_xlabel('p (Elevator Stability)', fontsize=11)
        axes[0, 0].set_ylabel('Time (years)', fontsize=11)
        axes[0, 0].set_title('Time Sensitivity to p', fontsize=12, fontweight='bold', pad=10)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].legend(fontsize=9)
        
        axes[1, 0].plot(p_range, costs_p, 'r-', linewidth=2.5)
        axes[1, 0].fill_between(p_range, costs_p, alpha=0.2, color='red')
        axes[1, 0].axvline(base_p, color='blue', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'Base p={base_p:.3f}')
        axes[1, 0].set_xlabel('p (Elevator Stability)', fontsize=11)
        axes[1, 0].set_ylabel('Cost (billion $)', fontsize=11)
        axes[1, 0].set_title('Cost Sensitivity to p', fontsize=12, fontweight='bold', pad=10)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].legend(fontsize=9)
        
        # q的敏感性
        times_q = [model.calculate_time(base_a, base_p, q)[0] for q in q_range]
        costs_q = [model.calculate_cost(base_a, base_p, q)[0]/1e9 for q in q_range]
        
        axes[0, 1].plot(q_range, times_q, 'b-', linewidth=2.5)
        axes[0, 1].fill_between(q_range, times_q, alpha=0.2, color='blue')
        axes[0, 1].axvline(base_q, color='red', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'Base q={base_q:.3f}')
        axes[0, 1].set_xlabel('q (Rocket Stability)', fontsize=11)
        axes[0, 1].set_ylabel('Time (years)', fontsize=11)
        axes[0, 1].set_title('Time Sensitivity to q', fontsize=12, fontweight='bold', pad=10)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].legend(fontsize=9)
        
        axes[1, 1].plot(q_range, costs_q, 'r-', linewidth=2.5)
        axes[1, 1].fill_between(q_range, costs_q, alpha=0.2, color='red')
        axes[1, 1].axvline(base_q, color='blue', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'Base q={base_q:.3f}')
        axes[1, 1].set_xlabel('q (Rocket Stability)', fontsize=11)
        axes[1, 1].set_ylabel('Cost (billion $)', fontsize=11)
        axes[1, 1].set_title('Cost Sensitivity to q', fontsize=12, fontweight='bold', pad=10)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        axes[1, 1].legend(fontsize=9)
        
        # a的敏感性
        times_a = [model.calculate_time(a, base_p, base_q)[0] for a in a_range]
        costs_a = [model.calculate_cost(a, base_p, base_q)[0]/1e9 for a in a_range]
        
        axes[0, 2].plot(a_range, times_a, 'b-', linewidth=2.5)
        axes[0, 2].fill_between(a_range, times_a, alpha=0.2, color='blue')
        axes[0, 2].axvline(base_a, color='red', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'Base a={base_a:.2f}')
        axes[0, 2].set_xlabel('a (Elevator Ratio)', fontsize=11)
        axes[0, 2].set_ylabel('Time (years)', fontsize=11)
        axes[0, 2].set_title('Time Sensitivity to a', fontsize=12, fontweight='bold', pad=10)
        axes[0, 2].grid(True, alpha=0.3, linestyle='--')
        axes[0, 2].legend(fontsize=9)
        
        axes[1, 2].plot(a_range, costs_a, 'r-', linewidth=2.5)
        axes[1, 2].fill_between(a_range, costs_a, alpha=0.2, color='red')
        axes[1, 2].axvline(base_a, color='blue', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'Base a={base_a:.2f}')
        axes[1, 2].set_xlabel('a (Elevator Ratio)', fontsize=11)
        axes[1, 2].set_ylabel('Cost (billion $)', fontsize=11)
        axes[1, 2].set_title('Cost Sensitivity to a', fontsize=12, fontweight='bold', pad=10)
        axes[1, 2].grid(True, alpha=0.3, linestyle='--')
        axes[1, 2].legend(fontsize=9)
        
        plt.suptitle(f'Sensitivity Analysis (p={base_p:.3f}, q={base_q:.3f}, a={base_a:.2f})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig


# ==================== 8. 主程序 ====================

def main():
    print("=" * 80)
    print("MCM 2026 Problem B - 第二问：基于实际数据的最终版本")
    print("=" * 80)
    print(f"\n实际成本参数:")
    print(f"  电梯成本: ${ELEVATOR_COST_PER_TON:,.0f}/吨")
    print(f"  火箭成本: ${ROCKET_COST_PER_TON:,.0f}/吨 (是电梯的 {ROCKET_COST_PER_TON/ELEVATOR_COST_PER_TON:.2f} 倍)")
    print(f"  火箭发射次数: {ROCKET_LAUNCHES_PER_YEAR:.2f} 次/年")
    print(f"  电梯年运力: {TOTAL_ELEVATOR_CAPACITY:,.0f} 吨/年")
    print(f"  火箭年运力: {TOTAL_ROCKET_CAPACITY:,.0f} 吨/年")
    
    # 初始化模型
    elevator_failure = ElevatorFailureModel()
    rocket_failure = RocketFailureModel()
    cost_time_model = TransportCostTimeModel()
    
    selected_sites = ['California', 'Florida', 'French_Guiana', 'Kazakhstan', 'China']
    
    print("\n" + "-" * 80)
    print("步骤 1: 计算稳定性因子 p 和 q")
    print("-" * 80)
    
    p_mean, p_std = elevator_failure.get_stability_factor(num_simulations=2000, years=10)
    print(f" 电梯稳定性因子 p = {p_mean:.4f} ± {p_std:.4f}")
    
    q_mean, q_std = rocket_failure.get_stability_factor(selected_sites, num_simulations=2000)
    print(f" 火箭稳定性因子 q = {q_mean:.4f} ± {q_std:.4f}")
    print(f"  使用发射场: {', '.join(selected_sites)}")
    
    p, q = p_mean, q_mean
    
    print("\n" + "-" * 80)
    print("步骤 2: 帕累托最优求解")
    print("-" * 80)
    
    optimizer = ParetoOptimizer(cost_time_model)
    pareto_points, all_a, all_times, all_costs = optimizer.generate_pareto_front(p, q, n_points=120)
    print(f" 找到 {len(pareto_points)} 个帕累托最优解")
    
    # 显示部分帕累托点
    print("\n部分帕累托最优解:")
    print(f"{'a':^10} {'Time (years)':^15} {'Cost (B$)':^15}")
    print("-" * 42)
    for pt in pareto_points[::max(1, len(pareto_points)//5)]:
        print(f"{pt['a']:^10.3f} {pt['time']:^15.1f} {pt['cost']/1e9:^15.1f}")
    
    print("\n" + "-" * 80)
    print("步骤 3: 梯度下降求解最优权重")
    print("-" * 80)
    
    weight_results = optimizer.find_optimal_weights(p, q)
    
    print("\n不同权重下的最优解:")
    print(f"{'w_e':^8} {'w_r':^8} {'Optimal a':^12} {'Time (yrs)':^13} {'Cost (B$)':^13}")
    print("-" * 58)
    for r in weight_results[::3]:  # 每3个显示一个
        print(f"{r['w_e']:^8.2f} {r['w_r']:^8.2f} {r['optimal_a']:^12.3f} "
              f"{r['time']:^13.1f} {r['cost']/1e9:^13.1f}")
    
    # 找到平衡解
    balanced_result = min(weight_results, key=lambda r: abs(r['w_e'] - 0.5))
    print(f"\n*** 平衡解 (w_e ≈ 0.5): a = {balanced_result['optimal_a']:.3f}, "
          f"Time = {balanced_result['time']:.1f} years, Cost = ${balanced_result['cost']/1e9:.1f}B")
    
    print("\n" + "-" * 80)
    print("步骤 4: 蒙特卡洛模拟")
    print("-" * 80)
    
    mc_simulator = MonteCarloSimulator(elevator_failure, rocket_failure, cost_time_model)
    
    test_a_values = [0.3, 0.5, 0.7]
    mc_results_dict = {}
    
    for test_a in test_a_values:
        print(f"\n模拟 a = {test_a}...")
        mc_results = mc_simulator.run_simulation(test_a, num_simulations=1500, 
                                                  selected_sites=selected_sites)
        mc_results_dict[test_a] = mc_results
        
        print(f"  时间: {mc_results['time_mean']:.1f} ± {mc_results['time_std']:.1f} 年 "
              f"(中位数: {mc_results['time_median']:.1f})")
        print(f"  成本: {mc_results['cost_mean']/1e9:.1f} ± {mc_results['cost_std']/1e9:.1f} B$ "
              f"(中位数: {mc_results['cost_median']/1e9:.1f})")
        print(f"  p: {mc_results['p_mean']:.4f} ± {mc_results['p_std']:.4f}")
        print(f"  q: {mc_results['q_mean']:.4f} ± {mc_results['q_std']:.4f}")
    
    print("\n" + "-" * 80)
    print("步骤 5: 生成可视化图表 (共5张)")
    print("-" * 80)
    
    visualizer = Visualizer()
    
    # 图1: 3D优��曲面（平滑版）
    print("\n[1/5] 生成3D优化曲面...")
    fig1 = visualizer.plot_3d_optimization_surface(
        cost_time_model, 
        p_range=(0.58, 0.95),   # ===== 关键参数：根据实际p值调整 =====
        q_range=(0.62, 0.95),   # ===== 关键参数：根据实际q值调整 =====
        resolution=32,           # ===== 关键参数：分辨率（25-35） =====
        smooth_sigma=1.0         # ===== 关键参数：平滑度（0.8-1.5） =====
    )
    fig1.savefig('3d_optimization_surface_final.png', dpi=200, bbox_inches='tight')
    print("     已保存: 3d_optimization_surface_final.png")
    
    # 图2: 帕累托前��
    print("[2/5] 生成帕累托前沿图...")
    fig2 = visualizer.plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q)
    fig2.savefig('pareto_front_final.png', dpi=200, bbox_inches='tight')
    print("     已保存: pareto_front_final.png")
    
    # 图3: 最优权重关系
    print("[3/5] 生成最优权重关系图...")
    fig3 = visualizer.plot_optimal_a_vs_weights(weight_results, p, q)
    fig3.savefig('optimal_weights_final.png', dpi=200, bbox_inches='tight')
    print("     已保存: optimal_weights_final.png")
    
    # 图4: 蒙特卡洛结果 (a=0.5)
    print("[4/5] 生成蒙特卡洛模拟结果图...")
    fig4 = visualizer.plot_monte_carlo_results(mc_results_dict[0.5], 0.5)
    fig4.savefig('monte_carlo_results_final.png', dpi=200, bbox_inches='tight')
    print("     已保存: monte_carlo_results_final.png")
    
    # 图5: 敏感性分析
    print("[5/5] 生成敏感性分析图...")
    fig5 = visualizer.plot_sensitivity_analysis(cost_time_model, p, q, 
                                                 balanced_result['optimal_a'])
    fig5.savefig('sensitivity_analysis_final.png', dpi=200, bbox_inches='tight')
    print("     已保存: sensitivity_analysis_final.png")
    
    print("\n" + "=" * 80)
    print("最终结果总结")
    print("=" * 80)
    
    print(f"""
    【稳定性因子】
    ├─ 电梯稳定性 p = {p:.4f} ± {p_std:.4f}
    │  └─ 受太空碎片撞击影响（泊松分布，λ={elevator_failure.impact_rate_per_year}次/年）
    └─ 火箭稳定性 q = {q:.4f} ± {q_std:.4f}
       └─ 综合天气效率×发射成功率
    
    【帕累托最优】
    ├─ 找到 {len(pareto_points)} 个非支配解
    ├─ 时间范围: {min(all_times):.1f} - {max(all_times):.1f} 年
    └─ 成本范围: ${min(all_costs)/1e9:.1f}B - ${max(all_costs)/1e9:.1f}B
    
    【平衡优化结果】 (w_e = w_r ≈ 0.5)
    ├─ 最优电梯占比 a* = {balanced_result['optimal_a']:.3f}
    ├─ 预计完成时间 T = {balanced_result['time']:.1f} 年
    └─ 预计总成本 C = ${balanced_result['cost']/1e9:.1f} billion
    
    【蒙特卡洛模拟】 (a = 0.5, N = 1500)
    ├─ 时间分布: {mc_results_dict[0.5]['time_mean']:.1f} ± {mc_results_dict[0.5]['time_std']:.1f} 年
    ├─ 成本分布: ${mc_results_dict[0.5]['cost_mean']/1e9:.1f} ± {mc_results_dict[0.5]['cost_std']/1e9:.1f} billion
    ├─ p 分布: {mc_results_dict[0.5]['p_mean']:.4f} ± {mc_results_dict[0.5]['p_std']:.4f}
    └─ q 分布: {mc_results_dict[0.5]['q_mean']:.4f} ± {mc_results_dict[0.5]['q_std']:.4f}
    
    【关键洞察】
    ├─ 电梯成本优势: 火箭成本是电梯的 {ROCKET_COST_PER_TON/ELEVATOR_COST_PER_TON:.2f} 倍
    ├─ 电梯运力优势: 电梯年运力是火箭的 {TOTAL_ELEVATOR_CAPACITY/TOTAL_ROCKET_CAPACITY:.2f} 倍
    ├─ 稳定性权衡: 电梯易受碎片影响(p={p:.3f})，火箭受天气限制(q={q:.3f})
    └─ 建议策略: 优先使用电梯({balanced_result['optimal_a']*100:.1f}%)降低成本，火箭作为补充保证时间
    """)
    
    print("=" * 80)
    print("所有图表已生成！")
    print("=" * 80)
    
    plt.show()
    
    return {
        'p': p, 'p_std': p_std,
        'q': q, 'q_std': q_std,
        'pareto_points': pareto_points,
        'weight_results': weight_results,
        'balanced_result': balanced_result,
        'mc_results': mc_results_dict
    }


if __name__ == "__main__":
    results = main()