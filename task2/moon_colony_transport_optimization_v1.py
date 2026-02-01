"""
MCM 2026 Problem B - 第二问：考虑外界干扰因素的月球殖民地运输优化模型
考虑因素：太空电梯失灵（太空碎片撞击）、天气影响、火箭发射失败
方法：蒙特卡洛模拟 + 离散事件模拟 + 帕累托最优 + 梯度下降
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson, norm
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 基础参数设置 ====================

# 总运输量 (百万吨)
TOTAL_CARGO = 100  # 100 million metric tons

# 太空电梯参数
ELEVATOR_CAPACITY_PER_YEAR = 179000  # metric tons per year per Galactic Harbour
NUM_GALACTIC_HARBOURS = 3
TOTAL_ELEVATOR_CAPACITY = ELEVATOR_CAPACITY_PER_YEAR * NUM_GALACTIC_HARBOURS  # 537,000 tons/year

# 火箭参数 (Falcon Heavy advanced)
ROCKET_PAYLOAD_MIN = 100  # metric tons
ROCKET_PAYLOAD_MAX = 150  # metric tons
ROCKET_PAYLOAD_AVG = (ROCKET_PAYLOAD_MIN + ROCKET_PAYLOAD_MAX) / 2  # 125 tons

# 成本参数（假设值，可根据实际数据调整）
ELEVATOR_COST_PER_TON = 2802100  # $/ton (电梯运输成本较低)
ROCKET_COST_PER_TON = 6993100   # $/ton (火箭运输成本较高)

# ==================== 2. 发射场气候数据 ====================

# 基于搜索结果的10个发射场气候效率数据
# weather_efficiency: 年均可发射天数比例
LAUNCH_SITES = {
    'Alaska': {
        'name': 'Pacific Spaceport Complex, Alaska',
        'weather_efficiency': 0.55,  # 较差：寒冷、多风、多雪、雾
        'best_season': 'summer',
        'monthly_efficiency': [0.3, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80, 0.75, 0.65, 0.50, 0.40, 0.30]
    },
    'California': {
        'name': 'Vandenberg SFB, California',
        'weather_efficiency': 0.72,  # 中等：沿海雾，但整体稳定
        'best_season': 'fall-spring',
        'monthly_efficiency': [0.75, 0.78, 0.80, 0.82, 0.70, 0.60, 0.55, 0.60, 0.75, 0.82, 0.80, 0.75]
    },
    'Texas': {
        'name': 'Starbase, Texas',
        'weather_efficiency': 0.70,  # 中等：夏季雷暴和飓风季节
        'best_season': 'winter-spring',
        'monthly_efficiency': [0.80, 0.82, 0.78, 0.72, 0.65, 0.55, 0.50, 0.52, 0.58, 0.70, 0.78, 0.82]
    },
    'Florida': {
        'name': 'Kennedy Space Center, Florida',
        'weather_efficiency': 0.65,  # 中等偏低：频繁雷暴、闪电、飓风
        'best_season': 'winter-spring',
        'monthly_efficiency': [0.80, 0.82, 0.78, 0.72, 0.58, 0.45, 0.42, 0.45, 0.50, 0.65, 0.75, 0.80]
    },
    'Virginia': {
        'name': 'Wallops Flight Facility, Virginia',
        'weather_efficiency': 0.68,  # 中等
        'best_season': 'spring-fall',
        'monthly_efficiency': [0.60, 0.62, 0.70, 0.78, 0.80, 0.75, 0.72, 0.75, 0.78, 0.75, 0.68, 0.58]
    },
    'Kazakhstan': {
        'name': 'Baikonur Cosmodrome, Kazakhstan',
        'weather_efficiency': 0.70,  # 中等：极端温度、沙尘暴
        'best_season': 'spring-fall',
        'monthly_efficiency': [0.50, 0.55, 0.70, 0.82, 0.85, 0.80, 0.78, 0.80, 0.82, 0.75, 0.60, 0.50]
    },
    'French_Guiana': {
        'name': 'Guiana Space Centre, French Guiana',
        'weather_efficiency': 0.78,  # 较好：热带，但雨季有影响
        'best_season': 'dry-season',
        'monthly_efficiency': [0.70, 0.65, 0.60, 0.58, 0.65, 0.75, 0.82, 0.88, 0.90, 0.88, 0.82, 0.75]
    },
    'India': {
        'name': 'Satish Dhawan Space Centre, India',
        'weather_efficiency': 0.65,  # 中等：季风季节影响大
        'best_season': 'post-monsoon',
        'monthly_efficiency': [0.80, 0.82, 0.78, 0.70, 0.60, 0.40, 0.35, 0.38, 0.50, 0.75, 0.82, 0.82]
    },
    'China': {
        'name': 'Taiyuan Satellite Launch Center, China',
        'weather_efficiency': 0.70,  # 中等：寒冷干燥冬季，沙尘暴
        'best_season': 'spring-fall',
        'monthly_efficiency': [0.55, 0.58, 0.70, 0.80, 0.85, 0.80, 0.75, 0.78, 0.82, 0.78, 0.65, 0.55]
    },
    'New_Zealand': {
        'name': 'Māhia Peninsula, New Zealand',
        'weather_efficiency': 0.72,  # 中等：海洋性气候，风雨
        'best_season': 'summer-fall',
        'monthly_efficiency': [0.80, 0.82, 0.78, 0.72, 0.65, 0.58, 0.55, 0.58, 0.65, 0.72, 0.78, 0.80]
    }
}

# ==================== 3. 故障模型参数 ====================

# 3.1 太空电梯故障模型 - 基于泊松分布的撞击事件
class ElevatorFailureModel:
    """太空电梯故障模型 - 考虑太空碎片撞击"""
    
    def __init__(self):
        # 撞击事件参数（每年平均撞击次数 - 基于LEO碎片密度模型）
        self.impact_rate_per_year = 2.5  # λ for Poisson distribution
        
        # 撞击强度分布参数（正态分布）
        self.impact_intensity_mean = 0.3  # 平均强度
        self.impact_intensity_std = 0.15  # 强度标准差
        
        # 累积损伤阈值（超过则失效）
        self.damage_threshold_critical = 1.0  # 完全失效
        self.damage_threshold_degraded = 0.5  # 性能下降
        
        # 故障等级及其效率因子
        self.failure_grades = {
            0: {'name': '正常运行', 'efficiency': 1.0, 'repair_time': 0},
            1: {'name': '轻微损伤', 'efficiency': 0.9, 'repair_time': 7},      # 7天修复
            2: {'name': '中度损伤', 'efficiency': 0.7, 'repair_time': 30},     # 30天修复
            3: {'name': '严重损伤', 'efficiency': 0.4, 'repair_time': 90},     # 90天修复
            4: {'name': '完全失效', 'efficiency': 0.0, 'repair_time': 365}     # 1年修复
        }
    
    def simulate_yearly_impacts(self, num_years=1):
        """模拟每年的撞击事件"""
        impacts = []
        for year in range(num_years):
            # 泊松分布：该年撞击次数
            num_impacts = np.random.poisson(self.impact_rate_per_year)
            year_impacts = []
            for _ in range(num_impacts):
                # 正态分布：每次撞击的强度
                intensity = max(0, np.random.normal(self.impact_intensity_mean, 
                                                     self.impact_intensity_std))
                year_impacts.append(intensity)
            impacts.append(year_impacts)
        return impacts
    
    def calculate_damage_level(self, cumulative_damage):
        """根据累积损伤计算故障等级"""
        if cumulative_damage < 0.2:
            return 0
        elif cumulative_damage < 0.4:
            return 1
        elif cumulative_damage < 0.6:
            return 2
        elif cumulative_damage < self.damage_threshold_critical:
            return 3
        else:
            return 4
    
    def get_stability_factor(self, num_simulations=1000, years=1):
        """
        通过蒙特卡洛模拟计算电梯稳定性因子 p
        返回：平均效率因子 p (0-1)
        """
        efficiency_samples = []
        
        for _ in range(num_simulations):
            cumulative_damage = 0
            yearly_efficiency = []
            
            for year in range(years):
                impacts = self.simulate_yearly_impacts(1)[0]
                
                for impact in impacts:
                    cumulative_damage += impact
                
                # 考虑自然恢复（假设每年恢复10%的累积损伤）
                cumulative_damage *= 0.9
                
                damage_level = self.calculate_damage_level(cumulative_damage)
                efficiency = self.failure_grades[damage_level]['efficiency']
                yearly_efficiency.append(efficiency)
            
            efficiency_samples.append(np.mean(yearly_efficiency))
        
        return np.mean(efficiency_samples), np.std(efficiency_samples)


# 3.2 火箭故障模型
class RocketFailureModel:
    """火箭故障模型 - 考虑天气和发射失败"""
    
    def __init__(self, launch_sites=LAUNCH_SITES):
        self.launch_sites = launch_sites
        # Falcon Heavy历史成功率（基于SpaceX数据，约95%以上）
        self.base_success_rate = 0.95
        # 到2050年预计提升到的成功率
        self.projected_success_rate_2050 = 0.98
        
    def get_weather_efficiency(self, selected_sites=None):
        """计算选定发射场的综合天气效率"""
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        efficiencies = [self.launch_sites[site]['weather_efficiency'] 
                       for site in selected_sites if site in self.launch_sites]
        return np.mean(efficiencies)
    
    def simulate_monthly_launches(self, site_name, num_launches_target):
        """模拟某发射场的月度发射情况"""
        if site_name not in self.launch_sites:
            return 0
        
        site = self.launch_sites[site_name]
        monthly_eff = site['monthly_efficiency']
        
        successful_launches = 0
        for month in range(12):
            # 该月可用的发射窗口比例
            weather_factor = monthly_eff[month]
            # 该月计划发射数
            monthly_target = num_launches_target / 12
            # 实际可进行的发射数
            actual_attempts = int(monthly_target * weather_factor)
            
            # 每次发射的成功概率
            for _ in range(actual_attempts):
                if np.random.random() < self.projected_success_rate_2050:
                    successful_launches += 1
        
        return successful_launches
    
    def get_stability_factor(self, selected_sites=None, num_simulations=1000):
        """
        计算火箭运输系统的稳定性因子 q
        综合考虑：天气效率 × 发射成功率
        """
        if selected_sites is None:
            selected_sites = list(self.launch_sites.keys())
        
        weather_eff = self.get_weather_efficiency(selected_sites)
        
        # 综合稳定性 = 天气效率 × 发射成功率
        q_mean = weather_eff * self.projected_success_rate_2050
        
        # 通过模拟计算标准差
        q_samples = []
        for _ in range(num_simulations):
            # 随机扰动
            weather_var = np.random.normal(weather_eff, 0.05)
            success_var = np.random.normal(self.projected_success_rate_2050, 0.02)
            q_samples.append(np.clip(weather_var * success_var, 0, 1))
        
        return q_mean, np.std(q_samples)


# ==================== 4. 成本与时间模型 ====================

class TransportCostTimeModel:
    """运输成本和时间模型"""
    
    def __init__(self, total_cargo=TOTAL_CARGO * 1e6):  # 转换为吨
        self.total_cargo = total_cargo
        self.elevator_capacity = TOTAL_ELEVATOR_CAPACITY
        self.rocket_payload = ROCKET_PAYLOAD_AVG
        
        # 基础成本
        self.elevator_cost_per_ton = ELEVATOR_COST_PER_TON
        self.rocket_cost_per_ton = ROCKET_COST_PER_TON
        
    def calculate_time(self, a, p, q, num_rockets_per_year=707.76):
        """
        计算完成运输任务所需时间
        a: 电梯运输占比
        p: 电梯稳定性因子
        q: 火箭稳定性因子
        """
        # 电梯部分运输量
        elevator_cargo = self.total_cargo * a
        # 火箭部分运输量
        rocket_cargo = self.total_cargo * (1 - a)
        
        # 实际有效年运输能力
        effective_elevator_capacity = self.elevator_capacity * p
        effective_rocket_capacity = num_rockets_per_year * self.rocket_payload * q
        
        # 计算各部分所需时间
        if effective_elevator_capacity > 0:
            time_elevator = elevator_cargo / effective_elevator_capacity
        else:
            time_elevator = float('inf') if elevator_cargo > 0 else 0
            
        if effective_rocket_capacity > 0:
            time_rocket = rocket_cargo / effective_rocket_capacity
        else:
            time_rocket = float('inf') if rocket_cargo > 0 else 0
        
        # 总时间取最大值（并行运输）
        total_time = max(time_elevator, time_rocket)
        
        return total_time, time_elevator, time_rocket
    
    def calculate_cost(self, a, p, q):
        """
        计算总成本
        考虑故障导致的额外成本
        """
        elevator_cargo = self.total_cargo * a
        rocket_cargo = self.total_cargo * (1 - a)
        
        # 基础成本
        base_elevator_cost = elevator_cargo * self.elevator_cost_per_ton
        base_rocket_cost = rocket_cargo * self.rocket_cost_per_ton
        
        # 故障额外成本（故障导致需要重试或修复）
        # 效率降低意味着需要更多资源来完成同样的任务
        elevator_overhead = (1 / p - 1) * base_elevator_cost * 0.3  # 30%的额外开销
        rocket_overhead = (1 / q - 1) * base_rocket_cost * 0.2     # 20%的额外开销
        
        total_cost = base_elevator_cost + base_rocket_cost + elevator_overhead + rocket_overhead
        
        return total_cost, base_elevator_cost + elevator_overhead, base_rocket_cost + rocket_overhead
    
    def normalize(self, time, cost, time_ref=None, cost_ref=None):
        """归一化处理"""
        if time_ref is None:
            time_ref = time
        if cost_ref is None:
            cost_ref = cost
        
        norm_time = time / time_ref if time_ref > 0 else 0
        norm_cost = cost / cost_ref if cost_ref > 0 else 0
        
        return norm_time, norm_cost
    
    def objective_function(self, params, p, q, w_e=0.5, w_r=0.5, 
                          time_ref=200, cost_ref=1e12):
        """
        目标函数：最小化加权的归一化时间和成本
        params[0] = a (电梯占比)
        """
        a = params[0]
        
        # 边界检查
        if a < 0 or a > 1:
            return 1e10
        
        time, _, _ = self.calculate_time(a, p, q)
        cost, _, _ = self.calculate_cost(a, p, q)
        
        norm_time, norm_cost = self.normalize(time, cost, time_ref, cost_ref)
        
        # 加权目标函数
        # w_e: 时间权重 (efficiency/time importance)
        # w_r: 成本权重 (resource/cost importance)
        objective = w_e * norm_time + w_r * norm_cost
        
        return objective


# ==================== 5. 帕累托最优求解 ====================

class ParetoOptimizer:
    """帕累托最优求解器"""
    
    def __init__(self, model):
        self.model = model
        
    def generate_pareto_front(self, p, q, n_points=100):
        """生成帕累托前沿"""
        a_values = np.linspace(0.01, 0.99, n_points)
        
        times = []
        costs = []
        
        for a in a_values:
            time, _, _ = self.model.calculate_time(a, p, q)
            cost, _, _ = self.model.calculate_cost(a, p, q)
            times.append(time)
            costs.append(cost)
        
        # 找到帕累托最优点
        pareto_points = []
        for i in range(len(a_values)):
            is_pareto = True
            for j in range(len(a_values)):
                if i != j:
                    # 检查是否被支配
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
    
    def find_optimal_weights(self, p, q, target_a=None):
        """
        找到最优的权重系数 w_e 和 w_r
        使用梯度下降法
        """
        # 参考值（用于归一化）
        time_ref, _, _ = self.model.calculate_time(0.5, p, q)
        cost_ref, _, _ = self.model.calculate_cost(0.5, p, q)
        
        best_results = []
        
        # 遍历不同的权重组合
        w_values = np.linspace(0.01, 0.99, 17)  # w_e从0.1到0.9
        
        for w_e in w_values:
            w_r = 1 - w_e
            
            # 使用scipy优化找最优a
            result = minimize(
                self.model.objective_function,
                x0=[0.5],
                args=(p, q, w_e, w_r, time_ref, cost_ref),
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)]
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


# # ==================== 6. 蒙特卡洛模拟 ====================

# class MonteCarloSimulator:
#     """蒙特卡洛模拟器"""
    
#     def __init__(self, elevator_model, rocket_model, cost_time_model):
#         self.elevator_model = elevator_model
#         self.rocket_model = rocket_model
#         self.cost_time_model = cost_time_model
        
#     def run_simulation(self, a, num_simulations=1000, selected_sites=None):
#         """
#         运行蒙特卡洛模拟
#         返回时间和成本的分布
#         """
#         times = []
#         costs = []
#         p_values = []
#         q_values = []
        
#         for _ in range(num_simulations):
#             # 模拟电梯稳定性
#             p, _ = self.elevator_model.get_stability_factor(num_simulations=1, years=1)
#             p = np.clip(p + np.random.normal(0, 0.05), 0.1, 1.0)
            
#             # 模拟火箭稳定性
#             q, _ = self.rocket_model.get_stability_factor(selected_sites, num_simulations=1)
#             q = np.clip(q + np.random.normal(0, 0.03), 0.1, 1.0)
            
#             # 计算时间和成本
#             time, _, _ = self.cost_time_model.calculate_time(a, p, q)
#             cost, _, _ = self.cost_time_model.calculate_cost(a, p, q)
            
#             times.append(time)
#             costs.append(cost)
#             p_values.append(p)
#             q_values.append(q)
        
#         return {
#             'times': np.array(times),
#             'costs': np.array(costs),
#             'p_values': np.array(p_values),
#             'q_values': np.array(q_values),
#             'time_mean': np.mean(times),
#             'time_std': np.std(times),
#             'cost_mean': np.mean(costs),
#             'cost_std': np.std(costs),
#             'p_mean': np.mean(p_values),
#             'q_mean': np.mean(q_values)
#         }

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


# ==================== 7. 可视化 ====================

class Visualizer:
    """可视化模块"""
    
    @staticmethod
    def plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q):
        """绘制帕累托前沿"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 所有点
        ax.scatter(all_times, np.array(all_costs)/1e9, alpha=0.5, label='All solutions')
        
        # 帕累托前沿点
        pareto_times = [pt['time'] for pt in pareto_points]
        pareto_costs = [pt['cost']/1e9 for pt in pareto_points]
        ax.scatter(pareto_times, pareto_costs, c='red', s=100, 
                  label='Pareto optimal', zorder=5)
        
        # 连接帕累托前沿
        sorted_pareto = sorted(pareto_points, key=lambda x: x['time'])
        pareto_times_sorted = [pt['time'] for pt in sorted_pareto]
        pareto_costs_sorted = [pt['cost']/1e9 for pt in sorted_pareto]
        ax.plot(pareto_times_sorted, pareto_costs_sorted, 'r--', alpha=0.7)
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Cost (billion $)', fontsize=12)
        ax.set_title(f'Pareto Front (p={p:.3f}, q={q:.3f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
        
        # 图1: 最优a vs w_e
        axes[0].plot(w_e_values, optimal_a_values, 'b-o', linewidth=2, markersize=8)
        axes[0].set_xlabel('$w_e$ (Time Weight)', fontsize=12)
        axes[0].set_ylabel('Optimal $a$ (Elevator Ratio)', fontsize=12)
        axes[0].set_title('Optimal Elevator Ratio vs Time Weight', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 图2: 时间 vs w_e
        axes[1].plot(w_e_values, times, 'g-s', linewidth=2, markersize=8)
        axes[1].set_xlabel('$w_e$ (Time Weight)', fontsize=12)
        axes[1].set_ylabel('Total Time (years)', fontsize=12)
        axes[1].set_title('Completion Time vs Time Weight', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 图3: 成本 vs w_e
        axes[2].plot(w_e_values, costs, 'r-^', linewidth=2, markersize=8)
        axes[2].set_xlabel('$w_e$ (Time Weight)', fontsize=12)
        axes[2].set_ylabel('Total Cost (billion $)', fontsize=12)
        axes[2].set_title('Total Cost vs Time Weight', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Optimization Results (p={p:.3f}, q={q:.3f})', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_monte_carlo_results(mc_results, a):
        """绘制蒙特卡洛模拟结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 时间分布
        axes[0, 0].hist(mc_results['times'], bins=50, density=True, alpha=0.7, color='blue')
        axes[0, 0].axvline(mc_results['time_mean'], color='red', linestyle='--', 
                          label=f"Mean: {mc_results['time_mean']:.1f} years")
        axes[0, 0].set_xlabel('Time (years)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Time Distribution')
        axes[0, 0].legend()
        
        # 成本分布
        axes[0, 1].hist(mc_results['costs']/1e9, bins=50, density=True, alpha=0.7, color='green')
        axes[0, 1].axvline(mc_results['cost_mean']/1e9, color='red', linestyle='--',
                          label=f"Mean: {mc_results['cost_mean']/1e9:.1f} billion $")
        axes[0, 1].set_xlabel('Cost (billion $)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Cost Distribution')
        axes[0, 1].legend()
        
        # p值分布
        axes[1, 0].hist(mc_results['p_values'], bins=50, density=True, alpha=0.7, color='purple')
        axes[1, 0].axvline(mc_results['p_mean'], color='red', linestyle='--',
                          label=f"Mean: {mc_results['p_mean']:.3f}")
        axes[1, 0].set_xlabel('Elevator Stability Factor (p)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Elevator Stability Distribution')
        axes[1, 0].legend()
        
        # q值分布
        axes[1, 1].hist(mc_results['q_values'], bins=50, density=True, alpha=0.7, color='orange')
        axes[1, 1].axvline(mc_results['q_mean'], color='red', linestyle='--',
                          label=f"Mean: {mc_results['q_mean']:.3f}")
        axes[1, 1].set_xlabel('Rocket Stability Factor (q)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Rocket Stability Distribution')
        axes[1, 1].legend()
        
        plt.suptitle(f'Monte Carlo Simulation Results (a={a:.2f})', fontsize=14)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_3d_optimization_surface(model, p_range, q_range):
        """绘制3D优化曲面"""
        from mpl_toolkits.mplot3d import Axes3D
        
        p_vals = np.linspace(p_range[0], p_range[1], 30)
        q_vals = np.linspace(q_range[0], q_range[1], 30)
        P, Q = np.meshgrid(p_vals, q_vals)
        
        # 对每个(p,q)组合找最优a
        optimal_A = np.zeros_like(P)
        Times = np.zeros_like(P)
        Costs = np.zeros_like(P)
        
        for i in range(len(p_vals)):
            for j in range(len(q_vals)):
                p, q = P[j, i], Q[j, i]
                
                # 简化：直接计算不同a的结果，取平衡点
                best_a = 0.5
                best_obj = float('inf')
                
                for a in np.linspace(0.1, 0.9, 20):
                    time, _, _ = model.calculate_time(a, p, q)
                    cost, _, _ = model.calculate_cost(a, p, q)
                    obj = time * cost  # 简单的时间-成本乘积作为目标
                    if obj < best_obj:
                        best_obj = obj
                        best_a = a
                
                optimal_A[j, i] = best_a
                time, _, _ = model.calculate_time(best_a, p, q)
                cost, _, _ = model.calculate_cost(best_a, p, q)
                Times[j, i] = time
                Costs[j, i] = cost / 1e9
        
        # 创建3D图
        fig = plt.figure(figsize=(15, 5))
        
        # 图1: 最优a曲面
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(P, Q, optimal_A, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('p (Elevator Stability)')
        ax1.set_ylabel('q (Rocket Stability)')
        ax1.set_zlabel('Optimal a')
        ax1.set_title('Optimal Elevator Ratio')
        fig.colorbar(surf1, ax=ax1, shrink=0.5)
        
        # 图2: 时间曲面
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(P, Q, Times, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('p (Elevator Stability)')
        ax2.set_ylabel('q (Rocket Stability)')
        ax2.set_zlabel('Time (years)')
        ax2.set_title('Completion Time')
        fig.colorbar(surf2, ax=ax2, shrink=0.5)
        
        # 图3: 成本曲面
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(P, Q, Costs, cmap='coolwarm', alpha=0.8)
        ax3.set_xlabel('p (Elevator Stability)')
        ax3.set_ylabel('q (Rocket Stability)')
        ax3.set_zlabel('Cost (billion $)')
        ax3.set_title('Total Cost')
        fig.colorbar(surf3, ax=ax3, shrink=0.5)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sensitivity_analysis(model, base_p, base_q, base_a):
        """敏感性分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 变化范围
        p_range = np.linspace(0.5, 1.0, 50)
        q_range = np.linspace(0.5, 1.0, 50)
        a_range = np.linspace(0.1, 0.9, 50)
        
        # p的敏感性
        times_p = [model.calculate_time(base_a, p, base_q)[0] for p in p_range]
        costs_p = [model.calculate_cost(base_a, p, base_q)[0]/1e9 for p in p_range]
        
        axes[0, 0].plot(p_range, times_p, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('p (Elevator Stability)')
        axes[0, 0].set_ylabel('Time (years)')
        axes[0, 0].set_title('Time Sensitivity to p')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[1, 0].plot(p_range, costs_p, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('p (Elevator Stability)')
        axes[1, 0].set_ylabel('Cost (billion $)')
        axes[1, 0].set_title('Cost Sensitivity to p')
        axes[1, 0].grid(True, alpha=0.3)
        
        # q的敏感性
        times_q = [model.calculate_time(base_a, base_p, q)[0] for q in q_range]
        costs_q = [model.calculate_cost(base_a, base_p, q)[0]/1e9 for q in q_range]
        
        axes[0, 1].plot(q_range, times_q, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('q (Rocket Stability)')
        axes[0, 1].set_ylabel('Time (years)')
        axes[0, 1].set_title('Time Sensitivity to q')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 1].plot(q_range, costs_q, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('q (Rocket Stability)')
        axes[1, 1].set_ylabel('Cost (billion $)')
        axes[1, 1].set_title('Cost Sensitivity to q')
        axes[1, 1].grid(True, alpha=0.3)
        
        # a的敏感性
        times_a = [model.calculate_time(a, base_p, base_q)[0] for a in a_range]
        costs_a = [model.calculate_cost(a, base_p, base_q)[0]/1e9 for a in a_range]
        
        axes[0, 2].plot(a_range, times_a, 'b-', linewidth=2)
        axes[0, 2].set_xlabel('a (Elevator Ratio)')
        axes[0, 2].set_ylabel('Time (years)')
        axes[0, 2].set_title('Time Sensitivity to a')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 2].plot(a_range, costs_a, 'r-', linewidth=2)
        axes[1, 2].set_xlabel('a (Elevator Ratio)')
        axes[1, 2].set_ylabel('Cost (billion $)')
        axes[1, 2].set_title('Cost Sensitivity to a')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Sensitivity Analysis (base: p={base_p:.2f}, q={base_q:.2f}, a={base_a:.2f})', 
                    fontsize=14)
        plt.tight_layout()
        return fig


# ==================== 8. 主程序 ====================

def main():
    print("=" * 80)
    print("MCM 2026 Problem B - 第二问：考虑外界干扰因素的月球殖民地运输优化模型")
    print("=" * 80)
    
    # 初始化模型
    elevator_failure = ElevatorFailureModel()
    rocket_failure = RocketFailureModel()
    cost_time_model = TransportCostTimeModel()
    
    # 选择使用的发射场（可以根据需要选择子集）
    selected_sites = ['California', 'Florida', 'French_Guiana', 'Kazakhstan', 'China']
    
    print("\n" + "-" * 40)
    print("1. 计算稳定性因子 p 和 q")
    print("-" * 40)
    
    # 计算电梯稳定性 p
    p_mean, p_std = elevator_failure.get_stability_factor(num_simulations=1000, years=10)
    print(f"电梯稳定性因子 p: {p_mean:.4f} ± {p_std:.4f}")
    
    # 计算火箭稳定性 q
    q_mean, q_std = rocket_failure.get_stability_factor(selected_sites, num_simulations=1000)
    print(f"火箭稳定性因子 q: {q_mean:.4f} ± {q_std:.4f}")
    print(f"选用发射场: {', '.join(selected_sites)}")
    
    # 使用计算得到的p和q
    p = p_mean
    q = q_mean
    
    print("\n" + "-" * 40)
    print("2. 时间和成本表达式")
    print("-" * 40)
    print("""
    时间模型 T(a, p, q):
    T = max(T_elevator, T_rocket)
    其中:
    - T_elevator = (a × M_total) / (C_elevator × p)
    - T_rocket = ((1-a) × M_total) / (C_rocket × q)
    
    成本模型 C(a, p, q):
    C = C_e + C_r + overhead
    其中:
    - C_e = a × M_total × cost_per_ton_elevator × (1 + 0.3×(1/p - 1))
    - C_r = (1-a) × M_total × cost_per_ton_rocket × (1 + 0.2×(1/q - 1))
    
    归一化目标函数:
    F(a, w_e, w_r) = w_e × (T/T_ref) + w_r × (C/C_ref)
    约束: w_e + w_r = 1
    """)
    
    print("\n" + "-" * 40)
    print("3. 帕累托最优求解")
    print("-" * 40)
    
    optimizer = ParetoOptimizer(cost_time_model)
    pareto_points, all_a, all_times, all_costs = optimizer.generate_pareto_front(p, q)
    
    print(f"找到 {len(pareto_points)} 个帕累托最优解")
    print("\n部分帕累托最优解:")
    print(f"{'a':^8} {'Time (years)':^15} {'Cost (billion $)':^20}")
    print("-" * 45)
    for pt in pareto_points[::max(1, len(pareto_points)//5)]:
        print(f"{pt['a']:^8.3f} {pt['time']:^15.1f} {pt['cost']/1e9:^20.2f}")
    
    print("\n" + "-" * 40)
    print("4. 梯度下降求解最优权重")
    print("-" * 40)
    
    weight_results = optimizer.find_optimal_weights(p, q)
    
    print("\n不同权重下的最优解:")
    print(f"{'w_e':^8} {'w_r':^8} {'Optimal a':^12} {'Time (yrs)':^12} {'Cost (B$)':^12}")
    print("-" * 55)
    for r in weight_results[::2]:  # 每隔一个打印
        print(f"{r['w_e']:^8.2f} {r['w_r']:^8.2f} {r['optimal_a']:^12.3f} "
              f"{r['time']:^12.1f} {r['cost']/1e9:^12.2f}")
    
    print("\n" + "-" * 40)
    print("5. 蒙特卡洛模拟")
    print("-" * 40)
    
    mc_simulator = MonteCarloSimulator(elevator_failure, rocket_failure, cost_time_model)
    
    # 对几个不同的a值进行模拟
    test_a_values = [0.3, 0.5, 0.7]
    
    for test_a in test_a_values:
        mc_results = mc_simulator.run_simulation(test_a, num_simulations=1000, 
                                                  selected_sites=selected_sites)
        print(f"\na = {test_a}:")
        print(f"  时间: {mc_results['time_mean']:.1f} ± {mc_results['time_std']:.1f} 年")
        print(f"  成本: {mc_results['cost_mean']/1e9:.2f} ± {mc_results['cost_std']/1e9:.2f} 十亿美元")
        print(f"  平均p: {mc_results['p_mean']:.4f}, 平均q: {mc_results['q_mean']:.4f}")
    
    print("\n" + "-" * 40)
    print("6. 生成可视化图表")
    print("-" * 40)
    
    visualizer = Visualizer()
    
    # 帕累托前沿图
    fig1 = visualizer.plot_pareto_front(pareto_points, all_a, all_times, all_costs, p, q)
    fig1.savefig('pareto_front.png', dpi=150, bbox_inches='tight')
    print("已保存: pareto_front.png")
    
    # 最优权重关系图
    fig2 = visualizer.plot_optimal_a_vs_weights(weight_results, p, q)
    fig2.savefig('optimal_weights.png', dpi=150, bbox_inches='tight')
    print("已保存: optimal_weights.png")
    
    # 蒙特卡洛结果图
    mc_results = mc_simulator.run_simulation(0.5, num_simulations=1000)
    fig3 = visualizer.plot_monte_carlo_results(mc_results, 0.5)
    fig3.savefig('monte_carlo_results.png', dpi=150, bbox_inches='tight')
    print("已保存: monte_carlo_results.png")
    
    # 3D优化曲面
    fig4 = visualizer.plot_3d_optimization_surface(cost_time_model, (0.5, 1.0), (0.5, 1.0))
    fig4.savefig('optimization_surface_3d.png', dpi=150, bbox_inches='tight')
    print("已保存: optimization_surface_3d.png")
    
    # 敏感性分析
    fig5 = visualizer.plot_sensitivity_analysis(cost_time_model, p, q, 0.5)
    fig5.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print("已保存: sensitivity_analysis.png")
    
    print("\n" + "-" * 40)
    print("7. 结果总结")
    print("-" * 40)
    
    # 找到平衡解（w_e = w_r = 0.5时的最优解）
    balanced_result = [r for r in weight_results if abs(r['w_e'] - 0.5) < 0.05][0]
    
    print(f"""
    模型关键参数:
    - 总运输量: 100 百万吨
    - 电梯年运力: {TOTAL_ELEVATOR_CAPACITY:,} 吨/年 (3个Galactic Harbour)
    - 火箭单次载荷: {ROCKET_PAYLOAD_AVG} 吨
    
    稳定性因子:
    - 电梯稳定性 p = {p:.4f} (受太空碎片撞击影响)
    - 火箭稳定性 q = {q:.4f} (受天气和发射成功率影响)
    
    平衡优化结果 (w_e = w_r = 0.5):
    - 最优电梯占比 a = {balanced_result['optimal_a']:.3f}
    - 预计完成时间: {balanced_result['time']:.1f} 年
    - 预计总成本: {balanced_result['cost']/1e9:.2f} 十亿美元
    
    建议:
    - 当更重视时间效率(w_e > 0.5)时，应增加火箭使用比例
    - 当更重视成本控制(w_r > 0.5)时，应增加电梯使用比例
    - 实际决策需要根据MCM Agency的具体优先级调整权重
    """)
    
    plt.show()
    
    return {
        'p': p,
        'q': q,
        'pareto_points': pareto_points,
        'weight_results': weight_results,
        'balanced_result': balanced_result
    }


if __name__ == "__main__":
    results = main()