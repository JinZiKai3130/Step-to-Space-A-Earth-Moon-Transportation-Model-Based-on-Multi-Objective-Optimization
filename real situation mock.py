import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. 参数设置 (Parameters Setup)
# ==========================================
TOTAL_PAYLOAD = 1e8  # 1亿吨 (Metric Tons)
SIMULATION_RUNS = 2000  # 蒙特卡洛模拟次数 (可调整为1000+以获得更平滑结果)

# --- 太空电梯 (Space Elevator, SE) 参数 ---
# 假设3个太空港，基准年运力 (根据题目源数据调整)
SE_BASE_CAPACITY_PER_YEAR = 179000 * 3
# SE_FIXED_COST = 500 * 10 ** 9  # 假设建设固定成本 (比如5000亿美元)
SE_OPS_COST_PER_YEAR = 10 * 10 ** 9  # 年运营成本
# 故障参数
SE_DEBRIS_LAMBDA = 2.0  # 泊松分布参数：平均每年发生严重撞击次数
SE_IMPACT_SEVERITY = 0.05  # 每次撞击导致的年效率下降百分比 (5%)

# --- 火箭 (Rocket) 参数 ---
# 假设使用Falcon Heavy级别
["cite_start"]
ROCKET_PAYLOAD = 125  # 单次运载 125吨 [cite: 19]
ROCKET_COST_PER_LAUNCH = 0.15 * 10 ** 9  # 单次发射成本 (1.5亿美元)
ROCKET_FAILURE_RATE = 0.04  # 技术故障率 4%
# 10个发射场的天气适航率 (Weather Availability)
# 假设数据：[Alaska, CA, TX, FL, VA, Kaz, FraGui, India, China, NZ]
LAUNCH_SITES_WEATHER = [0.7, 0.85, 0.8, 0.75, 0.75, 0.6, 0.9, 0.8, 0.7, 0.8]


# ==========================================
# 2. 核心模拟引擎 (Simulation Engine)
# ==========================================

def simulate_efficiency_factors():
    """
    模拟单次实验中的 p (电梯效率) 和 q (火箭综合效率)
    """
    # --- 计算 p (电梯) ---
    # 泊松分布生成当年的撞击次数
    debris_hits = np.random.poisson(SE_DEBRIS_LAMBDA)
    # 效率 p = 1 - (撞击次数 * 单次损失)
    # 设定下限，最差情况效率不低于 10% (0.1)
    p = max(0.1, 1.0 - (debris_hits * SE_IMPACT_SEVERITY))

    # --- 计算 q (火箭) ---
    # 随机选择一个发射场，或者取平均天气概率
    avg_weather_success = np.mean(LAUNCH_SITES_WEATHER)
    # 模拟天气波动 (正态分布扰动)
    weather_factor = np.clip(np.random.normal(avg_weather_success, 0.05), 0, 1)

    # 综合效率 q = 天气因子 * (1 - 故障率)
    q = weather_factor * (1 - ROCKET_FAILURE_RATE)

    return p, q


def calculate_mission_metrics(a, p_avg, q_avg):
    """
    基于给定的比例 a 和平均效率 p, q，计算总时间和总成本
    """
    # --- 任务分配 ---
    mass_se = TOTAL_PAYLOAD * a
    mass_rocket = TOTAL_PAYLOAD * (1 - a)

    # --- 太空电梯计算 ---
    if mass_se > 0:
        effective_capacity = SE_BASE_CAPACITY_PER_YEAR * p_avg
        time_se = mass_se / effective_capacity
        cost_se = SE_FIXED_COST + (SE_OPS_COST_PER_YEAR * time_se)
    else:
        time_se = 0
        cost_se = 0  # 假设不建电梯则无成本，或者可以设为固定建设成本

    # --- 火箭计算 ---
    if mass_rocket > 0:
        # 需要发射次数 = 质量 / 单次载重
        # 考虑故障：如果故障率是 f，则需要发射 N / (1-f) 次才能成功 N 次
        num_launches_needed = (mass_rocket / ROCKET_PAYLOAD)
        real_launches = num_launches_needed / (1 - ROCKET_FAILURE_RATE)

        # 成本计算
        cost_rocket = real_launches * ROCKET_COST_PER_LAUNCH

        # 时间计算 (假设全球10个发射场并发发射能力)
        # 假设全球每年最大发射能力为 N 次 (例如 1000次)
        GLOBAL_MAX_LAUNCHES_PER_YEAR = 1000
        # 实际每年发射能力受天气 q 影响
        effective_launches_per_year = GLOBAL_MAX_LAUNCHES_PER_YEAR * q_avg
        time_rocket = real_launches / effective_launches_per_year
    else:
        time_rocket = 0
        cost_rocket = 0

    # --- 系统总计 ---
    # 两个系统并行工作，总时间取决于慢的那个 (关键路径)
    total_time = max(time_se, time_rocket)
    total_cost = cost_se + cost_rocket

    return total_time, total_cost


# ==========================================
# 3. 蒙特卡洛与优化 (Monte Carlo & Optimization)
# ==========================================

# 离散化 a (从 0 到 1，步长 0.05)
a_values = np.linspace(0, 1, 21)
results = []

print("Running Simulation...")
for a in a_values:
    temp_times = []
    temp_costs = []

    # 蒙特卡洛循环
    for _ in range(SIMULATION_RUNS):
        p, q = simulate_efficiency_factors()
        t, c = calculate_mission_metrics(a, p, q)
        temp_times.append(t)
        temp_costs.append(c)

    # 取均值
    avg_time = np.mean(temp_times)
    avg_cost = np.mean(temp_costs)
    results.append({'a': a, 'Time': avg_time, 'Cost': avg_cost})

df = pd.DataFrame(results)

# --- 归一化 (Normalization) ---
# 使用 Min-Max 归一化，将时间和成本映射到 [0, 1] 区间
df['Time_Norm'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
df['Cost_Norm'] = (df['Cost'] - df['Cost'].min()) / (df['Cost'].max() - df['Cost'].min())

# ==========================================
# 4. 可视化 (Visualization)
# ==========================================

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# 图 1: 成本 vs 时间 (寻找帕累托前沿)
sc = ax[0].scatter(df['Time'], df['Cost'], c=df['a'], cmap='viridis', s=100)
ax[0].set_xlabel('Total Time (Years)')
ax[0].set_ylabel('Total Cost (Currency Units)')
ax[0].set_title('Trade-off Analysis: Time vs. Cost')
ax[0].grid(True, linestyle='--', alpha=0.6)
plt.colorbar(sc, ax=ax[0], label='Proportion of Space Elevator (a)')

# 标注关键点
for i, row in df.iterrows():
    if i % 2 == 0:  # 避免标签过密
        ax[0].annotate(f'a={row["a"]:.1f}', (row['Time'], row['Cost']), xytext=(5, 5), textcoords='offset points')

# 图 2: 综合得分随权重 w 的变化
# 计算不同权重下的最佳 a
weights = np.linspace(0, 1, 100)  # w 从 0 (只看成本) 到 1 (只看时间)
best_a_for_weights = []

for w in weights:
    # Score = w * T_norm + (1-w) * C_norm
    df['Score'] = w * df['Time_Norm'] + (1 - w) * df['Cost_Norm']
    # 找到 Score 最小的行
    best_row = df.loc[df['Score'].idxmin()]
    best_a_for_weights.append(best_row['a'])

ax[1].plot(weights, best_a_for_weights, 'r-', linewidth=3)
ax[1].set_xlabel('Weight on Time (w) [0=Cost Focus, 1=Time Focus]')
ax[1].set_ylabel('Optimal Proportion for Elevator (a)')
ax[1].set_title('Optimal Strategy Sensitivity Analysis')
ax[1].grid(True)
ax[1].set_yticks(np.linspace(0, 1, 11))

plt.tight_layout()
plt.show()

# 输出推荐策略
print("\n--- Recommendation Logic ---")
print("Extreme Time Priority (w=1.0): Best a =", best_a_for_weights[-1])
print("Extreme Cost Priority (w=0.0): Best a =", best_a_for_weights[0])
print("Balanced Strategy (w=0.5):     Best a =", best_a_for_weights[50])