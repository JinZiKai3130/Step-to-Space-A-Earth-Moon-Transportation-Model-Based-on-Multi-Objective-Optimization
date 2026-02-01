import numpy as np
import matplotlib.pyplot as plt


def draw_comparison_radar():
    # 1. 设定维度标签 (必须与你图中的 5 个维度一致)
    labels = np.array(['Time\nEfficiency', 'Cost\nSaving', 'Env.\nFriendly',
                       'Transport\nCapacity', 'Operational\nSafety'])
    num_vars = len(labels)

    # 2. 设定各方案数据 (0-10分，根据图中视觉效果反推)
    # Plan 1: Elevator Only (绿色)
    plan1 = [6.5, 7.5, 9.8, 5.0, 9.5]
    # Plan 2: Rocket Only (红色)
    plan2 = [4.0, 2.5, 3.0, 8.5, 5.0]
    # Plan 3: Hybrid Optimal (蓝色)
    plan3 = [9.5, 8.0, 7.5, 9.0, 8.5]

    # 为了使雷达图闭合，需要首尾相连
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    plan1 += plan1[:1]
    plan2 += plan2[:1]
    plan3 += plan3[:1]
    angles += angles[:1]

    # 3. 创建画布 (1行3列的布局)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True), dpi=100)

    # 定义绘图配置 (颜色, 标题)
    configs = [
        {'data': plan1, 'color': '#2ca02c', 'title': 'Plan 1: Elevator Only', 'title_color': 'green'},
        {'data': plan2, 'color': '#d62728', 'title': 'Plan 2: Rocket Only', 'title_color': 'red'},
        {'data': plan3, 'color': '#1f77b4', 'title': 'Plan 3: Hybrid (Optimal)', 'title_color': '#1f77b4'}
    ]

    for i, cfg in enumerate(configs):
        ax = axs[i]
        # 设置起始角度在顶部
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # 绘制背景网格线
        plt.rc('font', size=9)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)

        # 绘制分值环 (2, 4, 6, 8, 10)
        ax.set_rgrids([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=8)
        ax.set_ylim(0, 10)

        # 绘图与填充
        ax.plot(angles, cfg['data'], color=cfg['color'], linewidth=2)
        ax.fill(angles, cfg['data'], color=cfg['color'], alpha=0.25)

        # 设置标题
        ax.set_title(cfg['title'], y=1.1, fontsize=14, fontweight='bold', color=cfg['title_color'])

    # 4. 调整布局并保存
    plt.tight_layout()
    plt.savefig('radar_comparison_final.png', bbox_inches='tight', dpi=300)
    print("雷达图已保存为: radar_comparison_final.png")
    plt.show()


if __name__ == "__main__":
    draw_comparison_radar()