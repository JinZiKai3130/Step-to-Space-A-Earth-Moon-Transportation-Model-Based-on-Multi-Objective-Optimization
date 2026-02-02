import numpy as np
import matplotlib.pyplot as plt


def draw_comparison_radar():
    labels = np.array(['Time\nEfficiency', 'Cost\nSaving', 'Env.\nFriendly',
                       'Transport\nCapacity', 'Operational\nSafety'])
    num_vars = len(labels)

    plan1 = [6.5, 7.5, 9.8, 5.0, 9.5]
    plan2 = [4.0, 2.5, 3.0, 8.5, 5.0]
    plan3 = [9.5, 8.0, 7.5, 9.0, 8.5]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    plan1 += plan1[:1]
    plan2 += plan2[:1]
    plan3 += plan3[:1]
    angles += angles[:1]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True), dpi=100)

    configs = [
        {'data': plan1, 'color': '#2ca02c', 'title': 'Plan 1: Elevator Only', 'title_color': 'green'},
        {'data': plan2, 'color': '#d62728', 'title': 'Plan 2: Rocket Only', 'title_color': 'red'},
        {'data': plan3, 'color': '#1f77b4', 'title': 'Plan 3: Hybrid (Optimal)', 'title_color': '#1f77b4'}
    ]

    for i, cfg in enumerate(configs):
        ax = axs[i]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.rc('font', size=9)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)

        ax.set_rgrids([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=8)
        ax.set_ylim(0, 10)

        ax.plot(angles, cfg['data'], color=cfg['color'], linewidth=2)
        ax.fill(angles, cfg['data'], color=cfg['color'], alpha=0.25)

        ax.set_title(cfg['title'], y=1.1, fontsize=14, fontweight='bold', color=cfg['title_color'])

    plt.tight_layout()
    plt.savefig('radar_comparison_final.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_comparison_radar()