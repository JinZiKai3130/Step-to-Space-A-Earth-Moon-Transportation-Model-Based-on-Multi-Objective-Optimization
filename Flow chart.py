import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_integrated_flowchart():
    fig, ax = plt.subplots(figsize=(10, 15), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')

    COLOR_RED = '#E58E8E'
    COLOR_GREEN = '#B8D0C4'
    COLOR_BLUE = '#1A5F7A'
    TEXT_COLOR = '#333333'

    def add_node(text, x, y, fc, width=2.8, height=1.1, is_dark=False):
        shadow = patches.FancyBboxPatch(
            (x - width/2 + 0.05, y - height/2 - 0.05), width, height,
            boxstyle="round,pad=0.1", ec="none", fc='#CCCCCC', alpha=0.4
        )
        ax.add_patch(shadow)
        box = patches.FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1", ec="#444444", fc=fc, lw=1.2
        )
        ax.add_patch(box)
        t_color = 'white' if is_dark else TEXT_COLOR
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                color=t_color, fontweight='bold', linespacing=1.2)

    def add_arrow(start_coords, end_coords):
        ax.annotate('', xy=end_coords, xytext=start_coords,
                    arrowprops=dict(arrowstyle='-|>', color='#444444',
                                  lw=1.2, mutation_scale=15))

    add_node("divide the whole\nworkload into 2\ndivisions", 5, 17, COLOR_RED)
    add_node("Problem 1:\n3 plans", 2.5, 15, COLOR_GREEN)
    add_node("Problem 4:\nadd another standard:\npollution", 7.5, 15, COLOR_GREEN)
    add_node("only elevator system/\nonly rockets", 1.2, 13, COLOR_GREEN, width=2.2)
    add_node("hybrid plan", 3.8, 13, COLOR_GREEN, width=2.2)
    add_node("change the metrics\nof time, cost, pollution", 7.5, 13, COLOR_BLUE, is_dark=True)
    add_node("choose the best plan\nconsidering the cost\nand time", 2.5, 11, COLOR_BLUE, is_dark=True)
    add_node("P4 calculate\ncalculate the metrics\ntime,cost, pollution", 7.5, 11, COLOR_GREEN)
    add_node("Problem 2:\nadjust the weights of\n2 standards", 1.2, 9, COLOR_GREEN, width=2.2)
    add_node("Problem 3:\ncalculate the cost of\ntransportation", 3.8, 9, COLOR_GREEN, width=2.2)
    add_node("the environment\naffects on the earth", 7.5, 9, COLOR_GREEN)
    add_node("choose the best plan\nconsidering the cost\nand time", 1.2, 7, COLOR_RED, width=2.2)
    add_node("A better solution:\ngenerating water and\nother resources on\nthe moon", 3.8, 7, COLOR_RED, width=2.2)

    add_arrow((5, 16.4), (2.5, 15.6))
    add_arrow((5, 16.4), (7.5, 15.6))
    add_arrow((2.5, 14.4), (1.2, 13.6))
    add_arrow((2.5, 14.4), (3.8, 13.6))
    add_arrow((1.2, 12.4), (2.5, 11.6))
    add_arrow((3.8, 12.4), (2.5, 11.6))
    add_arrow((2.5, 10.4), (1.2, 9.6))
    add_arrow((2.5, 10.4), (3.8, 9.6))
    add_arrow((1.2, 8.4), (1.2, 7.6))
    add_arrow((3.8, 8.4), (3.8, 7.6))
    add_arrow((7.5, 14.4), (7.5, 13.6))
    add_arrow((7.5, 12.4), (7.5, 11.6))
    add_arrow((7.5, 10.4), (7.5, 9.6))

    plt.tight_layout()
    plt.savefig('logic_flowchart_v2.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_integrated_flowchart()