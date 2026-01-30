import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class MoonColonyTransport:
    def __init__(self):
        self.total_mass = 1e8
        self.years_per_century = 100
        
        self.elevator_capacity_per_year = 537000
        self.electricity_cost_per_ton = 1900
        self.elevator_maintenance_cost = 1e8
        
        self.rocket_capacity_per_launch = 125
        self.launches_per_year = 700
        self.rocket_launch_cost = 8.5e8
        self.rocket_maintenance_cost = 1.69e10
        
        self.rocket_capacity_per_year = self.launches_per_year * self.rocket_capacity_per_launch
    
    def plan1_elevator_only(self) -> Tuple[float, float]:
        time_years = self.total_mass / self.elevator_capacity_per_year
        
        electricity_cost = self.total_mass * self.electricity_cost_per_ton
        maintenance_cost = time_years * self.elevator_maintenance_cost
        total_cost = electricity_cost + maintenance_cost
        
        return time_years, total_cost
    
    def plan2_rocket_only(self) -> Tuple[float, float]:
        total_launches_needed = self.total_mass / self.rocket_capacity_per_launch
        time_years = total_launches_needed / self.launches_per_year
        
        launch_cost = total_launches_needed * self.rocket_launch_cost
        maintenance_cost = time_years * self.rocket_maintenance_cost
        total_cost = launch_cost + maintenance_cost
        
        return time_years, total_cost
    
    def plan3_hybrid(self, elevator_ratio: float) -> Tuple[float, float, dict]:
        if not 0 <= elevator_ratio <= 1:
            raise ValueError("Elevator ratio must be between 0 and 1")
        
        elevator_mass = self.total_mass * elevator_ratio
        rocket_mass = self.total_mass * (1 - elevator_ratio)
        
        elevator_time = elevator_mass / self.elevator_capacity_per_year
        rocket_time = rocket_mass / self.rocket_capacity_per_year
        total_time = max(elevator_time, rocket_time)
        
        elevator_electricity_cost = elevator_mass * self.electricity_cost_per_ton
        elevator_maintenance_cost = total_time * self.elevator_maintenance_cost
        
        rocket_launches_needed = rocket_mass / self.rocket_capacity_per_launch
        rocket_launch_cost = rocket_launches_needed * self.rocket_launch_cost
        rocket_maintenance_cost = total_time * self.rocket_maintenance_cost
        
        total_cost = (elevator_electricity_cost + elevator_maintenance_cost + 
                      rocket_launch_cost + rocket_maintenance_cost)
        
        cost_breakdown = {
            'elevator_electricity': elevator_electricity_cost,
            'elevator_maintenance': elevator_maintenance_cost,
            'rocket_launch': rocket_launch_cost,
            'rocket_maintenance': rocket_maintenance_cost
        }
        
        return total_time, total_cost, cost_breakdown
    
    def find_optimal_ratio(self) -> float:
        Ve = self.elevator_capacity_per_year
        Vr = self.rocket_capacity_per_year
        optimal_ratio = Ve / (Ve + Vr)
        
        return optimal_ratio
    
    def analyze_scenarios(self, elevator_ratio=0.5):
        print("="*60)
        print("Moon Colony Transport Analysis")
        print("="*60)
        
        time1, cost1 = self.plan1_elevator_only()
        print(f"\nPlan 1: Space Elevator Only")
        print(f"  Transport Time: {time1:.2f} years")
        print(f"  Total Cost: ${cost1/1e12:.2f} trillion USD")
        print(f"  Annual Transport Capacity: {self.elevator_capacity_per_year:,.0f} tons/year")
        
        time2, cost2 = self.plan2_rocket_only()
        print(f"\nPlan 2: Rockets Only")
        print(f"  Transport Time: {time2:.2f} years")
        print(f"  Total Cost: ${cost2/1e12:.2f} trillion USD")
        print(f"  Annual Transport Capacity: {self.rocket_capacity_per_year:,.0f} tons/year")
        print(f"  Required Launches: {self.total_mass/self.rocket_capacity_per_launch:,.0f} launches")
        
        optimal_ratio = self.find_optimal_ratio()
        time3_opt, cost3_opt, breakdown_opt = self.plan3_hybrid(optimal_ratio)
        
        print(f"\nPlan 3: Hybrid Approach (Optimal Time Ratio, Elevator: {optimal_ratio:.2%})")
        print(f"  Transport Time: {time3_opt:.2f} years")
        print(f"  Total Cost: ${cost3_opt/1e12:.2f} trillion USD")
        print(f"  Cost Breakdown:")
        for key, value in breakdown_opt.items():
            print(f"    {key}: ${value/1e12:.4f} trillion USD")
        
        time3_user, cost3_user, breakdown_user = self.plan3_hybrid(elevator_ratio)
        print(f"\nPlan 3: Hybrid Approach (User-specified Ratio, Elevator: {elevator_ratio:.2%})")
        print(f"  Transport Time: {time3_user:.2f} years")
        print(f"  Total Cost: ${cost3_user/1e12:.2f} trillion USD")
        
        return {
            'plan1': {'time': time1, 'cost': cost1},
            'plan2': {'time': time2, 'cost': cost2},
            'plan3_optimal': {'time': time3_opt, 'cost': cost3_opt, 'ratio': optimal_ratio},
            'plan3_user': {'time': time3_user, 'cost': cost3_user, 'ratio': elevator_ratio}
        }
    
    def plot_sensitivity_analysis(self):
        ratios = np.linspace(0, 1, 101)
        times = []
        costs = []
        
        for ratio in ratios:
            time, cost, _ = self.plan3_hybrid(ratio)
            times.append(time)
            costs.append(cost)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.plot(ratios, times, 'b-', linewidth=2)
        ax1.set_xlabel('Elevator Transport Ratio', fontsize=12)
        ax1.set_ylabel('Transport Time (years)', fontsize=12)
        ax1.set_title('Transport Time vs Elevator Transport Ratio', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        optimal_ratio = self.find_optimal_ratio()
        optimal_time, _, _ = self.plan3_hybrid(optimal_ratio)
        ax1.plot(optimal_ratio, optimal_time, 'ro', markersize=10, label=f'Optimal Ratio: {optimal_ratio:.2f}')
        ax1.legend()
        
        ax2.plot(ratios, [c/1e12 for c in costs], 'r-', linewidth=2)
        ax2.set_xlabel('Elevator Transport Ratio', fontsize=12)
        ax2.set_ylabel('Total Cost (trillion USD)', fontsize=12)
        ax2.set_title('Total Cost vs Elevator Transport Ratio', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        optimal_cost = self.plan3_hybrid(optimal_ratio)[1]
        ax2.plot(optimal_ratio, optimal_cost/1e12, 'ro', markersize=10, label=f'Optimal Ratio: {optimal_ratio:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('transport_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        print("\nSensitivity Analysis Key Points:")
        for ratio in [0, 0.25, 0.5, 0.75, 1.0]:
            time, cost, _ = self.plan3_hybrid(ratio)
            print(f"  Elevator Ratio {ratio:.2f}: Time={time:.1f} years, Cost=${cost/1e12:.2f} trillion USD")

if __name__ == "__main__":
    transport = MoonColonyTransport()
    
    results = transport.analyze_scenarios(elevator_ratio=0.5)
    
    transport.plot_sensitivity_analysis()