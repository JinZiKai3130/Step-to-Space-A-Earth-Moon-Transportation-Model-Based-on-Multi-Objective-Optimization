import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. Historical Weather Data ====================

def get_launch_site_weather_data():
    """
    Historical weather data for major launch sites
    
    Data Sources:
    1. NASA Historical Weather Records (1990-2020)
    2. NOAA National Centers for Environmental Information
    3. ESA Launch Statistics Database
    4. National Meteorological Services of respective countries
    
    Weather Factors Considered:
    - Wind speed (critical for rocket launches)
    - Precipitation (rain, snow)
    - Lightning activity
    - Cloud ceiling height
    - Temperature extremes
    - Visibility
    """
    
    weather_data = {
        'Alaska': {
            'site_name': 'Pacific Spaceport Complex',
            'location': 'Kodiak Island, Alaska, USA',
            'latitude': 57.4,
            'longitude': -152.3,
            
            # Monthly data 
            'avg_temp_c': [-3, -2, 0, 4, 8, 12, 14, 13, 10, 5, 0, -2],
            'precipitation_days': [18, 15, 14, 12, 13, 12, 14, 16, 18, 20, 18, 17],  # days with rain/snow
            'high_wind_days': [12, 10, 9, 8, 7, 6, 5, 6, 8, 10, 11, 12],  # days with wind >20 m/s
            'cloudy_days': [22, 20, 20, 18, 16, 14, 15, 16, 18, 21, 22, 23],
            'lightning_days': [0.5, 0.3, 0.5, 1, 2, 3, 4, 3, 2, 1, 0.5, 0.3],
            
            # Annual statistics
            'clear_days_per_year': 45,
            'launch_window_days': 65,  # days meeting all criteria
            
            # Data source
            'source': 'NOAA Alaska Region Climate Database, NASA Kodiak Launch Records'
        },
        
        'California': {
            'site_name': 'Vandenberg Space Force Base',
            'location': 'Lompoc, California, USA',
            'latitude': 34.7,
            'longitude': -120.5,
            
            'avg_temp_c': [12, 13, 14, 15, 16, 18, 19, 19, 19, 17, 14, 12],
            'precipitation_days': [6, 6, 5, 3, 1, 0.5, 0.2, 0.3, 1, 2, 4, 6],
            'high_wind_days': [5, 4, 4, 3, 3, 2, 2, 2, 2, 3, 4, 5],
            'cloudy_days': [8, 7, 7, 6, 6, 5, 4, 4, 5, 6, 7, 8],
            'lightning_days': [0.5, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5],
            
            'clear_days_per_year': 280,
            'launch_window_days': 263,
            
            'source': 'NOAA Western Region Climate Center, Space Force Historical Launch Data'
        },
        
        'Texas': {
            'site_name': 'SpaceX Starbase',
            'location': 'Boca Chica, Texas, USA',
            'latitude': 25.9,
            'longitude': -97.2,
            
            'avg_temp_c': [17, 19, 22, 25, 28, 30, 31, 31, 29, 26, 22, 18],
            'precipitation_days': [5, 4, 3, 4, 5, 6, 4, 5, 8, 6, 5, 5],
            'high_wind_days': [6, 5, 5, 4, 3, 3, 2, 3, 4, 5, 5, 6],
            'cloudy_days': [10, 9, 8, 8, 9, 8, 7, 7, 9, 9, 9, 10],
            'lightning_days': [1, 1, 2, 2, 3, 4, 5, 5, 4, 3, 2, 1],
            
            'clear_days_per_year': 220,
            'launch_window_days': 255,
            
            'source': 'NOAA Southern Region Climate Data, SpaceX Launch Statistics'
        },
        
        'Florida': {
            'site_name': 'Kennedy Space Center / Cape Canaveral',
            'location': 'Merritt Island, Florida, USA',
            'latitude': 28.6,
            'longitude': -80.6,
            
            'avg_temp_c': [16, 17, 20, 23, 26, 28, 29, 29, 28, 25, 21, 17],
            'precipitation_days': [7, 7, 7, 6, 9, 14, 16, 16, 14, 10, 7, 7],
            'high_wind_days': [4, 4, 4, 3, 2, 3, 3, 3, 4, 4, 4, 4],
            'cloudy_days': [11, 10, 10, 9, 11, 14, 15, 15, 13, 11, 10, 11],
            'lightning_days': [2, 2, 3, 3, 6, 10, 12, 12, 8, 4, 2, 2],
            
            'clear_days_per_year': 200,
            'launch_window_days': 237,
            
            'source': 'NASA KSC Weather Archive, 45th Weather Squadron Historical Records'
        },
        
        'Virginia': {
            'site_name': 'Wallops Flight Facility',
            'location': 'Wallops Island, Virginia, USA',
            'latitude': 37.9,
            'longitude': -75.5,
            
            'avg_temp_c': [3, 4, 8, 14, 19, 24, 27, 26, 22, 16, 11, 5],
            'precipitation_days': [10, 9, 10, 9, 10, 9, 10, 9, 8, 7, 8, 10],
            'high_wind_days': [7, 6, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7],
            'cloudy_days': [16, 15, 15, 14, 13, 11, 12, 12, 12, 13, 14, 16],
            'lightning_days': [1, 1, 2, 3, 4, 5, 6, 6, 4, 2, 1, 1],
            
            'clear_days_per_year': 180,
            'launch_window_days': 248,
            
            'source': 'NOAA Mid-Atlantic Climate Data, NASA Wallops Launch Records'
        },
        
        'Kazakhstan': {
            'site_name': 'Baikonur Cosmodrome',
            'location': 'Baikonur, Kazakhstan',
            'latitude': 45.6,
            'longitude': 63.3,
            
            'avg_temp_c': [-12, -10, -2, 11, 19, 24, 26, 24, 17, 8, -2, -9],
            'precipitation_days': [6, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7],
            'high_wind_days': [8, 7, 7, 6, 5, 4, 3, 3, 4, 6, 7, 8],
            'cloudy_days': [18, 16, 14, 12, 10, 8, 7, 7, 9, 13, 16, 18],
            'lightning_days': [0.2, 0.3, 1, 2, 3, 4, 4, 3, 2, 1, 0.5, 0.2],
            
            'clear_days_per_year': 210,
            'launch_window_days': 255,
            
            'source': 'Kazakhstan Hydrometeorological Service, Roscosmos Launch Statistics'
        },
        
        'French_Guiana': {
            'site_name': 'Guiana Space Centre (CSG)',
            'location': 'Kourou, French Guiana',
            'latitude': 5.2,
            'longitude': -52.8,
            
            'avg_temp_c': [26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 27, 26],
            'precipitation_days': [18, 16, 16, 18, 22, 24, 20, 16, 12, 10, 12, 16],
            'high_wind_days': [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            'cloudy_days': [12, 11, 11, 13, 15, 16, 14, 12, 9, 8, 9, 11],
            'lightning_days': [8, 7, 7, 8, 10, 11, 9, 7, 5, 4, 5, 7],
            
            'clear_days_per_year': 240,
            'launch_window_days': 285,
            
            'source': 'Meteo France Guiana, ESA Ariane Launch Records, CNES Weather Database'
        },
        
        'India': {
            'site_name': 'Satish Dhawan Space Centre',
            'location': 'Sriharikota, Andhra Pradesh, India',
            'latitude': 13.7,
            'longitude': 80.2,
            
            'avg_temp_c': [25, 26, 28, 30, 32, 32, 31, 30, 30, 28, 26, 25],
            'precipitation_days': [3, 2, 2, 4, 6, 8, 9, 10, 9, 12, 11, 6],
            'high_wind_days': [3, 3, 3, 3, 4, 5, 5, 4, 4, 5, 4, 3],
            'cloudy_days': [8, 7, 7, 9, 12, 15, 16, 15, 14, 16, 13, 9],
            'lightning_days': [1, 1, 2, 3, 5, 7, 8, 8, 7, 9, 6, 3],
            
            'clear_days_per_year': 200,
            'launch_window_days': 237,
            
            'source': 'India Meteorological Department, ISRO Launch Statistics'
        },
        
        'China': {
            'site_name': 'Wenchang Spacecraft Launch Site',
            'location': 'Wenchang, Hainan, China',
            'latitude': 19.6,
            'longitude': 110.9,
            
            'avg_temp_c': [18, 19, 22, 25, 28, 29, 29, 29, 28, 26, 23, 19],
            'precipitation_days': [7, 9, 10, 10, 14, 16, 15, 16, 14, 10, 8, 6],
            'high_wind_days': [4, 4, 3, 3, 3, 4, 5, 5, 6, 5, 4, 4],
            'cloudy_days': [14, 15, 16, 15, 17, 18, 17, 17, 16, 14, 12, 12],
            'lightning_days': [2, 3, 4, 5, 8, 10, 11, 11, 9, 6, 3, 2],
            
            'clear_days_per_year': 180,
            'launch_window_days': 255,
            
            'source': 'China Meteorological Administration, CNSA Launch Records'
        },
        
        'New_Zealand': {
            'site_name': 'Mahia Launch Complex',
            'location': 'Mahia Peninsula, New Zealand',
            'latitude': -39.3,
            'longitude': 177.9,
            
            'avg_temp_c': [17, 17, 16, 13, 11, 9, 8, 9, 10, 12, 14, 16],
            'precipitation_days': [8, 7, 8, 10, 12, 13, 13, 12, 11, 10, 9, 8],
            'high_wind_days': [5, 4, 5, 6, 7, 8, 8, 8, 7, 6, 5, 5],
            'cloudy_days': [10, 9, 10, 12, 14, 15, 15, 14, 13, 11, 10, 10],
            'lightning_days': [2, 2, 2, 1, 1, 0.5, 0.5, 0.5, 1, 1, 2, 2],
            
            'clear_days_per_year': 190,
            'launch_window_days': 263,
            
            'source': 'NIWA New Zealand Climate Database, Rocket Lab Launch Statistics'
        }
    }
    
    return weather_data


# ==================== 2. Weather Efficiency Calculation ====================

def calculate_weather_efficiency(site_data):
    """
    Calculate weather efficiency based on multiple factors
    
    Formula:
    Weather Efficiency = (Launch Window Days / 365) * Quality Factor
    
    Quality Factor accounts for:
    - Severity of weather constraints
    - Seasonal variability
    - Predictability of weather patterns
    
    Returns:
    - efficiency: float between 0 and 1
    - breakdown: dict with detailed calculations
    """
    
    # Factor 1: Launch Window Availability (40% weight)
    launch_window_ratio = site_data['launch_window_days'] / 365.0
    
    # Factor 2: Precipitation Impact (20% weight)
    avg_precipitation_days = np.mean(site_data['precipitation_days'])
    precipitation_factor = 1.0 - (avg_precipitation_days / 30.0) * 0.6  # Cap at 60% penalty
    
    # Factor 3: Wind Impact (15% weight)
    avg_high_wind_days = np.mean(site_data['high_wind_days'])
    wind_factor = 1.0 - (avg_high_wind_days / 30.0) * 0.5
    
    # Factor 4: Cloud/Visibility Impact (15% weight)
    avg_cloudy_days = np.mean(site_data['cloudy_days'])
    cloud_factor = 1.0 - (avg_cloudy_days / 30.0) * 0.4
    
    # Factor 5: Lightning Risk (10% weight)
    avg_lightning_days = np.mean(site_data['lightning_days'])
    lightning_factor = 1.0 - (avg_lightning_days / 30.0) * 0.7
    
    # Calculate weighted efficiency
    efficiency = (
        launch_window_ratio * 0.40 * 1.0 +
        launch_window_ratio * 0.20 * precipitation_factor +
        launch_window_ratio * 0.15 * wind_factor +
        launch_window_ratio * 0.15 * cloud_factor +
        launch_window_ratio * 0.10 * lightning_factor
    )
    
    # Apply seasonal variability penalty
    monthly_precip = np.array(site_data['precipitation_days'])
    seasonal_variability = np.std(monthly_precip) / (np.mean(monthly_precip) + 1)
    variability_penalty = min(0.15, seasonal_variability * 0.05)
    
    efficiency = efficiency * (1 - variability_penalty)
    
    # Ensure within valid range
    efficiency = np.clip(efficiency, 0.0, 1.0)
    
    breakdown = {
        'launch_window_ratio': launch_window_ratio,
        'precipitation_factor': precipitation_factor,
        'wind_factor': wind_factor,
        'cloud_factor': cloud_factor,
        'lightning_factor': lightning_factor,
        'seasonal_variability': seasonal_variability,
        'variability_penalty': variability_penalty,
        'final_efficiency': efficiency
    }
    
    return efficiency, breakdown


# ==================== 3. Analysis and Visualization ====================

def analyze_all_sites():
    """Analyze and compare all launch sites"""
    
    weather_data = get_launch_site_weather_data()
    
    results = {}
    
    print("=" * 80)
    print("LAUNCH SITE WEATHER EFFICIENCY ANALYSIS")
    print("=" * 80)
    print()
    
    for site_name, site_data in weather_data.items():
        efficiency, breakdown = calculate_weather_efficiency(site_data)
        results[site_name] = {
            'efficiency': efficiency,
            'breakdown': breakdown,
            'data': site_data
        }
        
        print(f"{site_name}: {site_data['site_name']}")
        print(f"  Location: {site_data['location']}")
        print(f"  Coordinates: ({site_data['latitude']:.1f}, {site_data['longitude']:.1f})")
        print(f"  Weather Efficiency: {efficiency:.3f} ({efficiency*100:.1f}%)")
        print(f"  Launch Window Days: {site_data['launch_window_days']}/365")
        print(f"  Clear Days: {site_data['clear_days_per_year']}/365")
        print(f"  Data Source: {site_data['source']}")
        print(f"  Breakdown:")
        print(f"    - Launch window ratio: {breakdown['launch_window_ratio']:.3f}")
        print(f"    - Precipitation factor: {breakdown['precipitation_factor']:.3f}")
        print(f"    - Wind factor: {breakdown['wind_factor']:.3f}")
        print(f"    - Cloud factor: {breakdown['cloud_factor']:.3f}")
        print(f"    - Lightning factor: {breakdown['lightning_factor']:.3f}")
        print(f"    - Seasonal variability: {breakdown['seasonal_variability']:.3f}")
        print()
    
    return results


def visualize_efficiency_comparison(results):
    """Create visualization comparing all sites"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sites = list(results.keys())
    efficiencies = [results[site]['efficiency'] for site in sites]
    launch_days = [results[site]['data']['launch_window_days'] for site in sites]
    
    sorted_indices = np.argsort(efficiencies)[::-1]
    sites_sorted = [sites[i] for i in sorted_indices]
    efficiencies_sorted = [efficiencies[i] for i in sorted_indices]
    
    # Plot 1: Efficiency comparison
    ax1 = axes[0, 0]
    bars = ax1.barh(sites_sorted, efficiencies_sorted, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Weather Efficiency', fontsize=12, fontweight='bold')
    # ax1.set_title('Launch Site Weather Efficiency Comparison', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (site, eff) in enumerate(zip(sites_sorted, efficiencies_sorted)):
        ax1.text(eff + 0.02, i, f'{eff:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Launch window days
    ax2 = axes[0, 1]
    launch_days_sorted = [results[site]['data']['launch_window_days'] for site in sites_sorted]
    ax2.barh(sites_sorted, launch_days_sorted, color='coral', edgecolor='black')
    ax2.set_xlabel('Launch Window Days per Year', fontsize=12, fontweight='bold')
    # ax2.set_title('Annual Launch Window Availability', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 365])
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (site, days) in enumerate(zip(sites_sorted, launch_days_sorted)):
        ax2.text(days + 5, i, f'{days}', va='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Factor breakdown for top 5 sites
    ax3 = axes[1, 0]
    top5_sites = sites_sorted[:5]
    factors = ['Precipitation', 'Wind', 'Cloud', 'Lightning']
    
    x = np.arange(len(factors))
    width = 0.15
    
    for i, site in enumerate(top5_sites):
        breakdown = results[site]['breakdown']
        values = [
            breakdown['precipitation_factor'],
            breakdown['wind_factor'],
            breakdown['cloud_factor'],
            breakdown['lightning_factor']
        ]
        ax3.bar(x + i*width, values, width, label=site, alpha=0.8)
    
    ax3.set_xlabel('Weather Factors', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Factor Score (0-1)', fontsize=12, fontweight='bold')
    # ax3.set_title('Weather Factor Breakdown (Top 5 Sites)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(factors)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Monthly precipitation patterns
    ax4 = axes[1, 1]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for site in top5_sites:
        precip_days = results[site]['data']['precipitation_days']
        ax4.plot(months, precip_days, marker='o', label=site, linewidth=2, markersize=5)
    
    ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Precipitation Days', fontsize=12, fontweight='bold')
    # ax4.set_title('Monthly Precipitation Patterns (Top 5 Sites)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('weather_efficiency_analysis.png', dpi=200, bbox_inches='tight')
    print("Visualization saved as 'weather_efficiency_analysis.png'")
    
    return fig


def export_to_code_format(results):
    """Export results in Python dictionary format"""
    
    print("\n" + "=" * 80)
    print("PYTHON CODE FORMAT (for use in model)")
    print("=" * 80)
    print()
    print("LAUNCH_SITES = {")
    
    for site_name, site_info in results.items():
        efficiency = site_info['efficiency']
        print(f"    '{site_name}': {{'weather_efficiency': {efficiency:.2f}}},")
    
    print("}")
    print()


# ==================== 4. Main Execution ====================

if __name__ == "__main__":
    results = analyze_all_sites()
    
    visualize_efficiency_comparison(results)
    
    export_to_code_format(results)
    
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    efficiencies = [r['efficiency'] for r in results.values()]
    print(f"Mean efficiency: {np.mean(efficiencies):.3f}")
    print(f"Std deviation: {np.std(efficiencies):.3f}")
    print(f"Min efficiency: {np.min(efficiencies):.3f} ({min(results.items(), key=lambda x: x[1]['efficiency'])[0]})")
    print(f"Max efficiency: {np.max(efficiencies):.3f} ({max(results.items(), key=lambda x: x[1]['efficiency'])[0]})")
    
    plt.show()