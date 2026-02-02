# Step to Space: An Earth-Moon Transportation Model Based on Multi-Objective Optimization
This repository implements the multi-objective optimization model for Earth-Moon transportation systems proposed in the corresponding research paper, aiming to provide a practical logistics framework for 2050 lunar colonization plans.

## 1. Project Overview
### 1.1 Research Background
By 2050, Earth will face increasing environmental pressure, and establishing sustainable lunar colonies has become a critical strategic choice for humanity. However, the transportation of supplies between Earth and the Moon is a major obstacle to lunar colonization. Traditional rocket-based transportation systems have inherent limitations; thus, this study integrates space elevators to build an efficient, economical, and environmentally friendly hybrid transportation system.

### 1.2 Research Objectives
- Address the limitations of traditional rocket transportation for Earth-Moon logistics.
- Propose a hybrid transportation system combining space elevators and rockets.
- Verify the superiority of the hybrid strategy through multi-dimensional evaluation (time, cost, environment).
- Provide technical support for long-term space missions (e.g., Mars colonization, interstellar exploration).

## 2. Core Research Content
The study addresses four key problems in Earth-Moon transportation system design, with corresponding modeling and analysis:

### 2.1 Problem 1: Global Launch Capacity Prediction & Logistics Plan Comparison
- **Method**: Monte Carlo simulation (incorporating Technology Bottleneck Coefficient and Sigmoid-based infrastructure saturation limits).
- **Key Results**:
  - Median global launch capacity by 2050: 707.76 launches/year.
  - Comparative analysis of three logistics plans:
    - Elevator-only approach: Cost-efficient but with scalability limits.
    - Hybrid Synergy Strategy: Reduces 100-million-ton mission duration by 14% compared to Plan 1.

### 2.2 Problem 2: System Reliability & Pareto Optimal Allocation
- **Method**:
  - Model space debris impacts on space elevators as a Poisson process.
  - Model rocket success rates via meteorological weighted indices.
  - Solve optimal allocation ratio of elevators/rockets using Pareto optimality.
- **Key Results**:
  - Sensitivity analysis confirms high model robustness.
  - Elevator stability is the primary factor affecting mission duration and economic cost.

### 2.3 Problem 3: Lunar Colony Water Demand & In-Situ Resource Utilization (ISRU)
- **Method**: Quantify water demand considering water recycling; cost comparison of Earth-Moon transport vs. local production.
- **Key Results**:
  - Annual supplementary water demand for lunar colonies: 4.73×10⁵ tons (covering domestic, agricultural, industrial needs).
  - In-situ resource utilization (ISRU) is economically viable (local production cost << Earth-to-Moon transport cost), ensuring long-term colony sustainability.

### 2.4 Problem 4: Multi-Dimensional Evaluation via TOPSIS
- **Method**:
  - Build Environmental Impact metric using Analytic Hierarchy Process (AHP) (weighting greenhouse gases, black carbon, etc.).
  - Multi-dimensional evaluation (Time, Cost, Environment) via TOPSIS method.
- **Key Results**: Hybrid transportation strategy is the most robust solution, balancing rapid expansion and ecological preservation.

## 3. Key Technologies & Methods
| Category               | Technologies/Methods                                                                 |
|------------------------|--------------------------------------------------------------------------------------|
| Simulation             | Monte Carlo Simulation, Poisson Process                                              |
| Optimization           | Pareto Optimality, Multi-Objective Optimization                                      |
| Evaluation             | TOPSIS Method, Analytic Hierarchy Process (AHP)                                      |
| Resource Management    | In-Situ Resource Utilization (ISRU), Water Recycle Modeling                          |

## 4. Usage Guide
### 4.1 Environment Configuration
- Python 3.8+
- Dependencies:
  ```bash
  pip install numpy scipy pandas scikit-learn matplotlib seaborn
  ```

### 4.2 Run the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/JinZiKai3130/An-Earth-Moon-Transportation-Model
   cd An-Earth-Moon-Transportation-Model
   ```

2. You can verify the data we got through the code in the repository.

## 5. Key Findings
- We predict global launch capacity using a Monte Carlo simulation. By incorporating a Technology Bottleneck Coefficient and Sigmoid-based infrastructure saturation limits , we forecast a median capacity of 707.76 launches per year by 2050. Comparative analysis of three logistical plans demonstrates that while an elevator-only approach is cost-efficient, a hybrid synergy strategy reduces the 100-million-ton mission duration by 14\% compared to Plan 1.
- We enhance the model by accounting for system reliability. We model space debris impacts on the elevator as a Poisson process and rocket success rates through meteorological weighted indices. Applying Pareto optimality, we solve for the optimal allocation ratio between the elevator and rockets. Sensitivity analysis confirms high model robustness, revealing that elevator stability is the primary leading factor for mission duration and economic cost.
- In-situ resource utilization (ISRU) is essential for long-term lunar colony sustainability (water production in particular).
- The model can be extended to Mars colonization and interstellar exploration missions with minor adaptations.
